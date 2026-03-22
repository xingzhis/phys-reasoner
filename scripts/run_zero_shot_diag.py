"""Zero-shot baseline: run a model on stratified sample, score with rule-tier verifier.

Uses vLLM for fast batch inference. Thinking mode enabled (Qwen3 default; matches
training/eval conditions).

Usage:
    python -u scripts/run_zero_shot.py \\
        --model Qwen/Qwen3.5-4B \\
        --n_per_tier 200 \\
        --output data/results/zero_shot_scores.parquet
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from phys_reasoner.training.reward import compute_score

_SYSTEM_PROMPT = (
    "Solve the physics problem step by step. Put your final answer in \\boxed{}."
)


def load_sample(parquet_path: str, n_per_tier: int, seed: int = 42,
                chunk_id: int = 0, n_chunks: int = 1) -> pd.DataFrame:
    df = pd.read_parquet(parquet_path)
    parts = []
    for _, grp in df.groupby(["source", "answer_type"]):
        parts.append(grp.sample(min(len(grp), n_per_tier), random_state=seed))
    sampled = pd.concat(parts).reset_index(drop=True)
    if n_chunks > 1:
        chunk_size = (len(sampled) + n_chunks - 1) // n_chunks  # ceiling div
        start = chunk_id * chunk_size
        end = min(start + chunk_size, len(sampled))
        sampled = sampled.iloc[start:end].reset_index(drop=True)
    return sampled


def build_prompt(problem: str, tokenizer) -> str:
    messages = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": problem},
    ]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True,  # Qwen3 thinking mode: natural mode, matches eval/training
    )


def strip_thinking(text: str) -> str:
    """Return only the text after </think>. If no </think> tag, return full text."""
    tag = "</think>"
    idx = text.rfind(tag)
    if idx == -1:
        return text
    return text[idx + len(tag):].strip()


def run_inference(df: pd.DataFrame, model_name: str, max_new_tokens: int) -> tuple[list[str], list[str], list[bool]]:
    """Run vLLM batch inference. Returns (predictions, raw_outputs, truncated_flags).

    predictions contain only the post-thinking text (after </think>), so that
    intermediate \\boxed{} expressions inside the thinking trace don't confuse
    the verifier. raw_outputs contain the full model output including the thinking
    trace, saved for diagnostics.
    """
    from vllm import LLM, SamplingParams

    hf_home = os.environ.get("HF_HOME", "data/hf_cache")
    print(f"Loading model {model_name} (download_dir={hf_home})...", flush=True)

    llm = LLM(
        model=model_name,
        dtype="bfloat16",
        download_dir=hf_home,
        trust_remote_code=True,
        max_model_len=max_new_tokens + 4096,  # context = prompt budget + output budget
    )
    tokenizer = llm.get_tokenizer()

    # Official Qwen3 thinking-mode sampling params (from HF model card)
    # max_tokens set to full budget — vLLM defaults to 16 if omitted, not unlimited
    # Model stops naturally at EOS well before this limit (diagnostic intent)
    sampling_params = SamplingParams(
        temperature=1.0,
        top_p=0.95,
        top_k=20,
        presence_penalty=1.5,
        max_tokens=max_new_tokens,
    )

    print(f"Building {len(df)} prompts...", flush=True)
    prompts = [build_prompt(row["problem"], tokenizer) for _, row in df.iterrows()]

    print(f"Running inference (max_new_tokens={max_new_tokens})...", flush=True)
    outputs = llm.generate(prompts, sampling_params)

    predictions: list[str] = []
    raw_outputs: list[str] = []
    truncated: list[bool] = []
    for out in outputs:
        full_text = out.outputs[0].text
        n_tokens = len(out.outputs[0].token_ids)
        raw_outputs.append(full_text)
        predictions.append(strip_thinking(full_text))
        truncated.append(n_tokens >= max_new_tokens)

    n_trunc = sum(truncated)
    if n_trunc:
        print(f"  Warning: {n_trunc}/{len(predictions)} outputs hit max_new_tokens={max_new_tokens} (truncated)", flush=True)

    return predictions, raw_outputs, truncated


def score_predictions(df: pd.DataFrame, predictions: list[str], raw_outputs: list[str], truncated: list[bool]) -> pd.DataFrame:
    records = []
    for (_, row), pred_text, raw_text, is_trunc in zip(df.iterrows(), predictions, raw_outputs, truncated):
        gold = row["answer"]
        if isinstance(gold, str):
            try:
                parsed = json.loads(gold)
                if isinstance(parsed, list):
                    gold = parsed
            except Exception:
                pass

        score = compute_score(
            solution_str=pred_text,
            ground_truth=gold,
            data_source=str(row.get("source", "")),
            answer_type=str(row.get("answer_type", "unknown")),
            unit=str(row.get("unit") or ""),
            tolerance=0.05,
            xverify_judge=None,
            problem=str(row.get("problem", "")),
        )
        records.append({
            "problem_id": row["problem_id"],
            "source": row["source"],
            "answer_type": row["answer_type"],
            "gold_answer": row["answer"],
            "pred_text": pred_text,       # post-think text only (used for scoring)
            "raw_output": raw_text,        # full output including <think>...</think>
            "score": score,
            "truncated": is_trunc,
        })
    return pd.DataFrame(records)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/processed/candidates_deduped.parquet")
    parser.add_argument("--model", default="Qwen/Qwen3.5-4B")
    parser.add_argument("--n_per_tier", type=int, default=200)
    parser.add_argument("--max_new_tokens", type=int, default=61440)  # → max_model_len=65536
    parser.add_argument("--output", default="data/results/zero_shot_scores.parquet")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--chunk_id", type=int, default=0, help="0-indexed chunk to process")
    parser.add_argument("--n_chunks", type=int, default=1, help="total number of chunks")
    parser.add_argument("--max_samples", type=int, default=None, help="hard cap on total rows (for quick smoke tests)")
    args = parser.parse_args()

    print(f"Loading sample from {args.input} (n_per_tier={args.n_per_tier}, chunk={args.chunk_id}/{args.n_chunks})...", flush=True)
    df = load_sample(args.input, args.n_per_tier, seed=args.seed,
                     chunk_id=args.chunk_id, n_chunks=args.n_chunks)
    if args.max_samples is not None:
        df = df.head(args.max_samples).reset_index(drop=True)
    print(f"  Sample size: {len(df)} rows", flush=True)

    print(f"Running inference with {args.model}...", flush=True)
    predictions, raw_outputs, truncated = run_inference(df, args.model, args.max_new_tokens)

    print("Scoring predictions...", flush=True)
    results_df = score_predictions(df, predictions, raw_outputs, truncated)

    # Per-source accuracy
    print("\n=== Zero-shot Accuracy (rule tier only, thinking enabled) ===", flush=True)
    overall = results_df["score"].mean()
    n_trunc = results_df["truncated"].sum()
    by_source = results_df.groupby("source")["score"].mean().sort_values(ascending=False)
    for src, acc in by_source.items():
        n = (results_df["source"] == src).sum()
        print(f"  {src:<30} {acc:.1%}  (n={n})", flush=True)
    print(f"  {'OVERALL':<30} {overall:.1%}  (n={len(results_df)})", flush=True)
    print(f"  Truncated outputs: {n_trunc}/{len(results_df)} ({n_trunc/len(results_df):.1%})", flush=True)

    # Per-sample token stats (diagnostic)
    print("\n=== Token usage per sample ===", flush=True)
    token_counts = []
    for _, row in results_df.iterrows():
        raw = row["raw_output"]
        has_think_end = "</think>" in raw
        stripped = raw[raw.rfind("</think>") + len("</think>"):].strip() if has_think_end else raw
        has_boxed = r"\boxed{" in stripped
        # token count stored in truncated col is a bool; recompute from raw_output length proxy
        print(
            f"  src={str(row['source']):<20}  type={str(row['answer_type']):<12}"
            f"  trunc={row['truncated']}  has_think_end={has_think_end}  has_boxed={has_boxed}"
            f"  score={row['score']}  preview={repr(stripped[:80])}",
            flush=True,
        )

    # Aggregate token stats from raw_output character lengths as proxy
    # (actual token counts would need the tokenizer; use truncated flag as proxy)
    n_has_think = sum(1 for r in results_df["raw_output"] if "</think>" in r)
    n_has_boxed = sum(
        1 for r in results_df["raw_output"]
        if r"\boxed{" in (r[r.rfind("</think>") + len("</think>"):].strip() if "</think>" in r else r)
    )
    print(f"\n=== Diagnostic Summary ===", flush=True)
    print(f"  has </think>  : {n_has_think}/{len(results_df)}", flush=True)
    print(f"  has \\boxed{{}} : {n_has_boxed}/{len(results_df)}", flush=True)
    print(f"  truncated     : {n_trunc}/{len(results_df)}", flush=True)
    print(f"  score mean    : {overall:.1%}", flush=True)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    results_df.to_parquet(args.output, index=False)
    print(f"\nSaved results to {args.output}", flush=True)


if __name__ == "__main__":
    main()