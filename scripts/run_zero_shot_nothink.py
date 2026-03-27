"""Zero-shot baseline with thinking mode DISABLED (Qwen3 non-thinking mode).

Same sampling and scoring logic as run_zero_shot.py, but uses enable_thinking=False
and Qwen3's recommended non-thinking sampling params (temperature=0.7, top_p=0.8,
top_k=20, no presence_penalty).  Outputs are shorter — no <think> trace — so
max_new_tokens default is lower (8192).

Usage:
    python -u scripts/run_zero_shot_nothink.py \\
        --model Qwen/Qwen3.5-4B \\
        --n_per_tier 200 \\
        --output data/results/zero_shot_nothink_scores.parquet
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import pandas as pd

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
        chunk_size = (len(sampled) + n_chunks - 1) // n_chunks
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
        enable_thinking=False,  # non-thinking mode
    )


def run_inference(df: pd.DataFrame, model_name: str, max_new_tokens: int,
                  gpu_memory_utilization: float = 0.9) -> tuple[list[str], list[bool]]:
    """Run vLLM batch inference. Returns (predictions, truncated_flags).

    In non-thinking mode there is no <think> trace so raw output == prediction.
    """
    from vllm import LLM, SamplingParams

    hf_home = os.environ.get("HF_HOME", "data/hf_cache")
    print(f"Loading model {model_name} (download_dir={hf_home})...", flush=True)

    llm = LLM(
        model=model_name,
        dtype="bfloat16",
        download_dir=hf_home,
        trust_remote_code=True,
        max_model_len=max_new_tokens + 4096,
        gpu_memory_utilization=gpu_memory_utilization,
    )
    tokenizer = llm.get_tokenizer()

    # Official Qwen3 non-thinking-mode sampling params (from HF model card)
    sampling_params = SamplingParams(
        temperature=0.7,
        top_p=0.8,
        top_k=20,
        max_tokens=max_new_tokens,
    )

    print(f"Building {len(df)} prompts...", flush=True)
    prompts = [build_prompt(row["problem"], tokenizer) for _, row in df.iterrows()]

    print(f"Running inference (thinking=OFF, max_new_tokens={max_new_tokens})...", flush=True)
    outputs = llm.generate(prompts, sampling_params)

    # Sanity-check: print first output so the user can confirm no <think> tag
    if outputs:
        first_text = outputs[0].outputs[0].text
        print("\n--- FIRST OUTPUT (sanity check, thinking should be OFF) ---", flush=True)
        print(first_text[:1000], flush=True)
        if "<think>" in first_text:
            print("WARNING: <think> tag found in output — thinking may still be on!", flush=True)
        else:
            print("OK: no <think> tag detected.", flush=True)
        print("--- END FIRST OUTPUT ---\n", flush=True)

    predictions: list[str] = []
    truncated: list[bool] = []
    token_counts: list[int] = []
    for out in outputs:
        full_text = out.outputs[0].text
        n_tokens = len(out.outputs[0].token_ids)
        predictions.append(full_text)
        truncated.append(n_tokens >= max_new_tokens)
        token_counts.append(n_tokens)

    n_trunc = sum(truncated)
    if n_trunc:
        print(f"  Warning: {n_trunc}/{len(predictions)} outputs hit max_new_tokens={max_new_tokens} (truncated)", flush=True)
    print(f"  Token counts — min: {min(token_counts)}  max: {max(token_counts)}  mean: {sum(token_counts)/len(token_counts):.0f}", flush=True)

    return predictions, truncated, token_counts


def score_predictions(df: pd.DataFrame, predictions: list[str], truncated: list[bool], token_counts: list[int]) -> pd.DataFrame:
    records = []
    for (_, row), pred_text, is_trunc, n_tok in zip(df.iterrows(), predictions, truncated, token_counts):
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
            "pred_text": pred_text,
            "raw_output": pred_text,   # same as pred_text — no thinking trace
            "n_tokens": n_tok,
            "score": score,
            "truncated": is_trunc,
        })
    return pd.DataFrame(records)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/processed/candidates_deduped.parquet")
    parser.add_argument("--model", default="Qwen/Qwen3.5-4B")
    parser.add_argument("--n_per_tier", type=int, default=200)
    parser.add_argument("--max_new_tokens", type=int, default=8192)
    parser.add_argument("--output", default="data/results/zero_shot_nothink_scores.parquet")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--chunk_id", type=int, default=0)
    parser.add_argument("--n_chunks", type=int, default=1)
    parser.add_argument("--max_samples", type=int, default=None, help="hard cap on rows, useful for smoke tests")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.9,
                        help="vLLM GPU memory fraction (default 0.9; reduce if OOM on shared nodes)")
    args = parser.parse_args()

    print(f"Loading sample from {args.input} (n_per_tier={args.n_per_tier}, chunk={args.chunk_id}/{args.n_chunks})...", flush=True)
    df = load_sample(args.input, args.n_per_tier, seed=args.seed,
                     chunk_id=args.chunk_id, n_chunks=args.n_chunks)
    if args.max_samples is not None:
        df = df.head(args.max_samples).reset_index(drop=True)
        print(f"  Capped to {args.max_samples} sample(s) for smoke test.", flush=True)
    print(f"  Sample size: {len(df)} rows", flush=True)

    print(f"Running inference with {args.model} (thinking OFF)...", flush=True)
    predictions, truncated, token_counts = run_inference(df, args.model, args.max_new_tokens,
                                                          gpu_memory_utilization=args.gpu_memory_utilization)

    print("Scoring predictions...", flush=True)
    results_df = score_predictions(df, predictions, truncated, token_counts)

    print("\n=== Zero-shot Accuracy (rule tier only, thinking DISABLED) ===", flush=True)
    overall = results_df["score"].mean()
    n_trunc = results_df["truncated"].sum()
    by_source = results_df.groupby("source")["score"].agg(["mean", "count"]).sort_values("mean", ascending=False)
    for src, (acc, n) in by_source.iterrows():
        print(f"  {src:<30} {acc:.1%}  (n={n})", flush=True)
    print(f"  {'OVERALL':<30} {overall:.1%}  (n={len(results_df)})", flush=True)
    print(f"  Truncated outputs: {n_trunc}/{len(results_df)} ({n_trunc/len(results_df):.1%})", flush=True)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    results_df.to_parquet(args.output, index=False)
    print(f"\nSaved results to {args.output}", flush=True)


if __name__ == "__main__":
    main()
