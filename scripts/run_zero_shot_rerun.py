"""Re-run inference on truncated samples with a higher max_new_tokens limit.

Reads existing zero_shot_chunk*.parquet files to find truncated rows, re-fetches
problem text from candidates_deduped.parquet, reruns vLLM inference at the
official Qwen3.5 complex-problem limit (81920 tokens), then saves a merged
parquet that replaces truncated rows with fresh results.

Usage:
    python -u scripts/run_zero_shot_rerun.py \\
        --chunks_glob "data/results/zero_shot_chunk*.parquet" \\
        --output data/results/zero_shot_rerun.parquet
"""

from __future__ import annotations

import argparse
import glob
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

# Official Qwen3.5 recommendation for complex problems (math/competition benchmarks)
QWEN35_COMPLEX_MAX_TOKENS = 81920


def load_truncated_ids(chunks_glob: str) -> tuple[pd.DataFrame, list[str]]:
    """Load all chunk parquets, return (full_df, truncated_problem_ids)."""
    paths = sorted(glob.glob(chunks_glob))
    if not paths:
        raise FileNotFoundError(f"No files matched: {chunks_glob}")
    print(f"Loading {len(paths)} chunk file(s): {paths}", flush=True)
    dfs = [pd.read_parquet(p) for p in paths]
    full_df = pd.concat(dfs).reset_index(drop=True)
    truncated_ids = full_df.loc[full_df["truncated"], "problem_id"].tolist()
    print(f"Total rows: {len(full_df)}  |  Truncated: {len(truncated_ids)} ({len(truncated_ids)/len(full_df):.1%})", flush=True)
    return full_df, truncated_ids


def load_problems(candidates_path: str, problem_ids: list[str]) -> pd.DataFrame:
    """Return subset of candidates parquet for the given problem_ids."""
    cands = pd.read_parquet(candidates_path)
    subset = cands[cands["problem_id"].isin(set(problem_ids))].copy()
    subset = subset.set_index("problem_id").loc[problem_ids].reset_index()
    print(f"Loaded {len(subset)} problem rows from candidates.", flush=True)
    return subset


def build_prompt(problem: str, tokenizer) -> str:
    messages = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": problem},
    ]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True,
    )


def strip_thinking(text: str) -> str:
    tag = "</think>"
    idx = text.rfind(tag)
    if idx == -1:
        return text
    return text[idx + len(tag):].strip()


def run_inference(df: pd.DataFrame, model_name: str, max_new_tokens: int) -> tuple[list[str], list[str], list[bool]]:
    from vllm import LLM, SamplingParams

    hf_home = os.environ.get("HF_HOME", "data/hf_cache")
    print(f"Loading model {model_name} (download_dir={hf_home})...", flush=True)

    llm = LLM(
        model=model_name,
        dtype="bfloat16",
        download_dir=hf_home,
        trust_remote_code=True,
        max_model_len=max_new_tokens + 4096,
    )
    tokenizer = llm.get_tokenizer()

    # Official Qwen3 thinking-mode sampling params (from HF model card)
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
        print(f"  Warning: {n_trunc}/{len(predictions)} still hit max_new_tokens={max_new_tokens}", flush=True)
    else:
        print(f"  All {len(predictions)} outputs completed before token limit.", flush=True)

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
            "raw_output": raw_text,
            "score": score,
            "truncated": is_trunc,
        })
    return pd.DataFrame(records)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--chunks_glob", default="data/results/zero_shot_chunk*.parquet",
                        help="Glob pattern for existing chunk parquets")
    parser.add_argument("--candidates", default="data/processed/candidates_deduped.parquet")
    parser.add_argument("--model", default="Qwen/Qwen3.5-4B")
    parser.add_argument("--max_new_tokens", type=int, default=QWEN35_COMPLEX_MAX_TOKENS,
                        help=f"Default: {QWEN35_COMPLEX_MAX_TOKENS} (Qwen3.5 official complex-problem recommendation)")
    parser.add_argument("--output", default="data/results/zero_shot_rerun.parquet",
                        help="Output parquet for the fresh rerun rows (not merged)")
    parser.add_argument("--merged_output", default="data/results/zero_shot_merged.parquet",
                        help="Output parquet: original rows with truncated rows replaced by rerun results")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Cap on truncated rows to rerun — useful for smoke tests (e.g. --max_samples 1)")
    parser.add_argument("--chunk_id", type=int, default=0, help="0-indexed chunk of truncated rows to process")
    parser.add_argument("--n_chunks", type=int, default=1, help="Total number of parallel chunks")
    args = parser.parse_args()

    # Step 1: find truncated problem_ids
    full_df, truncated_ids = load_truncated_ids(args.chunks_glob)
    if not truncated_ids:
        print("No truncated rows found. Nothing to do.", flush=True)
        return

    # Step 2: slice to this chunk, then apply max_samples cap
    if args.n_chunks > 1:
        chunk_size = (len(truncated_ids) + args.n_chunks - 1) // args.n_chunks
        start = args.chunk_id * chunk_size
        end = min(start + chunk_size, len(truncated_ids))
        truncated_ids = truncated_ids[start:end]
        print(f"Chunk {args.chunk_id}/{args.n_chunks}: rows {start}–{end-1} ({len(truncated_ids)} samples)", flush=True)
    if args.max_samples is not None:
        truncated_ids = truncated_ids[:args.max_samples]
        print(f"Capped to {args.max_samples} sample(s) for smoke test.", flush=True)
    problems_df = load_problems(args.candidates, truncated_ids)

    # Step 3: run inference at higher token limit
    print(f"\nRe-running {len(problems_df)} truncated samples at max_new_tokens={args.max_new_tokens}...", flush=True)
    predictions, raw_outputs, truncated_flags = run_inference(problems_df, args.model, args.max_new_tokens)

    # Step 4: score
    print("Scoring predictions...", flush=True)
    rerun_df = score_predictions(problems_df, predictions, raw_outputs, truncated_flags)

    # Save raw rerun results
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    rerun_df.to_parquet(args.output, index=False)
    print(f"Saved rerun results ({len(rerun_df)} rows) to {args.output}", flush=True)

    # Summary
    old_trunc_acc = full_df.loc[full_df["truncated"], "score"].mean()
    new_trunc_acc = rerun_df["score"].mean()
    still_trunc = rerun_df["truncated"].sum()
    print(f"\n=== Rerun Summary (chunk {args.chunk_id}/{args.n_chunks}) ===", flush=True)
    print(f"  Truncated rows rerun:        {len(rerun_df)}", flush=True)
    print(f"  Still truncated at {args.max_new_tokens}: {still_trunc}", flush=True)
    print(f"  Accuracy before (truncated): {old_trunc_acc:.1%}", flush=True)
    print(f"  Accuracy after  (rerun):     {new_trunc_acc:.1%}", flush=True)

    # Step 5: merge — only when running single-chunk (n_chunks=1); otherwise use merge_rerun_chunks.py
    if args.n_chunks == 1:
        rerun_lookup = rerun_df.set_index("problem_id")
        rerun_ids = set(rerun_df["problem_id"])
        merged = full_df.copy()
        mask = merged["problem_id"].isin(rerun_ids)
        for col in ["pred_text", "raw_output", "score", "truncated"]:
            merged.loc[mask, col] = merged.loc[mask, "problem_id"].map(rerun_lookup[col]).values
        merged.to_parquet(args.merged_output, index=False)
        print(f"Saved merged results ({len(merged)} rows) to {args.merged_output}", flush=True)
        overall = merged["score"].mean()
        n_still_trunc = merged["truncated"].sum()
        by_source = merged.groupby("source")["score"].agg(["mean", "count"]).sort_values("mean", ascending=False)
        print("\n=== Merged Dataset Accuracy ===", flush=True)
        for src, (acc, n) in by_source.iterrows():
            print(f"  {src:<30} {acc:.1%}  (n={n})", flush=True)
        print(f"  {'OVERALL':<30} {overall:.1%}  (n={len(merged)})", flush=True)
        print(f"  Still-truncated:             {n_still_trunc}/{len(merged)} ({n_still_trunc/len(merged):.1%})", flush=True)
    else:
        print(f"Multi-chunk mode: run merge_rerun_chunks.py after all {args.n_chunks} chunks complete.", flush=True)


if __name__ == "__main__":
    main()
