"""Merge zero_shot_rerun_chunk*.parquet outputs with the original zero_shot_chunk*.parquet baseline.

Run after all N_CHUNKS rerun jobs complete:
    python scripts/merge_rerun_chunks.py
"""

from __future__ import annotations

import glob
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).parent.parent


def main() -> None:
    # Load original baseline
    orig_paths = sorted(glob.glob(str(ROOT / "data/results/zero_shot_chunk*.parquet")))
    if not orig_paths:
        sys.exit("No zero_shot_chunk*.parquet found.")
    full_df = pd.concat([pd.read_parquet(p) for p in orig_paths]).reset_index(drop=True)
    print(f"Original baseline: {len(full_df)} rows, {full_df['truncated'].sum()} truncated")

    # Load all rerun chunks
    rerun_paths = sorted(glob.glob(str(ROOT / "data/results/zero_shot_rerun_chunk*.parquet")))
    if not rerun_paths:
        sys.exit("No zero_shot_rerun_chunk*.parquet found. Run the rerun jobs first.")
    rerun_df = pd.concat([pd.read_parquet(p) for p in rerun_paths]).reset_index(drop=True)
    print(f"Rerun chunks ({len(rerun_paths)} files): {len(rerun_df)} rows")

    # Check coverage
    expected_trunc = set(full_df.loc[full_df["truncated"], "problem_id"])
    got = set(rerun_df["problem_id"])
    missing = expected_trunc - got
    if missing:
        print(f"WARNING: {len(missing)} truncated problem_ids not covered by rerun chunks.")
        print(f"  Missing chunk IDs may still be running.")

    # Merge
    rerun_lookup = rerun_df.set_index("problem_id")
    rerun_ids = got & expected_trunc
    merged = full_df.copy()
    mask = merged["problem_id"].isin(rerun_ids)
    for col in ["pred_text", "raw_output", "score", "truncated"]:
        merged.loc[mask, col] = merged.loc[mask, "problem_id"].map(rerun_lookup[col]).values

    out = ROOT / "data/results/zero_shot_merged.parquet"
    merged.to_parquet(out, index=False)
    print(f"Saved merged results to {out}")

    # Summary
    overall = merged["score"].mean()
    n_still_trunc = merged["truncated"].sum()
    old_acc = full_df["score"].mean()
    rerun_acc = rerun_df["score"].mean()
    old_trunc_acc = full_df.loc[full_df["truncated"], "score"].mean()

    print(f"\n=== Accuracy comparison ===")
    print(f"  Original baseline (all):    {old_acc:.1%}  (n={len(full_df)})")
    print(f"  Original truncated only:    {old_trunc_acc:.1%}  (n={full_df['truncated'].sum()})")
    print(f"  Rerun (truncated, new):     {rerun_acc:.1%}  (n={len(rerun_df)})")
    print(f"  Merged (final):             {overall:.1%}  (n={len(merged)})")
    print(f"  Still truncated at 81920:   {n_still_trunc}/{len(merged)} ({n_still_trunc/len(merged):.1%})")

    print("\n=== By source ===")
    by_source = merged.groupby("source")["score"].agg(["mean", "count"]).sort_values("mean", ascending=False)
    for src, (acc, n) in by_source.iterrows():
        print(f"  {src:<30} {acc:.1%}  (n={n})")


if __name__ == "__main__":
    main()
