"""Merge probe_v5_A and probe_v5_B shards into a single rollouts.parquet.

Verifies that the two shards cover disjoint problem_idx ranges and that each
problem has the expected number of rollouts before concatenating.

Usage:
    python3 scripts/merge_probe_shards.py \\
        --shards outputs/probe_v5_A outputs/probe_v5_B \\
        --out outputs/probe_v5/rollouts.parquet
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import pandas as pd


def load_shard(shard_dir: str) -> pd.DataFrame:
    d = Path(shard_dir)
    chunks = sorted(d.glob("rollouts_chunk_*.parquet"))
    if not chunks:
        raise FileNotFoundError(f"No rollouts_chunk_*.parquet in {shard_dir}")
    dfs = [pd.read_parquet(c) for c in chunks]
    df = pd.concat(dfs, ignore_index=True)
    print(f"{shard_dir}: {len(chunks)} chunks, {len(df)} rollouts, "
          f"problem_idx range [{df['problem_idx'].min()}, {df['problem_idx'].max()}]")
    return df


def sanity_check(df: pd.DataFrame, name: str) -> None:
    total = len(df)
    # Count degenerate rollouts
    degen = sum(
        1 for _, row in df.iterrows()
        if any(str(row.get(c, "") or "").count("!") > 50
               for c in ["phase1_text", "phase2_text", "phase1b_text"])
    )
    has_code = sum(1 for _, row in df.iterrows() if row.get("code"))
    interrupted = sum(1 for _, row in df.iterrows() if row.get("interrupted"))
    print(f"  [{name}] total={total}, degen={degen} ({100*degen/total:.1f}%), "
          f"has_code={has_code} ({100*has_code/total:.1f}%), "
          f"interrupted={interrupted} ({100*interrupted/total:.1f}%)")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--shards", nargs="+", required=True, help="Shard directories")
    p.add_argument("--out", required=True, help="Output parquet path")
    args = p.parse_args()

    dfs = []
    seen_pids: set[int] = set()
    for shard in args.shards:
        df = load_shard(shard)
        sanity_check(df, os.path.basename(shard))
        pids = set(df["problem_idx"].unique())
        overlap = pids & seen_pids
        if overlap:
            raise ValueError(f"Overlapping problem_idx between shards: {sorted(overlap)[:5]}...")
        seen_pids |= pids
        dfs.append(df)

    merged = pd.concat(dfs, ignore_index=True)
    merged = merged.sort_values(["problem_idx", "rollout_idx"]).reset_index(drop=True)

    print(f"\nMerged: {len(merged)} rollouts, {merged['problem_idx'].nunique()} unique problems")
    sanity_check(merged, "MERGED")

    # Rollouts per problem
    rpp = merged.groupby("problem_idx").size()
    print(f"Rollouts per problem: min={rpp.min()}, max={rpp.max()}, "
          f"mean={rpp.mean():.1f}, mode={rpp.mode().iloc[0]}")
    if rpp.min() != rpp.max():
        print(f"  WARNING: variable rollouts/problem. Problems with non-mode counts: "
              f"{rpp[rpp != rpp.mode().iloc[0]].to_dict()}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    merged.to_parquet(args.out, index=False)
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
