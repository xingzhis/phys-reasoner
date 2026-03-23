"""Analyze zero-shot baseline results: rule vs xVerify-3B vs xVerify-7B.

Merges all 8 chunks, computes accuracy breakdowns by source / answer_type,
and handles truncated-vs-non-truncated comparison.

Usage (after all xVerify rescores are done):
    python scripts/analyze_zero_shot.py \
        --chunks data/results/zero_shot_chunk{0..7}.parquet \
        --rescore_3b data/results/rescore_3b_all8.parquet \
        --rescore_7b data/results/rescore_7b_all8.parquet

    # If only partial rescore is available (e.g. chunks 0-3 only):
    python scripts/analyze_zero_shot.py \
        --chunks data/results/zero_shot_chunk{0..7}.parquet \
        --rescore_3b data/results/rescore_3b.parquet \
        --rescore_7b data/results/rescore_7b.parquet
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import numpy as np


def load_chunks(paths: list[str]) -> pd.DataFrame:
    dfs = [pd.read_parquet(p) for p in paths]
    df = pd.concat(dfs, ignore_index=True)
    print(f"Loaded {len(df)} rows from {len(paths)} chunk file(s).")
    return df


def simplify_type(t: str) -> str:
    if str(t).startswith("["):
        try:
            parts = json.loads(t)
            return f"multi({len(parts)})-{parts[0]}"
        except Exception:
            return "multi"
    return str(t)


def fmt(val: float, n: int | None = None) -> str:
    s = f"{val:.1%}"
    if n is not None:
        s += f" ({int(round(val*n))}/{n})"
    return s


def report(df_all: pd.DataFrame, df_3b: pd.DataFrame | None, df_7b: pd.DataFrame | None) -> None:
    """Print full comparison report."""

    # Build unified dataframe: start from all 8 chunks (rule scores)
    df = df_all.copy()
    n_total = len(df)
    n_trunc = int(df["truncated"].sum())

    # Merge xVerify scores if available
    has_3b = df_3b is not None
    has_7b = df_7b is not None

    if has_3b:
        xv3b = df_3b[["problem_id", "score_xverify"]].rename(columns={"score_xverify": "score_3b"})
        df = df.merge(xv3b, on="problem_id", how="left")

    if has_7b:
        xv7b = df_7b[["problem_id", "score_xverify"]].rename(columns={"score_xverify": "score_7b"})
        df = df.merge(xv7b, on="problem_id", how="left")

    # --------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("ZERO-SHOT BASELINE — Full Analysis")
    print("=" * 70)

    # Truncation summary
    print(f"\n## Truncation Summary (max_new_tokens=32768)")
    print(f"  Total samples:   {n_total}")
    print(f"  Truncated:       {n_trunc} ({n_trunc/n_total:.1%})")
    print(f"  Not truncated:   {n_total - n_trunc} ({(n_total-n_trunc)/n_total:.1%})")

    # --------------------------------------------------------------------------
    def table_section(subset_df: pd.DataFrame, label: str) -> None:
        n = len(subset_df)
        rule = subset_df["score"].mean()
        cols = ["Rule-only"]
        vals = [rule]
        ns   = [n]
        if has_3b and "score_3b" in subset_df.columns:
            v3 = subset_df["score_3b"].dropna()
            cols.append(f"Rule+xVerify-3B (n={len(v3)})")
            vals.append(v3.mean() if len(v3) else float("nan"))
            ns.append(len(v3))
        if has_7b and "score_7b" in subset_df.columns:
            v7 = subset_df["score_7b"].dropna()
            cols.append(f"Rule+xVerify-7B (n={len(v7)})")
            vals.append(v7.mean() if len(v7) else float("nan"))
            ns.append(len(v7))

        print(f"\n  ### {label} (n={n})")
        for col, val, ni in zip(cols, vals, ns):
            print(f"    {col:<35} {val:.4f}  ({int(round(val*ni))}/{ni})")

    # --------------------------------------------------------------------------
    print("\n## Overall Accuracy")
    table_section(df, "All samples")

    non_trunc = df[~df["truncated"]]
    trunc_df  = df[df["truncated"]]
    table_section(non_trunc, "Non-truncated only")
    table_section(trunc_df,  "Truncated only")

    # --------------------------------------------------------------------------
    print("\n## By Source")
    for src in sorted(df["source"].unique()):
        sub = df[df["source"] == src]
        table_section(sub, f"source={src}")

    # --------------------------------------------------------------------------
    print("\n## By Answer Type")
    df["atype_simple"] = df["answer_type"].apply(simplify_type)
    type_counts = df["atype_simple"].value_counts()
    for atype in type_counts.index:
        sub = df[df["atype_simple"] == atype]
        table_section(sub, f"type={atype}")

    # --------------------------------------------------------------------------
    # Summary table
    print("\n## Summary Table (truncated vs non-truncated)")
    print(f"\n{'Subset':<22} {'n':>5} {'Rule':>8}", end="")
    if has_3b:
        print(f" {'3B-xV':>8}", end="")
    if has_7b:
        print(f" {'7B-xV':>8}", end="")
    print()
    print("-" * (22 + 6 + 9 + (9 if has_3b else 0) + (9 if has_7b else 0)))

    for label, sub in [
        ("ALL", df),
        ("non-truncated", non_trunc),
        ("truncated", trunc_df),
    ]:
        n = len(sub)
        rule = sub["score"].mean()
        row = f"{label:<22} {n:>5} {rule:>8.4f}"
        if has_3b and "score_3b" in sub.columns:
            v = sub["score_3b"].dropna().mean()
            row += f" {v:>8.4f}"
        if has_7b and "score_7b" in sub.columns:
            v = sub["score_7b"].dropna().mean()
            row += f" {v:>8.4f}"
        print(row)

    # Per-source summary table
    print(f"\n{'Source':<22} {'n':>5} {'Rule':>8}", end="")
    if has_3b:
        print(f" {'3B-xV':>8}", end="")
    if has_7b:
        print(f" {'7B-xV':>8}", end="")
    print()
    print("-" * (22 + 6 + 9 + (9 if has_3b else 0) + (9 if has_7b else 0)))

    for src in sorted(df["source"].unique()):
        sub = df[df["source"] == src]
        n = len(sub)
        rule = sub["score"].mean()
        row = f"{src:<22} {n:>5} {rule:>8.4f}"
        if has_3b and "score_3b" in sub.columns:
            v = sub["score_3b"].dropna().mean()
            row += f" {v:>8.4f}"
        if has_7b and "score_7b" in sub.columns:
            v = sub["score_7b"].dropna().mean()
            row += f" {v:>8.4f}"
        print(row)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--chunks", nargs="+",
                        default=[f"data/results/zero_shot_chunk{i}.parquet" for i in range(8)],
                        help="All zero_shot_chunk*.parquet files")
    parser.add_argument("--rescore_3b", default=None,
                        help="rescore parquet with 3B xVerify (score_xverify column)")
    parser.add_argument("--rescore_7b", default=None,
                        help="rescore parquet with 7B xVerify (score_xverify column)")
    args = parser.parse_args()

    df_all = load_chunks(args.chunks)

    df_3b = None
    if args.rescore_3b and Path(args.rescore_3b).exists():
        df_3b = pd.read_parquet(args.rescore_3b)
        print(f"Loaded 3B rescore: {len(df_3b)} rows")

    df_7b = None
    if args.rescore_7b and Path(args.rescore_7b).exists():
        df_7b = pd.read_parquet(args.rescore_7b)
        print(f"Loaded 7B rescore: {len(df_7b)} rows")

    report(df_all, df_3b, df_7b)


if __name__ == "__main__":
    main()
