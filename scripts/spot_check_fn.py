"""Spot-check false negatives in the zero-shot baseline.

Loads the rescored parquet (with rule + xVerify scores) and shows samples
that were scored as negative (0.0) by xVerify-7B. Useful for manual
inspection to identify residual false negatives.

Usage:
    python scripts/spot_check_fn.py \
        --rescore data/results/rescore_7b.parquet \
        [--source PHYSICS] \
        [--answer_type expression] \
        [--n 20] \
        [--seed 42]
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import pandas as pd


def simplify_type(t: str) -> str:
    if str(t).startswith("["):
        try:
            parts = json.loads(t)
            return "+".join(parts)
        except Exception:
            return "multi"
    return str(t)


def truncate(s: str, n: int = 300) -> str:
    s = str(s).strip()
    if len(s) > n:
        return s[:n] + "…"
    return s


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rescore", default="data/results/rescore_7b.parquet",
                        help="Rescored parquet file (must have score_xverify column)")
    parser.add_argument("--source", default=None, help="Filter by source")
    parser.add_argument("--answer_type", default=None, help="Filter by answer_type substring")
    parser.add_argument("--n", type=int, default=20, help="Number of FN samples to show")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--show_pred", action="store_true",
                        help="Also print the model's raw prediction text")
    parser.add_argument("--show_problem", action="store_true",
                        help="Also print the problem text")
    parser.add_argument("--rule_positive", action="store_true",
                        help="Show samples where rule=0 but rule_verify was -1 (unverifiable) — i.e., xVerify got a chance but still said 0")
    args = parser.parse_args()

    df = pd.read_parquet(args.rescore)
    print(f"Loaded {len(df)} rows from {args.rescore}")
    print(f"Columns: {list(df.columns)}")

    if "score_xverify" not in df.columns:
        print("ERROR: no score_xverify column. Run rescore_xverify.py first.")
        sys.exit(1)

    # False negatives: xVerify score = 0.0
    fn = df[df["score_xverify"] == 0.0].copy()
    print(f"\nTotal negatives (score_xverify=0): {len(fn)} / {len(df)}")

    if args.source:
        fn = fn[fn["source"] == args.source]
        print(f"After source={args.source}: {len(fn)}")
    if args.answer_type:
        fn = fn[fn["answer_type"].astype(str).str.contains(args.answer_type)]
        print(f"After answer_type~='{args.answer_type}': {len(fn)}")

    fn_sample = fn.sample(min(args.n, len(fn)), random_state=args.seed)

    # FN breakdown by source + type
    print("\n--- FN breakdown ---")
    fn["atype_simple"] = fn["answer_type"].apply(simplify_type)
    breakdown = fn.groupby(["source", "atype_simple"]).size().reset_index(name="n_fn")
    # Also compute FN rate per group from original df
    df["atype_simple"] = df["answer_type"].apply(simplify_type)
    total = df.groupby(["source", "atype_simple"]).size().reset_index(name="n_total")
    breakdown = breakdown.merge(total, on=["source", "atype_simple"])
    breakdown["fn_rate"] = breakdown["n_fn"] / breakdown["n_total"]
    breakdown = breakdown.sort_values("n_fn", ascending=False)
    print(breakdown.to_string(index=False))

    print(f"\n--- Sample of {len(fn_sample)} false negatives ---\n")
    for i, (_, row) in enumerate(fn_sample.iterrows()):
        print(f"{'='*70}")
        print(f"[{i+1}] problem_id={row['problem_id']} | source={row['source']} | type={row['answer_type']}")
        print(f"  Gold:  {truncate(str(row['gold_answer']))}")
        print(f"  Pred:  {truncate(str(row['pred_text']))}")
        print(f"  Rule score:    {row['score']:.1f}")
        print(f"  xVerify score: {row['score_xverify']:.1f}")
        if args.show_problem and "problem" in row:
            print(f"  Problem: {truncate(str(row['problem']), 500)}")
        if args.show_pred:
            print(f"  Full pred_text:\n{str(row['pred_text'])[:1000]}")
        print()

    print(f"{'='*70}")
    print(f"Shown {len(fn_sample)} of {len(fn)} false negatives")


if __name__ == "__main__":
    main()
