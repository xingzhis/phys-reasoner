"""Benchmark the numerical substitution equivalence checker on the zero-shot baseline.

Measures:
  1. FN rescues: rows where score_xverify=0.0 and new rule_verify returns True
     (verifier was wrong — model may have been right)
  2. FP check: rows where score_xverify=1.0 and new rule_verify returns False
     (would be a regression — verifier now says wrong when xVerify said right)

Usage:
    python scripts/benchmark_numerical_equiv.py \
        --input data/results/rescore_7b_all8_fixed.parquet \
        [--sample N]   # limit rows processed per group for speed
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from phys_reasoner.verifier.extract import extract_answer
from phys_reasoner.verifier.math_verify_wrapper import rule_verify


def get_pred_str(pred_text: str) -> str:
    """Extract the first boxed answer from model output, or raw text."""
    parts = extract_answer(pred_text)
    return parts[0] if parts else pred_text.strip()


def run(input_path: str, sample: int | None) -> None:
    df = pd.read_parquet(input_path)
    print(f"Loaded {len(df)} rows from {input_path}")

    # Confirm required columns
    for col in ("score_xverify", "answer_type", "gold_answer", "pred_text"):
        if col not in df.columns:
            print(f"ERROR: missing column '{col}'")
            sys.exit(1)

    # Focus on symbolic answer types
    sym_types = {"expression", "equation", "EX", "EQ"}
    df_sym = df[df["answer_type"].isin(sym_types)].copy()
    print(f"Symbolic rows (expression/equation): {len(df_sym)}")

    # -------------------------------------------------------------------------
    # 1. FN rescue check: xVerify said wrong (0.0) — does new checker say True?
    # -------------------------------------------------------------------------
    fn_rows = df_sym[df_sym["score_xverify"] == 0.0]
    if sample:
        fn_rows = fn_rows.sample(min(sample, len(fn_rows)), random_state=42)
    print(f"\n--- FN rescue check ({len(fn_rows)} rows) ---")

    rescues = []
    for _, row in fn_rows.iterrows():
        pred_str = get_pred_str(str(row["pred_text"]))
        gold_str = str(row["gold_answer"]) if not isinstance(row["gold_answer"], list) else row["gold_answer"][0]
        result = rule_verify(pred_str, gold_str)
        if result is True:
            rescues.append({
                "problem_id": row.get("problem_id", ""),
                "source": row.get("source", ""),
                "answer_type": row.get("answer_type", ""),
                "gold": gold_str[:80],
                "pred": pred_str[:80],
            })

    print(f"Rescued: {len(rescues)} / {len(fn_rows)} ({len(rescues)/max(len(fn_rows),1):.1%})")

    if rescues:
        print("\nSample rescued cases (up to 20):")
        rng = random.Random(0)
        sample_rescues = rng.sample(rescues, min(20, len(rescues)))
        for r in sample_rescues:
            print(f"  [{r['source']}] {r['answer_type']}")
            print(f"    gold: {r['gold']}")
            print(f"    pred: {r['pred']}")
            print()

    # -------------------------------------------------------------------------
    # 2. FP check: xVerify said correct (1.0) — does new checker say False?
    # -------------------------------------------------------------------------
    fp_rows = df_sym[df_sym["score_xverify"] == 1.0]
    if sample:
        fp_rows = fp_rows.sample(min(sample, len(fp_rows)), random_state=42)
    print(f"\n--- FP regression check ({len(fp_rows)} rows) ---")

    regressions = []
    for _, row in fp_rows.iterrows():
        pred_str = get_pred_str(str(row["pred_text"]))
        gold_str = str(row["gold_answer"]) if not isinstance(row["gold_answer"], list) else row["gold_answer"][0]
        result = rule_verify(pred_str, gold_str)
        if result is False:
            regressions.append({
                "problem_id": row.get("problem_id", ""),
                "source": row.get("source", ""),
                "answer_type": row.get("answer_type", ""),
                "gold": gold_str[:80],
                "pred": pred_str[:80],
            })

    print(f"Regressions (FP introduced): {len(regressions)} / {len(fp_rows)} ({len(regressions)/max(len(fp_rows),1):.1%})")
    if regressions:
        print("\nRegression cases (up to 10):")
        for r in regressions[:10]:
            print(f"  [{r['source']}] {r['answer_type']}")
            print(f"    gold: {r['gold']}")
            print(f"    pred: {r['pred']}")
            print()

    # -------------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------------
    print("\n=== Summary ===")
    print(f"FN rescue rate (expression/equation): {len(rescues)}/{len(fn_rows)} = {len(rescues)/max(len(fn_rows),1):.1%}")
    print(f"Regression rate (FP introduced):      {len(regressions)}/{len(fp_rows)} = {len(regressions)/max(len(fp_rows),1):.1%}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/results/rescore_7b_all8_fixed.parquet")
    parser.add_argument("--sample", type=int, default=None,
                        help="Limit N rows per group (FN / FP) for speed")
    args = parser.parse_args()
    run(args.input, args.sample)
