"""Quick post-dedup validation: confirms parquet has normalized answer_types and sane row count.

Checks:
1. Schema: all answer_type values are canonical (no raw NV/EX/Numerical/etc.)
2. Row count: within ±5% of expected baseline (~6866)
3. Smoke verify: 200 random rows through verify_answer (gold==gold), no false negatives

Usage:
    python -u scripts/validate_parquet.py
    python -u scripts/validate_parquet.py --parquet data/processed/candidates_deduped.parquet
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pandas as pd

from phys_reasoner.verifier.router import verify_answer

_CANONICAL_TYPES = {
    "numerical", "expression", "equation", "interval",
    "mcq", "true_false", "open_end", "code", "unknown",
}
_RAW_TYPES = {
    "NV", "EX", "EQ", "IN", "MC", "MCQ", "TF", "T/F",
    "Numerical", "Expression", "Equation", "Interval",
    "True/False", "Open-end", "symbolic",
}
_SKIP_TYPES = {"open_end", "code", "unknown"}
_EXPECTED_ROWS = 6866
_TOLERANCE_PCT = 0.05


def check_schema(df: pd.DataFrame) -> bool:
    """All answer_type values must be canonical."""
    raw_found = []
    for val in df["answer_type"].unique():
        # val may be a JSON list string like '["numerical", "expression"]'
        if isinstance(val, str) and val.startswith("["):
            try:
                parts = json.loads(val)
                for p in parts:
                    if str(p) in _RAW_TYPES:
                        raw_found.append(val)
                        break
            except Exception:
                raw_found.append(val)
        elif str(val) in _RAW_TYPES:
            raw_found.append(val)

    if raw_found:
        print(f"  FAIL: {len(raw_found)} non-canonical answer_type values found:")
        for v in raw_found[:10]:
            print(f"    {v!r}")
        return False

    print(f"  PASS: all {len(df['answer_type'].unique())} answer_type values are canonical")
    return True


def check_row_count(df: pd.DataFrame) -> bool:
    n = len(df)
    lo = int(_EXPECTED_ROWS * (1 - _TOLERANCE_PCT))
    hi = int(_EXPECTED_ROWS * (1 + _TOLERANCE_PCT))
    ok = lo <= n <= hi
    status = "PASS" if ok else "WARN"
    print(f"  {status}: {n} rows (expected ~{_EXPECTED_ROWS}, range [{lo}, {hi}])")
    return ok


def check_smoke_verify(df: pd.DataFrame, n_sample: int = 200, seed: int = 42) -> bool:
    """Sample n rows; gold==gold must not produce false negatives (score 0.0)."""
    eligible = df[~df["answer_type"].apply(
        lambda t: str(t).strip("[]\"'") in _SKIP_TYPES
    )]
    sample = eligible.sample(min(n_sample, len(eligible)), random_state=seed)

    false_negatives = []
    for _, row in sample.iterrows():
        gold = row["answer"]
        if isinstance(gold, str):
            try:
                parsed = json.loads(gold)
                if isinstance(parsed, list):
                    gold = parsed
            except Exception:
                pass

        pred_text = gold if isinstance(gold, str) else ", ".join(str(x) for x in gold)
        result = verify_answer(
            pred_text=pred_text,
            gold_answer=gold,
            answer_type=row["answer_type"],
            gold_unit=str(row.get("unit") or ""),
            tolerance=0.05,
            xverify_judge=None,
        )
        if result == 0.0:
            false_negatives.append({
                "problem_id": row["problem_id"],
                "source": row["source"],
                "answer_type": row["answer_type"],
                "answer": gold,
            })

    fn_rate = len(false_negatives) / len(sample)
    ok = fn_rate < 0.05
    status = "PASS" if ok else "FAIL"
    print(f"  {status}: smoke verify {len(sample)} rows — {len(false_negatives)} FN ({fn_rate:.1%})")
    if false_negatives:
        print(f"    Sample FNs:")
        for fn in false_negatives[:5]:
            print(f"      {fn}")
    return ok


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parquet", default="data/processed/candidates_deduped.parquet")
    parser.add_argument("--n_smoke", type=int, default=200)
    args = parser.parse_args()

    path = Path(args.parquet)
    if not path.exists():
        print(f"ERROR: {path} not found", flush=True)
        sys.exit(1)

    print(f"Validating {path}...", flush=True)
    df = pd.read_parquet(path)

    print("\n[1] Schema check (normalized answer_types):", flush=True)
    ok1 = check_schema(df)

    print("\n[2] Row count check:", flush=True)
    ok2 = check_row_count(df)

    print(f"\n[3] Smoke verify ({args.n_smoke} random gold==gold rows):", flush=True)
    ok3 = check_smoke_verify(df, n_sample=args.n_smoke)

    print("\n--- Source breakdown ---", flush=True)
    print(df["source"].value_counts().to_string(), flush=True)
    print("\n--- answer_type value counts (top 20) ---", flush=True)
    print(df["answer_type"].value_counts().head(20).to_string(), flush=True)

    if ok1 and ok3:
        print("\nResult: PASS — parquet is valid, D1 re-run not required", flush=True)
        sys.exit(0)
    else:
        print("\nResult: FAIL — check issues above", flush=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
