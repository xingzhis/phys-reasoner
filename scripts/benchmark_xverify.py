"""Benchmark xVerify-0.5B-I vs xVerify-3B-Ib on rule-tier failures.

Usage:
    python scripts/benchmark_xverify.py [--n_sample N]

Strategy:
  1. Load candidates_deduped.parquet
  2. Find rule-tier failures (verifiable types where rule returns None on gold==gold)
  3. Run both xVerify models on those cases
  4. Report accuracy (all should return True since pred==gold) + throughput
"""

from __future__ import annotations

import argparse
import os
import time

import pandas as pd

os.environ.setdefault(
    "HF_HOME",
    "/gpfs/radev/scratch/krishnaswamy_smita/xs272/phys-reasoner/hf_cache",
)

_PARQUET = (
    "/gpfs/radev/scratch/krishnaswamy_smita/xs272/phys-reasoner"
    "/data/processed/candidates_deduped.parquet"
)
_MODELS = [
    "IAAR-Shanghai/xVerify-0.5B-I",
    "IAAR-Shanghai/xVerify-3B-Ib",
]
_UNVERIFIABLE = {"open_end", "code", "unknown", "mcq", "true_false"}


def collect_rule_failures(df: pd.DataFrame, max_n: int) -> list[dict]:
    """Find rows where rule_verify(gold, gold) != 1.0 (rule can't self-verify)."""
    from phys_reasoner.verifier.math_verify_wrapper import rule_verify
    from phys_reasoner.verifier.router import _build_gold_parts, _extract_pred_parts

    failures = []
    for _, row in df.iterrows():
        atype = row.get("answer_type", "unknown")
        if isinstance(atype, list):
            primary = atype[0]
        else:
            primary = str(atype).split(",")[0].strip()
        if primary in _UNVERIFIABLE:
            continue

        gold = row.get("answer", "")
        if not gold or not str(gold).strip():
            continue

        gold_str = str(gold)
        gold_parts = _build_gold_parts(gold_str)
        if not gold_parts:
            continue

        # Only check single-part for simplicity
        if len(gold_parts) != 1:
            continue

        result = rule_verify(gold_parts[0], gold_parts[0])
        if result is not True:
            failures.append({
                "gold": gold_str,
                "answer_type": atype,
                "source": row.get("source", ""),
                "rule_result": result,
            })
            if len(failures) >= max_n:
                break

    return failures


def run_benchmark(model_name: str, cases: list[dict]) -> dict:
    """Run xVerify on cases (all gold==gold, all should be True). Return stats."""
    from phys_reasoner.verifier.xverify_judge import XVerifyJudge

    print(f"\nLoading {model_name} ...")
    judge = XVerifyJudge(model_name=model_name, device="cuda")

    correct = 0
    t0 = time.perf_counter()
    for case in cases:
        gold = case["gold"]
        result = judge(gold, gold)
        if result:
            correct += 1
    elapsed = time.perf_counter() - t0

    n = len(cases)
    return {
        "model": model_name,
        "n": n,
        "accuracy": correct / n if n else 0.0,
        "throughput_sps": n / elapsed if elapsed > 0 else 0.0,
        "elapsed_s": elapsed,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_sample", type=int, default=200,
                        help="Max rule-tier failure cases to benchmark (default: 200)")
    parser.add_argument("--models", nargs="+", default=_MODELS)
    args = parser.parse_args()

    print(f"Loading parquet: {_PARQUET}")
    df = pd.read_parquet(_PARQUET)
    print(f"  {len(df)} rows")

    print("\nCollecting rule-tier failures (gold==gold self-verify failures) ...")
    failures = collect_rule_failures(df, args.n_sample)
    print(f"  Found {len(failures)} rule failures")
    if not failures:
        print("No rule failures found — all gold answers self-verify. Nothing to benchmark.")
        return

    # Summary of failure types
    from collections import Counter
    type_counts = Counter(str(c["answer_type"]) for c in failures)
    print(f"  By type: {dict(type_counts)}")

    results = []
    for model_name in args.models:
        stats = run_benchmark(model_name, failures)
        results.append(stats)
        print(
            f"\n{model_name}:\n"
            f"  accuracy:   {stats['accuracy']:.1%} ({int(stats['accuracy']*stats['n'])}/{stats['n']})\n"
            f"  throughput: {stats['throughput_sps']:.1f} samples/sec\n"
            f"  elapsed:    {stats['elapsed_s']:.1f}s"
        )

    # Recommendation
    print("\n--- Recommendation ---")
    if len(results) == 2:
        acc_0, acc_1 = results[0]["accuracy"], results[1]["accuracy"]
        thr_0, thr_1 = results[0]["throughput_sps"], results[1]["throughput_sps"]
        gap = acc_1 - acc_0
        if gap < 0.02:
            print(f"Accuracy gap: {gap:.1%} < 2%  →  use {results[0]['model']} (faster: {thr_0:.1f} vs {thr_1:.1f} sps)")
        else:
            print(f"Accuracy gap: {gap:.1%} >= 2%  →  use {results[1]['model']} (better accuracy)")


if __name__ == "__main__":
    main()
