"""Gold-gold round-trip test of the full verifier pipeline (rule + xVerify) on Dr. SCI.

Loads drsci_physics_clean.parquet, wraps each gold answer in \\boxed{} to simulate
a perfect model output, and runs verify_answer with XVerifyJudge.

Focus: rows where rule tier returns -1.0 (equation/expression types) to measure
how many xVerify resolves.

Usage
-----
  # Quick sample test (200 rows, equation/expression heavy):
  python scripts/drsci_xverify_test.py

  # Full run on all rows:
  python scripts/drsci_xverify_test.py --n_samples 0

  # Larger sample:
  python scripts/drsci_xverify_test.py --n_samples 500
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

import pandas as pd
from tqdm import tqdm

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from drsci_audit import infer_answer_type
from phys_reasoner.verifier.router import verify_answer
from phys_reasoner.verifier.xverify_judge import XVerifyJudge


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", default="data/processed/drsci_physics_clean.parquet")
    parser.add_argument("--model", default="IAAR-Shanghai/xVerify-3B-Ib")
    parser.add_argument(
        "--n_samples", type=int, default=200,
        help="Number of rows to sample (0 = all). Sample is stratified by answer type.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tolerance", type=float, default=0.05)
    parser.add_argument(
        "--device", default="cuda",
        help="Device for xVerify model (cuda or cpu)",
    )
    args = parser.parse_args()

    hf_cache = ROOT / "hf_cache"

    # --- Load data ---
    print(f"Loading {args.input}...", flush=True)
    df = pd.read_parquet(args.input)
    print(f"  {len(df):,} rows")

    gt_col = "reward_model.ground_truth"
    q_col  = "extra_info.question"

    # Infer answer types
    df["_answer_type"] = [
        infer_answer_type(str(gt)) for gt in df[gt_col].fillna("").astype(str)
    ]

    # --- Sample ---
    if args.n_samples > 0:
        # Stratified: oversample equation/expression (the hard ones for rule-tier)
        type_counts = df["_answer_type"].value_counts()
        print(f"\nAnswer type distribution (full {len(df):,} rows):")
        for t, n in type_counts.items():
            print(f"  {t:<15} {n:>7}  ({n/len(df):.1%})")

        # Sample proportionally but ensure we get at least some of each type
        n_total = min(args.n_samples, len(df))
        sample_frames = []
        for atype, grp in df.groupby("_answer_type", observed=True):
            n_want = max(1, round(n_total * len(grp) / len(df)))
            n_take = min(n_want, len(grp))
            sample_frames.append(grp.sample(n_take, random_state=args.seed))
        sample = pd.concat(sample_frames).sample(frac=1, random_state=args.seed).reset_index(drop=True)
        print(f"\nUsing stratified sample of {len(sample):,} rows")
    else:
        sample = df.reset_index(drop=True)
        print(f"\nUsing all {len(sample):,} rows")

    # --- Load xVerify model ---
    print(f"\nLoading xVerify model: {args.model}", flush=True)
    import os
    os.environ["HF_HOME"] = str(hf_cache)

    # Resolve snapshot hash from the cache directory
    model_cache_dir = hf_cache / "hub" / f"models--{args.model.replace('/', '--')}" / "snapshots"
    snapshots = list(model_cache_dir.iterdir()) if model_cache_dir.exists() else []
    if snapshots:
        model_path = str(sorted(snapshots)[-1])  # use latest snapshot hash
        print(f"  Using cached snapshot: {model_path}", flush=True)
    else:
        model_path = args.model  # fall back to HF model name (requires network)
        print(f"  No local cache found, using model name: {model_path}", flush=True)

    xv_judge = XVerifyJudge(model_name=model_path, device=args.device)
    print("  xVerify model loaded.", flush=True)

    # --- Run gold-gold round-trip: both with and without xVerify ---
    scores_rule_only: list[float] = []
    scores_with_xv: list[float] = []

    print("\nRunning gold-gold round-trip verifier test...", flush=True)
    for _, row in tqdm(sample.iterrows(), total=len(sample)):
        gold = str(row[gt_col]).strip()
        atype = str(row["_answer_type"])
        question = str(row.get(q_col, "")) if q_col in sample.columns else ""
        pred_text = f"\\boxed{{{gold}}}"

        # Rule-only score
        score_rule = verify_answer(
            pred_text=pred_text,
            gold_answer=gold,
            answer_type=atype,
            tolerance=args.tolerance,
            xverify_judge=None,
        )
        scores_rule_only.append(score_rule)

        # With xVerify (only call if rule returned -1.0 — not verified)
        if score_rule == -1.0:
            score_xv = verify_answer(
                pred_text=pred_text,
                gold_answer=gold,
                answer_type=atype,
                tolerance=args.tolerance,
                xverify_judge=xv_judge,
                problem_text=question,
            )
        else:
            score_xv = score_rule
        scores_with_xv.append(score_xv)

    # --- Summary ---
    n = len(scores_rule_only)
    rule_pass  = scores_rule_only.count(1.0)
    rule_fail  = scores_rule_only.count(0.0)
    rule_unv   = scores_rule_only.count(-1.0)

    xv_pass    = scores_with_xv.count(1.0)
    xv_fail    = scores_with_xv.count(0.0)
    xv_unv     = scores_with_xv.count(-1.0)

    # Of the rule-unverifiable rows, how many did xVerify resolve?
    xv_resolved_pass  = sum(
        1 for r, x in zip(scores_rule_only, scores_with_xv) if r == -1.0 and x == 1.0
    )
    xv_resolved_fail  = sum(
        1 for r, x in zip(scores_rule_only, scores_with_xv) if r == -1.0 and x == 0.0
    )
    xv_still_unv      = sum(
        1 for r, x in zip(scores_rule_only, scores_with_xv) if r == -1.0 and x == -1.0
    )

    print(f"\n=== Verifier pipeline results (n={n:,}) ===")
    print(f"\nRule-tier only:")
    print(f"  pass        {rule_pass:>5}  ({rule_pass/n:.1%})")
    print(f"  fail        {rule_fail:>5}  ({rule_fail/n:.1%})")
    print(f"  unverifiable{rule_unv:>5}  ({rule_unv/n:.1%})")

    print(f"\nWith xVerify (on rule-unverifiable subset of {rule_unv:,}):")
    print(f"  resolved → pass  {xv_resolved_pass:>5}  ({xv_resolved_pass/max(rule_unv,1):.1%} of unv)")
    print(f"  resolved → fail  {xv_resolved_fail:>5}  ({xv_resolved_fail/max(rule_unv,1):.1%} of unv)")
    print(f"  still unv        {xv_still_unv:>5}  ({xv_still_unv/max(rule_unv,1):.1%} of unv)")

    print(f"\nCombined (rule + xVerify):")
    print(f"  pass        {xv_pass:>5}  ({xv_pass/n:.1%})")
    print(f"  fail        {xv_fail:>5}  ({xv_fail/n:.1%})")
    print(f"  unverifiable{xv_unv:>5}  ({xv_unv/n:.1%})")

    # --- By answer type ---
    print(f"\n=== By answer type (combined scores) ===")
    type_results: dict[str, list[float]] = {}
    for atype, score in zip(sample["_answer_type"].tolist(), scores_with_xv):
        type_results.setdefault(str(atype), []).append(score)

    for atype in ["numerical", "expression", "equation", "mcq", "unknown"]:
        if atype not in type_results:
            continue
        svec = type_results[atype]
        nn = len(svec)
        p = svec.count(1.0)
        f = svec.count(0.0)
        u = svec.count(-1.0)
        print(f"  {atype:<15} n={nn:>5}  pass={p/nn:.1%}  fail={f/nn:.1%}  unv={u/nn:.1%}")

    # --- Spot-check xVerify failures (gold-gold round-trip should never fail) ---
    failures = [
        (i, sample.iloc[i])
        for i, (r, x) in enumerate(zip(scores_rule_only, scores_with_xv))
        if x == 0.0  # xVerify said WRONG for gold-gold
    ]
    if failures:
        print(f"\n=== WARNING: {len(failures)} gold-gold FAILURES (xVerify said wrong) ===")
        print("  These are xVerify false negatives on gold-gold round-trips.")
        for idx, (i, row) in enumerate(failures[:10]):
            gold = str(row[gt_col])[:80]
            atype = str(row["_answer_type"])
            print(f"  [{idx+1}] type={atype}  gold={gold!r}")
    else:
        print(f"\nNo gold-gold failures (xVerify had no false negatives on this sample).")


if __name__ == "__main__":
    main()
