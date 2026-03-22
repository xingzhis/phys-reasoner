"""Systematic verifier diagnostic: trace exactly what happens at each pipeline stage."""
import os, sys, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pandas as pd
from phys_reasoner.data.normalize import normalize_answer_type
from phys_reasoner.verifier.router import (
    _build_gold_parts, _extract_pred_parts, verify_answer
)
from phys_reasoner.verifier.math_verify_wrapper import rule_verify
from phys_reasoner.verifier.xverify_judge import XVerifyJudge

ROOT = os.path.join(os.path.dirname(__file__), "..")
DIAG = os.path.join(ROOT, "data", "results", "zero_shot_diag.parquet")
SRC  = os.path.join(ROOT, "data", "processed", "candidates_deduped.parquet")

def main():
    df  = pd.read_parquet(DIAG)
    src = pd.read_parquet(SRC)[["problem_id", "problem"]].set_index("problem_id")

    print("Loading xVerify...", flush=True)
    judge = XVerifyJudge(model_name="IAAR-Shanghai/xVerify-3B-Ib", device="cuda")
    print("Done.\n")

    ISSUES = []

    for i, row in df.iterrows():
        atype   = row["answer_type"]
        gold    = row["gold_answer"]
        pred    = row["pred_text"]
        problem = src.loc[row["problem_id"], "problem"] if row["problem_id"] in src.index else ""

        norm_type   = normalize_answer_type(atype)
        gold_parts  = _build_gold_parts(gold)
        pred_parts  = _extract_pred_parts(pred)
        primary     = norm_type[0] if isinstance(norm_type, list) else norm_type

        rule_score  = verify_answer(pred, gold, atype, xverify_judge=None, problem_text=problem)
        xv_score    = verify_answer(pred, gold, atype, xverify_judge=judge, problem_text=problem)

        # xVerify per-pair calls (only when rule != True)
        pair_xv = []
        for gp in gold_parts:
            for pp in pred_parts:
                r = rule_verify(pp, gp, 0.05)
                if r is not True:
                    xv_call = judge(pp, gp, problem)
                    pair_xv.append((pp[:60], gp[:60], r, xv_call))

        print(f"{'='*70}")
        print(f"Sample {i}  |  raw_type={repr(atype)}  |  norm_type={norm_type}  |  primary={primary}")
        print(f"  gold_parts ({len(gold_parts)}): {[g[:80] for g in gold_parts]}")
        print(f"  pred_parts ({len(pred_parts)}): {[p[:80] for p in pred_parts]}")
        print(f"  parts_match: {len(pred_parts)} vs {len(gold_parts)} → {'OK' if len(pred_parts)==len(gold_parts) else 'MISMATCH → short-circuits to 0.0'}")
        print(f"  rule_score={rule_score:+.1f}   xv_score={xv_score:+.1f}", end="")
        if xv_score != rule_score:
            print(f"  ← xVerify FLIPPED", end="")
        print()
        if pair_xv:
            print(f"  xVerify pair calls:")
            for pp, gp, rule, xv in pair_xv:
                print(f"    rule={str(rule):<5}  xv={xv}  pred={repr(pp)}  gold={repr(gp)}")

        # Classify issue
        issues = []
        if primary in ("open_end", "code", "unknown"):
            issues.append(f"SKIP: primary_type={primary}")
        if len(pred_parts) != len(gold_parts):
            issues.append(f"PARTS_MISMATCH: {len(pred_parts)} pred vs {len(gold_parts)} gold — xVerify bypassed")
        if xv_score == -1.0:
            issues.append("XV_MINUS1: -1.0 even with xVerify (check primary_type or parts_mismatch→wrong branch)")
        for pp, gp, rule, xv in pair_xv:
            if rule is None and not xv:
                issues.append(f"RULE_NONE_XV_WRONG: rule couldn't parse, xVerify says wrong")
            if rule is False and not xv:
                issues.append(f"RULE_FALSE_XV_WRONG: rule says wrong, xVerify confirms")
        if issues:
            ISSUES.append((i, issues))
            print(f"  !! ISSUES: {issues}")
        print()

    print(f"\n{'='*70}")
    print("SYSTEMATIC ISSUE SUMMARY")
    print(f"{'='*70}")
    for i, issues in ISSUES:
        row = df.iloc[i]
        print(f"\nSample {i} (type={row['answer_type']}, rule={verify_answer(row['pred_text'], row['gold_answer'], row['answer_type']):.1f}):")
        for iss in issues:
            print(f"  • {iss}")

if __name__ == "__main__":
    main()