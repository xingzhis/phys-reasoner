"""Re-score a pass@k parquet (all_texts column) with xVerify enabled.

Reads a completed pass@k result file (from run_zero_shot_corpus_passk.py or
run_zero_shot_drsci.py), re-scores every completion in all_texts with the
given xVerify model, and rewrites pass_rate / n_correct / n_verifiable.

Original rule-only scores are preserved in scores_rule / pass_rate_rule.
New xVerify scores are written to scores / n_correct / n_verifiable / pass_rate
/ pass_rate_nontrunc (consistent with downstream Goldilocks analysis reading
the same column names).

Corpus schema (candidates_deduped.parquet):
  Join on problem_id to get: problem, unit, tolerance
  Score via compute_score() which handles units and multi-part answers.

Dr. SCI schema (zero_shot_drsci_sample.parquet):
  problem text is in extra_info.question (already in file); no unit field.
  Score via verify_answer() direct.

Usage
-----
  # Corpus pass@k:
  python scripts/rescore_passk_xverify.py \\
      --input  data/results/zero_shot_corpus_passk.parquet \\
      --orig   data/processed/candidates_deduped.parquet \\
      --model  IAAR-Shanghai/xVerify-7B-I \\
      --output data/results/zero_shot_corpus_passk_xv7b.parquet

  # Dr. SCI pass@k (no --orig needed):
  python scripts/rescore_passk_xverify.py \\
      --input  data/results/zero_shot_drsci_sample.parquet \\
      --schema drsci \\
      --model  IAAR-Shanghai/xVerify-7B-I \\
      --output data/results/zero_shot_drsci_sample_xv7b.parquet
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from phys_reasoner.training.reward import compute_score
from phys_reasoner.verifier.router import verify_answer
from phys_reasoner.verifier.xverify_judge import XVerifyJudge


# ---------------------------------------------------------------------------
# Rescoring
# ---------------------------------------------------------------------------

def rescore_corpus(
    df: pd.DataFrame,
    orig: pd.DataFrame,
    judge: XVerifyJudge,
    tolerance: float,
) -> pd.DataFrame:
    """Re-score corpus pass@k rows. Joins on problem_id for problem/unit."""
    df = df.merge(orig[["problem_id", "problem", "unit", "tolerance"]], on="problem_id", how="left")

    new_scores_list: list[list[float]] = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Rescoring (corpus)"):
        gold = row["gold_answer"]
        if isinstance(gold, str):
            try:
                parsed = json.loads(gold)
                if isinstance(parsed, list):
                    gold = parsed
            except Exception:
                pass

        try:
            raw_tol = row.get("tolerance")
            row_tol = float(str(raw_tol).lstrip(",").strip()) if pd.notna(raw_tol) else tolerance
        except (ValueError, TypeError):
            row_tol = tolerance

        at   = str(row.get("answer_type") or "unknown")
        unit = str(row.get("unit") or "")
        prob = str(row.get("problem") or "")

        texts = row["all_texts"]  # list of k strings
        scores_k: list[float] = []
        for text in texts:
            try:
                sc = compute_score(
                    solution_str=str(text),
                    ground_truth=gold,
                    answer_type=at,
                    unit=unit,
                    tolerance=row_tol,
                    xverify_judge=judge,
                    problem=prob,
                )
            except Exception:
                sc = -1.0
            scores_k.append(sc)
        new_scores_list.append(scores_k)

    return _apply_new_scores(df, new_scores_list)


def rescore_drsci(
    df: pd.DataFrame,
    judge: XVerifyJudge,
    tolerance: float,
) -> pd.DataFrame:
    """Re-score Dr. SCI pass@k rows. problem text is in extra_info.question."""
    new_scores_list: list[list[float]] = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Rescoring (drsci)"):
        gold = str(row.get("gold_answer") or "")
        at   = str(row.get("answer_type") or "numerical")
        prob = str(row.get("extra_info.question") or "")

        texts = row["all_texts"]
        scores_k: list[float] = []
        for text in texts:
            try:
                sc = verify_answer(
                    pred_text=str(text),
                    gold_answer=gold,
                    answer_type=at,
                    tolerance=tolerance,
                    xverify_judge=judge,
                    problem_text=prob,
                )
            except Exception:
                sc = -1.0
            scores_k.append(sc)
        new_scores_list.append(scores_k)

    return _apply_new_scores(df, new_scores_list)


def _apply_new_scores(df: pd.DataFrame, new_scores_list: list[list[float]]) -> pd.DataFrame:
    """Preserve old scores as *_rule columns and write new xVerify scores."""
    df = df.copy()

    # Preserve rule-only scores
    df["scores_rule"]           = df["scores"]
    df["pass_rate_rule"]        = df["pass_rate"]
    df["pass_rate_nontrunc_rule"] = df.get("pass_rate_nontrunc", float("nan"))
    df["n_correct_rule"]        = df["n_correct"]
    df["n_verifiable_rule"]     = df["n_verifiable"]

    # Write new xVerify scores
    df["scores"] = new_scores_list

    n_correct_list     = [s.count(1.0)                         for s in new_scores_list]
    n_verif_list       = [sum(1 for sc in s if sc != -1.0)     for s in new_scores_list]
    n_trunc_list       = df["n_truncated"].tolist() if "n_truncated" in df.columns else [0] * len(df)
    n_samples_list     = [len(s)                               for s in new_scores_list]

    pass_rate_list = [nc / ns for nc, ns in zip(n_correct_list, n_samples_list)]
    pass_rate_nontrunc_list = [
        nc / max(ns - nt, 1) if (ns - nt) > 0 else float("nan")
        for nc, ns, nt in zip(n_correct_list, n_samples_list, n_trunc_list)
    ]

    df["n_correct"]          = n_correct_list
    df["n_verifiable"]       = n_verif_list
    df["n_samples"]          = n_samples_list
    df["pass_rate"]          = pass_rate_list
    df["pass_rate_nontrunc"] = pass_rate_nontrunc_list

    return df


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def print_summary(df: pd.DataFrame, model_name: str, schema: str, k: int) -> None:
    pr_rule = df["pass_rate_rule"].mean()
    pr_xv   = df["pass_rate"].mean()
    gl_low, gl_high = 0.15, 0.85

    print(f"\n=== Pass@{k} Re-score with {model_name} ===")
    print(f"  Rule-only  pass@{k}: {pr_rule:.1%}  (n={len(df)})")
    print(f"  xVerify    pass@{k}: {pr_xv:.1%}")
    print(f"  Delta:              {pr_xv - pr_rule:+.1%}")

    n_unverif = (df["n_verifiable"] == 0).sum()
    print(f"  Unverifiable rows:  {n_unverif} ({n_unverif/len(df):.1%})")

    n_gl_rule = ((df["pass_rate_rule"] >= gl_low) & (df["pass_rate_rule"] <= gl_high)).sum()
    n_gl_xv   = ((df["pass_rate"] >= gl_low) & (df["pass_rate"] <= gl_high)).sum()
    print(f"\n  Goldilocks [15%–85%]:")
    print(f"    rule-only: {n_gl_rule}/{len(df)} ({n_gl_rule/len(df):.1%})")
    print(f"    xVerify:   {n_gl_xv}/{len(df)} ({n_gl_xv/len(df):.1%})")

    group_col = "drsci_from" if schema == "drsci" else "source"
    if group_col in df.columns:
        print(f"\n  By {group_col}:")
        for src, grp in df.groupby(group_col, observed=True):
            pr_r = grp["pass_rate_rule"].mean()
            pr_x = grp["pass_rate"].mean()
            n_gl = ((grp["pass_rate"] >= gl_low) & (grp["pass_rate"] <= gl_high)).sum()
            print(f"    {str(src):<30} rule={pr_r:.1%}  xv={pr_x:.1%}  "
                  f"goldilocks={n_gl/len(grp):.1%}  n={len(grp)}")

    at_col = "answer_type" if schema == "corpus" else "answer_type"
    if at_col in df.columns:
        print(f"\n  By answer_type:")
        for at, grp in df.groupby(at_col, observed=True):
            at_label = str(at)
            if at_label.startswith("["):
                at_label = "multi"
            pr_r = grp["pass_rate_rule"].mean()
            pr_x = grp["pass_rate"].mean()
            print(f"    {at_label:<15} rule={pr_r:.1%}  xv={pr_x:.1%}  n={len(grp)}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input",   required=True,
                        help="Pass@k parquet from run_zero_shot_corpus_passk.py or run_zero_shot_drsci.py")
    parser.add_argument("--orig",    default="data/processed/candidates_deduped.parquet",
                        help="Original corpus parquet for problem/unit (corpus schema only)")
    parser.add_argument("--schema",  choices=["corpus", "drsci"], default="corpus",
                        help="Input schema: corpus (default) or drsci")
    parser.add_argument("--model",   default="IAAR-Shanghai/xVerify-7B-I")
    parser.add_argument("--output",  required=True)
    parser.add_argument("--tolerance", type=float, default=0.05)
    args = parser.parse_args()

    print(f"Loading xVerify model: {args.model}...", flush=True)
    judge = XVerifyJudge(model_name=args.model, device="cuda")

    print(f"Loading pass@k results from {args.input}...", flush=True)
    df = pd.read_parquet(args.input)
    print(f"  {len(df)} rows, {df['n_samples'].iloc[0]} samples/row", flush=True)

    if "all_texts" not in df.columns:
        print("ERROR: 'all_texts' column not found. Re-run inference with the updated script "
              "that saves all k rollout texts.", flush=True)
        sys.exit(1)

    if args.schema == "corpus":
        print(f"Loading corpus for problem/unit join from {args.orig}...", flush=True)
        orig = pd.read_parquet(args.orig)
        rescored = rescore_corpus(df, orig, judge, args.tolerance)
    else:
        rescored = rescore_drsci(df, judge, args.tolerance)

    k = int(rescored["n_samples"].iloc[0])
    print_summary(rescored, args.model, args.schema, k)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    rescored.to_parquet(args.output, index=False)
    print(f"\nSaved → {args.output}", flush=True)


if __name__ == "__main__":
    main()
