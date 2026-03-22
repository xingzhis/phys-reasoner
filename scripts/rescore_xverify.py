"""Re-score existing zero-shot chunk parquets with xVerify enabled.

Loads pred_text / gold_answer from completed chunk files, joins with the
original dataset (for problem text + unit), then re-runs the verifier with
an XVerifyJudge instance.

Usage:
    python scripts/rescore_xverify.py \\
        --chunks data/results/zero_shot_chunk{0..3}.parquet \\
        --model IAAR-Shanghai/xVerify-3B-Ib \\
        --output data/results/rescore_3b.parquet

    # Compare 3B vs 7B:
    python scripts/rescore_xverify.py --model IAAR-Shanghai/xVerify-3B-Ib ...
    python scripts/rescore_xverify.py --model IAAR-Shanghai/xVerify-7B-I  ...
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from phys_reasoner.verifier.router import verify_answer
from phys_reasoner.verifier.xverify_judge import XVerifyJudge


def load_chunks(chunk_paths: list[str]) -> pd.DataFrame:
    dfs = [pd.read_parquet(p) for p in chunk_paths]
    df = pd.concat(dfs, ignore_index=True)
    print(f"Loaded {len(df)} rows from {len(chunk_paths)} chunk file(s).")
    return df


def load_orig(orig_path: str) -> pd.DataFrame:
    orig = pd.read_parquet(orig_path)[["problem_id", "problem", "unit", "tolerance"]]
    return orig


def rescore(
    combined: pd.DataFrame,
    orig: pd.DataFrame,
    xverify_judge: XVerifyJudge,
    tolerance: float,
) -> pd.DataFrame:
    """Re-score all rows and return a copy with added 'score_xverify' column."""
    merged = combined.merge(orig, on="problem_id", how="left")
    if merged["problem"].isna().any():
        n_missing = merged["problem"].isna().sum()
        print(f"  Warning: {n_missing} rows missing problem text after join (will use empty string).")

    scores_xv: list[float] = []

    for _, row in tqdm(merged.iterrows(), total=len(merged), desc="Rescoring"):
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

        score_xv = verify_answer(
            pred_text=str(row["pred_text"]),
            gold_answer=gold,
            answer_type=str(row["answer_type"]),
            gold_unit=str(row["unit"]) if pd.notna(row.get("unit")) else "",
            tolerance=row_tol,
            xverify_judge=xverify_judge,
            problem_text=str(row["problem"]) if pd.notna(row.get("problem")) else "",
        )
        # -1.0 (unverifiable) → 0.0, consistent with training reward
        scores_xv.append(max(score_xv, 0.0))

    merged["score_xverify"] = scores_xv
    return merged


def print_summary(df: pd.DataFrame, model_name: str) -> None:
    rule_acc = df["score"].mean()
    xv_acc   = df["score_xverify"].mean()

    print(f"\n=== Re-score summary ({model_name}) ===")
    print(f"  Rule-only accuracy:   {rule_acc:.4f}  ({df['score'].sum():.0f}/{len(df)})")
    print(f"  xVerify accuracy:     {xv_acc:.4f}  ({df['score_xverify'].sum():.0f}/{len(df)})")
    delta = xv_acc - rule_acc
    print(f"  Delta (xv - rule):    {delta:+.4f}")

    print("\n--- By source ---")
    by_src = df.groupby("source")[["score", "score_xverify"]].mean()
    by_src["delta"] = by_src["score_xverify"] - by_src["score"]
    by_src["n"] = df.groupby("source")["score"].count()
    print(by_src.sort_values("score_xverify", ascending=False).to_string(float_format="{:.4f}".format))

    print("\n--- By answer_type (simplified) ---")
    df2 = df.copy()
    # Simplify list-type answer_types for readability
    def simplify_type(t):
        if t.startswith("["):
            try:
                parts = json.loads(t)
                return f"multi({len(parts)})-{parts[0]}"
            except Exception:
                return "multi"
        return t
    df2["atype_simple"] = df2["answer_type"].apply(simplify_type)
    by_type = df2.groupby("atype_simple")[["score", "score_xverify"]].mean()
    by_type["delta"] = by_type["score_xverify"] - by_type["score"]
    by_type["n"] = df2.groupby("atype_simple")["score"].count()
    print(by_type.sort_values("n", ascending=False).head(15).to_string(float_format="{:.4f}".format))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--chunks", nargs="+", required=True,
                        help="Paths to zero_shot_chunk*.parquet files")
    parser.add_argument("--orig", default="data/processed/candidates_deduped.parquet",
                        help="Original dataset parquet (for problem text + unit)")
    parser.add_argument("--model", default="IAAR-Shanghai/xVerify-3B-Ib",
                        help="xVerify HuggingFace model ID")
    parser.add_argument("--output", required=True,
                        help="Output parquet path")
    parser.add_argument("--tolerance", type=float, default=0.05)
    args = parser.parse_args()

    print(f"Loading xVerify model: {args.model}")
    judge = XVerifyJudge(model_name=args.model, device="cuda")

    combined = load_chunks(args.chunks)
    orig = load_orig(args.orig)

    rescored = rescore(combined, orig, judge, args.tolerance)
    print_summary(rescored, args.model)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    rescored.to_parquet(args.output, index=False)
    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()