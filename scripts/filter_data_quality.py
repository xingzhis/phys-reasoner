"""Data quality filter for the candidates parquet.

Applies three cleaning passes:

  1. REMOVE — Explanation-primary questions: problem starts with Explain/Describe/Discuss.
     These are fundamentally open-ended; the verifier cannot reliably check them even
     when a numerical gold exists (the question asks for the derivation, not the number).

  2. REMOVE — Placeholder gold answers: gold is literally C_1, C_2 (no actual expressions).

  3. STRIP  — Garbled unit fields (non-ASCII, non-Angstrom): the unit string was corrupted
     during Chinese→LaTeX conversion.  Rather than remove the question, we zero-out the
     unit so xVerify handles comparison without a broken unit hint.

Usage:
    python scripts/filter_data_quality.py \\
        --input  data/processed/candidates_deduped.parquet \\
        --output data/processed/candidates_filtered.parquet

    # Dry-run (print report only, no file written):
    python scripts/filter_data_quality.py --dry-run
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# Filter definitions
# ---------------------------------------------------------------------------

# Rule 1: problem starts with an explanation verb
_EXPLAIN_RE = re.compile(r"^(explain|describe|discuss)\b", re.IGNORECASE)

# Rule 2: gold answer is a placeholder like C_1, C_2 / C_{1}, C_{2}
_PLACEHOLDER_GOLD_RE = re.compile(
    r"^\\boxed\{C_?\{?[12]\}?,\s*C_?\{?[12]\}?\}$"
)

# Rule 3: unit field contains non-ASCII characters (garbled encoding).
# Å (U+00C5) appears as \text{Å} or $Å$ and is a valid physics unit — keep those.
_ANGSTROM_RE = re.compile(r"Å|\\text\{Å\}|\$Å\$")


def _is_garbled_unit(unit: str) -> bool:
    if not unit:
        return False
    if not any(ord(c) > 127 for c in unit):
        return False
    # Allow Angstrom (Å, U+00C5) — it's a legitimate non-ASCII physics unit
    cleaned = _ANGSTROM_RE.sub("", unit)
    return any(ord(c) > 127 for c in cleaned)


def apply_filters(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (cleaned_df, removed_df) with a 'filter_reason' column on removed_df."""
    # Answer column may be 'answer' (candidates) or 'gold_answer' (rescore output)
    answer_col = "answer" if "answer" in df.columns else "gold_answer"

    reasons: list[str | None] = []
    strip_unit_mask = pd.Series(False, index=df.index)

    for _, row in df.iterrows():
        problem = str(row.get("problem", "")).strip()
        gold = str(row.get(answer_col, "")).strip()
        unit = str(row.get("unit", "") or "")

        if _EXPLAIN_RE.match(problem):
            reasons.append("explanation_primary")
        elif _PLACEHOLDER_GOLD_RE.match(gold):
            reasons.append("placeholder_gold")
        else:
            reasons.append(None)
            if _is_garbled_unit(unit):
                strip_unit_mask.at[_] = True

    reason_series = pd.Series(reasons, index=df.index)
    remove_mask = reason_series.notna()

    removed = df[remove_mask].copy()
    removed["filter_reason"] = reason_series[remove_mask]

    kept = df[~remove_mask].copy()
    # Strip garbled unit fields in kept rows
    n_stripped = strip_unit_mask[~remove_mask].sum()
    kept.loc[strip_unit_mask[~remove_mask], "unit"] = ""

    return kept, removed, int(n_stripped)


def print_report(df_orig: pd.DataFrame, kept: pd.DataFrame,
                 removed: pd.DataFrame, n_stripped: int) -> None:
    answer_col = "answer" if "answer" in df_orig.columns else "gold_answer"

    print(f"\n{'='*60}")
    print(f"  Data Quality Filter Report")
    print(f"{'='*60}")
    print(f"  Input rows  : {len(df_orig):>6}")
    print(f"  Removed     : {len(removed):>6}")
    print(f"  Unit stripped:{n_stripped:>6}  (kept, unit zeroed)")
    print(f"  Output rows : {len(kept):>6}")
    print()

    for reason, group in removed.groupby("filter_reason"):
        print(f"  [{reason}]  ({len(group)} rows)")
        by_src = group["source"].value_counts()
        for src, n in by_src.items():
            print(f"    {src:<30} {n}")
        if len(group) <= 30:
            for _, row in group.iterrows():
                pid = row["problem_id"]
                gold = str(row.get(answer_col, ""))[:60]
                prob = str(row.get("problem", ""))[:80]
                print(f"    • {pid}")
                print(f"      Q:    {prob}")
                print(f"      Gold: {gold}")
        print()

    print(f"  Accuracy impact on rescore (if available):")
    if "score_xverify" in df_orig.columns:
        orig_acc = df_orig["score_xverify"].mean()
        kept_acc = kept["score_xverify"].mean()
        print(f"    Original : {orig_acc:.4f}  (n={len(df_orig)})")
        print(f"    Filtered : {kept_acc:.4f}  (n={len(kept)})")
    else:
        print("    (no score_xverify column in input — run on rescore output for accuracy delta)")
    print(f"{'='*60}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", default="data/processed/candidates_deduped.parquet")
    parser.add_argument("--output", default="data/processed/candidates_filtered.parquet")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print report without writing output file")
    args = parser.parse_args()

    df = pd.read_parquet(args.input)
    print(f"Loaded {len(df)} rows from {args.input}")

    kept, removed, n_stripped = apply_filters(df)
    print_report(df, kept, removed, n_stripped)

    if args.dry_run:
        print("Dry-run mode — no file written.")
        return

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    kept.to_parquet(args.output, index=False)
    print(f"Saved filtered dataset to {args.output}")

    removed_path = Path(args.output).with_stem(Path(args.output).stem + "_removed")
    removed.to_parquet(removed_path, index=False)
    print(f"Saved removed rows to    {removed_path}")


if __name__ == "__main__":
    main()
