"""Data quality filter for the candidates parquet.

Applies cleaning passes:

  1. REMOVE — Explanation-primary questions: problem starts with Explain/Describe/Discuss.
     These are fundamentally open-ended; the verifier cannot reliably check them even
     when a numerical gold exists (the question asks for the derivation, not the number).

  2. REMOVE — Placeholder gold answers: gold is literally C_1, C_2 (no actual expressions).

  3. STRIP  — Garbled unit fields (non-ASCII, non-Angstrom): the unit string was corrupted
     during Chinese→LaTeX conversion.  Rather than remove the question, we zero-out the
     unit so xVerify handles comparison without a broken unit hint.

  4. NORMALIZE + REMOVE — MCQ gold answers: strip \\boxed{}, \\text{}, parens, trailing junk
     to produce a clean single letter (e.g. \\boxed{B} → B, (A) → A, A. → A, A *** → A,
     \\text{(B) \\lambda/(4n)} → B).  Rows that cannot be reduced to 1–3 letters are removed.

  5. NORMALIZE + REMOVE — True/False gold answers: strip \\boxed{}, map yes/no/Y/N/T/F to
     True/False (e.g. \\boxed{Yes} → True, \\boxed{No, formula} → False, F → False).
     Rows with no recognizable T/F token are removed.

  6. REMOVE — Non-Latin-script gold answers: gold contains Chinese, Japanese, Korean, or
     Arabic characters.  These are untranslatable by the verifier.
     NOTE: valid Unicode math (×, ≤, ², Fe²⁺, Å) is intentionally kept.

  7. REMOVE — Non-English corpus rows (language column != 'en', when present).

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


# ---------------------------------------------------------------------------
# Rule 6: Non-Latin script detection
# Matches CJK, Arabic, Devanagari, Hangul. Does NOT match Unicode math/science
# notation (×, ≤, ², Fe²⁺, Å, Greek letters) which are valid physics answers.
# ---------------------------------------------------------------------------

_NON_LATIN_SCRIPT_RE = re.compile(
    r"[\u4e00-\u9fff"   # CJK Unified Ideographs
    r"\u3040-\u30ff"    # Hiragana + Katakana
    r"\u3400-\u4dbf"    # CJK Extension A
    r"\u0600-\u06ff"    # Arabic
    r"\u0900-\u097f"    # Devanagari
    r"\uac00-\ud7af"    # Hangul Syllables
    r"]"
)


def _has_non_latin_script(s: str) -> bool:
    return bool(_NON_LATIN_SCRIPT_RE.search(s))


# ---------------------------------------------------------------------------
# Rules 4 & 5: MCQ and True/False gold normalization
# ---------------------------------------------------------------------------

def _unbox_gold(s: str) -> str:
    """Strip outermost \\boxed{...} with depth-aware brace tracking."""
    idx = s.find(r"\boxed{")
    if idx == -1:
        return s
    start = idx + len(r"\boxed{")
    depth = 1
    i = start
    while i < len(s) and depth > 0:
        if s[i] == "{":
            depth += 1
        elif s[i] == "}":
            depth -= 1
        i += 1
    if depth == 0:
        return s[start : i - 1].strip()
    return s


def normalize_mcq_gold(s: str) -> str | None:
    """Normalize MCQ gold to 1–3 uppercase letters, or None if unrecoverable.

    Handles all observed patterns:
      \\boxed{B}              → B
      (A)                    → A
      A.  / A)  / A:         → A
      A ***  / A - text      → A
      (A) some text          → A
      \\text{B}              → B
      (\\mathrm{c})          → C
      \\text{(B) \\lambda/…} → B
      ABD  (multi-select)    → ABD
    """
    raw = s.strip()

    # Unbox \boxed{...}
    unboxed = _unbox_gold(raw)

    # Strip \text{...} or \mathrm{...} wrappers
    m = re.match(r"^\\(?:text|mathrm|mathbf)\{(.*)\}$", unboxed.strip(), re.DOTALL)
    if m:
        unboxed = m.group(1).strip()

    # Strip outer parens wrapping a \mathrm or \text: (\mathrm{c}) → c
    m = re.match(r"^\(\\?(?:mathrm|text|mathbf)\{([A-Za-z]{1,3})\}\)$", unboxed)
    if m:
        return m.group(1).upper()

    # Plain (ABC)
    m = re.match(r"^\(([A-Za-z]{1,3})\)$", unboxed.strip())
    if m:
        return m.group(1).upper()

    # Extract leading letter(s), optionally wrapped in parens, followed by separator or end
    # Handles: A, A., A), A:, A ***, (A), (A) text, A - text, ABD
    m = re.match(
        r"^\(?([A-Za-z]{1,3})\)?(?:[.):\s*\-]|$)",
        unboxed.strip(),
    )
    if m:
        letter = m.group(1).upper()
        if re.match(r"^[A-Z]{1,3}$", letter):
            return letter

    return None  # unrecoverable


_TF_MAP: dict[str, str] = {
    "yes": "True", "y": "True", "t": "True", "true": "True",
    "no": "False", "n": "False", "f": "False", "false": "False",
}


def normalize_tf_gold(s: str) -> str | None:
    """Normalize TF gold to 'True' or 'False', or None if unrecoverable.

    Handles: \\boxed{Yes} → True, \\boxed{No, formula} → False,
             T/F/Y/N → True/False, yes/no → True/False.
    """
    original = s.strip()

    if original in ("True", "False"):
        return original

    # Unbox
    unboxed = _unbox_gold(original)

    # Extract first word token (handles "Yes, formula" → "Yes")
    first_token = re.split(r"[\s,;({\[]", unboxed)[0].strip().rstrip(".,")
    canonical = _TF_MAP.get(first_token.lower())
    if canonical:
        return canonical

    # Search anywhere for a T/F word (for compound strings like "E_1=3, ..., No")
    m = re.search(r"\b(yes|no|true|false)\b", unboxed, re.IGNORECASE)
    if m:
        return _TF_MAP[m.group(1).lower()]

    return None  # unrecoverable


def apply_filters(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, int]:
    """Return (cleaned_df, removed_df, n_units_stripped).

    cleaned_df has MCQ/TF gold answers normalized in-place.
    removed_df has a 'filter_reason' column.
    """
    # Answer column may be 'answer' (candidates) or 'gold_answer' (rescore output)
    answer_col = "answer" if "answer" in df.columns else "gold_answer"

    reasons: list[str | None] = []
    strip_unit_mask = pd.Series(False, index=df.index)
    normalized_answers: dict[int, str] = {}  # index → new gold value

    for idx, row in df.iterrows():
        problem = str(row.get("problem", "")).strip()
        gold = str(row.get(answer_col, "")).strip()
        unit = str(row.get("unit", "") or "")
        atype = str(row.get("answer_type", "") or "").strip().lower()
        lang = str(row.get("language", "en") or "en").strip().lower()

        # Rule 7: non-English corpus rows
        if lang not in ("en", "", "nan"):
            reasons.append("non_english_language")
            continue

        # Rule 1: explanation-primary
        if _EXPLAIN_RE.match(problem):
            reasons.append("explanation_primary")
            continue

        # Rule 2: placeholder gold
        if _PLACEHOLDER_GOLD_RE.match(gold):
            reasons.append("placeholder_gold")
            continue

        # Rule 6: non-Latin script in gold answer
        if _has_non_latin_script(gold):
            reasons.append("non_latin_script_gold")
            continue

        # Rule 4: MCQ normalization
        if atype == "mcq":
            norm = normalize_mcq_gold(gold)
            if norm is None:
                reasons.append("mcq_unrecoverable")
                continue
            if norm != gold:
                normalized_answers[idx] = norm

        # Rule 5: True/False normalization
        elif atype == "true_false":
            norm = normalize_tf_gold(gold)
            if norm is None:
                reasons.append("tf_unrecoverable")
                continue
            if norm != gold:
                normalized_answers[idx] = norm

        reasons.append(None)
        if _is_garbled_unit(unit):
            strip_unit_mask.at[idx] = True

    reason_series = pd.Series(reasons, index=df.index)
    remove_mask = reason_series.notna()

    removed = df[remove_mask].copy()
    removed["filter_reason"] = reason_series[remove_mask]

    kept = df[~remove_mask].copy()

    # Apply MCQ/TF normalizations in kept rows
    for idx, new_val in normalized_answers.items():
        if idx in kept.index:
            kept.at[idx, answer_col] = new_val

    # Strip garbled unit fields
    n_stripped = int(strip_unit_mask[~remove_mask].sum())
    kept.loc[strip_unit_mask[~remove_mask], "unit"] = ""

    return kept, removed, n_stripped


def print_report(df_orig: pd.DataFrame, kept: pd.DataFrame,
                 removed: pd.DataFrame, n_stripped: int) -> None:
    answer_col = "answer" if "answer" in df_orig.columns else "gold_answer"

    # Count normalizations (rows where answer changed vs original)
    n_normalized = 0
    if answer_col in df_orig.columns and answer_col in kept.columns:
        shared_idx = kept.index.intersection(df_orig.index)
        n_normalized = int((kept.loc[shared_idx, answer_col] != df_orig.loc[shared_idx, answer_col]).sum())

    print(f"\n{'='*60}")
    print(f"  Data Quality Filter Report")
    print(f"{'='*60}")
    print(f"  Input rows  : {len(df_orig):>6}")
    print(f"  Removed     : {len(removed):>6}  ({len(removed)/len(df_orig):.1%})")
    print(f"  Normalized  : {n_normalized:>6}  (MCQ/TF gold fixed in-place)")
    print(f"  Unit stripped:{n_stripped:>6}  (kept, unit zeroed)")
    print(f"  Output rows : {len(kept):>6}")
    print()

    for reason, group in removed.groupby("filter_reason"):
        print(f"  [{reason}]  ({len(group)} rows, {len(group)/len(df_orig):.1%})")
        if "source" in group.columns:
            by_src = group["source"].value_counts()
            for src, n in by_src.items():
                print(f"    {src:<30} {n}")
        if len(group) <= 30:
            for _, row in group.iterrows():
                pid = row.get("problem_id", "?")
                gold = str(row.get(answer_col, ""))[:60]
                prob = str(row.get("problem", ""))[:80]
                print(f"    • {pid}")
                print(f"      Q:    {prob}")
                print(f"      Gold: {gold}")
        print()

    # Spot-check normalizations
    if n_normalized and answer_col in df_orig.columns:
        shared_idx = kept.index.intersection(df_orig.index)
        changed = kept.loc[shared_idx][kept.loc[shared_idx, answer_col] != df_orig.loc[shared_idx, answer_col]]
        print(f"  Normalization spot-check (first 20):")
        for idx, row in changed.head(20).iterrows():
            orig = str(df_orig.at[idx, answer_col])[:50]
            new  = str(row[answer_col])
            atype = str(row.get("answer_type", ""))
            print(f"    [{atype:12s}] {repr(orig):<55} → {repr(new)}")
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
