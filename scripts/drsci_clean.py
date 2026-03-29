"""Clean Dr. SCI ground_truth strings before use in training.

Fixes four categories of formatting artifacts found in drsci_physics_deduped.parquet:

  1. Prose-wrapped answers (6.6%)
     "Therefore, the final answer is: $\\boxed{X}$"  →  "X"
     "$\\boxed{X}$"  →  "X"
     Affects: MegaScience (2,695) and natural_reasoning (4,764)

  2. Single display-math extraction (0.6%)
     Prose strings that contain exactly one \\(...\\) or \\[...\\] math block.
     "The specific heat is \\(C_V = 3Nk_B\\)."  →  "C_V = 3Nk_B"
     Only fires when the block is NOT the entire string (would be handled by step 3).

  3. Double-escaped LaTeX (0.1%)
     Ground truths where every backslash was escaped twice during JSON serialisation,
     e.g. "\\\\frac{1}{2}" (4 backslashes) instead of "\\frac{1}{2}" (2 backslashes).
     Heuristic: if every LaTeX command is preceded by ≥2 backslashes, halve them.

  4. Truncated answers (0.0%, 26 rows)
     Answers with unclosed braces — string was cut off mid-expression.
     These are dropped rather than mangled.

Outputs data/processed/drsci_physics_clean.parquet.

Usage
-----
  python scripts/drsci_clean.py
  python scripts/drsci_clean.py --report   # stats only, no save
"""

from __future__ import annotations

import argparse
import re
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent))
from drsci_audit import infer_answer_type


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extract_boxed_content(s: str) -> str | None:
    """Extract the outermost \\boxed{...} content from a string.

    Handles nested braces correctly.  Returns None if no \\boxed{ found.
    """
    idx = s.find(r"\boxed{")
    if idx == -1:
        return None
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
    return None  # unmatched brace


def _strip_dollar_wrappers(s: str) -> str:
    """Strip leading/trailing $...$ or $$...$$ math delimiters."""
    s = s.strip()
    # $$...$$ first (longer pattern)
    if s.startswith("$$") and s.endswith("$$") and len(s) > 4:
        s = s[2:-2].strip()
    # $...$
    elif s.startswith("$") and s.endswith("$") and len(s) > 2:
        s = s[1:-1].strip()
    # \[...\]
    elif s.startswith(r"\[") and s.endswith(r"\]"):
        s = s[2:-2].strip()
    return s


def _extract_single_display_math(s: str) -> str | None:
    """If the string contains exactly one \\(...\\) or \\[...\\] block embedded
    in surrounding prose, extract and return the math content.

    Returns None if:
    - There are zero or multiple math blocks (ambiguous / multi-part).
    - The block IS the entire string (handled upstream by _strip_dollar_wrappers).
    """
    blocks: list[str] = []
    blocks += re.findall(r"\\\((.*?)\\\)", s, re.DOTALL)
    blocks += re.findall(r"\\\[(.*?)\\\]", s, re.DOTALL)

    if len(blocks) != 1:
        return None

    content = blocks[0].strip()
    if not content:
        return None

    # Skip if the block IS the whole string — step above handles that.
    if s.strip() in (rf"\({content}\)", rf"\[{content}\]"):
        return None

    return content


_PROSE_PATTERNS = [
    # "Therefore, the final answer is: $\boxed{X}$"
    re.compile(r"^.*?(?:final answer is|answer is|answer:|result is)[:\s]*\$?\\?boxed\{", re.IGNORECASE | re.DOTALL),
    # "The answer is X" without boxed
    re.compile(r"^.*?(?:final answer is|the answer is)[:\s]+", re.IGNORECASE | re.DOTALL),
]

_PROSE_BOXED_RE = re.compile(
    r"""
    (?:                       # optional prose prefix
        .*?                   # any chars (non-greedy)
        (?:final\s+answer\s+is|answer\s+is|result\s+is|:\s*)  # prose marker
        [:\s]*
    )?
    \$?                       # optional $
    \\boxed\{                 # \boxed{
    """,
    re.IGNORECASE | re.DOTALL | re.VERBOSE,
)


def clean_ground_truth(s: str) -> tuple[str, str]:
    """Clean a single ground_truth string.

    Returns (cleaned_string, change_tag) where change_tag is one of:
      "unchanged", "prose_extracted", "dollar_stripped",
      "double_unescaped", "truncated_dropped"
    """
    original = s
    s = s.strip()

    # --- 1. Truncated: unclosed braces → mark for dropping ---
    depth = sum(1 if c == "{" else -1 if c == "}" else 0 for c in s)
    if depth != 0:
        return original, "truncated_dropped"

    # --- 2. Prose-wrapped + boxed: extract \boxed{} content ---
    # Detect: string contains \boxed{ and has prose before it, or has $\boxed{...}$ pattern
    boxed = _extract_boxed_content(s)
    if boxed is not None and boxed.strip():
        # Only apply if the string is more than just the boxed content
        # i.e., there's a wrapper/prose to strip.
        # Guard: skip empty-boxed cases like "$\boxed{}\n\begin{aligned}..." where
        # the real content is AFTER the box — extraction would produce an empty string.
        is_purely_boxed = s.strip() in (
            rf"\boxed{{{boxed}}}",
            rf"$\boxed{{{boxed}}}$",
            rf"$$\boxed{{{boxed}}}$$",
        )
        has_wrapper = not is_purely_boxed
        if has_wrapper:
            result = _strip_dollar_wrappers(boxed)
            if result.strip():  # only accept non-empty extractions
                return result, "prose_extracted"

    # --- 3. Dollar-wrapped (no \boxed{}) ---
    stripped = _strip_dollar_wrappers(s)
    if stripped != s:
        return stripped, "dollar_stripped"

    # --- 3b. Single display-math extraction ---
    # Fires for prose strings with exactly one \(...\) or \[...\] block.
    extracted = _extract_single_display_math(s)
    if extracted is not None:
        return extracted, "display_math_extracted"

    # --- 4. Double-escaped LaTeX ---
    # Heuristic: if the string contains \\\\<letter> (4+ backslashes before a command)
    # but NOT \\\\<newline-equivalent> (matrix row separators are legitimately \\),
    # then halve the backslashes.
    # We detect double-escaping by checking if ALL LaTeX commands use ≥2 backslashes.
    # Specifically: if every occurrence of \\ is followed by another \\ (i.e. \\\\cmd)
    # and there are NO single-\\ commands, it's double-escaped.
    if "\\\\" in s:
        # Count single-backslash commands: \cmd (2 chars: one \, one letter)
        single_bs_cmds = len(re.findall(r"(?<!\\)\\(?!\\)[a-zA-Z]", s))
        # Count double-backslash commands: \\cmd (4 chars: two \, one letter)
        double_bs_cmds = len(re.findall(r"\\\\[a-zA-Z]", s))
        if double_bs_cmds > 0 and single_bs_cmds == 0:
            # All LaTeX commands are double-escaped → halve the backslashes
            # Replace \\\\ (4 backslashes = two BS chars) with \\ (2 backslashes = one BS)
            # But preserve \\\\\\\\ (matrix row separator \\) → \\
            fixed = re.sub(r"\\\\", r"\\", s)
            return fixed, "double_unescaped"

    # --- 5. Prose leakthrough: still looks like prose after all extraction steps ---
    # Multi-block \(...\) answers and bare multi-line equations that we can't
    # reliably parse into a single verifiable answer.
    if re.search(r"\n\s*[A-Za-z]", s) or (re.match(r"^[A-Z][a-z]+ [a-z]", s) and len(s) > 30):
        return original, "prose_dropped"

    return original, "unchanged"


# ---------------------------------------------------------------------------
# Brace depth check
# ---------------------------------------------------------------------------

def _has_unclosed_braces(s: str) -> bool:
    depth = sum(1 if c == "{" else -1 if c == "}" else 0 for c in s)
    return depth > 0


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--input",  default="data/processed/drsci_physics_deduped.parquet",
    )
    parser.add_argument(
        "--output", default="data/processed/drsci_physics_clean.parquet",
    )
    parser.add_argument("--report", action="store_true",
                        help="Print stats only — do not save")
    args = parser.parse_args()

    print(f"Loading {args.input}...", flush=True)
    df = pd.read_parquet(args.input)
    print(f"  {len(df):,} rows")

    gt_col  = "reward_model.ground_truth"
    ref_col = "extra_info.reference_answer"
    from_col = "extra_info.from"

    # --- Apply cleaning ---
    print("Cleaning ground_truth strings...", flush=True)
    cleaned = []
    tags = []
    for gt in df[gt_col].fillna("").astype(str):
        c, tag = clean_ground_truth(gt)
        cleaned.append(c)
        tags.append(tag)

    df = df.copy()
    df["_clean_tag"]        = tags
    df["_gt_original"]      = df[gt_col]
    df[gt_col]              = cleaned
    # Mirror clean to reference_answer if it matches original
    if ref_col in df.columns:
        orig_ref = df[ref_col].fillna("").astype(str)
        orig_gt  = df["_gt_original"].fillna("").astype(str)
        same_as_gt = orig_ref == orig_gt
        df.loc[same_as_gt, ref_col] = df.loc[same_as_gt, gt_col]

    # --- Report ---
    tag_counts = Counter(tags)
    total = len(df)
    print(f"\n=== Cleaning summary ===")
    for tag in ["unchanged", "prose_extracted", "display_math_extracted",
                "dollar_stripped", "double_unescaped",
                "truncated_dropped", "prose_dropped"]:
        n = tag_counts.get(tag, 0)
        print(f"  {tag:<25} {n:>6}  ({n/total:.1%})")

    print(f"\n  By source:")
    for src, grp in df.groupby(from_col, observed=True):
        src_tags = Counter(grp["_clean_tag"])
        changed = sum(v for k, v in src_tags.items() if k != "unchanged")
        print(f"  {str(src):<35}  changed={changed:>5} ({changed/len(grp):.1%})")

    # --- Drop truncated + prose ---
    drop_tags = {"truncated_dropped", "prose_dropped"}
    n_trunc = tag_counts.get("truncated_dropped", 0)
    n_prose_drop = tag_counts.get("prose_dropped", 0)
    df_clean = df[~df["_clean_tag"].isin(drop_tags)].copy()
    print(f"\n  Dropped truncated rows:  {n_trunc}")
    print(f"  Dropped prose rows:      {n_prose_drop}")
    print(f"  Final clean rows: {len(df_clean):,}")

    # --- Spot-check prose extractions ---
    extracted = df[df["_clean_tag"] == "prose_extracted"]
    if len(extracted):
        print(f"\n=== Spot-check: prose extractions (first 10) ===")
        for i, (_, row) in enumerate(extracted.head(10).iterrows()):
            orig = str(row["_gt_original"])[:80]
            new  = str(row[gt_col])[:80]
            print(f"  [{i+1}]")
            print(f"    BEFORE: {orig!r}")
            print(f"    AFTER:  {new!r}")

    # --- Spot-check display-math extractions ---
    dm_extracted = df[df["_clean_tag"] == "display_math_extracted"]
    if len(dm_extracted):
        print(f"\n=== Spot-check: display_math_extracted (first 8) ===")
        for i, (_, row) in enumerate(dm_extracted.head(8).iterrows()):
            orig = str(row["_gt_original"])[:80]
            new  = str(row[gt_col])[:80]
            print(f"  [{i+1}]")
            print(f"    BEFORE: {orig!r}")
            print(f"    AFTER:  {new!r}")

    # --- Spot-check double-unescape ---
    unescaped = df[df["_clean_tag"] == "double_unescaped"]
    if len(unescaped):
        print(f"\n=== Spot-check: double-unescape (first 5) ===")
        for i, (_, row) in enumerate(unescaped.head(5).iterrows()):
            orig = str(row["_gt_original"])[:80]
            new  = str(row[gt_col])[:80]
            print(f"  [{i+1}] BEFORE: {orig!r}")
            print(f"       AFTER:  {new!r}")

    # --- Re-run answer type distribution after cleaning ---
    print(f"\n=== Answer type distribution after cleaning ===")
    from collections import Counter as C
    types_after = [infer_answer_type(gt) for gt in df_clean[gt_col].fillna("").astype(str)]
    tc = C(types_after)
    for at in ["numerical", "expression", "equation", "mcq", "unknown"]:
        n = tc.get(at, 0)
        print(f"  {at:<15} {n:>7}  ({n/len(df_clean):.1%})")

    # --- Save ---
    if not args.report:
        # Drop internal helper columns before saving
        df_save = df_clean.drop(columns=["_clean_tag", "_gt_original"], errors="ignore")
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        df_save.to_parquet(args.output, index=False)
        print(f"\nSaved {len(df_save):,} rows → {args.output}")


if __name__ == "__main__":
    main()
