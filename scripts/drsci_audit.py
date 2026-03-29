"""Answer-format audit and answer_type inference for Dr. SCI physics rows.

Runs on a stratified sample of the (optionally deduped) Dr. SCI parquet and
reports:
  - answer_type distribution (numerical / expression / equation / unknown)
  - gold-gold round-trip pass rate (verify_answer(gold, gold, answer_type))
  - pass rate by source (extra_info.from) and difficulty bucket
  - language / non-English row count
  - whether ground_truth == reference_answer at scale
  - source quality stratification with drop recommendations

Phases covered: 1b, 1c, 1d, 2a, 2b, 2c

Usage
-----
  # Audit a stratified 500-row sample from the deduped set:
  python scripts/drsci_audit.py

  # Audit the full set (slow, but gives exact numbers):
  python scripts/drsci_audit.py --input data/processed/drsci_physics.parquet --n_sample 0

  # Write answer_type labels back to a new parquet:
  python scripts/drsci_audit.py --save_typed data/processed/drsci_physics_typed.parquet
"""

from __future__ import annotations

import argparse
import re
import sys
import unicodedata
from collections import Counter
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from phys_reasoner.verifier.router import verify_answer


# ---------------------------------------------------------------------------
# Answer type inference (Phase 2a)
# ---------------------------------------------------------------------------

# Matches a LaTeX scientific-notation number: e.g. "6.48 \times 10^{-3}"
_SCI_NOTATION_RE = re.compile(
    r"[\d.]+\s*\\(?:times|cdot)\s*10\^\{?[+-]?[\d.]+\}?"
)
# LaTeX operators that indicate a symbolic expression
_LATEX_OPS_RE = re.compile(r"\\(?:frac|sqrt|pi|alpha|beta|gamma|delta|omega|"
                            r"lambda|mu|nu|sigma|phi|psi|theta|int|sum|prod|"
                            r"partial|nabla|infty|pm|mp|leq|geq|neq)")
# Detect CJK characters (non-English)
_CJK_RE = re.compile(r"[\u4e00-\u9fff\u3400-\u4dbf\uf900-\ufaff"
                     r"\u3000-\u303f\uff00-\uffef]")


def infer_answer_type(ground_truth: str) -> str:
    """Infer answer_type from a Dr. SCI ground_truth string.

    Priority order:
      1. Single uppercase letter (A–Z) or (A)–(Z) → "mcq"
      2. Bare percentage N% → "numerical"  (verifier needs % stripped externally)
      3. Parseable as float after stripping LaTeX sci-notation → "numerical"
      4. Contains top-level '=' → "equation"
      5. Contains LaTeX operators / structural chars → "expression"
      6. Non-LaTeX expression: digit + operator/letter, no comma, ≤60 chars → "expression"
      7. Short symbolic: sign + letter(s), e.g. "-q", "+e" → "expression"
      8. Otherwise → "unknown"
    """
    s = ground_truth.strip()
    if not s:
        return "unknown"

    # --- 1. MCQ: bare single uppercase letter or (A) form ---
    # Dr. SCI MCQ options are uppercase; lowercase single letters are physics
    # variables (c, g, r…) and fall through to expression/unknown detection.
    if re.fullmatch(r"[A-Z]", s) or re.fullmatch(r"\([A-Z]\)", s):
        return "mcq"

    # --- 2. Bare percentage: "88.4%", "60%", "0.000001%" ---
    # Classified as numerical; the caller must strip % before passing to the
    # verifier (rule_verify cannot parse "88.4%" directly).
    _pct = re.fullmatch(r"([+-]?[\d.]+)\s*%", s)
    if _pct:
        try:
            float(_pct.group(1))
            return "numerical"
        except ValueError:
            pass

    # --- 3. Float parse after stripping LaTeX sci-notation / units ---
    # First: if the whole string is a LaTeX sci-notation number (e.g. "8.2 \times 10^{-6}"),
    # classify directly as numerical before the LaTeX-marker check fires on the backslash.
    _sci_full = re.fullmatch(
        r"([+-]?[\d.]+)\s*\\(?:times|cdot)\s*10\^\{?([+-]?[\d.]+)\}?", s
    )
    if _sci_full:
        try:
            float(_sci_full.group(1))  # validates coefficient
            return "numerical"
        except ValueError:
            pass

    plain = _SCI_NOTATION_RE.sub("", s).strip()
    plain = re.sub(r"\\pm\s*[\d.]+", "", plain).strip()
    plain = re.sub(r"(\d)\s*[A-Za-z][A-Za-z0-9]*\s*$", r"\1", plain).strip()
    try:
        float(plain)
        return "numerical"
    except ValueError:
        pass

    # --- 4. Equation: top-level '=' (ignoring '=' inside braces) ---
    depth = 0
    for ch in s:
        if ch in "({[":
            depth += 1
        elif ch in ")}]":
            depth -= 1
        elif ch == "=" and depth == 0:
            return "equation"

    # --- 5. LaTeX expression markers ---
    if _LATEX_OPS_RE.search(s) or "^" in s or "_" in s or "\\" in s:
        return "expression"

    # --- 6. Non-LaTeX expression fallback ---
    # Catches "2.5(1 - cos(theta))", "(M + 2m)g", "1/r^2" (already above via "^"),
    # complex numbers "10 + j5", etc.
    # Guard: must have digit + operator/letter, no comma (comma → multi-part, leave
    # as unknown for now), short enough to not be prose.
    if (re.search(r"\d", s)
            and re.search(r"[a-zA-Z()\+\-\*/]", s)
            and not re.search(r",", s)
            and len(s) <= 60):
        return "expression"

    # --- 7. Short symbolic: e.g. "-q", "+e", "mc" ---
    if re.fullmatch(r"[+-]?[a-zA-Z]\w*", s):
        return "expression"

    # --- 8. Bare e-notation / Unicode multiply ---
    try:
        float(s.replace("×", "e").replace("x", "e"))
        return "numerical"
    except ValueError:
        pass

    return "unknown"


# ---------------------------------------------------------------------------
# Stratified sample
# ---------------------------------------------------------------------------

def stratified_sample(df: pd.DataFrame, n: int, seed: int = 42) -> pd.DataFrame:
    """Sample n rows stratified by (extra_info.from, difficulty bucket)."""
    if n == 0 or n >= len(df):
        return df.copy()

    diff_col = "extra_info.difficulty"
    from_col = "extra_info.from"

    # Discretize difficulty into 5 buckets
    if diff_col in df.columns:
        df = df.copy()
        df["_diff_bucket"] = pd.cut(df[diff_col].fillna(0.0),
                                    bins=[-0.001, 0.1, 0.25, 0.5, 0.75, 1.01],
                                    labels=["0.0", "0.1-0.25", "0.25-0.5", "0.5-0.75", "0.75+"])
        strat_col = "_diff_bucket"
    else:
        strat_col = None

    if from_col in df.columns and strat_col is not None:
        group_cols = [from_col, strat_col]
    elif from_col in df.columns:
        group_cols = [from_col]
    else:
        return df.sample(min(n, len(df)), random_state=seed).reset_index(drop=True)

    parts = []
    for _, grp in df.groupby(group_cols, observed=True):
        n_take = max(1, round(n * len(grp) / len(df)))
        parts.append(grp.sample(min(len(grp), n_take), random_state=seed))
    sampled = pd.concat(parts).reset_index(drop=True)
    # Trim or top-up to exactly n
    if len(sampled) > n:
        sampled = sampled.sample(n, random_state=seed).reset_index(drop=True)
    elif len(sampled) < n:
        leftover = df.drop(sampled.index, errors="ignore")
        extra = leftover.sample(min(n - len(sampled), len(leftover)), random_state=seed)
        sampled = pd.concat([sampled, extra]).reset_index(drop=True)
    return sampled


# ---------------------------------------------------------------------------
# Gold-gold round-trip
# ---------------------------------------------------------------------------

def run_round_trip(row: pd.Series) -> tuple[float, str]:
    """Run verify_answer(gold, gold, answer_type) for one Dr. SCI row.

    Returns (score, inferred_answer_type).
    score: 1.0 = pass, 0.0 = fail, -1.0 = unverifiable.
    """
    gt = str(row.get("reward_model.ground_truth") or "")
    if not gt:
        return -1.0, "unknown"

    at = infer_answer_type(gt)
    # For gold-gold the "pred_text" is just the gold wrapped in \boxed{} so the
    # extractor finds it; no model output to parse here.
    pred_text = f"\\boxed{{{gt}}}"
    try:
        score = verify_answer(
            pred_text=pred_text,
            gold_answer=gt,
            answer_type=at,
            tolerance=0.05,
            xverify_judge=None,
        )
    except Exception:
        score = -1.0
    return score, at


# ---------------------------------------------------------------------------
# Language check
# ---------------------------------------------------------------------------

def is_non_english(text: str) -> bool:
    """Return True if the text contains CJK or other non-Latin characters."""
    return bool(_CJK_RE.search(text))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--input", default="data/processed/drsci_physics_deduped.parquet",
        help="Dr. SCI parquet to audit (use drsci_physics.parquet for pre-dedup audit)",
    )
    parser.add_argument(
        "--n_sample", type=int, default=500,
        help="Stratified sample size (0 = full dataset, slower)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
    )
    parser.add_argument(
        "--save_typed", default=None,
        help="If set, save input parquet with 'inferred_answer_type' column added",
    )
    args = parser.parse_args()

    if not Path(args.input).exists():
        # Fall back to pre-dedup file
        fallback = "data/processed/drsci_physics.parquet"
        if Path(fallback).exists():
            print(f"  {args.input} not found — falling back to {fallback}", flush=True)
            args.input = fallback
        else:
            print(f"ERROR: {args.input} not found. Run drsci_dedup.py first.", flush=True)
            sys.exit(1)

    print(f"Loading {args.input}...", flush=True)
    df = pd.read_parquet(args.input)
    print(f"  Total rows: {len(df)}")

    # --- 1d. Language filter ---
    q_col = "extra_info.question"
    if q_col in df.columns:
        non_english = df[q_col].apply(lambda x: is_non_english(str(x) if pd.notna(x) else ""))
        n_non_en = non_english.sum()
        pct_non_en = n_non_en / max(len(df), 1)
        print(f"\n=== Language check ===")
        print(f"  Non-English (CJK) rows: {n_non_en} ({pct_non_en:.2%})")
        if pct_non_en > 0.001:
            print(f"  ACTION NEEDED: >0.1% non-English — add language filter")
        else:
            print(f"  OK: <0.1% non-English")

    # --- 1b. ground_truth vs reference_answer equivalence ---
    if "extra_info.reference_answer" in df.columns and "reward_model.ground_truth" in df.columns:
        gt_col = df["reward_model.ground_truth"].astype(str)
        ref_col = df["extra_info.reference_answer"].astype(str)
        n_mismatch = (gt_col != ref_col).sum()
        print(f"\n=== ground_truth vs reference_answer ===")
        print(f"  Mismatches: {n_mismatch} / {len(df)} ({n_mismatch / max(len(df), 1):.2%})")

    # --- Sample for the remaining checks ---
    if args.n_sample > 0:
        print(f"\nSampling {args.n_sample} rows (stratified by source + difficulty)...")
        sample = stratified_sample(df, args.n_sample, seed=args.seed)
    else:
        print(f"\nRunning on full dataset ({len(df)} rows)...")
        sample = df.copy()
    print(f"  Sample size: {len(sample)} rows")

    # --- 2a. Answer type inference ---
    print(f"\n=== Answer type inference (Phase 2a) ===", flush=True)
    scores = []
    answer_types = []
    for _, row in sample.iterrows():
        score, at = run_round_trip(row)
        scores.append(score)
        answer_types.append(at)
    sample = sample.copy()
    sample["inferred_answer_type"] = answer_types
    sample["round_trip_score"] = scores

    at_counts = Counter(answer_types)
    print("  Answer type distribution:")
    for at, n in at_counts.most_common():
        print(f"    {at:<15} {n}  ({n / max(len(sample), 1):.1%})")

    # --- 2b. Gold-gold round-trip pass rate ---
    pass_rate = sum(1 for s in scores if s == 1.0) / max(len(scores), 1)
    fail_rate = sum(1 for s in scores if s == 0.0) / max(len(scores), 1)
    unverif_rate = sum(1 for s in scores if s == -1.0) / max(len(scores), 1)

    print(f"\n=== Gold-gold round-trip (Phase 2b) ===")
    print(f"  Pass (1.0):          {sum(s == 1.0 for s in scores):4d}  ({pass_rate:.1%})")
    print(f"  Fail (0.0):          {sum(s == 0.0 for s in scores):4d}  ({fail_rate:.1%})")
    print(f"  Unverifiable (-1.0): {sum(s == -1.0 for s in scores):4d}  ({unverif_rate:.1%})")

    if pass_rate < 0.90:
        print(f"  WARNING: gold-gold pass rate {pass_rate:.1%} < 90% — "
              "answer normalization step needed before using Dr. SCI")
    else:
        print(f"  OK: gold-gold pass rate {pass_rate:.1%} ≥ 90%")

    # Pass rate per answer type
    print("\n  Round-trip by answer type:")
    for at in at_counts:
        at_rows = [(s, t) for s, t in zip(scores, answer_types) if t == at]
        if not at_rows:
            continue
        at_pass = sum(1 for s, _ in at_rows if s == 1.0) / len(at_rows)
        at_unverif = sum(1 for s, _ in at_rows if s == -1.0) / len(at_rows)
        print(f"    {at:<15} pass={at_pass:.1%}  unverif={at_unverif:.1%}  (n={len(at_rows)})")

    # --- 1c. Source quality stratification ---
    from_col = "extra_info.from"
    if from_col in sample.columns:
        print(f"\n=== Source quality stratification (Phase 1c) ===")
        for src, grp in sample.groupby(from_col, observed=True):
            src_scores = grp["round_trip_score"].tolist()
            src_pass = sum(s == 1.0 for s in src_scores) / max(len(src_scores), 1)
            src_unverif = sum(s == -1.0 for s in src_scores) / max(len(src_scores), 1)
            flag = ""
            # Threshold is on confirmed-wrong rate, not rule-tier pass rate.
            # High unverifiable fractions are expected for equation-heavy sources
            # (natural_reasoning is 52% equations — unverifiable ≠ wrong).
            src_wrong = sum(s == 0.0 for s in src_scores) / max(len(src_scores), 1)
            if src_wrong > 0.05:
                flag = f"  *** DROP RECOMMENDATION: confirmed-wrong rate {src_wrong:.1%} > 5%"
            elif src_unverif > 0.60:
                flag = "  (high unverif — xVerify needed in reward loop)"
            print(f"  {str(src):<35} pass={src_pass:.1%}  unverif={src_unverif:.1%}  "
                  f"n={len(src_scores)}{flag}")

    # --- Pass rate by difficulty bucket ---
    diff_col = "extra_info.difficulty"
    if diff_col in sample.columns:
        print(f"\n=== Round-trip by difficulty bucket ===")
        sample_cp = sample.copy()
        sample_cp["_diff_bucket"] = pd.cut(
            sample_cp[diff_col].fillna(0.0),
            bins=[-0.001, 0.1, 0.25, 0.5, 0.75, 1.01],
            labels=["0.0", "0.1-0.25", "0.25-0.5", "0.5-0.75", "0.75+"],
        )
        for bucket, grp in sample_cp.groupby("_diff_bucket", observed=True):
            b_scores = grp["round_trip_score"].tolist()
            b_pass = sum(s == 1.0 for s in b_scores) / max(len(b_scores), 1)
            print(f"  diff={bucket:<10} pass={b_pass:.1%}  n={len(b_scores)}")

    # --- 2c. xVerify need assessment ---
    unverif_count = sum(s == -1.0 for s in scores)
    unverif_pct = unverif_count / max(len(scores), 1)
    print(f"\n=== xVerify need assessment (Phase 2c) ===")
    print(f"  Unverifiable by rule tier: {unverif_count} ({unverif_pct:.1%})")
    if unverif_pct > 0.20:
        print(f"  ACTION: >20% unverifiable — xVerify needed in reward loop for Dr. SCI")
    elif unverif_pct > 0.05:
        print(f"  NOTE: {unverif_pct:.1%} unverifiable — xVerify adds latency but may be worthwhile")
    else:
        print(f"  OK: <5% unverifiable — Dr. SCI cleaner than existing 6.8k")

    # --- Sample failures for manual inspection ---
    fail_rows = sample[sample["round_trip_score"] != 1.0]
    if len(fail_rows) > 0:
        print(f"\n=== Sample failures (up to 20, for manual inspection) ===")
        for i, (_, row) in enumerate(fail_rows.head(20).iterrows()):
            gt = str(row.get("reward_model.ground_truth", ""))[:80]
            at = row.get("inferred_answer_type", "?")
            sc = row.get("round_trip_score", "?")
            src = row.get("extra_info.from", "?")
            print(f"  [{i+1:2d}] score={sc}  type={at:<12}  from={str(src):<30}  gt={gt!r}")

    # --- Save typed parquet ---
    if args.save_typed:
        print(f"\nWriting inferred_answer_type to full dataset...")
        full_df = pd.read_parquet(args.input)
        # Infer answer types for the full dataset
        all_types = []
        for _, row in full_df.iterrows():
            gt = str(row.get("reward_model.ground_truth") or "")
            all_types.append(infer_answer_type(gt))
        full_df = full_df.copy()
        full_df["inferred_answer_type"] = all_types
        Path(args.save_typed).parent.mkdir(parents=True, exist_ok=True)
        full_df.to_parquet(args.save_typed, index=False)
        at_full_counts = Counter(all_types)
        print(f"  Saved {len(full_df)} rows → {args.save_typed}")
        print("  Full answer type distribution:")
        for at, n in at_full_counts.most_common():
            print(f"    {at:<15} {n}  ({n / max(len(full_df), 1):.1%})")


if __name__ == "__main__":
    main()
