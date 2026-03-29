"""Explore Dr. SCI answer format distribution and compare to existing 6.8k corpus.

Answers four questions:
  1. Unit stripping — are units still embedded in ground_truth strings?
     (Detecting bare unit suffixes, dimensional strings, SI symbols)
  2. Answer type diversity — numeric / expression / equation distribution
     vs our curated 6.8k (which has all types including expression and equation)
  3. Distribution implications — what answer types are missing or over-represented,
     and what that means for the verifier and training reward signal
  4. Spot-checks — sample rows for each answer type and unit pattern

Usage
-----
  python scripts/drsci_explore_answers.py
  python scripts/drsci_explore_answers.py --input data/processed/drsci_physics.parquet
  python scripts/drsci_explore_answers.py --n_sample 0   # full dataset (slow)
"""

from __future__ import annotations

import argparse
import re
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Reuse answer_type inference from audit script
sys.path.insert(0, str(Path(__file__).parent))
from drsci_audit import infer_answer_type


# ---------------------------------------------------------------------------
# Unit detection heuristics
# ---------------------------------------------------------------------------

# Common SI / physics unit symbols (bare, not in LaTeX commands)
_SI_UNITS = {
    # Length
    "m", "km", "cm", "mm", "nm", "pm", "fm", "um",
    "Angstrom", "A",
    # Mass
    "kg", "g", "mg", "ug",
    # Time
    "s", "ms", "us", "ns", "ps", "min", "h", "hr",
    # Current
    "A",  # ampere (also Angstrom — ambiguous)
    # Temperature
    "K", "C",  # Celsius — bare C is risky
    # Frequency
    "Hz", "kHz", "MHz", "GHz",
    # Energy
    "J", "kJ", "MJ", "eV", "keV", "MeV", "GeV",
    # Power
    "W", "kW", "MW",
    # Force
    "N", "kN",
    # Pressure
    "Pa", "kPa", "MPa", "atm", "bar",
    # Charge
    "C",  # coulombs
    # Voltage / EMF
    "V", "mV", "kV",
    # Resistance
    "Ohm",
    # Capacitance
    "F", "pF", "nF", "uF",
    # Magnetic flux
    "T", "mT", "uT",
    # Angle
    "rad", "deg",
    # Speed
    "c",  # speed of light
}

# Pattern: digit(s) immediately followed by a known unit (word boundary)
# e.g. "9.8m/s", "100N", "6.4eV", "273K"
_BARE_UNIT_RE = re.compile(
    r"\d\s*("
    + "|".join(sorted(_SI_UNITS, key=len, reverse=True))  # longest match first
    + r")(?:\b|$|[/\s,;])"
)

# LaTeX unit commands (unit embedded in LaTeX math)
_LATEX_UNIT_RE = re.compile(
    r"\\(?:text|mathrm|rm)\{([^}]+)\}|"  # \text{m}, \mathrm{kg}
    r"\\(?:si|SI)\{[^}]+\}|"             # siunitx \si{...}
    r"\\(?:meter|kilogram|second|ampere|kelvin|mole|candela|"
    r"newton|pascal|joule|watt|coulomb|volt|ohm|farad|tesla|"
    r"hertz|electronvolt|angstrom)"
)


def detect_unit(s: str) -> str:
    """Classify whether a ground_truth string has units.

    Returns one of:
      'bare_unit'   — digit followed by a bare SI symbol (e.g. "9.8m", "100N")
      'latex_unit'  — unit embedded in LaTeX command (e.g. "\\text{m}")
      'fraction_unit' — looks like "value unit" with a space before letters
      'no_unit'     — no detectable unit (pure number or symbolic expression)
    """
    s_stripped = s.strip()

    # LaTeX unit command check
    if _LATEX_UNIT_RE.search(s_stripped):
        return "latex_unit"

    # Bare unit attached to digit
    if _BARE_UNIT_RE.search(s_stripped):
        return "bare_unit"

    # "3.14 m" or "9.8 m/s^2" — value space unit
    # Heuristic: ends with letter(s) after a space following digits
    if re.search(r"\d\s+[A-Za-z][A-Za-z0-9/^]*\s*$", s_stripped):
        return "fraction_unit"

    return "no_unit"


# ---------------------------------------------------------------------------
# Compare with existing 6.8k corpus
# ---------------------------------------------------------------------------

def load_corpus_answer_types(corpus_path: str) -> Counter:
    """Load answer_type distribution from candidates_deduped.parquet."""
    df = pd.read_parquet(corpus_path)
    counts: Counter = Counter()
    for val in df["answer_type"]:
        if isinstance(val, str):
            try:
                import json
                parsed = json.loads(val)
                if isinstance(parsed, list):
                    for t in parsed:
                        counts[str(t).lower()] += 1
                    continue
            except Exception:
                pass
        counts[str(val).lower()] += 1
    return counts


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--input", default="data/processed/drsci_physics.parquet",
        help="Dr. SCI parquet to explore",
    )
    parser.add_argument(
        "--corpus", default="data/processed/candidates_deduped.parquet",
        help="Existing 6.8k training corpus for comparison",
    )
    parser.add_argument(
        "--n_sample", type=int, default=2000,
        help="Rows to sample for analysis (0 = full dataset)",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print(f"Loading {args.input}...", flush=True)
    df = pd.read_parquet(args.input)
    print(f"  Total rows: {len(df):,}")

    if args.n_sample > 0 and args.n_sample < len(df):
        df_sample = df.sample(args.n_sample, random_state=args.seed).reset_index(drop=True)
        print(f"  Using {len(df_sample):,}-row random sample")
    else:
        df_sample = df.copy()
        print(f"  Using full dataset")

    gt_col = "reward_model.ground_truth"
    ref_col = "extra_info.reference_answer"
    from_col = "extra_info.from"
    diff_col = "extra_info.difficulty"

    if gt_col not in df_sample.columns:
        print(f"ERROR: column '{gt_col}' not found. Available: {list(df_sample.columns)}")
        sys.exit(1)

    gts = df_sample[gt_col].fillna("").astype(str).tolist()

    # ---------------------------------------------------------------------------
    # 1. Answer type distribution
    # ---------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("1. ANSWER TYPE DISTRIBUTION (inferred)")
    print("=" * 70)

    answer_types = [infer_answer_type(gt) for gt in gts]
    at_counts = Counter(answer_types)
    total = len(answer_types)

    for at in ["numerical", "expression", "equation", "unknown"]:
        n = at_counts.get(at, 0)
        print(f"  {at:<15} {n:>6}  ({n/total:.1%})")
    print(f"  {'TOTAL':<15} {total:>6}")

    # ---------------------------------------------------------------------------
    # 2. Unit detection
    # ---------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("2. UNIT DETECTION IN ground_truth")
    print("=" * 70)

    unit_tags = [detect_unit(gt) for gt in gts]
    unit_counts = Counter(unit_tags)

    for tag in ["no_unit", "bare_unit", "fraction_unit", "latex_unit"]:
        n = unit_counts.get(tag, 0)
        print(f"  {tag:<20} {n:>6}  ({n/total:.1%})")

    n_with_units = sum(v for k, v in unit_counts.items() if k != "no_unit")
    print(f"\n  Rows WITH detectable units: {n_with_units} ({n_with_units/total:.1%})")
    print(f"  Rows WITHOUT detectable units (pure number/expr): {unit_counts['no_unit']} "
          f"({unit_counts['no_unit']/total:.1%})")

    if n_with_units / total > 0.05:
        print(f"\n  NOTE: >5% of answers have units embedded in the string.")
        print(f"  IMPLICATION: Our verifier will strip bare units (FP2 guard) and return None,")
        print(f"  falling through to xVerify. Unit-bearing rows will need xVerify in the reward loop.")
    else:
        print(f"\n  OK: <5% unit-bearing rows — unit stripping is not a major issue.")

    # ---------------------------------------------------------------------------
    # 3. Unit detection by answer type
    # ---------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("3. UNIT PRESENCE BY ANSWER TYPE")
    print("=" * 70)

    df_sample = df_sample.copy()
    df_sample["_at"] = answer_types
    df_sample["_unit_tag"] = unit_tags

    for at in ["numerical", "expression", "equation", "unknown"]:
        subset = df_sample[df_sample["_at"] == at]
        if len(subset) == 0:
            continue
        u_counts = Counter(subset["_unit_tag"])
        n_with = sum(v for k, v in u_counts.items() if k != "no_unit")
        print(f"  {at:<15} n={len(subset):>5}  "
              f"with_unit={n_with:>4} ({n_with/len(subset):.1%})  "
              f"no_unit={u_counts['no_unit']:>4} ({u_counts['no_unit']/len(subset):.1%})")

    # ---------------------------------------------------------------------------
    # 4. Answer type by source
    # ---------------------------------------------------------------------------
    if from_col in df_sample.columns:
        print("\n" + "=" * 70)
        print("4. ANSWER TYPE DISTRIBUTION BY SOURCE (extra_info.from)")
        print("=" * 70)
        for src, grp in df_sample.groupby(from_col, observed=True):
            at_by_src = Counter(grp["_at"])
            n = len(grp)
            parts = "  ".join(
                f"{at}={at_by_src.get(at, 0)/n:.0%}"
                for at in ["numerical", "expression", "equation", "unknown"]
            )
            print(f"  {str(src):<35}  n={n:>5}  {parts}")

    # ---------------------------------------------------------------------------
    # 5. Comparison with existing 6.8k corpus
    # ---------------------------------------------------------------------------
    if Path(args.corpus).exists():
        print("\n" + "=" * 70)
        print("5. COMPARISON WITH EXISTING 6.8k CORPUS")
        print("=" * 70)
        corpus_counts = load_corpus_answer_types(args.corpus)
        corpus_total = sum(corpus_counts.values())

        print(f"  {'Answer type':<20} {'Dr. SCI':>12}  {'6.8k corpus':>12}")
        print(f"  {'-'*20} {'-'*12}  {'-'*12}")
        for at in ["numerical", "expression", "equation", "symbolic", "mcq",
                   "true_false", "interval", "open_end", "unknown"]:
            drsci_n = at_counts.get(at, 0)
            corpus_n = corpus_counts.get(at, 0)
            if drsci_n == 0 and corpus_n == 0:
                continue
            drsci_pct = f"{drsci_n/total:.1%}" if total else "–"
            corpus_pct = f"{corpus_n/corpus_total:.1%}" if corpus_total else "–"
            print(f"  {at:<20} {drsci_n:>6} ({drsci_pct:>5})  "
                  f"{corpus_n:>6} ({corpus_pct:>5})")

        print(f"\n  Implications:")

        # Numerical dominance
        drsci_num_pct = at_counts.get("numerical", 0) / total
        corpus_num_pct = corpus_counts.get("numerical", 0) / corpus_total
        if drsci_num_pct > corpus_num_pct + 0.10:
            print(f"  [!] Dr. SCI is NUMERICAL-HEAVY ({drsci_num_pct:.0%} vs corpus {corpus_num_pct:.0%}).")
            print(f"      Routing policy may overfit to numerical problems if Dr. SCI is the primary source.")
            print(f"      Consider: upsample corpus expression/equation rows, or add expression-heavy supplements.")

        # Expression coverage
        drsci_expr_pct = at_counts.get("expression", 0) / total
        corpus_expr_pct = corpus_counts.get("expression", 0) / corpus_total
        if drsci_expr_pct < corpus_expr_pct - 0.10:
            print(f"  [!] Dr. SCI has FEWER expression-type answers ({drsci_expr_pct:.0%} vs {corpus_expr_pct:.0%}).")
            print(f"      Tool-Check and Check actions are most useful for expression/equation types.")
            print(f"      If Dr. SCI becomes primary, Tool-Check training signal will be weaker.")

        # Equation coverage
        drsci_eq_pct = at_counts.get("equation", 0) / total
        corpus_eq_pct = corpus_counts.get("equation", 0) / corpus_total
        if drsci_eq_pct < corpus_eq_pct - 0.05:
            print(f"  [!] Dr. SCI has fewer equation-type answers ({drsci_eq_pct:.0%} vs {corpus_eq_pct:.0%}).")

        # Unknown / unverifiable
        drsci_unk_pct = at_counts.get("unknown", 0) / total
        if drsci_unk_pct > 0.10:
            print(f"  [!] {drsci_unk_pct:.0%} of Dr. SCI answers are 'unknown' type.")
            print(f"      These will return -1.0 from rule_verify and need xVerify in the reward loop.")

    # ---------------------------------------------------------------------------
    # 6. Spot-checks (sample rows per answer type and unit pattern)
    # ---------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("6. SPOT-CHECKS (5 samples per answer type)")
    print("=" * 70)

    for at in ["numerical", "expression", "equation", "unknown"]:
        subset = df_sample[df_sample["_at"] == at]
        if len(subset) == 0:
            continue
        print(f"\n  --- {at.upper()} ({len(subset)} rows in sample) ---")
        for i, (_, row) in enumerate(subset.head(5).iterrows()):
            gt = str(row.get(gt_col, ""))[:80]
            unit_tag = row.get("_unit_tag", "?")
            src = str(row.get(from_col, "?"))[:30]
            diff = row.get(diff_col, "?")
            print(f"    [{i+1}] gt={gt!r:<60}  unit={unit_tag}  src={src}  diff={diff}")

    print("\n  --- ROWS WITH DETECTED UNITS ---")
    unit_rows = df_sample[df_sample["_unit_tag"] != "no_unit"]
    for i, (_, row) in enumerate(unit_rows.head(10).iterrows()):
        gt = str(row.get(gt_col, ""))[:80]
        unit_tag = row.get("_unit_tag", "?")
        at = row.get("_at", "?")
        src = str(row.get(from_col, "?"))[:30]
        print(f"  [{i+1}] {unit_tag:<15} type={at:<12}  gt={gt!r}  src={src}")

    # ---------------------------------------------------------------------------
    # 7. Summary for corpus decision
    # ---------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("7. SUMMARY FOR CORPUS INTEGRATION DECISION")
    print("=" * 70)

    print(f"  Answer type diversity:")
    for at in ["numerical", "expression", "equation", "unknown"]:
        n = at_counts.get(at, 0)
        print(f"    {at:<15} {n/total:.1%}")

    print(f"\n  Unit stripping:")
    print(f"    Rows with units in ground_truth: {n_with_units/total:.1%}")
    if n_with_units / total < 0.05:
        print(f"    -> Units are largely stripped. Verifier can score directly.")
    else:
        print(f"    -> Some units present. Verifier FP2 guard will route to xVerify.")

    n_numerical = at_counts.get("numerical", 0)
    n_expr = at_counts.get("expression", 0)
    n_eq = at_counts.get("equation", 0)
    n_unk = at_counts.get("unknown", 0)

    print(f"\n  For the four-action routing policy:")
    print(f"    Answer/Check actions: work well for numerical-type problems ({n_numerical/total:.0%} of Dr. SCI)")
    print(f"    Tool-Check (check_equation, compare_expr): needs expression/equation types "
          f"({(n_expr+n_eq)/total:.0%} of Dr. SCI)")
    print(f"    xVerify fallback needed for: unknown type + unit-bearing rows "
          f"(~{(n_unk+n_with_units)/total:.0%} combined)")

    if (n_expr + n_eq) / total < 0.15:
        print(f"\n  [RECOMMENDATION] Dr. SCI is numerically-dominated.")
        print(f"  Retain existing 6.8k as a hard-problem + expression/equation supplement")
        print(f"  to ensure the Tool-Check action has enough training signal.")


if __name__ == "__main__":
    main()
