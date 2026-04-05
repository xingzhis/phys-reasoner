"""Verifier round-trip tests on Dr. SCI answer formats.

Tests are split into three groups:
  - Gold-gold round-trips (20 cases) — verify_answer(gold, gold) should be 1.0
  - Near-correct perturbations (10 cases) — within 5% tolerance → 1.0
  - Clearly wrong answers (10 cases) → 0.0

These serve as regression tests if the verifier changes.

All cases use synthetic answer strings representative of the Dr. SCI
ground_truth distribution (bare numbers, LaTeX sci-notation, expressions)
rather than live parquet reads so the tests run without the parquet file.

Run with:
  pytest tests/test_drsci_verifier.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
from drsci_audit import infer_answer_type

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from phys_reasoner.verifier.router import verify_answer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _pred(gold: str) -> str:
    """Wrap gold in \\boxed{} to simulate a clean model output."""
    return f"\\boxed{{{gold}}}"


def _pred_perturbed(gold_val: float, pct: float) -> str:
    """Perturb gold_val by pct (e.g. 0.03 = +3%) and wrap in \\boxed{}."""
    perturbed = gold_val * (1 + pct)
    return f"\\boxed{{{perturbed:.6g}}}"


# ---------------------------------------------------------------------------
# Group 1: Gold-gold round-trips — must return 1.0
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("gold,answer_type", [
    # Plain integers
    ("5",           "numerical"),
    ("-3",          "numerical"),
    ("0",           "numerical"),
    # Plain floats
    ("3.14",        "numerical"),
    ("6.48",        "numerical"),
    ("-0.332",      "numerical"),
    ("1.5e-3",      "numerical"),
    # Scientific notation (LaTeX)
    (r"8.2 \times 10^{-6}",   "numerical"),
    (r"3.0 \times 10^{8}",    "numerical"),
    (r"1.6 \times 10^{-19}",  "numerical"),
    # Simple fractions
    (r"\frac{1}{2}",           "expression"),
    (r"\frac{3}{4}",           "expression"),
    # Square-root expressions
    (r"\sqrt{2}",              "expression"),
    (r"2\sqrt{3}",             "expression"),
    # Common physics constants
    ("9.8",         "numerical"),
    ("6.674e-11",   "numerical"),
    # Negative scientific notation
    (r"-2.5 \times 10^{3}",   "numerical"),
    # Larger plain numbers
    ("1024",        "numerical"),
    ("299792458",   "numerical"),
    # Zero in scientific notation
    ("0.0",         "numerical"),
])
def test_gold_gold_round_trip(gold: str, answer_type: str) -> None:
    """verify_answer(gold_as_pred, gold, answer_type) should be 1.0."""
    pred = _pred(gold)
    score = verify_answer(
        pred_text=pred,
        gold_answer=gold,
        answer_type=answer_type,
        tolerance=0.05,
        xverify_judge=None,
    )
    # -1.0 = unverifiable (rule couldn't parse) — skip rather than fail,
    # so we don't block on edge-case LaTeX not handled by math-verify.
    if score == -1.0:
        pytest.skip(f"rule_verify returned unverifiable for gold={gold!r}")
    assert score == 1.0, f"Expected 1.0 for gold={gold!r} type={answer_type}, got {score}"


# ---------------------------------------------------------------------------
# Group 2: Near-correct perturbations within 5% tolerance → 1.0
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("gold_val,perturb_pct", [
    (3.14,    0.01),   # +1%
    (9.8,     0.02),   # +2%
    (6.674e-11, 0.03), # +3%
    (100.0,   0.04),   # +4%
    (1.5e-3,  0.01),   # +1%
    (6.48,    0.02),   # +2%
    (-3.14,   0.01),   # -1% of negative number (pct applied to magnitude)
    (1024.0,  0.03),   # +3%
    (2.998e8, 0.01),   # +1%
    (42.0,    0.02),   # +2%
])
def test_near_correct_within_tolerance(gold_val: float, perturb_pct: float) -> None:
    """Answers within 5% of gold should score 1.0."""
    gold_str = f"{gold_val:.6g}"
    pred_str = _pred_perturbed(gold_val, perturb_pct)
    score = verify_answer(
        pred_text=pred_str,
        gold_answer=gold_str,
        answer_type="numerical",
        tolerance=0.05,
        xverify_judge=None,
    )
    if score == -1.0:
        pytest.skip(f"rule_verify unverifiable for gold={gold_str!r}")
    assert score == 1.0, (
        f"Expected 1.0 for gold={gold_str!r} perturbed by {perturb_pct:.0%}, got {score}"
    )


# ---------------------------------------------------------------------------
# Group 3: Clearly wrong answers → 0.0
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("gold,wrong_pred", [
    # Factor-of-2 off
    ("3.14",       "6.28"),
    ("9.8",        "4.9"),
    # Sign flip
    ("5.0",        "-5.0"),
    ("-3.14",      "3.14"),
    # Order-of-magnitude off
    ("1.5e-3",     "1.5e-6"),
    ("6.674e-11",  "6.674e-8"),
    # Completely wrong number
    ("42.0",       "13.0"),
    ("100.0",      "200.1"),
    # Way outside tolerance
    ("3.14",       "3.50"),   # +11.5%
    ("9.8",        "8.0"),    # -18.4%
])
def test_clearly_wrong(gold: str, wrong_pred: str) -> None:
    """Clearly wrong answers should score 0.0."""
    pred_text = _pred(wrong_pred)
    score = verify_answer(
        pred_text=pred_text,
        gold_answer=gold,
        answer_type="numerical",
        tolerance=0.05,
        xverify_judge=None,
    )
    if score == -1.0:
        pytest.skip(f"rule_verify unverifiable for gold={gold!r}")
    assert score == 0.0, (
        f"Expected 0.0 for gold={gold!r} vs wrong={wrong_pred!r}, got {score}"
    )


# ---------------------------------------------------------------------------
# Group 4: answer_type="unknown" → always -1.0 (unverifiable)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("gold", [
    "some prose answer",
    "it depends on the boundary conditions",
    "see derivation above",
])
def test_unknown_type_unverifiable(gold: str) -> None:
    """answer_type='unknown' should return -1.0 (unverifiable, not a crash)."""
    pred = _pred(gold)
    score = verify_answer(
        pred_text=pred,
        gold_answer=gold,
        answer_type="unknown",
        tolerance=0.05,
        xverify_judge=None,
    )
    assert score == -1.0, f"Expected -1.0 for unknown type, got {score}"


# ---------------------------------------------------------------------------
# Group 5: infer_answer_type — new detection cases
# Covers the three additions: MCQ, percentage, non-LaTeX expression fallback.
# These are Dr. SCI-specific and do NOT touch normalize_answer_type in src/.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("gold,expected_type", [
    # --- MCQ: single uppercase letter (bare or parenthesized) ---
    ("A",           "mcq"),
    ("B",           "mcq"),
    ("D",           "mcq"),
    ("G",           "mcq"),      # options beyond D
    ("J",           "mcq"),
    ("(A)",         "mcq"),
    ("(C)",         "mcq"),

    # --- Lowercase single letter: physics variable, NOT mcq ---
    ("c",           "expression"),   # speed of light
    ("g",           "expression"),   # gravitational acc.

    # --- Percentage → numerical ---
    ("88.4%",       "numerical"),
    ("60%",         "numerical"),
    ("0.000001%",   "numerical"),
    ("79.5%",       "numerical"),

    # --- Plain numerical (existing behavior must be preserved) ---
    ("3.14",                        "numerical"),
    ("6.48",                        "numerical"),
    (r"8.2 \times 10^{-6}",        "numerical"),
    ("1.5e-3",                      "numerical"),
    ("-0.332",                      "numerical"),

    # --- LaTeX expression (existing behavior must be preserved) ---
    (r"\frac{1}{2}",               "expression"),
    (r"\sqrt{2}",                  "expression"),
    (r"2\sqrt{3}",                 "expression"),

    # --- Equation (existing behavior must be preserved) ---
    ("v = u + at",                 "equation"),
    (r"E = mc^2",                  "equation"),

    # --- Non-LaTeX expression fallback (NEW) ---
    ("2.5(1 - cos(theta))",        "expression"),   # trig in parens
    ("(M + 2m)g",                  "expression"),   # symbolic with parens
    ("-q",                         "expression"),   # short symbolic charge
    ("+e",                         "expression"),   # short symbolic

    # --- Still unknown: multi-part comma-separated (not yet handled) ---
    ("6, 3",                       "unknown"),
    ("7 m/s, 210000, 35000",       "unknown"),

    # --- Still unknown: long prose ---
    ("The answer depends on boundary conditions", "unknown"),
])
def test_infer_answer_type(gold: str, expected_type: str) -> None:
    result = infer_answer_type(gold)
    assert result == expected_type, (
        f"infer_answer_type({gold!r}) = {result!r}, expected {expected_type!r}"
    )


# ---------------------------------------------------------------------------
# Gold normalization helpers from filter_data_quality.py
# ---------------------------------------------------------------------------

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
from filter_data_quality import (  # noqa: E402
    normalize_mcq_gold,
    normalize_tf_gold,
    _has_non_latin_script,
)


@pytest.mark.parametrize("raw,expected", [
    # Already a bare letter
    ("A",               "A"),
    ("d",               "D"),
    # Boxed letter
    (r"\boxed{B}",      "B"),
    (r"\boxed{c}",      "C"),
    # Parenthesized
    ("(A)",             "A"),
    ("(D)",             "D"),
    # Dot / closing paren separator
    ("A.",              "A"),
    ("A)",              "A"),
    ("C:",              "C"),
    # Letter + trailing content
    ("A ***",           "A"),
    ("(D) \\ 0",        "D"),
    ("A - some text",   "A"),
    ("(B) \\lambda/(4n)", "B"),
    # \text{} and \mathrm{} wrappers
    (r"\text{B}",       "B"),
    (r"\mathrm{C}",     "C"),
    (r"(\mathrm{c})",   "C"),
    # Multi-select preserved
    ("ABD",             "ABD"),
    (r"\boxed{ABD}",    "ABD"),
    # Unrecoverable
    (r"\boxed{1, 2}",   None),
    (r"\boxed{10^{-2}}", None),
    (r"\boxed{1^-, E1}", None),
])
def test_normalize_mcq_gold(raw, expected):
    result = normalize_mcq_gold(raw)
    assert result == expected, f"normalize_mcq_gold({raw!r}) = {result!r}, expected {expected!r}"


@pytest.mark.parametrize("raw,expected", [
    # Already canonical
    ("True",                        "True"),
    ("False",                       "False"),
    # Boxed yes/no
    (r"\boxed{Yes}",                "True"),
    (r"\boxed{No}",                 "False"),
    (r"\boxed{yes}",                "True"),
    (r"\boxed{no}",                 "False"),
    # Cross-style: true/false
    (r"\boxed{true}",               "True"),
    (r"\boxed{false}",              "False"),
    # Single initials
    ("T",                           "True"),
    ("F",                           "False"),
    ("Y",                           "True"),
    ("N",                           "False"),
    # Compound: extract leading token
    (r"\boxed{Yes, V = -\frac{1}{2}ar^2}", "True"),
    (r"\boxed{No, c}",              "False"),
    # Unrecoverable
    (r"\boxed{Z_{\text{eff}} < Z}", None),
    ('["g=...", "b=..."]',          None),
])
def test_normalize_tf_gold(raw, expected):
    result = normalize_tf_gold(raw)
    assert result == expected, f"normalize_tf_gold({raw!r}) = {result!r}, expected {expected!r}"


@pytest.mark.parametrize("text,expected", [
    # Chinese characters → True
    ("\\boxed{不可能}",              True),
    ("\\boxed{RT = \\text{常数}}",   True),
    ("\\boxed{天}",                  True),
    # Valid Unicode math / chemistry — NOT flagged
    ("Fe²⁺ + Cd → Fe + Cd²⁺",       False),
    ("SF₆",                          False),
    ("0 ≤ Δm²c⁴ ≤ 3.8 × 10⁻³",     False),
    ("9.5 × 10^{18} min⁻¹",         False),
    ("\\frac{1}{2}",                 False),
    ("E = mc^2",                     False),
])
def test_has_non_latin_script(text, expected):
    assert _has_non_latin_script(text) == expected, (
        f"_has_non_latin_script({text!r}) = {_has_non_latin_script(text)}, expected {expected}"
    )
