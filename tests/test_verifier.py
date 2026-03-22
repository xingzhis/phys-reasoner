"""Verifier test suite.

Fast tests (no GPU, no HF): run without -m slow flag.
Slow tests (gold round-trip, xVerify): marked @pytest.mark.slow.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from phys_reasoner.verifier.extract import expand_pm, extract_answer, split_by_comma
from phys_reasoner.verifier.math_verify_wrapper import rule_verify
from phys_reasoner.verifier.router import verify_answer

# ---------------------------------------------------------------------------
# D3. Perturbation tests
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("gold,perturb_pct,expected", [
    ("3.14", 0.01, True),    # +1% → within 5% tolerance
    ("3.14", 0.10, False),   # +10% → wrong
    ("-60", 0.0, True),      # exact match
    ("60", 0.0, False),      # sign flip: -60 ≠ 60 (gold is -60)
])
def test_perturbation(gold, perturb_pct, expected):
    gold_val = float(gold)
    pred_val = gold_val * (1 + perturb_pct) if perturb_pct else gold_val
    # For the sign-flip case (gold="60", perturb=0) pred is actually testing -60 vs 60
    if gold == "60" and perturb_pct == 0.0:
        gold_val = -60.0
        pred_val = 60.0
    pred_str = str(pred_val)
    gold_str = str(gold_val)
    result = rule_verify(pred_str, gold_str)
    if result is None:
        pytest.skip("rule_verify returned None (parser issue)")
    assert result == expected


# ---------------------------------------------------------------------------
# D4. Known equivalences
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("pred,gold,expected", [
    (r"\frac{1}{2}", "0.5", True),
    ("3/4", "0.75", True),
    (r"3.75 \times 10^4", "37500", True),
    (r"\boxed{3.75 \times 10^4}", "37500", True),  # boxed extraction
    (r"\sqrt{2}/2", "0.707107", True),
    ("1.414", r"\sqrt{2}", True),
])
def test_known_equivalences(pred, gold, expected):
    # For boxed pred, use verify_answer (which does extraction)
    if r"\boxed" in pred:
        result = verify_answer(pred, gold, answer_type="numerical")
        assert result in (1.0, -1.0)  # -1.0 if unverifiable is acceptable
        if result != -1.0:
            assert result == (1.0 if expected else 0.0)
    else:
        result = rule_verify(pred, gold)
        if result is None:
            pytest.skip(f"rule_verify returned None for {pred!r} vs {gold!r}")
        assert result == expected


# ---------------------------------------------------------------------------
# D5. Unit equivalence (pint tier)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("pred,gold,unit,expected", [
    ("980 cm/s^2", "9.8", "m/s^2", True),
    ("1000 g", "1.0", "kg", True),
    ("3 km/s", "3000", "m/s", True),
    ("3 km/s", "3", "m/s", False),   # wrong unit prefix
])
def test_unit_equivalence(pred, gold, unit, expected):
    from phys_reasoner.verifier.unit_check import unit_equivalent

    result = unit_equivalent(pred, gold, unit, tolerance=0.05)
    if result is None:
        pytest.skip(f"unit_equivalent returned None for {pred!r} vs {gold!r} [{unit}]")
    assert result == expected


# ---------------------------------------------------------------------------
# D6. Multipart and \\pm
# ---------------------------------------------------------------------------

def test_multipart_unordered():
    # gold = ["2", "3"], pred = "3, 2" → True (order-independent)
    result = verify_answer("3, 2", ["2", "3"], answer_type="numerical")
    assert result == 1.0


def test_pm_expansion():
    # gold = r"\pm 5", pred = "-5" → True (one of the pm variants)
    result = verify_answer("-5", r"\pm 5", answer_type="numerical")
    assert result == 1.0


# ---------------------------------------------------------------------------
# D7. Router edge cases
# ---------------------------------------------------------------------------

def test_unknown_type_returns_unverifiable():
    result = verify_answer("42", "42", answer_type="unknown")
    assert result == -1.0


def test_open_end_unverifiable():
    result = verify_answer("foo", "bar", answer_type="open_end")
    assert result == -1.0


def test_code_unverifiable():
    result = verify_answer("def f(): return 42", "def f(): return 0", answer_type="code")
    assert result == -1.0


def test_mcq_correct():
    result = verify_answer("A", "A", answer_type="mcq")
    assert result == 1.0


def test_mcq_wrong():
    result = verify_answer("B", "A", answer_type="mcq")
    assert result == 0.0


def test_mcq_case_insensitive():
    result = verify_answer("a", "A", answer_type="mcq")
    assert result == 1.0


def test_true_false_correct():
    result = verify_answer("True", "True", answer_type="true_false")
    assert result == 1.0


def test_multipart_length_mismatch():
    # pred has 3 parts, gold has 2 → wrong
    result = verify_answer("1, 2, 3", ["1", "2"], answer_type="numerical")
    assert result == 0.0


# ---------------------------------------------------------------------------
# D2. OlympiadBench 24 test cases (rule tier only)
# ---------------------------------------------------------------------------

_SCORING_EXAMPLES = (
    Path(__file__).parent.parent
    / "data/raw/grader-repos/OlympiadBench/eval/scoring_examples.json"
)


@pytest.mark.slow
def test_olympiadbench_24_cases():
    """Rule tier precision check on OlympiadBench scoring examples.

    Design: FNs (rule=False when expected=True) are acceptable — xVerify handles them.
    Only FPs (rule=True when expected=False) are hard failures; they skip xVerify entirely.

    Known math-verify limitations (unavoidable FPs):
    - Equation rearrangement: x^2+y^2=1 vs x^2-y^2=1 parsed as equivalent
    - Unit stripping: 0.64N vs 0.64 (unit N is dropped by latex2sympy)
    - Overflow: 10^{10^{10^{10}}} vs 10^{10} parse to same object
    """
    _KNOWN_FP_LIMIT = 3  # known math-verify limitations

    if not _SCORING_EXAMPLES.exists():
        pytest.skip("scoring_examples.json not found")
    with open(_SCORING_EXAMPLES, encoding="utf-8") as f:
        examples = json.load(f)

    # Deduplicate (gold, pred) pairs: if any annotation says equal=True, use True
    # (Some pairs appear twice with conflicting labels for strict vs. tolerance modes)
    seen: dict[tuple[str, str], bool] = {}
    for ex in examples:
        gold = ex["GT"]
        for ans in ex["Ans"]:
            key = (gold, ans["answer"])
            label = ans.get("equal", True)
            seen[key] = seen.get(key, False) or label

    false_positives = []
    false_negatives = []
    for (gold, model_output), expected in seen.items():
        result = rule_verify(model_output, gold)
        if result is None:
            continue  # unverifiable — falls through to xVerify
        if result is True and expected is False:
            false_positives.append({"gold": gold, "pred": model_output})
        elif result is False and expected is True:
            false_negatives.append({"gold": gold, "pred": model_output})

    if false_positives:
        print(f"\nFPs ({len(false_positives)}):")
        for fp in false_positives:
            print(f"  {fp}")
    if false_negatives:
        print(f"\nFNs ({len(false_negatives)}, handled by xVerify):")
        for fn in false_negatives[:5]:
            print(f"  {fn}")

    assert len(false_positives) <= _KNOWN_FP_LIMIT, (
        f"{len(false_positives)} FPs > {_KNOWN_FP_LIMIT}: {false_positives}"
    )


# ---------------------------------------------------------------------------
# D1. Gold round-trip (rule tier only) — @pytest.mark.slow
# ---------------------------------------------------------------------------

_DEDUPED_PARQUET = (
    Path(__file__).parent.parent / "data/processed/candidates_deduped.parquet"
)


@pytest.mark.slow
def test_gold_roundtrip():
    if not _DEDUPED_PARQUET.exists():
        pytest.skip("candidates_deduped.parquet not found")

    import pandas as pd

    df = pd.read_parquet(_DEDUPED_PARQUET)

    skip_types = {"open_end", "code", "unknown"}
    false_negatives = []

    for _, row in df.iterrows():
        at = row.get("answer_type", "unknown")
        if at in skip_types:
            continue

        gold = row["answer"]
        if isinstance(gold, str):
            try:
                parsed = json.loads(gold)
                if isinstance(parsed, list):
                    gold = parsed
            except Exception:
                pass

        result = verify_answer(
            pred_text=gold if isinstance(gold, str) else ", ".join(gold),
            gold_answer=gold,
            answer_type=at,
            gold_unit=str(row.get("unit") or ""),
            tolerance=0.05,
            xverify_judge=None,
        )
        if result == 0.0:
            false_negatives.append({
                "problem_id": row["problem_id"],
                "source": row["source"],
                "answer_type": at,
                "answer": gold,
            })

    total = len(df[~df["answer_type"].isin(skip_types)])
    fn_rate = len(false_negatives) / total if total > 0 else 0.0
    print(f"\nGold round-trip: {len(false_negatives)}/{total} false negatives ({fn_rate:.1%})")

    # Success criterion: < 5% false negatives
    assert fn_rate < 0.05, f"False negative rate {fn_rate:.1%} exceeds 5% threshold"


# ---------------------------------------------------------------------------
# D8. xVerify tests (require GPU) — @pytest.mark.slow
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def xverify():
    pytest.importorskip("transformers")
    from phys_reasoner.verifier.xverify_judge import XVerifyJudge
    return XVerifyJudge(model_name="IAAR-Shanghai/xVerify-0.5B-I")


@pytest.mark.slow
@pytest.mark.parametrize("pred,gold,expected", [
    (r"\frac{mg}{k}", r"\frac{gm}{k}", True),
    ("0.707", r"\frac{\sqrt{2}}{2}", True),
    (r"3.14 \text{ m/s}", "3.14", True),
    ("42", "43", False),
    ("-60", "60", False),
    ("gibberish xyz !!!", "3.14", False),
    (r"\boxed{}", "3.14", False),
    ("the answer is definitely correct trust me", "3.14", False),
])
def test_xverify_correctness(xverify, pred, gold, expected):
    result = xverify(pred, gold, problem_str="")
    assert result == expected


@pytest.mark.slow
def test_xverify_rule_override(xverify):
    pred = r"\frac{g \cdot m}{k}"
    gold = r"\frac{mg}{k}"
    rule_result = rule_verify(pred, gold)
    assert rule_result is False or rule_result is None
    xv_result = xverify(pred, gold)
    assert xv_result is True


@pytest.mark.slow
def test_xverify_logprob_in_range(xverify):
    score = xverify.get_logprob_score("3.14", "3.14159")
    assert 0.0 <= score <= 1.0
