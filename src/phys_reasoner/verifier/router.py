"""Main verifier router: dispatches to unit-check, rule-verify, and xVerify."""

from __future__ import annotations

import itertools

from phys_reasoner.data.normalize import normalize_answer_type
from phys_reasoner.verifier.extract import _extract_boxed, expand_pm, extract_answer, split_by_comma
from phys_reasoner.verifier.math_verify_wrapper import rule_verify
from phys_reasoner.verifier.unit_check import unit_equivalent


def verify_answer(
    pred_text: str,
    gold_answer: str | list[str],
    answer_type: str | list[str],
    gold_unit: str = "",
    tolerance: float = 0.05,
    xverify_judge=None,
    problem_text: str = "",
    always_use_xverify: bool = False,
) -> float:
    """Verify a model answer against the gold answer.

    Returns:
        1.0  — correct
        0.0  — confirmed wrong
        -1.0 — unverifiable (open_end, code, or rule couldn't parse with no xVerify)
    """
    # --- Normalize answer_type (handles raw parquet labels like "NV", "MCQ", "EX", etc.) ---
    answer_type = normalize_answer_type(answer_type)

    # --- Unverifiable types ---
    primary_type = answer_type[0] if isinstance(answer_type, list) else answer_type
    if primary_type in ("open_end", "code", "unknown"):
        return -1.0

    # --- MCQ / True-False: exact string match, no rule engine needed ---
    if primary_type in ("mcq", "true_false"):
        pred_parts = _extract_pred_parts(pred_text)
        pred_str = pred_parts[0] if pred_parts else pred_text.strip()
        gold_str = gold_answer[0] if isinstance(gold_answer, list) else gold_answer
        gold_str = _unbox(str(gold_str))
        return 1.0 if _exact_match(pred_str, gold_str) else 0.0

    # --- Build gold_parts: unbox + re-split (UGPhysics packs multi-part into \boxed{a, b}) ---
    gold_parts = _build_gold_parts(gold_answer)

    # --- Build pred_parts: extract boxed, then split by comma ---
    pred_parts = _extract_pred_parts(pred_text)

    return _match_parts(
        pred_parts,
        gold_parts,
        answer_type=answer_type,
        gold_unit=gold_unit,
        tolerance=tolerance,
        xverify_judge=xverify_judge,
        problem_text=problem_text,
        always_use_xverify=always_use_xverify,
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _unbox(s: str) -> str:
    """Strip a \\boxed{...} wrapper if the entire string is one boxed expression."""
    s = s.strip()
    boxed = _extract_boxed(s)
    if len(boxed) == 1:
        return boxed[0].strip()
    return s


def _build_gold_parts(gold_answer: str | list[str]) -> list[str]:
    """Build gold_parts from gold_answer.

    Handles:
    - List inputs: each element may be individually boxed, e.g. ["\\boxed{1}", "\\boxed{2}"]
    - String inputs: may be boxed with commas inside, e.g. "\\boxed{1, 2}" → ["1", "2"]
    """
    if isinstance(gold_answer, list):
        sources = list(gold_answer)
    else:
        # Split at top-level commas first (depth-aware: comma inside {} is not top-level)
        sources = split_by_comma(str(gold_answer)) or [str(gold_answer)]

    parts: list[str] = []
    for s in sources:
        unboxed = _unbox(s)
        # Re-split after unboxing: multi-part answers packed in one \boxed{a, b} become ["a", "b"]
        sub = split_by_comma(unboxed)
        parts.extend(sub if sub else [unboxed])
    return [p for p in parts if p.strip()]


def _extract_pred_parts(pred_text: str) -> list[str]:
    """Extract answer parts from raw model output."""
    raw = extract_answer(pred_text)
    parts: list[str] = []
    for item in raw:
        sub = split_by_comma(item)
        parts.extend(sub if sub else [item])
    return [p for p in parts if p.strip()]


def _exact_match(pred: str, gold: str) -> bool:
    return pred.strip().lower() == gold.strip().lower()


def _match_parts(
    pred_parts: list[str],
    gold_parts: list[str],
    answer_type: str | list[str],
    gold_unit: str,
    tolerance: float,
    xverify_judge,
    problem_text: str,
    always_use_xverify: bool,
) -> float:
    """Try to match pred_parts against gold_parts (order-independent for N<=4).

    Returns 1.0, 0.0, or -1.0 (unverifiable).
    """
    if len(pred_parts) != len(gold_parts):
        return 0.0

    n = len(pred_parts)
    orderings = (
        list(itertools.permutations(range(n))) if n <= 4 else [tuple(range(n))]
    )

    best = 0.0  # worst case unless a permutation gives better
    has_unverifiable = False

    for perm in orderings:
        result = _check_ordered(
            [pred_parts[i] for i in perm],
            gold_parts,
            answer_type=answer_type,
            gold_unit=gold_unit,
            tolerance=tolerance,
            xverify_judge=xverify_judge,
            problem_text=problem_text,
            always_use_xverify=always_use_xverify,
        )
        if result == 1.0:
            return 1.0
        if result == -1.0:
            has_unverifiable = True

    return -1.0 if has_unverifiable else 0.0


def _check_ordered(
    pred_parts: list[str],
    gold_parts: list[str],
    answer_type: str | list[str],
    gold_unit: str,
    tolerance: float,
    xverify_judge,
    problem_text: str,
    always_use_xverify: bool,
) -> float:
    """Check pred_parts vs gold_parts in the given order.

    Returns 1.0 (all correct), 0.0 (at least one confirmed wrong), -1.0 (some unverifiable).
    """
    any_unverifiable = False

    for idx, (pred_part, gold_part) in enumerate(zip(pred_parts, gold_parts)):
        part_type = (
            answer_type[idx]
            if isinstance(answer_type, list) and idx < len(answer_type)
            else (answer_type[0] if isinstance(answer_type, list) else answer_type)
        )
        part_result = _check_part(
            pred_part,
            gold_part,
            part_type=part_type,
            gold_unit=gold_unit,
            tolerance=tolerance,
            xverify_judge=xverify_judge,
            problem_text=problem_text,
            always_use_xverify=always_use_xverify,
        )
        if part_result is True:
            continue
        elif part_result is False:
            return 0.0  # confirmed wrong
        else:
            any_unverifiable = True  # None: rule couldn't parse

    return -1.0 if any_unverifiable else 1.0


def _check_part(
    pred_part: str,
    gold_part: str,
    part_type: str,
    gold_unit: str,
    tolerance: float,
    xverify_judge,
    problem_text: str,
    always_use_xverify: bool,
) -> bool | None:
    """Check one pred part against one gold part (with \\pm expansion).

    Returns True (correct), False (confirmed wrong), None (unverifiable).
    """
    gold_variants = expand_pm([gold_part])

    has_none = False
    for gold_variant in gold_variants:
        result = _check_pair(
            pred_part,
            gold_variant,
            part_type=part_type,
            gold_unit=gold_unit,
            tolerance=tolerance,
            xverify_judge=xverify_judge,
            problem_text=problem_text,
            always_use_xverify=always_use_xverify,
        )
        if result is True:
            return True
        if result is None:
            has_none = True

    # No True found: if any variant was unverifiable, escalate to unverifiable
    return None if has_none else False


def _check_pair(
    pred_str: str,
    gold_str: str,
    part_type: str,
    gold_unit: str,
    tolerance: float,
    xverify_judge,
    problem_text: str,
    always_use_xverify: bool,
) -> bool | None:
    """Check a single pred/gold pair.

    Returns True (pass), False (confirmed wrong), None (couldn't verify).
    Rule is only a fast-positive shortcut; xVerify handles rule=False|None too.
    """
    # Unit check (fast path for numerical with known unit)
    if gold_unit:
        unit_result = unit_equivalent(pred_str, gold_str, gold_unit, tolerance)
        if unit_result is True and not always_use_xverify:
            return True
        # unit_result False or None: fall through

    # Rule-based check
    rule_result = rule_verify(pred_str, gold_str, tolerance)
    if rule_result is True and not always_use_xverify:
        return True

    # xVerify fallback (called for rule=False or rule=None — prevents false negatives)
    if xverify_judge is not None:
        return xverify_judge(pred_str, gold_str, problem_text)

    # No xVerify
    if rule_result is False:
        return False
    return None  # rule was None = couldn't parse
