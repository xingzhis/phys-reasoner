"""VeRL-compatible reward function wrapping the physics answer verifier."""

from __future__ import annotations


def compute_score(
    solution_str: str,
    ground_truth: str | list[str],
    answer_type: str,
    unit: str = "",
    tolerance: float = 0.05,
    xverify_judge=None,
    problem: str = "",
) -> float:
    """VeRL reward function.

    Returns 1.0 (correct), 0.0 (wrong or unverifiable).
    Unverifiable answers (-1.0 from verify_answer) are treated as wrong during training.

    xverify_judge=None (default): rule-only verification. Correct for smoke tests and
    numerical-only curriculum. Rule verifier has ~68% FN rate on expression types —
    for production expression-type training, pass a warm xVerify-7B judge loaded on a
    dedicated reward GPU. See docs/training-decisions.md § "xVerify in reward function".
    """
    from phys_reasoner.verifier.router import verify_answer

    score = verify_answer(
        pred_text=solution_str,
        gold_answer=ground_truth,
        answer_type=answer_type,
        gold_unit=unit,
        tolerance=tolerance,
        xverify_judge=xverify_judge,
        problem_text=problem,
    )
    if score == -1.0:
        return 0.0
    return score
