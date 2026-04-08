"""VeRL-compatible reward function wrapping the physics answer verifier."""

from __future__ import annotations

import os

# Lazy singleton: created on first use, reused for the lifetime of the worker.
# Set XVERIFY_URL=http://<host>:<port>/judge to route the verifier's xVerify
# fallback to a remote service. Unset → rule-only verification (legacy path).
_XVERIFY_CLIENT = None
_XVERIFY_INIT_DONE = False


def _get_xverify_judge():
    global _XVERIFY_CLIENT, _XVERIFY_INIT_DONE
    if _XVERIFY_INIT_DONE:
        return _XVERIFY_CLIENT
    _XVERIFY_INIT_DONE = True
    url = os.environ.get("XVERIFY_URL", "").strip()
    if not url:
        return None
    from phys_reasoner.verifier.xverify_client import XVerifyHTTPClient

    _XVERIFY_CLIENT = XVerifyHTTPClient(url)
    return _XVERIFY_CLIENT


def compute_score(
    solution_str: str,
    ground_truth: str | list[str],
    # VeRL's naive reward manager passes these kwargs; metadata lives in extra_info.
    data_source: str = "",
    extra_info: dict | None = None,
    # Kept for direct callers (tests, stage0_probe); VeRL does NOT pass these.
    answer_type: str = "numerical",
    unit: str = "",
    tolerance: float = 0.05,
    xverify_judge=None,
    problem: str = "",
    **_kwargs,
) -> float:
    """VeRL reward function.

    Returns 1.0 (correct), 0.0 (wrong or unverifiable).
    Unverifiable answers (-1.0 from verify_answer) are treated as wrong during training.

    VeRL's naive reward manager calls compute_score(solution_str, ground_truth, data_source,
    extra_info). The verifier fields (answer_type, unit, tolerance) come from extra_info
    which is populated by build_training_parquets.py / the smoke parquet generator.

    xverify_judge=None (default): rule-only verification. Correct for smoke tests and
    numerical-only curriculum. Rule verifier has ~68% FN rate on expression types.
    """
    from phys_reasoner.verifier.router import verify_answer

    if extra_info:
        answer_type = extra_info.get("answer_type", answer_type)
        unit = extra_info.get("unit", unit)
        tolerance = float(extra_info.get("tolerance", tolerance))
        problem = extra_info.get("problem", problem)

    if xverify_judge is None:
        xverify_judge = _get_xverify_judge()

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
