"""VeRL-compatible reward function wrapping the physics answer verifier."""

from __future__ import annotations

import os

# Lazy singleton: created on first successful URL discovery, reused for the
# lifetime of the worker. URL is discovered via (1) XVERIFY_URL env var, OR
# (2) a rendezvous file at outputs/xverify_endpoints/current.url written by
# the companion serve_xverify.sbatch job. The file fallback exists because in
# the multi-node async path, Ray actor processes inherit env from the Ray
# daemons — which are started BEFORE the companion xverify job has written
# its URL. Without the file fallback, the singleton initializes to None and
# stays None for the rest of the process, silently disabling xverify.
#
# We do NOT short-circuit on a "tried once" flag; if the URL is not yet
# available on the first call, subsequent calls retry. The check is cheap
# (env lookup + os.path.isfile) so per-step overhead is negligible.
_XVERIFY_CLIENT = None
# Tracks whether the current singleton has been health-checked against the
# server. Separate from _XVERIFY_CLIENT so that the check only runs once per
# process (not per compute_score call).
_XVERIFY_HEALTH_CHECKED = False


def _require_xverify() -> bool:
    """Whether production-mode xverify enforcement is active.

    Set ``PHYS_REQUIRE_XVERIFY=1`` on production training jobs. When active:
      * A missing URL raises RuntimeError (loud fail at first compute_score
        call — catches the silent rule-only-fallback mode).
      * An unreachable/unhealthy URL also raises.
      * Test/smoke runs leave this unset so they can run without the companion.
    """
    return os.environ.get("PHYS_REQUIRE_XVERIFY", "").strip() == "1"


def _discover_xverify_url() -> str:
    """Return the xVerify URL from env or rendezvous file, or empty string."""
    url = os.environ.get("XVERIFY_URL", "").strip()
    if url:
        return url
    # Rendezvous file fallback. Path can be overridden via XVERIFY_URL_FILE.
    rendezvous = os.environ.get("XVERIFY_URL_FILE", "").strip()
    if not rendezvous:
        # Default location relative to repo root. Walk up from this file:
        # src/phys_reasoner/training/reward.py → repo root.
        here = os.path.abspath(os.path.dirname(__file__))
        repo_root = os.path.abspath(os.path.join(here, "..", "..", ".."))
        rendezvous = os.path.join(repo_root, "outputs", "xverify_endpoints", "current.url")
    if not os.path.isfile(rendezvous):
        return ""
    try:
        with open(rendezvous, "r") as f:
            line = f.readline().strip()
        return line if line and not line.startswith("#") else ""
    except OSError:
        return ""


def _get_xverify_judge():
    global _XVERIFY_CLIENT, _XVERIFY_HEALTH_CHECKED
    if _XVERIFY_CLIENT is not None:
        return _XVERIFY_CLIENT
    url = _discover_xverify_url()
    require = _require_xverify()
    if not url:
        if require:
            raise RuntimeError(
                "PHYS_REQUIRE_XVERIFY=1 is set but no xverify URL was "
                "discovered (checked XVERIFY_URL env var and default "
                "rendezvous file outputs/xverify_endpoints/current.url, "
                "plus XVERIFY_URL_FILE override). Production training must "
                "not silently fall back to rule-based reward: the rule "
                "verifier has ~68% false-negative rate on expression-type "
                "answers, which would train the model on a degraded signal. "
                "Verify the companion serve_xverify.sbatch job is running "
                "and has written its rendezvous file."
            )
        return None
    from phys_reasoner.verifier.xverify_client import XVerifyHTTPClient

    client = XVerifyHTTPClient(url)
    if require and not _XVERIFY_HEALTH_CHECKED:
        if not client.health_check():
            raise RuntimeError(
                f"PHYS_REQUIRE_XVERIFY=1 is set and an xverify URL was "
                f"discovered ({url}) but the server did not pass its "
                f"health check. It may be queue-stuck, crashed, or still "
                f"loading its model. Check the companion serve_xverify "
                f"job status and logs before restarting training."
            )
        _XVERIFY_HEALTH_CHECKED = True
    _XVERIFY_CLIENT = client
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
