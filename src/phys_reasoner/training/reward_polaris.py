"""VeRL-compatible reward function for POLARIS-Dataset-53K style math problems.

Rule-based boxed-answer extraction + string equality with ground_truth.
Intended for the K-anchor / C3 experiments that train on POLARIS parquet
(where verl's default_compute_score can't route because data_source is empty).

Matches POLARIS's own scoring convention: extract final \\boxed{...}, compare
to ground_truth modulo whitespace. Returns 1.0 on match, 0.0 otherwise.
"""

from __future__ import annotations

import re


_BOXED_PATTERN = re.compile(r"\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}")


def _extract_last_boxed(text: str) -> str:
    """Extract content of the last \\boxed{...} in text, handling nested braces."""
    # Find all last-innermost boxed answers; pattern handles one level of nesting.
    matches = _BOXED_PATTERN.findall(text or "")
    if matches:
        return matches[-1].strip()
    return ""


def _normalize(s: str) -> str:
    """Collapse whitespace and strip for tolerant string comparison."""
    return re.sub(r"\s+", "", (s or "").strip())


def compute_score(
    solution_str: str,
    ground_truth: str,
    data_source: str = "",
    extra_info=None,
    **kwargs,
) -> float:
    """Score a single rollout against the POLARIS-style ground truth.

    Signature matches what verl's naive reward manager passes. Extra kwargs
    are tolerated for forward-compat (e.g. sandbox_fusion_url).

    Handles ground_truth stored with OR without ``\\boxed{...}`` wrapper:
    DAPO-17k stores it wrapped (``\\boxed{8}``); POLARIS-53K and DeepScaleR
    store bare (``50\\sqrt{5}``, ``-\\frac{2}{3}``).
    """
    pred = _extract_last_boxed(solution_str)
    if not pred:
        return 0.0
    gt = str(ground_truth)
    gt_unwrapped = _extract_last_boxed(gt) or gt  # strip wrapper if present
    if _normalize(pred) == _normalize(gt_unwrapped):
        return 1.0
    return 0.0
