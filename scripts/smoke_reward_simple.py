"""Standalone smoke reward — zero imports from phys_reasoner.

Extracts \\boxed{...} from the response and compares numerically to ground_truth.
Returns 1.0 on match, 0.0 otherwise.

Used only in smoke_v2.sh (Step 1). Replaced by the real reward in Step 2.
"""
import re


def compute_score(solution_str: str, ground_truth: str, **kwargs) -> float:
    match = re.search(r"\\boxed\{([^}]+)\}", solution_str)
    if not match:
        return 0.0
    pred = match.group(1).strip()
    try:
        return 1.0 if abs(float(pred) - float(str(ground_truth).strip())) < 0.01 else 0.0
    except ValueError:
        return 1.0 if pred == str(ground_truth).strip() else 0.0
