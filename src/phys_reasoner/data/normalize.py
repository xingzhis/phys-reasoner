"""Answer type normalization: maps raw source labels to canonical AnswerType strings."""

from __future__ import annotations

import json

_RAW_TO_CANONICAL: dict[str, str] = {
    # Numerical
    "Numerical": "numerical",
    "NV": "numerical",
    "numerical": "numerical",
    # Expression / symbolic
    "Expression": "expression",
    "EX": "expression",
    "expression": "expression",
    "symbolic": "expression",
    # Equation
    "Equation": "equation",
    "EQ": "equation",
    "equation": "equation",
    # Interval
    "Interval": "interval",
    "IN": "interval",
    "interval": "interval",
    # MCQ
    "MCQ": "mcq",
    "MC": "mcq",
    "mcq": "mcq",
    # True/False
    "True/False": "true_false",
    "TF": "true_false",
    "T/F": "true_false",
    "true_false": "true_false",
    # Open-ended
    "Open-end": "open_end",
    "open_end": "open_end",
    # Code
    "code": "code",
    # Already canonical pass-through
    "equation": "equation",
    "interval": "interval",
    "mcq": "mcq",
    "true_false": "true_false",
    "open_end": "open_end",
    "unknown": "unknown",
}


def _map_single(raw: str) -> str:
    return _RAW_TO_CANONICAL.get(raw.strip(), "unknown")


def normalize_answer_type(raw: str | list, source: str = "") -> str | list[str]:  # noqa: C901
    """Map raw source answer_type labels to canonical AnswerType strings.

    Handles:
    - Single strings: "Numerical" -> "numerical"
    - Lists (OlympiadBench multi-part): ["NV", "EX"] -> ["numerical", "expression"]
    - Comma-joined strings (UGPhysics multi-part): "NV, EX" -> ["numerical", "expression"]

    Returns a single str if only one type; list[str] if multiple.
    """
    if isinstance(raw, list):
        normalized = [_map_single(str(r)) for r in raw]
        return normalized[0] if len(normalized) == 1 else normalized

    raw_str = str(raw)

    # Handle JSON-encoded list strings e.g. '["expression", "numerical"]'
    if raw_str.startswith("["):
        try:
            items = json.loads(raw_str)
            if isinstance(items, list):
                normalized = [_map_single(str(r)) for r in items]
                return normalized[0] if len(normalized) == 1 else normalized
        except (json.JSONDecodeError, TypeError):
            pass

    if "," in raw_str:
        parts = [p.strip() for p in raw_str.split(",")]
        normalized = [_map_single(p) for p in parts if p]
        if len(normalized) == 1:
            return normalized[0]
        return normalized

    return _map_single(raw_str)
