"""Pint-based unit-aware numerical comparison for physics answers."""

from __future__ import annotations

import re

import pint

_NUMBER_RE = re.compile(r"^\s*([+-]?\d+\.?\d*(?:[eE][+-]?\d+)?)", re.IGNORECASE)
_UREG = pint.UnitRegistry()


def unit_equivalent(
    pred_str: str,
    gold_str: str,
    gold_unit: str,
    tolerance: float = 0.05,
) -> bool | None:
    """Check if pred_str equals gold_str when physical units are considered.

    Returns:
        True  — equivalent within tolerance
        False — clearly wrong (different value after unit conversion)
        None  — can't parse, fall through to math-verify
    """
    ureg = _UREG

    try:
        gold_unit_parsed = ureg.parse_units(gold_unit)
    except Exception:
        return None

    # Parse gold numeric value
    gold_val = _parse_number(gold_str)
    if gold_val is None:
        return None
    gold_qty = gold_val * gold_unit_parsed

    # Parse pred as quantity (number + optional unit)
    pred_qty = _parse_quantity(pred_str, gold_unit_parsed, ureg)
    if pred_qty is None:
        return None

    try:
        diff = abs((pred_qty - gold_qty).to(gold_unit_parsed).magnitude)
        if abs(gold_val) < 1e-12:
            return diff < tolerance
        return (diff / abs(gold_val)) <= tolerance
    except Exception:
        return None


def _parse_number(s: str) -> float | None:
    try:
        return float(s.strip())
    except ValueError:
        m = _NUMBER_RE.match(s)
        if m:
            try:
                return float(m.group(1))
            except ValueError:
                pass
    return None


def _parse_quantity(s: str, fallback_unit, ureg):
    """Try to parse s as a pint Quantity. If no unit, attach fallback_unit."""
    try:
        qty = ureg.parse_expression(s)
        if not hasattr(qty, "dimensionality"):
            # plain number
            val = _parse_number(s)
            if val is None:
                return None
            return val * fallback_unit
        return qty
    except Exception:
        val = _parse_number(s)
        if val is None:
            return None
        return val * fallback_unit
