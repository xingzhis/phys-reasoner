"""Pint-based unit-aware numerical comparison for physics answers."""

from __future__ import annotations

import re

import pint

_NUMBER_RE = re.compile(r"^\s*([+-]?\d+\.?\d*(?:[eE][+-]?\d+)?)", re.IGNORECASE)
_UREG = pint.UnitRegistry()

# LaTeX command with argument: \mathrm{x} → x
_LATEX_CMD_RE = re.compile(r"\\(?:mathrm|text|mathbf|mathit|operatorname)\{([^}]*)\}")
# LaTeX math-mode $ delimiters only (keep ^, braces for exponent handling)
_LATEX_DOLLAR_RE = re.compile(r"\$")
# LaTeX scientific notation: 3.14 \times 10^{-5} or 3.14 \times 10^{5}
_LATEX_SCI_RE = re.compile(
    r"([+-]?\d+\.?\d*)\s*\\times\s*10\^(?:\{([+-]?\d+)\}|([+-]?\d+))"
)
# Standalone no-arg LaTeX commands like \cdot \, \! \approx (not \times — handled above)
_LATEX_CMD_NOARG_RE = re.compile(r"\\(?!times)[a-zA-Z,!]+")


def strip_latex_unit(u: str) -> str:
    """Strip LaTeX formatting from a unit string so pint can parse it.

    Examples:
        '$\\mathrm{~nm}$'               -> 'nm'
        '$\\mathrm{m} / \\mathrm{s}^2$' -> 'm / s^2'
        '$\\mathrm{~kJ} \\mathrm{~mol}^{-1}$' -> 'kJ mol^-1'
        '$10^6$ m'                       -> '10^6 m'
        ' J '                            -> 'J'
    """
    # Unwrap \mathrm{x} → x (and similar)
    u = _LATEX_CMD_RE.sub(r"\1", u)
    # Strip $ signs
    u = _LATEX_DOLLAR_RE.sub("", u)
    # Remove standalone commands (\cdot etc.) but keep braces for ^{-1} → handled by pint
    u = _LATEX_CMD_NOARG_RE.sub(" ", u)
    # Replace LaTeX thin space (~) with regular space
    u = u.replace("~", " ")
    # Normalise whitespace
    u = re.sub(r" +", " ", u).strip()
    return u


def _strip_latex_expr(s: str) -> str:
    """Normalise a pred expression string: convert LaTeX sci-notation and strip wrappers.

    Examples:
        '3.32 \\times 10^{-10} \\text{ m}' -> '3.32e-10 m'
        '5.0e-10 m'                          -> '5.0e-10 m'  (unchanged)
    """
    # Convert \times 10^{N} → eN
    s = _LATEX_SCI_RE.sub(lambda m: f"{m.group(1)}e{m.group(2) or m.group(3)}", s)
    # Then apply unit-level stripping (handles \text{}, \mathrm{}, etc.)
    s = _LATEX_CMD_RE.sub(r"\1", s)
    s = _LATEX_DOLLAR_RE.sub("", s)
    s = _LATEX_CMD_NOARG_RE.sub(" ", s)
    s = s.replace("~", " ")
    s = re.sub(r" +", " ", s).strip()
    return s


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

    cleaned_unit = strip_latex_unit(gold_unit)
    try:
        gold_unit_parsed = ureg.parse_units(cleaned_unit)
    except Exception:
        return None

    # Parse gold numeric value
    gold_val = _parse_number(gold_str)
    if gold_val is None:
        return None
    gold_qty = gold_val * gold_unit_parsed

    # Parse pred as quantity (number + optional unit), normalising LaTeX first
    pred_qty = _parse_quantity(_strip_latex_expr(pred_str), gold_unit_parsed, ureg)
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
