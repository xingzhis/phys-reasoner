"""Thin wrapper around math-verify for rule-based answer equivalence checking."""

from __future__ import annotations

import re

# Matches: coefficient \times 10^{exp} or 10^exp (handles optional braces and negative exponents)
_SCI_NOTATION_RE = re.compile(
    r"([\d.]+)\s*\\(?:times|cdot)\s*10\^(?:\{([+-]?\d+)\}|([+-]?\d+))"
)

# Detects nested exponentiation: ^{...^{ — causes overflow / wrong parse
_NESTED_EXP_RE = re.compile(r"\^\{[^}]*\^\{")

# Detects trailing bare unit suffix: digit(s) then letters with no intervening brace
# Matches e.g. "0.64N", "9.8m", but not "\sqrt{2}" or "v_{0}"
_TRAILING_UNIT_RE = re.compile(r"\d\s*[A-Za-z][A-Za-z0-9]*\s*$")


def _preprocess(s: str) -> str:
    """Normalize LaTeX patterns that math-verify can't handle natively.

    Converts 'N.NN \\times 10^K' / 'N.NN \\cdot 10^K' to float literals so
    that latex2sympy doesn't lose the coefficient when parsing `* 10^{K}`.
    """
    def _sci_to_float(m: re.Match) -> str:
        coef = float(m.group(1))
        exp_str = m.group(2) if m.group(2) is not None else m.group(3)
        try:
            return repr(coef * 10 ** int(exp_str))
        except (OverflowError, ValueError):
            return m.group(0)  # leave untouched on overflow

    return _SCI_NOTATION_RE.sub(_sci_to_float, s)


def _strip_boxed(s: str) -> str:
    """Strip a single \\boxed{} wrapper if present (for guard checks only)."""
    s = s.strip()
    if s.startswith("\\boxed{") and s.endswith("}"):
        return s[7:-1].strip()
    return s


def _has_trailing_unit(s: str) -> bool:
    """Return True if string ends with a bare unit-like suffix (digit then letters).

    Catches '0.64N', '9.8m', but not '\\sqrt{2}' or 'v_{0}' or '\\frac{1}{2}'.
    """
    s = _strip_boxed(s)
    # Must end in alpha chars, and the last non-alpha char before them must be a digit
    return bool(_TRAILING_UNIT_RE.search(s)) and not s.endswith("}")


def _lhs(s: str) -> str:
    """Return the part of an equation string before the first bare '=' sign."""
    # Skip LaTeX commands that contain '=' (like \geq, \leq, \neq, \approx)
    # Only split on '=' not preceded by <, >, !, \
    parts = re.split(r"(?<![<>!\\])=", s, maxsplit=1)
    return parts[0].strip() if len(parts) > 1 else ""


def rule_verify(pred_str: str, gold_str: str, tolerance: float = 0.05) -> bool | None:
    """Check answer equivalence using the math-verify rule engine.

    Guards against known math-verify failure modes:
    - FP1 (equation LHS confusion): math-verify extracts only RHS constant, so
      x^2+y^2=1 and x^2-y^2=1 both parse to 1. Guard: if LHS structures differ, return None.
    - FP2 (unit stripping): '0.64N' parses as 0.64, so '0.64N' == '0.64'. Guard: if one
      side has a trailing bare unit suffix the other lacks, return None.
    - FP3 (nested exponent overflow): 10^{10^{10^{10}}} overflows to 10. Guard: detect
      nested ^{..^{ pattern and return None.

    Returns:
        True  — equivalent
        False — not equivalent
        None  — couldn't parse / known failure mode (fall through to xVerify)
    """
    def _within_tolerance(g: float, p: float) -> bool:
        if abs(g) < 1e-12:
            return abs(p) < tolerance
        return abs(p - g) / abs(g) <= tolerance

    # --- Guard FP3: nested exponentiation → overflow ---
    if _NESTED_EXP_RE.search(gold_str) or _NESTED_EXP_RE.search(pred_str):
        return None

    # --- Guard FP2: trailing unit mismatch ---
    if _has_trailing_unit(gold_str) != _has_trailing_unit(pred_str):
        return None

    # --- Numerical percentage tolerance pre-check (raw strings) ---
    try:
        if _within_tolerance(float(gold_str.strip()), float(pred_str.strip())):
            return True
    except (ValueError, TypeError):
        pass

    # --- Preprocess scientific notation → float literals ---
    gold_pre = _preprocess(gold_str)
    pred_pre = _preprocess(pred_str)

    # --- Tolerance pre-check on preprocessed strings (catches N×10^K vs M×10^K) ---
    if gold_pre != gold_str or pred_pre != pred_str:
        try:
            if _within_tolerance(float(gold_pre.strip()), float(pred_pre.strip())):
                return True
        except (ValueError, TypeError):
            pass

    # --- Symbolic rule engine ---
    try:
        from math_verify import parse, verify
    except ImportError:
        return None

    # --- Guard FP1: equation LHS confusion ---
    # math-verify extracts only the RHS constant from equations, so two equations with
    # different LHS but same RHS (e.g. x^2+y^2=1 vs x^2-y^2=1) both parse to 1.
    # Also catches equation vs non-equation mismatch (one has '=', other doesn't).
    gold_lhs = _lhs(gold_pre)
    pred_lhs = _lhs(pred_pre)
    if bool(gold_lhs) != bool(pred_lhs):
        return None  # one is an equation, other isn't — incompatible
    if gold_lhs and pred_lhs and gold_lhs != pred_lhs:
        return None  # both equations, different LHS structures

    gold_parsed = parse(gold_pre)
    pred_parsed = parse(pred_pre)
    if not gold_parsed or not pred_parsed:
        return None

    # Guard against coefficient-extraction FP: math-verify sometimes extracts a simple
    # integer coefficient from complex expressions (e.g. "2 \cosh(...)" → 2, "R \ln 2" → 2).
    # If the parsed value is a small integer but the original string is complex (contains
    # LaTeX commands), the parse discarded structural information → can't trust the result.
    def _is_suspicious_simplification(parsed, original: str) -> bool:
        if len(parsed) < 1:
            return False
        val = parsed[0]
        try:
            from sympy import Integer
            if not isinstance(val, Integer):
                return False
            # Small integer result from a complex expression
            return abs(int(val)) <= 100 and len(original) > 15 and "\\" in original
        except Exception:
            return False

    if _is_suspicious_simplification(gold_parsed, gold_pre) or \
       _is_suspicious_simplification(pred_parsed, pred_pre):
        return None

    # Numeric tolerance check on sympy-parsed values (catches \frac{10}{7} vs 1.43)
    try:
        from sympy import N as sympyN
        g_num = float(sympyN(gold_parsed[0]))
        p_num = float(sympyN(pred_parsed[0]))
        if _within_tolerance(g_num, p_num):
            return True
    except Exception:
        pass

    try:
        return verify(gold_parsed, pred_parsed, float_rounding=6, timeout_seconds=5)
    except Exception:
        return None
