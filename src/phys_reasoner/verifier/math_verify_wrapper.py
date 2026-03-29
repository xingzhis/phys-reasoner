"""Thin wrapper around math-verify for rule-based answer equivalence checking."""

from __future__ import annotations

import re

# Matches: coefficient \times 10^{exp} or 10^exp (handles optional braces, negative
# exponents, and decimal exponents like 10^{14.5})
_SCI_NOTATION_RE = re.compile(
    r"([\d.]+)\s*\\(?:times|cdot)\s*10\^(?:\{([+-]?[\d.]+)\}|([+-]?[\d.]+))"
)

# Matches bare 10^{x.y} (decimal exponent, no preceding coefficient) — e.g. 10^{14.5}.
# Integer-only exponents (10^{14}) are already handled fine by latex2sympy; this only
# fires when there is a decimal point so we don't over-convert normal expressions.
_BARE_DECIMAL_POW_RE = re.compile(
    r"(?<![\\d.])10\^(?:\{([+-]?\d+\.\d+)\}|([+-]?\d+\.\d+))"
)

# Detects nested exponentiation: ^{...^{ — causes overflow / wrong parse
_NESTED_EXP_RE = re.compile(r"\^\{[^}]*\^\{")

# Detects trailing bare unit suffix: digit(s) then letters with no intervening brace
# Matches e.g. "0.64N", "9.8m", but not "\sqrt{2}" or "v_{0}"
_TRAILING_UNIT_RE = re.compile(r"\d\s*[A-Za-z][A-Za-z0-9]*\s*$")


def _preprocess(s: str) -> str:
    """Normalize LaTeX patterns that math-verify can't handle natively.

    1. 'N.NN \\times 10^K' / 'N.NN \\cdot 10^K'  → float literal
       (including decimal exponents like 10^{14.5})
    2. Bare '10^{x.y}' (decimal exponent, no coefficient) → float literal,
       so that expressions like '\\frac{\\sqrt{\\pi}}{2} \\times 10^{14.5}'
       can be evaluated numerically by sympy after substitution.
    """
    def _sci_to_float(m: re.Match) -> str:
        coef = float(m.group(1))
        exp_str = m.group(2) if m.group(2) is not None else m.group(3)
        try:
            return repr(coef * 10 ** float(exp_str))
        except (OverflowError, ValueError):
            return m.group(0)  # leave untouched on overflow

    def _bare_decimal_pow_to_float(m: re.Match) -> str:
        exp_str = m.group(1) if m.group(1) is not None else m.group(2)
        try:
            return repr(10 ** float(exp_str))
        except (OverflowError, ValueError):
            return m.group(0)

    s = _SCI_NOTATION_RE.sub(_sci_to_float, s)
    s = _BARE_DECIMAL_POW_RE.sub(_bare_decimal_pow_to_float, s)
    return s


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


def _sympy_numerical_equiv(
    gold_str: str,
    pred_str: str,
    tolerance: float = 0.05,
    n_trials: int = 10,
) -> bool | None:
    """Numerical substitution check for symbolic equivalence.

    Handles expressions (no '=') and equations (LHS = RHS):
    - Expressions: substitute random values, check f_gold ≈ f_pred at each point.
    - Equations: normalize f = LHS−RHS for both, check f_gold/f_pred is constant.

    Fast-path: try sympy.expand / sympy.cancel before random trials.

    Only proceeds when both sides have the same non-empty free symbol set.

    Returns:
        True  — all trials agree (or fast symbolic check passed)
        False — a trial conclusively disagrees
        None  — can't determine (parse failure, symbol mismatch, too few valid trials)
    """
    try:
        from latex2sympy2_extended import latex2sympy
        from sympy import Eq, N as sympyN, cancel, expand
    except ImportError:
        return None

    import random

    try:
        gold_expr = latex2sympy(gold_str)
        pred_expr = latex2sympy(pred_str)
    except Exception:
        return None

    if gold_expr is None or pred_expr is None:
        return None

    # Normalise equations to f = LHS − RHS
    gold_is_eq = isinstance(gold_expr, Eq)
    pred_is_eq = isinstance(pred_expr, Eq)
    if gold_is_eq != pred_is_eq:
        return None  # type mismatch: one equation, one expression

    if gold_is_eq:
        try:
            f_gold = gold_expr.lhs - gold_expr.rhs
            f_pred = pred_expr.lhs - pred_expr.rhs
        except Exception:
            return None  # e.g. RHS is a FiniteSet (solution set) — can't subtract
    else:
        f_gold = gold_expr
        f_pred = pred_expr

    gold_syms = f_gold.free_symbols
    pred_syms = f_pred.free_symbols

    if not gold_syms and not pred_syms:
        return None  # pure numeric — already handled by upstream numeric path

    if gold_syms != pred_syms:
        return None  # different variable sets — can't align safely

    symbols = list(gold_syms)

    # --- Fast symbolic pre-check (O(polynomial)-bounded, unlike simplify) ---
    try:
        diff = f_gold - f_pred
        if expand(diff) == 0 or cancel(diff) == 0:
            return True
    except Exception:
        pass

    if gold_is_eq:
        # Equations equivalent up to nonzero constant scale: check f_gold/f_pred is constant
        try:
            ratio_sym = cancel(f_gold / f_pred)
            if not ratio_sym.free_symbols:
                return True
        except Exception:
            pass

    # --- Numerical substitution ---
    rng = random.Random(42)  # deterministic seed for reproducibility
    valid_trials = 0
    ratio_ref: float | None = None

    for _ in range(n_trials):
        sub_dict = {s: rng.uniform(0.5, 3.0) for s in symbols}
        try:
            g_val = float(sympyN(f_gold.subs(sub_dict)))
            p_val = float(sympyN(f_pred.subs(sub_dict)))
        except Exception:
            continue  # div-by-zero, complex result, etc. — skip trial

        if not gold_is_eq:
            # Expression: check f_gold ≈ f_pred
            if abs(g_val) < 1e-10 and abs(p_val) < 1e-10:
                valid_trials += 1
                continue  # both near zero — not informative but not wrong
            if abs(g_val) < 1e-10 or abs(p_val) < 1e-10:
                return False  # one is zero, other is not
            rel_err = abs(p_val - g_val) / abs(g_val)
            if rel_err > tolerance:
                return False
            valid_trials += 1
        else:
            # Equation: check f_gold / f_pred is constant across trials
            if abs(p_val) < 1e-8 or abs(g_val) < 1e-8:
                continue  # near-zero denominator / numerator — unstable ratio
            ratio = g_val / p_val
            if ratio_ref is None:
                ratio_ref = ratio
            else:
                if abs(ratio - ratio_ref) / max(abs(ratio_ref), 1e-10) > tolerance:
                    return False
            valid_trials += 1

    if valid_trials < 3:
        return None  # too few valid trials to be confident
    return True


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
        g_plain = float(gold_str.strip())
        p_plain = float(pred_str.strip())
        if _within_tolerance(g_plain, p_plain):
            return True
        # Short-circuit False: both are plain Python floats that differ beyond tolerance.
        # Without this guard, math_verify.parse() silently drops Python e-notation exponents
        # (e.g. "1.5e-3" and "1.5e-6" both parse to 1.5), producing a false positive.
        return False
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

    # Fallback: direct latex2sympy numerical evaluation on the preprocessed strings.
    # Handles cases where math_verify's parse() silently drops symbolic terms, e.g.
    # "\frac{\sqrt{\pi}}{2} \times 10^{14.5}" → parse() extracts only 10^14.5,
    # but latex2sympy evaluates the full expression correctly.
    try:
        from latex2sympy2_extended import latex2sympy
        from sympy import N as sympyN
        g_num2 = float(sympyN(latex2sympy(gold_pre)))
        p_num2 = float(sympyN(latex2sympy(pred_pre)))
        if _within_tolerance(g_num2, p_num2):
            return True
    except Exception:
        pass

    # Numerical substitution for symbolic expressions / equations with free variables.
    # Rescues FNs where algebraically equivalent forms differ in structure, e.g.
    # a*(b+c) vs a*b+a*c, or r^2=x^2+y^2 vs x^2+y^2=r^2.
    num_sub = _sympy_numerical_equiv(gold_pre, pred_pre, tolerance)
    if num_sub is True:
        return True

    try:
        return verify(gold_parsed, pred_parsed, float_rounding=6, timeout_seconds=5)
    except Exception:
        return None
