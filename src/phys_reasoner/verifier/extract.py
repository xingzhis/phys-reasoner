"""Answer extraction from model output text."""

from __future__ import annotations

import re

_SPECIAL_MAP = {
    "\\left": "",
    "\\right": "",
    "$": "",
    "\\approx": "=",
    "\\simeq": "=",
    "\\sim": "=",
    "^\\prime": "'",
    "^{\\prime}": "'",
    "^\\circ": "",
    "%": "",
}

_MATHRM_RE = re.compile(r"\\(?:mathrm|mathbf)\{~?([^}]*)\}")


def extract_answer(text: str) -> list[str]:
    """Extract answer(s) from model output.

    1. Find all \\boxed{...} using brace-depth stack
    2. Fallback: keyword search ("answer is", "answer:")
    3. Last resort: return [text.strip()]
    """
    boxed = _extract_boxed(text)
    if boxed:
        return boxed

    lower = text.lower()
    for kw in ("the answer is", "answer is", "answer:"):
        idx = lower.rfind(kw)
        if idx != -1:
            tail = text[idx + len(kw):].strip()
            if tail:
                return [tail.split("\n")[0].strip()]

    return [text.strip()]


def _extract_boxed(text: str) -> list[str]:
    """Extract all \\boxed{...} contents using brace-depth stack."""
    results = []
    i = 0
    while i < len(text):
        idx = text.find("\\boxed{", i)
        if idx == -1:
            break
        start = idx + len("\\boxed{")
        depth = 1
        j = start
        while j < len(text) and depth > 0:
            if text[j] == "{":
                depth += 1
            elif text[j] == "}":
                depth -= 1
            j += 1
        if depth == 0:
            results.append(text[start : j - 1])
        i = j
    return results


def split_by_comma(expr: str) -> list[str]:
    """Split expression by commas that are outside brackets/braces."""
    parts: list[str] = []
    depth = 0
    start = 0
    for i, ch in enumerate(expr):
        if ch in ("(", "[", "{"):
            depth += 1
        elif ch in (")", "]", "}"):
            depth -= 1
        elif ch == "," and depth == 0:
            # Skip \, (LaTeX thin space) — backslash-comma is not a real separator
            if i > 0 and expr[i - 1] == "\\":
                continue
            parts.append(expr[start:i].strip())
            start = i + 1
    if start < len(expr):
        parts.append(expr[start:].strip())
    return [p for p in parts if p]


def expand_pm(expr_list: list[str]) -> list[str]:
    """Expand \\pm into all sign combinations.

    For k occurrences of \\pm, generates 2^k variants covering every +/- combo.
    E.g. "a \\pm b \\pm c" → ["a+b+c", "a+b-c", "a-b+c", "a-b-c"].

    Handles both '\\pm ' (with space) and '\\pm' (without) to avoid producing
    '+ 5' / '- 5' which confuse LaTeX parsers (sign gets lost).
    """
    import itertools

    result = []
    for expr in expr_list:
        # Normalise to a single token form for splitting
        normalised = expr.replace("\\pm ", "\\pm")
        count = normalised.count("\\pm")
        if count == 0:
            result.append(expr)
            continue
        # Generate all 2^k sign combinations
        for signs in itertools.product(("+", "-"), repeat=count):
            variant = normalised
            for sign in signs:
                variant = variant.replace("\\pm", sign, 1)
            result.append(variant)
    return result


def normalize_latex(expr: str) -> str:
    """Strip common LaTeX wrappers to prepare for comparison."""
    # Strip \\in prefix (set membership notation like "x \\in ...")
    if "\\in " in expr:
        expr = expr.split("\\in ")[1]

    for token, replacement in _SPECIAL_MAP.items():
        expr = expr.replace(token, replacement)

    expr = expr.strip("\n$,.:;^_=+`!@#%^&*~")
    expr = _MATHRM_RE.sub(r"\1", expr)
    return expr
