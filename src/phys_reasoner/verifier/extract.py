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
    """Expand \\pm into separate + and - versions.

    Handles both '\\pm 5' (with space) and '\\pm5' (without) to avoid
    producing '+ 5' / '- 5' which confuses LaTeX parsers (the sign gets lost).
    """
    result = []
    for expr in expr_list:
        if "\\pm" in expr:
            # Replace '\\pm ' (with trailing space) first to avoid '+ 5'/'- 5'
            pos = expr.replace("\\pm ", "+").replace("\\pm", "+")
            neg = expr.replace("\\pm ", "-").replace("\\pm", "-")
            result.append(pos)
            result.append(neg)
        else:
            result.append(expr)
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
