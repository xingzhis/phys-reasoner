"""Tests for the TIR pipeline: sandbox, prompts, stage0 helpers.

No GPU required, no VeRL required.
VeRL-dependent agent loop tests are in tests/test_tir_verl.py.

Run:
    python -m pytest tests/test_tir.py -v
"""

from __future__ import annotations

import pytest

from phys_reasoner.tir.sandbox import ALLOWED_PACKAGES, SandboxResult, execute_code, _check_imports
from phys_reasoner.tir.prompts import (
    ALLOWED_PACKAGES_STR,
    CODE_STOP,
    TIR_SYSTEM_PROMPT,
    extract_code,
)
from phys_reasoner.eval.stage0_probe import _extract_boxed, build_tir_prompt


# ---------------------------------------------------------------------------
# sandbox._check_imports
# ---------------------------------------------------------------------------

class TestSandboxCheckImports:
    def test_allowed_numpy(self):
        assert _check_imports("import numpy as np") is None

    def test_allowed_sympy(self):
        assert _check_imports("from sympy import symbols, solve") is None

    def test_allowed_math(self):
        assert _check_imports("import math") is None

    def test_allowed_nested(self):
        assert _check_imports("from numpy.linalg import norm") is None

    def test_disallowed_torch(self):
        assert _check_imports("import torch") == "torch"

    def test_disallowed_os(self):
        assert _check_imports("import os") == "os"

    def test_disallowed_subprocess(self):
        assert _check_imports("import subprocess") == "subprocess"

    def test_disallowed_requests(self):
        assert _check_imports("import requests") == "requests"

    def test_syntax_error_rejected(self):
        assert _check_imports("def f(: pass") == "<syntax error>"


# ---------------------------------------------------------------------------
# sandbox.execute_code
# ---------------------------------------------------------------------------

class TestExecuteCode:
    def test_basic_print(self):
        r = execute_code("print(2 + 2)")
        assert r.stdout == "4"
        assert not r.error
        assert not r.timed_out

    def test_sympy_differentiation(self):
        r = execute_code(
            "import sympy; x = sympy.Symbol('x'); print(sympy.diff(x**2, x))"
        )
        assert r.stdout == "2*x"
        assert not r.error

    def test_numpy_calculation(self):
        r = execute_code("import numpy as np; print(round(float(np.sqrt(2)), 6))")
        assert r.stdout == "1.414214"
        assert not r.error

    def test_timeout(self):
        r = execute_code("import time; time.sleep(40)", timeout=3.0)
        assert r.timed_out
        assert r.error

    def test_whitelist_rejection_no_subprocess(self):
        # Rejected pre-subprocess — no process should be spawned
        r = execute_code("import torch; print(torch.__version__)")
        assert r.error
        assert not r.timed_out
        assert "torch" in r.stderr

    def test_whitelist_rejects_os(self):
        r = execute_code("import os; os.system('echo bad')")
        assert r.error
        assert not r.timed_out

    def test_output_truncation(self):
        r = execute_code("print('x' * 10000)", max_output_bytes=100)
        assert len(r.stdout.encode()) <= 120
        assert "truncated" in r.stdout

    def test_runtime_error(self):
        r = execute_code("1 / 0")
        assert r.error
        assert not r.timed_out

    def test_syntax_error(self):
        r = execute_code("def f(: pass")
        assert r.error
        assert not r.timed_out

    def test_no_output_is_not_error(self):
        r = execute_code("x = 1 + 1")
        assert not r.error
        assert r.stdout == ""

    def test_multiline_output(self):
        r = execute_code("print('a'); print('b'); print('c')")
        assert r.stdout == "a\nb\nc"

    def test_physics_calculation(self):
        r = execute_code("m = 5.0; a = 9.8; F = m * a; print(f'{F:.1f}')")
        assert r.stdout == "49.0"
        assert not r.error

    def test_never_raises_on_none(self):
        # execute_code must never raise
        try:
            execute_code(None)  # type: ignore
        except Exception:
            pytest.fail("execute_code raised an exception on None input")

    def test_pint_allowed(self):
        r = execute_code(
            "import pint; ureg = pint.UnitRegistry(); q = 1 * ureg.meter; print(q)"
        )
        assert not r.error


# ---------------------------------------------------------------------------
# prompts constants
# ---------------------------------------------------------------------------

class TestPrompts:
    def test_code_stop_token(self):
        assert CODE_STOP == "[/code]"

    def test_no_answer_stop_token(self):
        # Phase 2 has no stop token — generation runs to EOS/max_tokens.
        # Occam's razor: chat-fine-tuned models emit EOS naturally after response.
        import phys_reasoner.tir.prompts as p
        assert not hasattr(p, "ANSWER_STOP"), "ANSWER_STOP should not exist in prompts"

    def test_system_prompt_has_format_tags(self):
        for tag in ["[code]", "[/code]", "[answer]"]:
            assert tag in TIR_SYSTEM_PROMPT, f"Tag '{tag}' missing from system prompt"
        assert r"\boxed" in TIR_SYSTEM_PROMPT

    def test_system_prompt_instructs_print(self):
        assert "print(" in TIR_SYSTEM_PROMPT

    def test_system_prompt_lists_key_packages(self):
        for pkg in ["numpy", "scipy", "sympy"]:
            assert pkg in TIR_SYSTEM_PROMPT, f"'{pkg}' not mentioned in system prompt"

    def test_allowed_packages_str_consistent_with_sandbox(self):
        for pkg in ALLOWED_PACKAGES_STR.replace(" ", "").split(","):
            assert pkg in ALLOWED_PACKAGES, (
                f"'{pkg}' in prompts.ALLOWED_PACKAGES_STR but not in sandbox.ALLOWED_PACKAGES"
            )


# ---------------------------------------------------------------------------
# prompts.extract_code
# ---------------------------------------------------------------------------

class TestExtractCode:
    def test_basic(self):
        assert extract_code("[code] print(42) [/code]") == "print(42)"

    def test_multiline(self):
        text = "[code]\nimport sympy\nprint(1)\n[/code]"
        result = extract_code(text)
        assert result is not None and "import sympy" in result

    def test_returns_last_block(self):
        assert extract_code("[code] x=1 [/code] text [code] x=2 [/code]") == "x=2"

    def test_no_block(self):
        assert extract_code("no code here") is None

    def test_unclosed_block(self):
        assert extract_code("[code] x=1") is None

    def test_empty_block(self):
        # Empty block is valid — sandbox will return empty stdout
        result = extract_code("[code][/code]")
        assert result == ""


# ---------------------------------------------------------------------------
# stage0_probe helpers
# ---------------------------------------------------------------------------

class TestExtractBoxed:
    def test_simple(self):
        assert _extract_boxed(r"\boxed{42}") == "42"

    def test_expression(self):
        result = _extract_boxed(r"\boxed{9.8 \, \mathrm{m/s^2}}")
        assert result is not None and "9.8" in result

    def test_nested_braces(self):
        assert _extract_boxed(r"\boxed{\frac{1}{2}mv^2}") == r"\frac{1}{2}mv^2"

    def test_returns_last(self):
        assert _extract_boxed(r"first \boxed{1} then \boxed{2}") == "2"

    def test_none_when_absent(self):
        assert _extract_boxed("no boxed here") is None

    def test_full_trajectory_format(self):
        # The answer comes BETWEEN [answer] and [/answer] — extract_boxed must find it
        text = r"[think] done [answer] \boxed{42 \, \mathrm{J}} [/answer]"
        result = _extract_boxed(text)
        assert result is not None and "42" in result


class TestBuildTirPrompt:
    def test_structure(self):
        msgs = build_tir_prompt("What is 2+2?")
        assert len(msgs) == 2
        assert msgs[0]["role"] == "system"
        assert msgs[1]["role"] == "user"
        assert msgs[1]["content"] == "What is 2+2?"

    def test_system_content_is_tir_prompt(self):
        msgs = build_tir_prompt("x")
        assert msgs[0]["content"] == TIR_SYSTEM_PROMPT
