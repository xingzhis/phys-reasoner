"""Tests for the TIR pipeline: sandbox, prompts, stage0 helpers.

No GPU required, no VeRL required.
VeRL-dependent tool tests are in tests/test_tir_verl.py.

Run:
    python -m pytest tests/test_tir.py -v
"""

from __future__ import annotations

import pytest

from phys_reasoner.tir.sandbox import ALLOWED_PACKAGES, SandboxResult, execute_code, _check_imports
from phys_reasoner.tir.prompts import (
    ALLOWED_PACKAGES_STR,
    PYTHON_TOOL_SCHEMA,
    TOOL_CALL_STOP,
    TIR_SYSTEM_PROMPT,
    extract_tool_call_code,
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
    def test_tool_call_stop_token(self):
        assert TOOL_CALL_STOP == "</tool_call>"

    def test_no_answer_stop_token(self):
        # Phase 2 has no stop token — generation runs to EOS/max_tokens.
        import phys_reasoner.tir.prompts as p
        assert not hasattr(p, "ANSWER_STOP"), "ANSWER_STOP should not exist in prompts"

    def test_no_code_stop_token(self):
        # Old [/code] stop token is gone.
        import phys_reasoner.tir.prompts as p
        assert not hasattr(p, "CODE_STOP"), "CODE_STOP should not exist — use TOOL_CALL_STOP"

    def test_system_prompt_has_boxed(self):
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

    def test_python_tool_schema_structure(self):
        assert PYTHON_TOOL_SCHEMA["type"] == "function"
        fn = PYTHON_TOOL_SCHEMA["function"]
        assert fn["name"] == "python"
        assert "code" in fn["parameters"]["properties"]
        assert "code" in fn["parameters"]["required"]


# ---------------------------------------------------------------------------
# prompts.extract_tool_call_code
# ---------------------------------------------------------------------------

class TestExtractToolCallCode:
    def _wrap_json(self, code: str) -> str:
        import json
        return f'<tool_call>\n{json.dumps({"name": "python", "arguments": {"code": code}})}\n</tool_call>'

    def _wrap_xml(self, code: str) -> str:
        return f"<tool_call>\n<function=python>\n<parameter=code>\n{code}\n</parameter>\n</function>\n</tool_call>"

    def test_basic_xml(self):
        assert extract_tool_call_code(self._wrap_xml("print(42)")) == "print(42)"

    def test_multiline_xml(self):
        code = "import sympy\nprint(1)"
        result = extract_tool_call_code(self._wrap_xml(code))
        assert result is not None and "import sympy" in result

    def test_json_still_supported(self):
        assert extract_tool_call_code(self._wrap_json("print(42)")) == "print(42)"

    def test_returns_last_block(self):
        block1 = self._wrap_xml("x=1")
        block2 = self._wrap_xml("x=2")
        assert extract_tool_call_code(f"{block1} text {block2}") == "x=2"

    def test_no_block(self):
        assert extract_tool_call_code("no tool call here") is None

    def test_malformed_json(self):
        assert extract_tool_call_code("<tool_call>not json</tool_call>") is None

    def test_missing_code_key(self):
        assert extract_tool_call_code('<tool_call>{"name":"python","arguments":{}}</tool_call>') is None
        assert extract_tool_call_code("<tool_call><function=python></function></tool_call>") is None

    def test_empty_code(self):
        result = extract_tool_call_code(self._wrap_xml(""))
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
        text = r"The distance is \boxed{42 \, \mathrm{J}}"
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
