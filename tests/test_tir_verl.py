"""PythonSandboxTool tests requiring VeRL. Skipped if VeRL is not installed.

Run with the -017.img overlay (has VeRL):
    python -m pytest tests/test_tir_verl.py -v

Covers two levels:
  - YAML config loading: verifies physcode_tools.yaml is valid and PythonSandboxTool
    is instantiable via VeRL's initialize_tools_from_config() — same code path as training.
  - Tool execution: verifies sandbox execution, error handling, schema correctness.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

pytest.importorskip("verl", reason="VeRL not installed")

from verl.tools.schemas import ToolResponse  # noqa: E402
from verl.tools.utils.tool_registry import initialize_tools_from_config  # noqa: E402
from phys_reasoner.tir.python_sandbox_tool import PythonSandboxTool  # noqa: E402

# Path to the tool config used in training
TOOLS_YAML = Path(__file__).parent.parent / "scripts" / "physcode_tools.yaml"


# ---------------------------------------------------------------------------
# Fixture
# ---------------------------------------------------------------------------

@pytest.fixture()
def tool() -> PythonSandboxTool:
    return PythonSandboxTool(
        config={"type": "native", "timeout": 5.0, "max_output_bytes": 4096},
        tool_schema=None,  # get_openai_tool_schema() provides it
    )


def _run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


# ---------------------------------------------------------------------------
# YAML config loading — same path as training (initialize_tools_from_config)
# ---------------------------------------------------------------------------

def test_tools_yaml_exists():
    assert TOOLS_YAML.exists(), f"physcode_tools.yaml not found at {TOOLS_YAML}"


def test_tools_yaml_loads_python_sandbox_tool():
    """Verify physcode_tools.yaml is parseable and produces a PythonSandboxTool."""
    tools = initialize_tools_from_config(str(TOOLS_YAML))
    assert len(tools) == 1, f"Expected 1 tool, got {len(tools)}"
    assert isinstance(tools[0], PythonSandboxTool)


def test_tools_yaml_tool_name_is_python():
    """Tool name must be 'python' — that's what Qwen3.5 emits in <tool_call>."""
    tools = initialize_tools_from_config(str(TOOLS_YAML))
    assert tools[0].name == "python"


def test_tools_yaml_tool_has_code_parameter():
    """Tool schema must have 'code' as a required parameter."""
    tools = initialize_tools_from_config(str(TOOLS_YAML))
    schema = tools[0].tool_schema.model_dump()
    props = schema["function"]["parameters"]["properties"]
    assert "code" in props
    assert "code" in schema["function"]["parameters"]["required"]


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

def test_tool_schema_name_is_python(tool):
    assert tool.name == "python"


def test_tool_schema_has_code_parameter(tool):
    schema = tool.tool_schema.model_dump()
    props = schema["function"]["parameters"]["properties"]
    assert "code" in props
    assert "code" in schema["function"]["parameters"]["required"]


# ---------------------------------------------------------------------------
# Execution
# ---------------------------------------------------------------------------

def test_tool_basic_execution(tool):
    response, reward, metrics = _run(tool.execute("id1", {"code": "print(99)"}))
    assert isinstance(response, ToolResponse)
    assert response.text == "99"
    assert reward == 0.0
    assert metrics["exec_success"] is True


def test_tool_multiline_output(tool):
    response, _, metrics = _run(tool.execute("id1", {"code": "print('a'); print('b')"}))
    assert "a" in response.text and "b" in response.text
    assert metrics["exec_success"] is True


def test_tool_exec_error(tool):
    response, _, metrics = _run(tool.execute("id1", {"code": "1 / 0"}))
    assert "(execution error)" in response.text
    assert metrics["exec_success"] is False


def test_tool_syntax_error(tool):
    response, _, metrics = _run(tool.execute("id1", {"code": "def f(: pass"}))
    assert "(execution error)" in response.text
    assert metrics["exec_success"] is False


def test_tool_timeout(tool):
    # Fixture sets timeout=5.0 — override via a short-timeout tool
    short_tool = PythonSandboxTool(
        config={"type": "native", "timeout": 1.0, "max_output_bytes": 4096},
        tool_schema=None,
    )
    response, _, metrics = _run(short_tool.execute("id1", {"code": "import time; time.sleep(30)"}))
    assert "(execution error)" in response.text
    assert metrics["exec_success"] is False


def test_tool_disallowed_package(tool):
    response, _, metrics = _run(tool.execute("id1", {"code": "import torch; print(torch.__version__)"}))
    assert "(execution error)" in response.text
    assert metrics["exec_success"] is False


def test_tool_no_output(tool):
    # No print() — not an error, just empty stdout → "(no output)"
    response, _, metrics = _run(tool.execute("id1", {"code": "x = 1 + 1"}))
    assert response.text == "(no output)"
    assert metrics["exec_success"] is True  # no error; caller sees "(no output)" text


def test_tool_sympy(tool):
    code = "import sympy; x = sympy.Symbol('x'); print(sympy.diff(x**3, x))"
    response, _, metrics = _run(tool.execute("id1", {"code": code}))
    assert "3*x**2" in response.text
    assert metrics["exec_success"] is True
