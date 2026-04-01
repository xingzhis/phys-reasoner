"""VeRL BaseTool wrapping phys_reasoner.tir.sandbox.execute_code.

Registered via scripts/physcode_tools.yaml and loaded by ToolAgentLoop.
The tool name "python" matches what Qwen3.5 emits in its native tool-call format.
"""

from __future__ import annotations

from verl.tools.base_tool import BaseTool
from verl.tools.schemas import OpenAIFunctionToolSchema, ToolResponse


class PythonSandboxTool(BaseTool):
    """Executes Python code in the phys_reasoner sandbox and returns stdout."""

    def get_openai_tool_schema(self) -> OpenAIFunctionToolSchema:
        return OpenAIFunctionToolSchema.model_validate({
            "type": "function",
            "function": {
                "name": "python",
                "description": (
                    "Execute Python code. "
                    "Allowed packages: numpy, scipy, sympy, pint, math, cmath, "
                    "statistics, fractions, decimal, itertools, functools, "
                    "collections, random, re, json, io, typing, time. "
                    "Must call print() to output the result."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "code": {
                            "type": "string",
                            "description": "Python code to execute. Must call print() to produce output.",
                        }
                    },
                    "required": ["code"],
                },
            },
        })

    async def execute(
        self, instance_id: str, parameters: dict, **kwargs
    ) -> tuple[ToolResponse, float, dict]:
        from phys_reasoner.tir.sandbox import execute_code  # noqa: PLC0415

        code = parameters.get("code", "")
        timeout = self.config.get("timeout", 30.0)
        max_output_bytes = self.config.get("max_output_bytes", 4096)

        result = execute_code(code, timeout=timeout, max_output_bytes=max_output_bytes)

        if result.error:
            text = f"(execution error)\n{result.stderr[:300]}"
        else:
            text = result.stdout or "(no output)"

        return ToolResponse(text=text), 0.0, {"exec_success": not result.error}
