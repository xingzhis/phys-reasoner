"""Centralized TIR format constants: system prompt, stop tokens, tool schema, allowed packages.

Import from here in both stage0_probe.py and any custom TIR code.

CRITICAL — THINKING MODE:
    Qwen3.5 uses <think>...</think> (angle brackets) for native chain-of-thought.
    We MUST disable this everywhere via enable_thinking=False in apply_chat_template.
    Failure to do so causes the model to enter native CoT mode and emit stray
    </think> tokens mid-response.

    Required in every call site:
      - stage0_probe.py: tokenizer.apply_chat_template(..., enable_thinking=False)
      - grpo_train.sh: data.apply_chat_template_kwargs.enable_thinking=False
                       actor_rollout_ref.model.enable_thinking=False

FORMAT:
    Qwen3/Qwen3.5 may appear in two closely related tool-call formats:
      XML-style native Qwen format:
        <tool_call>
        <function=python>
        <parameter=code>
        ...
        </parameter>
        </function>
        </tool_call>

      Hermes/JSON format:
        <tool_call>
        {"name": "python", "arguments": {"code": "..."}}
        </tool_call>

      <tool_response>
      result
      </tool_response>
      interpretation... \\boxed{answer}

    The stop token </tool_call> is a special token — no confusion with HTML.
    VeRL's ToolAgentLoop handles injection and mask=0 automatically.
    stage0_probe.py does manual injection (vLLM, no VeRL).
"""

from __future__ import annotations

import re

# ---------------------------------------------------------------------------
# Stop tokens
# ---------------------------------------------------------------------------

TOOL_CALL_STOP = "</tool_call>"
# Phase 1 stops at </tool_call> — special token, model has strong prior on it.
# Phase 2 has NO stop token — generation runs to EOS or max_tokens.

# ---------------------------------------------------------------------------
# Allowed packages (must stay in sync with sandbox.py ALLOWED_PACKAGES)
# ---------------------------------------------------------------------------

ALLOWED_PACKAGES_STR = (
    "numpy, scipy, sympy, pint, math, cmath, statistics, "
    "fractions, decimal, itertools, functools, collections, random, "
    "re, json, io, typing, time"
)

# ---------------------------------------------------------------------------
# Tool schema (OpenAI function format) — used by stage0_probe and PythonSandboxTool
# ---------------------------------------------------------------------------

PYTHON_TOOL_SCHEMA: dict = {
    "type": "function",
    "function": {
        "name": "python",
        "description": (
            f"Execute Python code. Allowed packages: {ALLOWED_PACKAGES_STR}. "
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
}

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

TIR_SYSTEM_PROMPT = (
    "You are an expert physics problem solver.\n"
    "Use the Python tool exactly once to compute the answer. "
    "After seeing the result, give your final answer as \\boxed{<value>}.\n"
    "\n"
    f"Allowed packages: {ALLOWED_PACKAGES_STR}\n"
    "Your code must call print() to output the result. Define all variables inside the block.\n"
    "\n"
    "Example:\n"
    "A ball falls from rest for t=2s under g=9.8 m/s². Distance = 0.5·g·t².\n"
    "[Calls python tool with: g=9.8; t=2.0; print(0.5*g*t**2)]\n"
    "[Tool returns: 19.6]\n"
    "The distance is 19.6 m. \\boxed{19.6 \\, \\mathrm{m}}\n"
)

# ---------------------------------------------------------------------------
# TIR format parsing helpers (used by stage0_probe for manual 2-phase vLLM loop)
# ---------------------------------------------------------------------------


def extract_tool_call_code(text: str) -> str | None:
    """Extract Python code from the last <tool_call> block.

    Supports both:
      - native Qwen XML tool calls
      - Hermes/JSON tool calls
    """
    matches = re.findall(r"<tool_call>(.*?)</tool_call>", text, re.DOTALL)
    if not matches:
        return None
    last_block = matches[-1].strip()

    xml_match = re.search(
        r"<function=python>\s*<parameter=code>\s*(.*?)\s*</parameter>\s*</function>",
        last_block,
        re.DOTALL,
    )
    if xml_match:
        return xml_match.group(1)

    # Fallback: Hermes/JSON shape, matched permissively without fully decoding JSON.
    json_match = re.search(
        r'"name"\s*:\s*"python".*?"code"\s*:\s*"((?:\\.|[^"\\])*)"',
        last_block,
        re.DOTALL,
    )
    if json_match:
        # Unescape JSON string contents without depending on stricter whole-object parsing.
        return bytes(json_match.group(1), "utf-8").decode("unicode_escape")

    return None
