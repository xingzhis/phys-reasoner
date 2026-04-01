"""Centralized TIR format constants: system prompt, stop tokens, allowed packages.

Import from here in both stage0_probe.py and tir_agent_loop.py.

CRITICAL — THINKING MODE:
    Qwen3.5 uses <think>...</think> (angle brackets) for native chain-of-thought.
    We MUST disable this everywhere via enable_thinking=False in apply_chat_template.
    Failure to do so causes the model to enter native CoT mode, ignore our TIR
    format, and emit stray </think> tokens mid-response.

    The TIR format deliberately uses NO [think]/[/think] tags to avoid any
    collision with the native thinking tokens.

    Required in every call site:
      - stage0_probe.py: tokenizer.apply_chat_template(..., enable_thinking=False)
      - grpo_train.sh: data.apply_chat_template_kwargs.enable_thinking=False
      - tir_agent_loop.py: passes through apply_chat_template_kwargs from data config
"""

from __future__ import annotations

import re

# ---------------------------------------------------------------------------
# Stop tokens
# ---------------------------------------------------------------------------

CODE_STOP = "[/code]"
# Phase 1 stops at [/code] so the complete code block is included in the trajectory.

# Phase 2 has NO stop token — generation runs to EOS or max_tokens.
# The model (chat-fine-tuned) naturally emits EOS after completing its response.
# Answer extraction uses _extract_boxed() on the full phase 2 output.
# [answer] is kept in the prompt as a prefix MARKER only (tells the model where
# to write the final answer), not as a generation stop signal.

# ---------------------------------------------------------------------------
# Allowed packages (must stay in sync with sandbox.py ALLOWED_PACKAGES)
# ---------------------------------------------------------------------------

ALLOWED_PACKAGES_STR = (
    "numpy, scipy, sympy, pint, math, cmath, statistics, "
    "fractions, decimal, itertools, functools, collections, random, "
    "re, json, io, typing, time"
)

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

TIR_SYSTEM_PROMPT = (
    "You are an expert physics problem solver.\n"
    "Respond using EXACTLY this format — do not describe or explain it, just use it directly:\n"
    "\n"
    "<step-by-step reasoning>\n"
    "[code]\n"
    "<python code that prints the answer>\n"
    "[/code]\n"
    "[output] <filled in automatically — do not write this line>\n"
    "<interpret the output and confirm the answer>\n"
    "[answer] \\boxed{{<final answer>}}\n"
    "\n"
    "Code block rules:\n"
    f"  Allowed packages: {ALLOWED_PACKAGES_STR}\n"
    "  Must call print() to output the final value. Define all variables inside the block.\n"
    "\n"
    "Example:\n"
    "A ball falls from rest for t=2s under g=9.8 m/s^2. Distance = 0.5*g*t^2.\n"
    "[code]\n"
    "g = 9.8\n"
    "t = 2.0\n"
    "print(0.5 * g * t**2)\n"
    "[/code]\n"
    "[output] 19.6\n"
    "The code gives 19.6 m. [answer] \\boxed{{19.6 \\, \\mathrm{{m}}}}\n"
)

# ---------------------------------------------------------------------------
# TIR format parsing helpers (shared by stage0_probe and tir_agent_loop)
# ---------------------------------------------------------------------------


def extract_code(text: str) -> str | None:
    """Extract the last [code]...[/code] block from text."""
    matches = re.findall(r"\[code\](.*?)\[/code\]", text, re.DOTALL)
    return matches[-1].strip() if matches else None
