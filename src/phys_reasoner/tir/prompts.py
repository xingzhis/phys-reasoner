"""Centralized TIR format constants: system prompt, stop tokens, allowed packages.

Import from here in both stage0_probe.py and tir_agent_loop.py.
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
    "Reason step by step, then write Python code to compute the answer.\n"
    "\n"
    "Follow this EXACT format — every tag is required:\n"
    "  [think] your reasoning [code] your Python code [/code]\n"
    "  [output] (the code output is filled in here automatically)\n"
    "  [think] interpret the result [answer] \\boxed{{final answer}}\n"
    "\n"
    "Rules for the [code] block:\n"
    f"  - Allowed packages: {ALLOWED_PACKAGES_STR}\n"
    "  - You MUST print your final computed value with print(), e.g. print(result)\n"
    "  - Do not import packages not in the allowed list\n"
    "  - The code runs in isolation; define all variables inside it\n"
    "\n"
    "Rules for the [answer] block:\n"
    "  - Write the final answer as \\boxed{{value}} immediately after [answer]\n"
    "  - Include units if relevant, e.g. \\boxed{{9.8 \\, \\mathrm{{m/s^2}}}}\n"
    "  - Use the code output to verify your answer before writing it\n"
    "  - Stop after writing \\boxed{{...}}"
)

# ---------------------------------------------------------------------------
# TIR format parsing helpers (shared by stage0_probe and tir_agent_loop)
# ---------------------------------------------------------------------------


def extract_code(text: str) -> str | None:
    """Extract the last [code]...[/code] block from text."""
    matches = re.findall(r"\[code\](.*?)\[/code\]", text, re.DOTALL)
    return matches[-1].strip() if matches else None
