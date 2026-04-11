"""Centralized TIR format constants: system prompt, stop tokens, tool schema, allowed packages.

Import from here in both stage0_probe.py and any custom TIR code.

THINKING MODE — enable_thinking=True for phase 1, False for phase 2:
    Confirmed 2026-04-02 by A/B smoke test (dump_rollouts.py, Qwen3.5-4B, ball-drop problem).

    WHY thinking must be ON for phase 1 (the tool-call turn):
      In Qwen3.5's tool-call format the assistant turn has no "plain content" slot —
      the model is expected to emit a <tool_call> block directly. With thinking
      suppressed (enable_thinking=False), there is nowhere for the model to reason,
      so it moves all reasoning into Python code comments. With thinking enabled, it
      reasons in a proper <think>...</think> block, then calls the tool with clean
      minimal code. Observed: </think> closes cleanly before <tool_call> — no leakage.

    WHY thinking must be OFF for phase 2 (the final-answer turn):
      After the <tool_response> is injected, the model just needs to state the answer.
      Thinking here adds tokens without value and risks re-deriving the answer rather
      than trusting the tool output. The injection (_make_tool_injection) already
      passes enable_thinking=False, so this is handled automatically.

    TOKEN LENGTH CAVEAT — thinking traces on hard problems can be long.
      Planned mitigations (not yet implemented):
        - Token-budget checkpoint: inject "<!-- budget: N tokens remaining -->" into
          the system prompt so the model self-regulates thinking length.
        - Stop-thinking injection: if phase 1 hits a token threshold, inject </think>
          to force the model to proceed to the tool call.
        - Soft prompt / instruction: add a "think briefly" instruction to the system
          prompt to discourage verbose reasoning.
        - RL length penalty: add a token-count penalty term to the reward in late
          GRPO ablations (λ > 0) to discourage unnecessarily long trajectories.

    Required settings per call site:
      - phase 1 prompts (stage0_probe.py, dump_rollouts.py):
            tokenizer.apply_chat_template(..., enable_thinking=True)
      - phase 2 injection (_make_tool_injection in dump_rollouts.py):
            tokenizer.apply_chat_template(..., enable_thinking=False)   ← already correct
      - training (grpo_train.sh):
            +data.apply_chat_template_kwargs.enable_thinking=True
      NOTE: actor_rollout_ref.model.enable_thinking is NOT a valid VeRL field in this
            build and crashes worker init if added — do not use it.

FORMAT — CRITICAL: Qwen3 vs Qwen3.5 use DIFFERENT tool-call formats
    Confirmed 2026-04-01 by inspecting rendered prompts via dump_rollouts.py.

    Qwen3 (0.6B, 1.7B, 4B, ...) → hermes/JSON format:
        <tool_call>
        {"name": "python", "arguments": {"code": "..."}}
        </tool_call>
        VeRL: multi_turn.format=hermes
        dump_rollouts / stage0_probe: extract_tool_call_code json_match branch

    Qwen3.5 (0.8B, 4B, ...) → Qwen XML/coder format:
        <tool_call>
        <function=python>
        <parameter=code>
        ...
        </parameter>
        </function>
        </tool_call>
        VeRL: multi_turn.format=qwen3_coder
        dump_rollouts / stage0_probe: extract_tool_call_code xml_match branch

    Using the wrong VeRL format causes silent failures: ToolAgentLoop cannot
    parse the tool call, no tool ever runs, and all rollouts get zero reward.

    The stop token </tool_call> is the same for both formats.
    extract_tool_call_code() handles both via separate regex branches.

    <tool_response>
    result
    </tool_response>
    interpretation... \\boxed{answer}

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

# Think-interrupt phrase — injected when thinking exceeds budget (mask=0 in training).
# Must match verl/verl/experimental/agent_loop/tool_agent_loop.py exactly.
THINK_INTERRUPT_PHRASE = "\nOkay, I've thought enough. Time to write my response.\n</think>\n"

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

def make_system_prompt(max_tool_calls: int = 1) -> str:
    """Return the TIR system prompt with the given tool-call budget.

    Use this anywhere MAX_TOOL_TURNS != 1.  All existing call sites that
    import TIR_SYSTEM_PROMPT directly get the default (max_tool_calls=1).
    """
    if max_tool_calls == 1:
        budget_line = "You may execute code at most once. Once you have the result, give your final answer directly.\n"
    else:
        budget_line = (
            f"You may execute code at most {max_tool_calls} times. "
            "Use a second call only to fix a runtime error in the first — "
            "not to refine a working result. "
            "Once you have a successful result, give your final answer directly.\n"
        )
    return (
        "You are an expert physics problem solver.\n"
        # "You MUST call the python tool exactly once to compute the answer. "
        # "Do NOT solve the problem analytically without using the tool. "
        # "After seeing the tool result, give your final answer as \\boxed{<value>}.\n"
        "You have access to a Python interpreter. Use it when it helps — "
        "for numerical computation, symbolic algebra, or unit conversion. "
        "You are not required to use it.\n"
        "Only call the tool if the code performs actual computation — "
        "numerical evaluation, symbolic solving, or unit conversion. "
        "Do not use it to print a formula or expression you derived in text.\n"
        "\n"
        f"Allowed packages: {ALLOWED_PACKAGES_STR}\n"
        "If you write code, it must call print() to output the result. "
        "Define all variables inside the code block.\n"
        "\n"
        + budget_line +
        "\n"
        "End with your final answer as \\boxed{<value>}.\n"
        # No format example here — the tool schema injected by apply_chat_template(tools=[...])
        # is sufficient to prime the model's tool-call format. An explicit example would need
        # to be format-specific (hermes vs qwen3_coder) and TIR_SYSTEM_PROMPT is shared across
        # both Qwen3 and Qwen3.5. Validated without example on Qwen3-0.6B (smoke_tir.sh).
    )


# Default prompt (max_tool_calls=1). All existing call sites use this directly.
# If MAX_TOOL_TURNS=2 is used in training/smoke, call make_system_prompt(2) instead.
TIR_SYSTEM_PROMPT = make_system_prompt(max_tool_calls=1)

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
