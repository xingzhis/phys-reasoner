"""PhysCode TIR Agent Loop — VeRL integration.

Subclasses AgentLoopBase and registers as "physcode_tir".

Trajectory format:
  reasoning text [code] python code [/code]
  [output] execution result
  interpretation text [answer] \\boxed{final answer}

CRITICAL — THINKING MODE: Qwen3.5 native <think>...</think> CoT MUST be disabled.
    enable_thinking=False is passed via data.apply_chat_template_kwargs in the training
    config (grpo_train.sh). The TIR format has NO [think]/[/think] tags — they collide
    with native thinking tokens and cause </think> leakage and format breakdown.
    See prompts.py module docstring for full explanation.

Phase 1: generate until [/code]  → response_mask all 1s
  execute_code()
  inject "[output] {result}\\n"  → response_mask all 0s (no gradient)
Phase 2: generate to EOS/max_tokens → response_mask all 1s
  Answer extracted via _extract_boxed() from full phase 2 output.

Activation: in the training launch script, add:
    import phys_reasoner.tir.tir_agent_loop  # triggers @register
"""

from __future__ import annotations

import logging
import os
from typing import Any
from uuid import uuid4

from verl.experimental.agent_loop.agent_loop import AgentLoopBase, AgentLoopOutput, register
from verl.workers.rollout.replica import TokenOutput

from phys_reasoner.tir.prompts import CODE_STOP, extract_code
from phys_reasoner.tir.sandbox import SandboxResult, execute_code

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

OUTPUT_PREFIX = "[output] "
OUTPUT_SUFFIX = "\n"


@register("physcode_tir")
class PhysCodeTIRAgentLoop(AgentLoopBase):
    """Single-block TIR agent loop for physics problem solving."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.prompt_length = self.rollout_config.prompt_length
        self.response_length = self.rollout_config.response_length

    async def run(self, sampling_params: dict[str, Any], **kwargs) -> AgentLoopOutput:
        messages = list(kwargs["raw_prompt"])
        request_id = uuid4().hex
        metrics: dict[str, Any] = {}

        # 1. Apply chat template
        prompt_ids = await self.apply_chat_template(messages)

        # 2. Phase 1: generate until [/code] (stop string included in output)
        phase1_params = {
            **sampling_params,
            "stop": [CODE_STOP],
            "include_stop_str_in_output": True,
            "max_tokens": self.response_length,
        }
        p1_out: TokenOutput = await self.server_manager.generate(
            request_id, prompt_ids=prompt_ids, sampling_params=phase1_params
        )
        phase1_ids = list(p1_out.token_ids)
        phase1_mask = [1] * len(phase1_ids)

        # 3. Extract and execute code
        phase1_text = self.tokenizer.decode(phase1_ids, skip_special_tokens=False)
        code_str = extract_code(phase1_text)

        # stop_reason == "completed" → [/code] was hit; "length" → budget exhausted
        if code_str and p1_out.stop_reason == "completed":
            result: SandboxResult = execute_code(code_str)
            if not result.error and result.stdout:
                output_text = result.stdout
            else:
                output_text = "(execution error)"
                logger.debug("Sandbox error: %s", result.stderr[:200])
        elif not code_str:
            output_text = "(no code block)"
            logger.debug("Phase 1: no [code]..[/code] block (stop_reason=%s)", p1_out.stop_reason)
        else:
            output_text = "(phase1 truncated)"
            logger.warning("Phase 1 hit token budget without closing [/code]")

        injection_text = f"{OUTPUT_PREFIX}{output_text}{OUTPUT_SUFFIX}"
        injection_ids = list(self.tokenizer.encode(injection_text, add_special_tokens=False))
        injection_mask = [0] * len(injection_ids)

        # 4. Phase 2: no stop token — generate to EOS or max_tokens.
        # The chat-fine-tuned model emits EOS after completing its response.
        # Answer is extracted by reward.py via _extract_boxed() on the full output.
        remaining = self.response_length - len(phase1_ids) - len(injection_ids)
        response_ids = phase1_ids + injection_ids
        response_mask = phase1_mask + injection_mask

        if remaining > 0:
            phase2_prompt_ids = prompt_ids + response_ids
            phase2_params = {
                **sampling_params,
                "max_tokens": remaining,
            }
            p2_out: TokenOutput = await self.server_manager.generate(
                request_id, prompt_ids=phase2_prompt_ids, sampling_params=phase2_params
            )
            phase2_ids = list(p2_out.token_ids)
            response_ids = response_ids + phase2_ids
            response_mask = response_mask + [1] * len(phase2_ids)

            if p2_out.stop_reason == "length":
                logger.warning(
                    "Phase 2 hit max_tokens=%d; \\boxed{} may be truncated", remaining
                )
        else:
            logger.warning(
                "No budget left for phase 2 (phase1=%d inj=%d budget=%d)",
                len(phase1_ids), len(injection_ids), self.response_length,
            )

        # 5. Truncate to response_length budget
        response_ids = response_ids[: self.response_length]
        response_mask = response_mask[: self.response_length]

        return AgentLoopOutput(
            prompt_ids=prompt_ids,
            response_ids=response_ids,
            response_mask=response_mask,
            num_turns=2,
            metrics=metrics,
            extra_fields={"turn_scores": [], "tool_rewards": []},
        )
