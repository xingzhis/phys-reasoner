"""Agent loop tests requiring VeRL. Skipped if VeRL is not installed.

Run with the -017.img overlay (has VeRL):
    python -m pytest tests/test_tir_verl.py -v
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from unittest.mock import AsyncMock, MagicMock

import pytest

pytest.importorskip("verl", reason="VeRL not installed")

from verl.experimental.agent_loop.agent_loop import AgentLoopOutput  # noqa: E402
from phys_reasoner.tir.prompts import CODE_STOP  # noqa: E402
from phys_reasoner.tir.tir_agent_loop import PhysCodeTIRAgentLoop  # noqa: E402


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@dataclass
class FakeTokenOutput:
    token_ids: list
    stop_reason: str = "completed"
    log_probs: list = None
    routed_experts: list = None
    extra_fields: dict = None
    num_preempted: int = None

    def __post_init__(self):
        if self.extra_fields is None:
            self.extra_fields = {}


class FakeTok:
    """Minimal tokenizer mock: 1 token per character."""

    def encode(self, text, add_special_tokens=False):
        return list(range(len(text)))

    def decode(self, ids, skip_special_tokens=False):
        return "[think] reasoning [code]\nprint(99)\n[/code]"

    def apply_chat_template(self, msgs, tokenize=True, add_generation_prompt=True):
        return list(range(10)) if tokenize else "<prompt>"

    def pad(self, inputs, **kwargs):
        return inputs


@pytest.fixture()
def agent_loop():
    p1_text = "[think] reasoning [code]\nprint(99)\n[/code]"
    p2_text = r"[think] got 99 [answer] \boxed{99} [/answer]"

    server = MagicMock()
    server.generate = AsyncMock(
        side_effect=[
            FakeTokenOutput(token_ids=list(range(len(p1_text))), stop_reason="completed"),
            FakeTokenOutput(token_ids=list(range(len(p2_text))), stop_reason="completed"),
        ]
    )

    rollout_cfg = MagicMock()
    rollout_cfg.prompt_length = 512
    rollout_cfg.response_length = 2048

    loop = PhysCodeTIRAgentLoop.__new__(PhysCodeTIRAgentLoop)
    loop.config = MagicMock()
    loop.rollout_config = rollout_cfg
    loop.prompt_length = rollout_cfg.prompt_length
    loop.response_length = rollout_cfg.response_length
    loop.server_manager = server
    loop.tokenizer = FakeTok()
    loop.processor = None
    loop.dataset_cls = MagicMock()
    loop.data_config = MagicMock()
    loop.apply_chat_template_kwargs = {}
    loop.system_prompt = None
    loop.loop = asyncio.new_event_loop()

    async def _apply_chat_template(messages, **kwargs):
        return list(range(10))

    loop.apply_chat_template = _apply_chat_template
    return loop


def _run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


RAW_PROMPT = {"raw_prompt": [{"role": "user", "content": "What is 99?"}]}

# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_run_returns_agent_loop_output(agent_loop):
    output = _run(agent_loop.run(sampling_params={"temperature": 0.7}, **RAW_PROMPT))
    assert isinstance(output, AgentLoopOutput)


def test_response_mask_has_zeros_for_injection(agent_loop):
    """Injected [output] tokens must have mask=0 (no gradient)."""
    output = _run(agent_loop.run(sampling_params={"temperature": 0.7}, **RAW_PROMPT))
    assert 0 in output.response_mask, "No injection tokens (no zeros in response_mask)"
    assert 1 in output.response_mask, "No LLM tokens (no ones in response_mask)"


def test_response_ids_within_budget(agent_loop):
    output = _run(agent_loop.run(sampling_params={"temperature": 0.7}, **RAW_PROMPT))
    assert len(output.response_ids) <= agent_loop.rollout_config.response_length
    assert len(output.response_ids) == len(output.response_mask)


def test_server_called_twice(agent_loop):
    """Phase 1 and Phase 2 each call server.generate once."""
    _run(agent_loop.run(sampling_params={"temperature": 0.7}, **RAW_PROMPT))
    assert agent_loop.server_manager.generate.call_count == 2


def test_phase1_uses_code_stop(agent_loop):
    _run(agent_loop.run(sampling_params={}, **RAW_PROMPT))
    sp = agent_loop.server_manager.generate.call_args_list[0].kwargs["sampling_params"]
    assert CODE_STOP in sp["stop"]


def test_phase2_has_no_stop_token(agent_loop):
    """Phase 2 must NOT have a stop token — generation runs to EOS/max_tokens.
    Answer is extracted via _extract_boxed() on the full output.
    """
    _run(agent_loop.run(sampling_params={}, **RAW_PROMPT))
    sp = agent_loop.server_manager.generate.call_args_list[1].kwargs["sampling_params"]
    assert "stop" not in sp or not sp.get("stop"), (
        f"Phase 2 should have no stop tokens, but got stop={sp.get('stop')}"
    )


def test_injection_mask_zeros_are_contiguous(agent_loop):
    """The injected [output] block should appear as a contiguous run of zeros
    between the phase1 ones and the phase2 ones."""
    output = _run(agent_loop.run(sampling_params={}, **RAW_PROMPT))
    mask = output.response_mask

    # Find the zero block
    zero_indices = [i for i, m in enumerate(mask) if m == 0]
    if not zero_indices:
        pytest.fail("No zeros in response_mask — injection did not happen")

    # Contiguity: no gaps in zero_indices
    for a, b in zip(zero_indices, zero_indices[1:]):
        assert b == a + 1, f"Zero block not contiguous: gap at positions {a}-{b}"
