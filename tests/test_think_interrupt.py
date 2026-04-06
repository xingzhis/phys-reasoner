"""Unit tests for the think-interrupt logic in ToolAgentLoop._handle_generating_state.

Tests are CPU-only and mock all VeRL / vLLM dependencies.
Token IDs are derived from the real Qwen3.5-4B tokenizer so they stay consistent
with production.

Run with:
    python -m pytest tests/test_think_interrupt.py -v
"""

from __future__ import annotations

import asyncio
import os
from unittest.mock import AsyncMock, MagicMock

import pytest
import torch

pytest.importorskip("verl", reason="VeRL not installed")

from transformers import AutoTokenizer  # noqa: E402

from verl.experimental.agent_loop.tool_agent_loop import (  # noqa: E402
    THINK_INTERRUPT_PHRASE,
    AgentData,
    AgentState,
    ToolAgentLoop,
)
from verl.trainer.ppo.core_algos import agg_loss  # noqa: E402
from verl.workers.rollout.replica import TokenOutput  # noqa: E402

# ---------------------------------------------------------------------------
# Load tokenizer once for the whole module
# ---------------------------------------------------------------------------
_MODEL = os.environ.get("MODEL", "Qwen/Qwen3.5-4B")


@pytest.fixture(scope="module")
def tokenizer():
    return AutoTokenizer.from_pretrained(_MODEL)


@pytest.fixture(scope="module")
def think_end_id(tokenizer):
    return tokenizer.convert_tokens_to_ids("</think>")


@pytest.fixture(scope="module")
def interrupt_ids(tokenizer):
    return tokenizer.encode(THINK_INTERRUPT_PHRASE, add_special_tokens=False)


# ---------------------------------------------------------------------------
# Budget constants consistent with the budget identity:
#   response_length = thinking_budget + len(interrupt_ids)
#                   + tool_call_budget + max_tool_response_length
#                   + answer_budget   (must be > 0)
# ---------------------------------------------------------------------------
THINKING_BUDGET = 10
TOOL_CALL_BUDGET = 8
MAX_TOOL_RESPONSE_LEN = 4
ANSWER_BUDGET = 20


def _response_length(interrupt_ids):
    return THINKING_BUDGET + len(interrupt_ids) + TOOL_CALL_BUDGET + MAX_TOOL_RESPONSE_LEN + ANSWER_BUDGET


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_token_output(token_ids: list[int], log_probs=None) -> TokenOutput:
    out = MagicMock(spec=TokenOutput)
    out.token_ids = token_ids
    out.log_probs = log_probs
    out.num_preempted = 0
    out.extra_fields = {}
    out.routed_experts = None
    return out


def _make_agent_data(prompt_ids: list[int]) -> AgentData:
    data = MagicMock(spec=AgentData)
    data.prompt_ids = list(prompt_ids)
    data.response_ids = []
    data.response_mask = []
    data.response_logprobs = []
    data.assistant_turns = 0
    data.user_turns = 0
    data.metrics = {}
    data.extra_fields = {}
    data.routed_experts = None
    data.tool_calls = []
    data.image_data = None
    data.video_data = None
    data.request_id = "test-req"
    return data


def _make_loop(interrupt_ids, think_end_id, thinking_budget=None, tool_call_budget=None) -> ToolAgentLoop:
    response_length = _response_length(interrupt_ids)
    loop = object.__new__(ToolAgentLoop)

    cfg = MagicMock()
    cfg.multi_turn.max_user_turns = 1
    cfg.multi_turn.max_assistant_turns = 4
    cfg.multi_turn.max_parallel_calls = 1
    cfg.multi_turn.max_tool_response_length = MAX_TOOL_RESPONSE_LEN
    cfg.multi_turn.tool_response_truncate_side = "right"
    cfg.multi_turn.format = "qwen3_coder"
    cfg.multi_turn.tool_config_path = None
    cfg.multi_turn.interaction_config_path = None
    cfg.multi_turn.thinking_budget = thinking_budget
    cfg.multi_turn.tool_call_budget = tool_call_budget
    cfg.prompt_length = 1024
    cfg.response_length = response_length
    loop.rollout_config = cfg

    tok = MagicMock()
    tok.convert_tokens_to_ids.return_value = think_end_id
    tok.encode.return_value = interrupt_ids
    loop.tokenizer = tok

    loop.server_manager = MagicMock()
    loop.loop = asyncio.get_event_loop()

    tp = MagicMock()
    tp.extract_tool_calls = AsyncMock(return_value=(None, [MagicMock()]))
    loop.tool_parser = tp
    loop.tool_parser_name = "qwen3_coder"
    loop.tools = {}
    loop.tool_schemas = []

    loop.max_user_turns = 1
    loop.max_assistant_turns = 4
    loop.max_parallel_calls = 1
    loop.max_tool_response_length = MAX_TOOL_RESPONSE_LEN
    loop.tool_response_truncate_side = "right"
    loop.prompt_length = 1024
    loop.response_length = response_length
    loop.interaction_config_file = None

    loop.thinking_budget = thinking_budget
    loop.tool_call_budget = tool_call_budget
    if thinking_budget is not None:
        loop._interrupt_ids = interrupt_ids
        loop._think_end_id = think_end_id

    return loop


def _run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


# ---------------------------------------------------------------------------
# Feature disabled
# ---------------------------------------------------------------------------

class TestThinkInterruptDisabled:
    def test_no_interrupt_when_budget_none(self, interrupt_ids, think_end_id):
        lp = _make_loop(interrupt_ids, think_end_id, thinking_budget=None)
        output = _make_token_output(list(range(THINKING_BUDGET + 5)))
        lp.server_manager.generate = AsyncMock(return_value=output)
        data = _make_agent_data([10, 20])
        _run(lp._handle_generating_state(data, {"max_tokens": lp.response_length}))
        assert lp.server_manager.generate.call_count == 1

    def test_sub_call1_uses_original_params_when_disabled(self, interrupt_ids, think_end_id):
        """When disabled, sampling_params must not be modified."""
        lp = _make_loop(interrupt_ids, think_end_id, thinking_budget=None)
        output = _make_token_output([1, 2, 3])
        lp.server_manager.generate = AsyncMock(return_value=output)
        data = _make_agent_data([10])
        original_params = {"max_tokens": lp.response_length, "temperature": 0.9}
        _run(lp._handle_generating_state(data, original_params))
        used = lp.server_manager.generate.call_args[1]["sampling_params"]
        assert used is original_params


# ---------------------------------------------------------------------------
# Interrupt fires
# ---------------------------------------------------------------------------

class TestThinkInterruptEnabled:
    def test_sub_call1_capped_at_thinking_budget(self, interrupt_ids, think_end_id):
        """Sub-call 1 must use max_tokens=thinking_budget, not the full response window."""
        lp = _make_loop(interrupt_ids, think_end_id, THINKING_BUDGET, TOOL_CALL_BUDGET)
        output = _make_token_output(list(range(THINKING_BUDGET + 1)))
        lp.server_manager.generate = AsyncMock(return_value=output)
        data = _make_agent_data([10])
        _run(lp._handle_generating_state(data, {"max_tokens": lp.response_length}))
        sc1_params = lp.server_manager.generate.call_args_list[0][1]["sampling_params"]
        assert sc1_params["max_tokens"] == THINKING_BUDGET

    def test_interrupt_fires_when_truncated(self, interrupt_ids, think_end_id):
        lp = _make_loop(interrupt_ids, think_end_id, THINKING_BUDGET, TOOL_CALL_BUDGET)
        output1 = _make_token_output(list(range(THINKING_BUDGET)))   # no think_end_id
        output2 = _make_token_output([200, 201, 202])
        lp.server_manager.generate = AsyncMock(side_effect=[output1, output2])
        data = _make_agent_data([10])
        _run(lp._handle_generating_state(data, {"max_tokens": THINKING_BUDGET}))
        assert lp.server_manager.generate.call_count == 2

    def test_sub_call2_capped_at_tool_call_budget(self, interrupt_ids, think_end_id):
        lp = _make_loop(interrupt_ids, think_end_id, THINKING_BUDGET, TOOL_CALL_BUDGET)
        output1 = _make_token_output(list(range(THINKING_BUDGET)))
        output2 = _make_token_output([200])
        lp.server_manager.generate = AsyncMock(side_effect=[output1, output2])
        data = _make_agent_data([10])
        _run(lp._handle_generating_state(data, {"max_tokens": THINKING_BUDGET}))
        sc2_params = lp.server_manager.generate.call_args_list[1][1]["sampling_params"]
        assert sc2_params["max_tokens"] == TOOL_CALL_BUDGET

    def test_interrupt_ids_appended_with_zero_mask(self, interrupt_ids, think_end_id):
        lp = _make_loop(interrupt_ids, think_end_id, THINKING_BUDGET, TOOL_CALL_BUDGET)
        output1 = _make_token_output(list(range(THINKING_BUDGET)))
        output2 = _make_token_output([200, 201])
        lp.server_manager.generate = AsyncMock(side_effect=[output1, output2])
        data = _make_agent_data([10])
        _run(lp._handle_generating_state(data, {"max_tokens": THINKING_BUDGET}))
        expected = [1] * THINKING_BUDGET + [0] * len(interrupt_ids) + [1] * 2
        assert data.response_mask == expected

    def test_response_ids_overwritten_with_sub_call2(self, interrupt_ids, think_end_id):
        lp = _make_loop(interrupt_ids, think_end_id, THINKING_BUDGET, TOOL_CALL_BUDGET)
        output1 = _make_token_output(list(range(THINKING_BUDGET)))
        output2 = _make_token_output([200, 201])
        lp.server_manager.generate = AsyncMock(side_effect=[output1, output2])
        data = _make_agent_data([10])
        _run(lp._handle_generating_state(data, {"max_tokens": THINKING_BUDGET}))
        assert data.response_ids == [200, 201]

    def test_prompt_ids_contain_full_sequence(self, interrupt_ids, think_end_id):
        lp = _make_loop(interrupt_ids, think_end_id, THINKING_BUDGET, TOOL_CALL_BUDGET)
        thinking_ids = list(range(THINKING_BUDGET))
        output1 = _make_token_output(thinking_ids)
        output2 = _make_token_output([200, 201])
        lp.server_manager.generate = AsyncMock(side_effect=[output1, output2])
        data = _make_agent_data([10])
        _run(lp._handle_generating_state(data, {"max_tokens": THINKING_BUDGET}))
        assert data.prompt_ids == [10] + thinking_ids + list(interrupt_ids) + [200, 201]

    def test_no_interrupt_when_think_end_present(self, interrupt_ids, think_end_id):
        """If think_end_id is in the output, no interrupt fires."""
        lp = _make_loop(interrupt_ids, think_end_id, THINKING_BUDGET, TOOL_CALL_BUDGET)
        ids = list(range(THINKING_BUDGET - 2)) + [think_end_id, 500]
        output1 = _make_token_output(ids)
        lp.server_manager.generate = AsyncMock(return_value=output1)
        data = _make_agent_data([10])
        _run(lp._handle_generating_state(data, {"max_tokens": THINKING_BUDGET}))
        assert lp.server_manager.generate.call_count == 1
        assert all(m == 1 for m in data.response_mask)

    def test_no_interrupt_when_output_shorter_than_budget(self, interrupt_ids, think_end_id):
        lp = _make_loop(interrupt_ids, think_end_id, THINKING_BUDGET, TOOL_CALL_BUDGET)
        output1 = _make_token_output([1, 2, 3])   # len=3 < THINKING_BUDGET
        lp.server_manager.generate = AsyncMock(return_value=output1)
        data = _make_agent_data([10])
        _run(lp._handle_generating_state(data, {"max_tokens": THINKING_BUDGET}))
        assert lp.server_manager.generate.call_count == 1

    def test_logprobs_padded_with_zeros_for_interrupt(self, interrupt_ids, think_end_id):
        lp = _make_loop(interrupt_ids, think_end_id, THINKING_BUDGET, TOOL_CALL_BUDGET)
        output1 = _make_token_output(list(range(THINKING_BUDGET)), log_probs=[-0.1] * THINKING_BUDGET)
        output2 = _make_token_output([200, 201], log_probs=[-0.5, -0.6])
        lp.server_manager.generate = AsyncMock(side_effect=[output1, output2])
        data = _make_agent_data([10])
        _run(lp._handle_generating_state(data, {"max_tokens": THINKING_BUDGET}))
        expected = [-0.1] * THINKING_BUDGET + [0.0] * len(interrupt_ids) + [-0.5, -0.6]
        assert data.response_logprobs == pytest.approx(expected)

    def test_interrupt_ids_precomputed_not_per_call(self, interrupt_ids, think_end_id):
        """tokenizer.encode must not be called inside the hot path."""
        lp = _make_loop(interrupt_ids, think_end_id, THINKING_BUDGET, TOOL_CALL_BUDGET)
        output1 = _make_token_output(list(range(THINKING_BUDGET)))
        output2 = _make_token_output([200])
        lp.server_manager.generate = AsyncMock(side_effect=[output1, output2])
        data = _make_agent_data([10])
        _run(lp._handle_generating_state(data, {"max_tokens": THINKING_BUDGET}))
        lp.tokenizer.encode.assert_not_called()

    def test_case_d_no_tool_call_after_interrupt(self, interrupt_ids, think_end_id):
        """Case D: interrupt fires but sub-call 2 produces no tool call → TERMINATED."""
        lp = _make_loop(interrupt_ids, think_end_id, THINKING_BUDGET, TOOL_CALL_BUDGET)
        output1 = _make_token_output(list(range(THINKING_BUDGET)))
        output2 = _make_token_output([200, 201])
        lp.server_manager.generate = AsyncMock(side_effect=[output1, output2])
        lp.tool_parser.extract_tool_calls = AsyncMock(return_value=(None, []))
        data = _make_agent_data([10])
        state = _run(lp._handle_generating_state(data, {"max_tokens": THINKING_BUDGET}))
        assert state == AgentState.TERMINATED
        assert data.response_mask == [1] * THINKING_BUDGET + [0] * len(interrupt_ids) + [1] * 2

    # --- logprobs length-consistency tests ---

    def test_logprobs_length_matches_mask_when_sc2_returns_empty(self, interrupt_ids, think_end_id):
        """sc1 has logprobs, sc2 returns None/empty → sc2 slots must be zero-padded
        so len(response_logprobs) == len(response_mask)."""
        lp = _make_loop(interrupt_ids, think_end_id, THINKING_BUDGET, TOOL_CALL_BUDGET)
        output1 = _make_token_output(list(range(THINKING_BUDGET)), log_probs=[-0.1] * THINKING_BUDGET)
        output2 = _make_token_output([200, 201], log_probs=None)
        lp.server_manager.generate = AsyncMock(side_effect=[output1, output2])
        data = _make_agent_data([10])
        _run(lp._handle_generating_state(data, {"max_tokens": THINKING_BUDGET}))
        assert len(data.response_logprobs) == len(data.response_mask)

    def test_logprobs_length_matches_mask_when_sc2_returns_probs(self, interrupt_ids, think_end_id):
        """sc1 has logprobs AND sc2 has logprobs → all three segments tracked."""
        lp = _make_loop(interrupt_ids, think_end_id, THINKING_BUDGET, TOOL_CALL_BUDGET)
        output1 = _make_token_output(list(range(THINKING_BUDGET)), log_probs=[-0.1] * THINKING_BUDGET)
        output2 = _make_token_output([200, 201], log_probs=[-0.5, -0.6])
        lp.server_manager.generate = AsyncMock(side_effect=[output1, output2])
        data = _make_agent_data([10])
        _run(lp._handle_generating_state(data, {"max_tokens": THINKING_BUDGET}))
        assert len(data.response_logprobs) == len(data.response_mask)

    def test_logprobs_empty_when_sc1_disabled(self, interrupt_ids, think_end_id):
        """sc1 returns no logprobs → response_logprobs stays empty (no tracking activated)."""
        lp = _make_loop(interrupt_ids, think_end_id, THINKING_BUDGET, TOOL_CALL_BUDGET)
        output1 = _make_token_output(list(range(THINKING_BUDGET)), log_probs=None)
        output2 = _make_token_output([200, 201], log_probs=[-0.5, -0.6])
        lp.server_manager.generate = AsyncMock(side_effect=[output1, output2])
        data = _make_agent_data([10])
        _run(lp._handle_generating_state(data, {"max_tokens": THINKING_BUDGET}))
        # Tracking never activated — response_logprobs must remain empty.
        assert data.response_logprobs == []

    def test_sc2_zeros_used_when_sc2_probs_empty(self, interrupt_ids, think_end_id):
        """When sc2 returns no logprobs, the fallback zeros occupy the sc2 token slots."""
        sc2_len = 3
        lp = _make_loop(interrupt_ids, think_end_id, THINKING_BUDGET, TOOL_CALL_BUDGET)
        output1 = _make_token_output(list(range(THINKING_BUDGET)), log_probs=[-0.1] * THINKING_BUDGET)
        output2 = _make_token_output(list(range(200, 200 + sc2_len)), log_probs=None)
        lp.server_manager.generate = AsyncMock(side_effect=[output1, output2])
        data = _make_agent_data([10])
        _run(lp._handle_generating_state(data, {"max_tokens": THINKING_BUDGET}))
        expected = [-0.1] * THINKING_BUDGET + [0.0] * len(interrupt_ids) + [0.0] * sc2_len
        assert data.response_logprobs == pytest.approx(expected)


# ---------------------------------------------------------------------------
# agg_loss handles non-contiguous masks correctly
# ---------------------------------------------------------------------------

class TestAggLossNonContiguousMask:
    def test_zero_mask_positions_dont_contribute(self):
        """Interrupt tokens (mask=0 in the middle) must not affect the loss."""
        loss_mat = torch.tensor([[1.0, 2.0, 999.0, 999.0, 999.0, 3.0, 4.0]])
        mask = torch.tensor([[1, 1, 0, 0, 0, 1, 1]], dtype=torch.float32)
        loss_mat_zeroed = loss_mat.clone()
        loss_mat_zeroed[0, 2:5] = 0.0
        r1 = agg_loss(loss_mat, mask, loss_agg_mode="token-mean")
        r2 = agg_loss(loss_mat_zeroed, mask, loss_agg_mode="token-mean")
        assert r1.item() == pytest.approx(r2.item())

    def test_token_mean_matches_manual_calculation(self):
        loss_mat = torch.tensor([[2.0, 4.0, 0.0, 0.0, 6.0]])
        mask = torch.tensor([[1, 1, 0, 0, 1]], dtype=torch.float32)
        result = agg_loss(loss_mat, mask, loss_agg_mode="token-mean")
        assert result.item() == pytest.approx((2.0 + 4.0 + 6.0) / 3)
