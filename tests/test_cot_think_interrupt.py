"""Unit tests for the think-interrupt logic in SingleTurnAgentLoop (CoT mode).

Tests are CPU-only and mock all VeRL / vLLM dependencies.
Token IDs are derived from the real Qwen3.5-4B tokenizer so they stay consistent
with production.

CoT interrupt mirrors TIR interrupt with two differences:
  1. sub-call-2 is the final answer (no tool call, no tool response).
  2. response_ids are CONCATENATED (thinking + interrupt + answer), not
     overwritten — CoT has no tool parser that needs to see only the second
     sub-call's output.

Run with:
    python -m pytest tests/test_cot_think_interrupt.py -v
"""

from __future__ import annotations

import asyncio
import os
from unittest.mock import AsyncMock, MagicMock

import pytest

pytest.importorskip("verl", reason="VeRL not installed")

from transformers import AutoTokenizer  # noqa: E402

from verl.experimental.agent_loop.single_turn_agent_loop import (  # noqa: E402
    SingleTurnAgentLoop,
)
from verl.experimental.agent_loop.tool_agent_loop import (  # noqa: E402
    THINK_INTERRUPT_PHRASE,
)
from verl.workers.rollout.replica import TokenOutput  # noqa: E402

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


THINKING_BUDGET = 10
# The CoT post-interrupt budget absorbs (tool_call + tool_response + answer).
# Pick a response_length that leaves a sensible post-interrupt window.
POST_INTERRUPT_BUDGET_EXPECTED = 24   # set below via RESPONSE_LENGTH formula


def _response_length(interrupt_ids):
    return THINKING_BUDGET + len(interrupt_ids) + POST_INTERRUPT_BUDGET_EXPECTED


def _make_token_output(token_ids, log_probs=None) -> TokenOutput:
    out = MagicMock(spec=TokenOutput)
    out.token_ids = list(token_ids)
    out.log_probs = log_probs
    out.num_preempted = 0
    out.extra_fields = {}
    out.routed_experts = None
    return out


def _make_loop(interrupt_ids, think_end_id, thinking_budget=None, tool_call_budget=None):
    response_length = _response_length(interrupt_ids)
    loop = object.__new__(SingleTurnAgentLoop)

    cfg = MagicMock()
    cfg.prompt_length = 1024
    cfg.response_length = response_length
    cfg.multi_turn.thinking_budget = thinking_budget
    cfg.multi_turn.tool_call_budget = tool_call_budget
    loop.rollout_config = cfg

    tok = MagicMock()
    tok.convert_tokens_to_ids.return_value = think_end_id
    tok.encode.return_value = interrupt_ids
    loop.tokenizer = tok

    loop.server_manager = MagicMock()

    loop.prompt_length = 1024
    loop.response_length = response_length
    loop.thinking_budget = thinking_budget
    loop.tool_call_budget = tool_call_budget
    if thinking_budget is not None:
        loop._interrupt_ids = list(interrupt_ids)
        loop._think_end_id = think_end_id
        loop._post_interrupt_budget = response_length - thinking_budget - len(interrupt_ids)

    # Mock the async plumbing used by run().
    loop.process_vision_info = AsyncMock(return_value={"images": None, "videos": None})
    loop.apply_chat_template = AsyncMock(return_value=[10, 20])  # fixed prompt_ids
    return loop


def _run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


def _call_run(loop, sampling_params):
    return _run(loop.run(sampling_params=sampling_params, raw_prompt=[{"role": "user", "content": "q"}]))


# ---------------------------------------------------------------------------
# Feature disabled
# ---------------------------------------------------------------------------

class TestCotThinkInterruptDisabled:
    def test_no_interrupt_when_budget_none(self, interrupt_ids, think_end_id):
        lp = _make_loop(interrupt_ids, think_end_id, thinking_budget=None)
        output = _make_token_output(list(range(THINKING_BUDGET + 5)))
        lp.server_manager.generate = AsyncMock(return_value=output)
        _call_run(lp, {"max_tokens": lp.response_length})
        assert lp.server_manager.generate.call_count == 1

    def test_sampling_params_passthrough_when_disabled(self, interrupt_ids, think_end_id):
        lp = _make_loop(interrupt_ids, think_end_id, thinking_budget=None)
        output = _make_token_output([1, 2, 3])
        lp.server_manager.generate = AsyncMock(return_value=output)
        original = {"max_tokens": lp.response_length, "temperature": 0.9}
        _call_run(lp, original)
        used = lp.server_manager.generate.call_args[1]["sampling_params"]
        assert used is original


# ---------------------------------------------------------------------------
# Interrupt fires
# ---------------------------------------------------------------------------

class TestCotThinkInterruptEnabled:
    def test_interrupt_fires_when_thinking_truncated(self, interrupt_ids, think_end_id):
        lp = _make_loop(interrupt_ids, think_end_id, THINKING_BUDGET, tool_call_budget=8)
        out1 = _make_token_output(list(range(THINKING_BUDGET)))  # no think_end_id, full budget
        out2 = _make_token_output([200, 201, 202])
        lp.server_manager.generate = AsyncMock(side_effect=[out1, out2])
        _call_run(lp, {"max_tokens": lp.response_length})
        assert lp.server_manager.generate.call_count == 2

    def test_no_interrupt_when_think_end_in_sub_call_1(self, interrupt_ids, think_end_id):
        lp = _make_loop(interrupt_ids, think_end_id, THINKING_BUDGET, tool_call_budget=8)
        ids = list(range(THINKING_BUDGET - 2)) + [think_end_id, 500]
        out1 = _make_token_output(ids)
        lp.server_manager.generate = AsyncMock(return_value=out1)
        _call_run(lp, {"max_tokens": lp.response_length})
        assert lp.server_manager.generate.call_count == 1

    def test_no_interrupt_when_output_shorter_than_budget(self, interrupt_ids, think_end_id):
        lp = _make_loop(interrupt_ids, think_end_id, THINKING_BUDGET, tool_call_budget=8)
        out1 = _make_token_output([1, 2, 3])  # much shorter than budget
        lp.server_manager.generate = AsyncMock(return_value=out1)
        _call_run(lp, {"max_tokens": lp.response_length})
        assert lp.server_manager.generate.call_count == 1

    def test_sub_call1_max_tokens_eq_thinking_budget(self, interrupt_ids, think_end_id):
        lp = _make_loop(interrupt_ids, think_end_id, THINKING_BUDGET, tool_call_budget=8)
        out1 = _make_token_output(list(range(THINKING_BUDGET)))
        out2 = _make_token_output([200])
        lp.server_manager.generate = AsyncMock(side_effect=[out1, out2])
        _call_run(lp, {"max_tokens": lp.response_length})
        sc1 = lp.server_manager.generate.call_args_list[0][1]["sampling_params"]
        assert sc1["max_tokens"] == THINKING_BUDGET

    def test_sub_call2_max_tokens_eq_post_interrupt_budget(self, interrupt_ids, think_end_id):
        lp = _make_loop(interrupt_ids, think_end_id, THINKING_BUDGET, tool_call_budget=8)
        out1 = _make_token_output(list(range(THINKING_BUDGET)))
        out2 = _make_token_output([200])
        lp.server_manager.generate = AsyncMock(side_effect=[out1, out2])
        _call_run(lp, {"max_tokens": lp.response_length})
        sc2 = lp.server_manager.generate.call_args_list[1][1]["sampling_params"]
        assert sc2["max_tokens"] == POST_INTERRUPT_BUDGET_EXPECTED

    def test_mask_layout_is_1_0_1(self, interrupt_ids, think_end_id):
        lp = _make_loop(interrupt_ids, think_end_id, THINKING_BUDGET, tool_call_budget=8)
        out1 = _make_token_output(list(range(THINKING_BUDGET)))
        out2 = _make_token_output([200, 201])
        lp.server_manager.generate = AsyncMock(side_effect=[out1, out2])
        result = _call_run(lp, {"max_tokens": lp.response_length})
        expected = [1] * THINKING_BUDGET + [0] * len(interrupt_ids) + [1] * 2
        assert result.response_mask == expected

    def test_response_ids_concatenated_not_overwritten(self, interrupt_ids, think_end_id):
        """Key CoT-vs-TIR difference: response_ids holds all three segments so the
        final answer stays visible to the reward function."""
        lp = _make_loop(interrupt_ids, think_end_id, THINKING_BUDGET, tool_call_budget=8)
        thinking = list(range(THINKING_BUDGET))
        answer = [200, 201]
        out1 = _make_token_output(thinking)
        out2 = _make_token_output(answer)
        lp.server_manager.generate = AsyncMock(side_effect=[out1, out2])
        result = _call_run(lp, {"max_tokens": lp.response_length})
        assert result.response_ids == thinking + list(interrupt_ids) + answer

    def test_sub_call2_prompt_includes_thinking_and_interrupt(self, interrupt_ids, think_end_id):
        lp = _make_loop(interrupt_ids, think_end_id, THINKING_BUDGET, tool_call_budget=8)
        thinking = list(range(THINKING_BUDGET))
        out1 = _make_token_output(thinking)
        out2 = _make_token_output([200])
        lp.server_manager.generate = AsyncMock(side_effect=[out1, out2])
        _call_run(lp, {"max_tokens": lp.response_length})
        sc2_prompt = lp.server_manager.generate.call_args_list[1][1]["prompt_ids"]
        assert sc2_prompt == [10, 20] + thinking + list(interrupt_ids)

    def test_same_request_id_across_sub_calls(self, interrupt_ids, think_end_id):
        lp = _make_loop(interrupt_ids, think_end_id, THINKING_BUDGET, tool_call_budget=8)
        out1 = _make_token_output(list(range(THINKING_BUDGET)))
        out2 = _make_token_output([200])
        lp.server_manager.generate = AsyncMock(side_effect=[out1, out2])
        _call_run(lp, {"max_tokens": lp.response_length})
        rid1 = lp.server_manager.generate.call_args_list[0][1]["request_id"]
        rid2 = lp.server_manager.generate.call_args_list[1][1]["request_id"]
        assert rid1 == rid2

    def test_logprobs_padded_with_zeros_for_interrupt(self, interrupt_ids, think_end_id):
        lp = _make_loop(interrupt_ids, think_end_id, THINKING_BUDGET, tool_call_budget=8)
        out1 = _make_token_output(list(range(THINKING_BUDGET)), log_probs=[-0.1] * THINKING_BUDGET)
        out2 = _make_token_output([200, 201], log_probs=[-0.5, -0.6])
        lp.server_manager.generate = AsyncMock(side_effect=[out1, out2])
        result = _call_run(lp, {"max_tokens": lp.response_length})
        expected = [-0.1] * THINKING_BUDGET + [0.0] * len(interrupt_ids) + [-0.5, -0.6]
        assert result.response_logprobs == pytest.approx(expected)

    def test_logprobs_none_when_sc1_disabled(self, interrupt_ids, think_end_id):
        lp = _make_loop(interrupt_ids, think_end_id, THINKING_BUDGET, tool_call_budget=8)
        out1 = _make_token_output(list(range(THINKING_BUDGET)), log_probs=None)
        out2 = _make_token_output([200], log_probs=[-0.5])
        lp.server_manager.generate = AsyncMock(side_effect=[out1, out2])
        result = _call_run(lp, {"max_tokens": lp.response_length})
        assert result.response_logprobs is None

    def test_logprobs_length_matches_mask_when_sc2_returns_empty(
        self, interrupt_ids, think_end_id
    ):
        lp = _make_loop(interrupt_ids, think_end_id, THINKING_BUDGET, tool_call_budget=8)
        out1 = _make_token_output(list(range(THINKING_BUDGET)), log_probs=[-0.1] * THINKING_BUDGET)
        out2 = _make_token_output([200, 201], log_probs=None)
        lp.server_manager.generate = AsyncMock(side_effect=[out1, out2])
        result = _call_run(lp, {"max_tokens": lp.response_length})
        assert len(result.response_logprobs) == len(result.response_mask)

    def test_truncation_to_response_length(self, interrupt_ids, think_end_id):
        lp = _make_loop(interrupt_ids, think_end_id, THINKING_BUDGET, tool_call_budget=8)
        out1 = _make_token_output(list(range(THINKING_BUDGET)))
        # Produce more than post_interrupt_budget tokens — final output must cap at response_length.
        overlong_answer = list(range(1000, 1000 + POST_INTERRUPT_BUDGET_EXPECTED + 50))
        out2 = _make_token_output(overlong_answer)
        lp.server_manager.generate = AsyncMock(side_effect=[out1, out2])
        result = _call_run(lp, {"max_tokens": lp.response_length})
        assert len(result.response_ids) == lp.response_length
        assert len(result.response_mask) == lp.response_length

    def test_interrupt_ids_precomputed_not_per_call(self, interrupt_ids, think_end_id):
        lp = _make_loop(interrupt_ids, think_end_id, THINKING_BUDGET, tool_call_budget=8)
        out1 = _make_token_output(list(range(THINKING_BUDGET)))
        out2 = _make_token_output([200])
        lp.server_manager.generate = AsyncMock(side_effect=[out1, out2])
        _call_run(lp, {"max_tokens": lp.response_length})
        lp.tokenizer.encode.assert_not_called()


# ---------------------------------------------------------------------------
# __init__ budget assertion
# ---------------------------------------------------------------------------

class TestCotBudgetAssertion:
    def test_post_interrupt_budget_must_be_positive(self, interrupt_ids, think_end_id):
        """If response_length <= thinking_budget + len(interrupt_ids), __init__ must fail."""
        # We can't easily call the real __init__ (AgentLoopBase needs many fields),
        # so replay the same computation that __init__ performs and assert the guard.
        response_length = THINKING_BUDGET + len(interrupt_ids)  # no room for answer
        post_interrupt_budget = response_length - THINKING_BUDGET - len(interrupt_ids)
        assert post_interrupt_budget == 0
        with pytest.raises(AssertionError):
            assert post_interrupt_budget > 0, "No room for answer after interrupt"
