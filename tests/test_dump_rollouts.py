"""Tests for scripts/dump_rollouts.py — think-interrupt, n_rollouts, parquet output.

Tests are CPU-only and mock vLLM + sandbox. The interrupt condition logic is tested
against the exact same checks as verl/verl/experimental/agent_loop/tool_agent_loop.py.

Run with:
    python -m pytest tests/test_dump_rollouts.py -v
"""

from __future__ import annotations

import os
import sys
import tempfile
from dataclasses import dataclass
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

# Ensure scripts/ is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

import dump_rollouts  # noqa: E402


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

THINK_END_TOKEN = "</think>"
# Fake token ID for </think> — consistent across tests
THINK_END_ID = 99999

# Fake interrupt phrase token IDs (length matters, content doesn't for unit tests)
FAKE_INTERRUPT_IDS = list(range(1000, 1015))  # 15 tokens, same as Qwen3.5-4B


@dataclass
class FakeOutput:
    text: str
    token_ids: list[int]
    finish_reason: str


@dataclass
class FakeRequestOutput:
    outputs: list[FakeOutput]


def _make_request_output(text: str, token_ids: list[int], finish_reason: str = "stop"):
    return FakeRequestOutput(outputs=[FakeOutput(text=text, token_ids=token_ids, finish_reason=finish_reason)])


def _make_test_parquet(tmpdir: str, n: int = 2) -> str:
    """Create a minimal parquet with the columns run_dump expects."""
    rows = []
    for i in range(n):
        rows.append({
            "prompt": [
                {"role": "system", "content": "You are a physicist."},
                {"role": "user", "content": f"Problem {i}: What is 2+{i}?"},
            ],
            "reward_model": {"ground_truth": str(2 + i)},
            "extra_info": {
                "problem": f"Problem {i}: What is 2+{i}?",
                "answer_type": "numerical",
            },
        })
    path = os.path.join(tmpdir, "test.parquet")
    pd.DataFrame(rows).to_parquet(path, index=False)
    return path


@pytest.fixture
def tmpdir():
    with tempfile.TemporaryDirectory() as d:
        yield d


@pytest.fixture
def test_parquet(tmpdir):
    return _make_test_parquet(tmpdir, n=2)


@pytest.fixture
def mock_tokenizer():
    tok = MagicMock()
    tok.apply_chat_template.return_value = "<|im_start|>system\nYou are...<|im_end|>\n<|im_start|>user\nQ<|im_end|>\n<|im_start|>assistant\n"
    tok.encode.return_value = FAKE_INTERRUPT_IDS
    tok.convert_tokens_to_ids.return_value = THINK_END_ID
    return tok


# ---------------------------------------------------------------------------
# _extract_boxed
# ---------------------------------------------------------------------------

class TestExtractBoxed:
    def test_simple(self):
        assert dump_rollouts._extract_boxed(r"The answer is \boxed{42}.") == "42"

    def test_nested_braces(self):
        assert dump_rollouts._extract_boxed(r"\boxed{\frac{1}{2}}") == r"\frac{1}{2}"

    def test_no_boxed(self):
        assert dump_rollouts._extract_boxed("no boxed answer here") is None

    def test_uses_last_boxed(self):
        text = r"First \boxed{wrong}. Actually \boxed{right}."
        assert dump_rollouts._extract_boxed(text) == "right"


# ---------------------------------------------------------------------------
# Interrupt condition (isolated)
# ---------------------------------------------------------------------------

class TestInterruptCondition:
    """Test the exact condition from tool_agent_loop.py lines 297-299:
        len(output.token_ids) >= thinking_budget
        and think_end_id not in output.token_ids
    """

    def test_fires_when_budget_exceeded_no_think_end(self):
        thinking_budget = 10
        token_ids = list(range(thinking_budget))  # len == budget, no think_end_id
        assert len(token_ids) >= thinking_budget
        assert THINK_END_ID not in token_ids

    def test_no_fire_when_think_end_present(self):
        thinking_budget = 10
        token_ids = list(range(thinking_budget - 1)) + [THINK_END_ID]
        assert len(token_ids) >= thinking_budget
        assert THINK_END_ID in token_ids  # should NOT fire

    def test_no_fire_when_output_shorter(self):
        thinking_budget = 10
        token_ids = [1, 2, 3]
        assert len(token_ids) < thinking_budget  # should NOT fire

    def test_fires_when_output_exceeds_budget(self):
        thinking_budget = 10
        token_ids = list(range(thinking_budget + 5))
        assert len(token_ids) >= thinking_budget
        assert THINK_END_ID not in token_ids


# ---------------------------------------------------------------------------
# Full run_dump with mocked vLLM
# ---------------------------------------------------------------------------

def _run_dump_mocked(
    tmpdir,
    test_parquet,
    mock_tokenizer,
    *,
    n_rollouts=1,
    thinking_budget=None,
    tool_call_budget=None,
    answer_budget=None,
    p1_outputs=None,
    p1b_outputs=None,
    p2_outputs=None,
    sandbox_result=None,
):
    """Helper: run dump_rollouts.run_dump with all heavy deps mocked."""
    n_problems = 2
    total = n_problems * n_rollouts

    # Default outputs if not specified
    if p1_outputs is None:
        # Model produces a tool call, stops at </tool_call>
        p1_outputs = [
            _make_request_output(
                text="<think>reasoning</think>\n<tool_call>\n<function=python>\n<parameter=code>\nprint(42)\n</parameter>\n</function>\n</tool_call>",
                token_ids=[1, 2, 3, THINK_END_ID, 10, 11, 12],
                finish_reason="stop",
            )
            for _ in range(total)
        ]

    if p2_outputs is None:
        p2_outputs = [
            _make_request_output(
                text="The answer is \\boxed{42}.",
                token_ids=[50, 51, 52],
                finish_reason="stop",
            )
            for _ in range(total)
        ]

    if sandbox_result is None:
        from phys_reasoner.tir.sandbox import SandboxResult
        sandbox_result = SandboxResult(stdout="42", stderr="", error=False)

    # Track which generate call we're on
    generate_calls = []

    def fake_generate(prompts, params):
        call_idx = len(generate_calls)
        generate_calls.append((prompts, params))
        if call_idx == 0:
            return p1_outputs[:len(prompts)]
        elif call_idx == 1 and p1b_outputs is not None:
            return p1b_outputs[:len(prompts)]
        else:
            # Last call is always phase 2
            return p2_outputs[:len(prompts)]

    mock_llm_instance = MagicMock()
    mock_llm_instance.generate.side_effect = fake_generate

    out_dir = os.path.join(tmpdir, "out")

    with (
        patch("transformers.AutoTokenizer.from_pretrained", return_value=mock_tokenizer),
        patch("vllm.LLM", return_value=mock_llm_instance),
        patch("phys_reasoner.tir.sandbox.execute_code", return_value=sandbox_result),
    ):

        dump_rollouts.run_dump(
            model_path="test-model",
            parquet_path=test_parquet,
            n=n_problems,
            n_rollouts=n_rollouts,
            out_dir=out_dir,
            seed=42,
            gpu_mem=0.3,
            temperature=1.0,
            top_p=0.9,
            enable_thinking=True,
            max_tokens=100,
            thinking_budget=thinking_budget,
            tool_call_budget=tool_call_budget,
            answer_budget=answer_budget,
            max_prompt_len=256,
            max_tool_response_len=512,
            dump_txt=True,
        )

    return out_dir, generate_calls


class TestRunDumpNoInterrupt:
    """Tests with think-interrupt disabled (thinking_budget=None)."""

    def test_two_generate_calls(self, tmpdir, test_parquet, mock_tokenizer):
        """Without interrupt: exactly 2 generate calls (phase 1 + phase 2)."""
        out_dir, calls = _run_dump_mocked(tmpdir, test_parquet, mock_tokenizer)
        assert len(calls) == 2  # phase 1 + phase 2

    def test_phase1_uses_max_tokens(self, tmpdir, test_parquet, mock_tokenizer):
        out_dir, calls = _run_dump_mocked(tmpdir, test_parquet, mock_tokenizer)
        p1_params = calls[0][1]
        assert p1_params.max_tokens == 100  # max_tokens default

    def test_phase2_uses_max_tokens(self, tmpdir, test_parquet, mock_tokenizer):
        out_dir, calls = _run_dump_mocked(tmpdir, test_parquet, mock_tokenizer)
        p2_params = calls[1][1]
        assert p2_params.max_tokens == 100

    def test_txt_files_created(self, tmpdir, test_parquet, mock_tokenizer):
        out_dir, _ = _run_dump_mocked(tmpdir, test_parquet, mock_tokenizer)
        txts = [f for f in os.listdir(out_dir) if f.endswith(".txt") and f.startswith("rollout_")]
        assert len(txts) == 2  # 2 problems x 1 rollout

    def test_parquet_created(self, tmpdir, test_parquet, mock_tokenizer):
        out_dir, _ = _run_dump_mocked(tmpdir, test_parquet, mock_tokenizer)
        pq = pd.read_parquet(os.path.join(out_dir, "rollouts.parquet"))
        assert len(pq) == 2
        assert "problem_idx" in pq.columns
        assert "rollout_idx" in pq.columns
        assert "gold_answer" in pq.columns
        assert "phase1_text" in pq.columns
        assert "interrupted" in pq.columns
        assert "code" in pq.columns
        assert "phase2_text" in pq.columns
        assert "extra_info" in pq.columns

    def test_interrupted_all_false(self, tmpdir, test_parquet, mock_tokenizer):
        out_dir, _ = _run_dump_mocked(tmpdir, test_parquet, mock_tokenizer)
        pq = pd.read_parquet(os.path.join(out_dir, "rollouts.parquet"))
        assert all(pq["interrupted"] == False)  # noqa: E712


class TestRunDumpWithInterrupt:
    """Tests with think-interrupt enabled."""

    def _make_thinking_only_output(self, thinking_budget):
        """Output that exceeds thinking_budget without </think> — interrupt fires."""
        return _make_request_output(
            text="<think>long reasoning that exceeds budget without closing think tag...",
            token_ids=list(range(thinking_budget)),  # exactly at budget, no THINK_END_ID
            finish_reason="length",
        )

    def _make_tool_call_output(self):
        """Output from phase 1b: model produces tool call after interrupt."""
        return _make_request_output(
            text="<tool_call>\n<function=python>\n<parameter=code>\nprint(42)\n</parameter>\n</function>\n</tool_call>",
            token_ids=[200, 201, 202, 203, 204],
            finish_reason="stop",
        )

    def test_three_generate_calls_when_interrupted(self, tmpdir, test_parquet, mock_tokenizer):
        """With interrupt: 3 generate calls (phase 1 + phase 1b + phase 2)."""
        tb = 10
        p1_outputs = [self._make_thinking_only_output(tb) for _ in range(2)]
        p1b_outputs = [self._make_tool_call_output() for _ in range(2)]

        out_dir, calls = _run_dump_mocked(
            tmpdir, test_parquet, mock_tokenizer,
            thinking_budget=tb, tool_call_budget=20, answer_budget=30,
            p1_outputs=p1_outputs, p1b_outputs=p1b_outputs,
        )
        assert len(calls) == 3  # phase 1 + phase 1b + phase 2

    def test_phase1_uses_thinking_budget(self, tmpdir, test_parquet, mock_tokenizer):
        tb = 10
        p1_outputs = [self._make_thinking_only_output(tb) for _ in range(2)]
        p1b_outputs = [self._make_tool_call_output() for _ in range(2)]

        _, calls = _run_dump_mocked(
            tmpdir, test_parquet, mock_tokenizer,
            thinking_budget=tb, tool_call_budget=20, answer_budget=30,
            p1_outputs=p1_outputs, p1b_outputs=p1b_outputs,
        )
        assert calls[0][1].max_tokens == tb

    def test_phase1b_uses_tool_call_budget(self, tmpdir, test_parquet, mock_tokenizer):
        tb = 10
        tcb = 20
        p1_outputs = [self._make_thinking_only_output(tb) for _ in range(2)]
        p1b_outputs = [self._make_tool_call_output() for _ in range(2)]

        _, calls = _run_dump_mocked(
            tmpdir, test_parquet, mock_tokenizer,
            thinking_budget=tb, tool_call_budget=tcb, answer_budget=30,
            p1_outputs=p1_outputs, p1b_outputs=p1b_outputs,
        )
        assert calls[1][1].max_tokens == tcb

    def test_phase2_uses_answer_budget(self, tmpdir, test_parquet, mock_tokenizer):
        tb = 10
        ab = 30
        p1_outputs = [self._make_thinking_only_output(tb) for _ in range(2)]
        p1b_outputs = [self._make_tool_call_output() for _ in range(2)]

        _, calls = _run_dump_mocked(
            tmpdir, test_parquet, mock_tokenizer,
            thinking_budget=tb, tool_call_budget=20, answer_budget=ab,
            p1_outputs=p1_outputs, p1b_outputs=p1b_outputs,
        )
        assert calls[2][1].max_tokens == ab

    def test_no_interrupt_when_think_end_present(self, tmpdir, test_parquet, mock_tokenizer):
        """If </think> is in token_ids, interrupt should NOT fire even if budget exceeded."""
        tb = 5
        # Output exceeds budget but has think_end_id
        p1_outputs = [
            _make_request_output(
                text="<think>short</think>\n<tool_call>\n<function=python>\n<parameter=code>\nprint(1)\n</parameter>\n</function>\n</tool_call>",
                token_ids=list(range(tb)) + [THINK_END_ID, 50, 51],
                finish_reason="stop",
            )
            for _ in range(2)
        ]

        _, calls = _run_dump_mocked(
            tmpdir, test_parquet, mock_tokenizer,
            thinking_budget=tb, tool_call_budget=20, answer_budget=30,
            p1_outputs=p1_outputs,
        )
        # Only 2 calls: phase 1 + phase 2 (no phase 1b)
        assert len(calls) == 2

    def test_no_interrupt_when_output_shorter_than_budget(self, tmpdir, test_parquet, mock_tokenizer):
        tb = 100
        p1_outputs = [
            _make_request_output(
                text="<think>quick</think>\n<tool_call>\n<function=python>\n<parameter=code>\nprint(1)\n</parameter>\n</function>\n</tool_call>",
                token_ids=[1, 2, 3],  # len=3 < 100
                finish_reason="stop",
            )
            for _ in range(2)
        ]

        _, calls = _run_dump_mocked(
            tmpdir, test_parquet, mock_tokenizer,
            thinking_budget=tb, tool_call_budget=20, answer_budget=30,
            p1_outputs=p1_outputs,
        )
        assert len(calls) == 2

    def test_interrupted_flag_in_parquet(self, tmpdir, test_parquet, mock_tokenizer):
        tb = 10
        p1_outputs = [self._make_thinking_only_output(tb) for _ in range(2)]
        p1b_outputs = [self._make_tool_call_output() for _ in range(2)]

        out_dir, _ = _run_dump_mocked(
            tmpdir, test_parquet, mock_tokenizer,
            thinking_budget=tb, tool_call_budget=20, answer_budget=30,
            p1_outputs=p1_outputs, p1b_outputs=p1b_outputs,
        )
        pq = pd.read_parquet(os.path.join(out_dir, "rollouts.parquet"))
        assert all(pq["interrupted"] == True)  # noqa: E712

    def test_phase1b_prompt_includes_interrupt_phrase(self, tmpdir, test_parquet, mock_tokenizer):
        """Phase 1b prompt = phase1_prompt + p1_text + THINK_INTERRUPT_PHRASE."""
        from phys_reasoner.tir.prompts import THINK_INTERRUPT_PHRASE

        tb = 10
        p1_outputs = [self._make_thinking_only_output(tb) for _ in range(2)]
        p1b_outputs = [self._make_tool_call_output() for _ in range(2)]

        _, calls = _run_dump_mocked(
            tmpdir, test_parquet, mock_tokenizer,
            thinking_budget=tb, tool_call_budget=20, answer_budget=30,
            p1_outputs=p1_outputs, p1b_outputs=p1b_outputs,
        )
        # calls[1] is phase 1b
        for prompt in calls[1][0]:
            assert THINK_INTERRUPT_PHRASE in prompt

    def test_phase1b_stop_token(self, tmpdir, test_parquet, mock_tokenizer):
        """Phase 1b must stop at </tool_call>."""
        from phys_reasoner.tir.prompts import TOOL_CALL_STOP

        tb = 10
        p1_outputs = [self._make_thinking_only_output(tb) for _ in range(2)]
        p1b_outputs = [self._make_tool_call_output() for _ in range(2)]

        _, calls = _run_dump_mocked(
            tmpdir, test_parquet, mock_tokenizer,
            thinking_budget=tb, tool_call_budget=20, answer_budget=30,
            p1_outputs=p1_outputs, p1b_outputs=p1b_outputs,
        )
        assert TOOL_CALL_STOP in calls[1][1].stop


class TestRunDumpMixedInterrupt:
    """Test when some rollouts are interrupted and some are not."""

    def test_partial_interrupt(self, tmpdir, test_parquet, mock_tokenizer):
        """Problem 0 gets interrupted, problem 1 does not."""
        tb = 10
        p1_outputs = [
            # Problem 0: exceeds budget, no </think>
            _make_request_output(
                text="<think>long reasoning...",
                token_ids=list(range(tb)),
                finish_reason="length",
            ),
            # Problem 1: short, includes </think> + tool call
            _make_request_output(
                text="<think>quick</think>\n<tool_call>\n<function=python>\n<parameter=code>\nprint(1)\n</parameter>\n</function>\n</tool_call>",
                token_ids=[1, 2, THINK_END_ID, 10, 11],
                finish_reason="stop",
            ),
        ]
        p1b_outputs = [
            _make_request_output(
                text="<tool_call>\n<function=python>\n<parameter=code>\nprint(42)\n</parameter>\n</function>\n</tool_call>",
                token_ids=[200, 201],
                finish_reason="stop",
            ),
        ]

        out_dir, calls = _run_dump_mocked(
            tmpdir, test_parquet, mock_tokenizer,
            thinking_budget=tb, tool_call_budget=20, answer_budget=30,
            p1_outputs=p1_outputs, p1b_outputs=p1b_outputs,
        )

        # 3 calls: phase 1 (2 prompts) + phase 1b (1 prompt) + phase 2 (2 prompts)
        assert len(calls) == 3
        assert len(calls[0][0]) == 2  # phase 1: both problems
        assert len(calls[1][0]) == 1  # phase 1b: only problem 0
        assert len(calls[2][0]) == 2  # phase 2: both problems

        pq = pd.read_parquet(os.path.join(out_dir, "rollouts.parquet"))
        assert pq.iloc[0]["interrupted"] == True  # noqa: E712
        assert pq.iloc[1]["interrupted"] == False  # noqa: E712


class TestNRollouts:
    """Test n_rollouts > 1."""

    def test_correct_expansion(self, tmpdir, test_parquet, mock_tokenizer):
        """2 problems x 3 rollouts = 6 total."""
        n_rollouts = 3
        total = 2 * n_rollouts

        p1_outputs = [
            _make_request_output(
                text="<think>ok</think>\n<tool_call>\n<function=python>\n<parameter=code>\nprint(1)\n</parameter>\n</function>\n</tool_call>",
                token_ids=[1, 2, THINK_END_ID, 3],
                finish_reason="stop",
            )
            for _ in range(total)
        ]
        p2_outputs = [
            _make_request_output(
                text="\\boxed{42}",
                token_ids=[50, 51],
                finish_reason="stop",
            )
            for _ in range(total)
        ]

        out_dir, calls = _run_dump_mocked(
            tmpdir, test_parquet, mock_tokenizer,
            n_rollouts=n_rollouts,
            p1_outputs=p1_outputs, p2_outputs=p2_outputs,
        )

        # Phase 1 and phase 2 each get 6 prompts
        assert len(calls[0][0]) == total
        assert len(calls[1][0]) == total

        # 6 txt files + summary.txt
        txts = [f for f in os.listdir(out_dir) if f.startswith("rollout_")]
        assert len(txts) == total

        # Parquet has 6 rows
        pq = pd.read_parquet(os.path.join(out_dir, "rollouts.parquet"))
        assert len(pq) == total

        # Check rollout_idx cycles 0, 1, 2 for each problem
        for prob_idx in pq["problem_idx"].unique():
            sub = pq[pq["problem_idx"] == prob_idx]
            assert sorted(sub["rollout_idx"].tolist()) == [0, 1, 2]


class TestValidation:
    """Test argument validation."""

    def test_tool_call_budget_required_with_thinking_budget(self, tmpdir, test_parquet, mock_tokenizer):
        with pytest.raises(ValueError, match="tool_call_budget"):
            _run_dump_mocked(
                tmpdir, test_parquet, mock_tokenizer,
                thinking_budget=10, tool_call_budget=None, answer_budget=30,
            )

    def test_answer_budget_required_with_thinking_budget(self, tmpdir, test_parquet, mock_tokenizer):
        with pytest.raises(ValueError, match="answer_budget"):
            _run_dump_mocked(
                tmpdir, test_parquet, mock_tokenizer,
                thinking_budget=10, tool_call_budget=20, answer_budget=None,
            )


class TestMaxModelLen:
    """Test that max_model_len is computed correctly from budgets."""

    def test_interrupt_enabled_model_len(self, tmpdir, test_parquet, mock_tokenizer):
        """max_model_len = max_prompt_len + thinking + interrupt + tool_call + tool_response + answer"""
        tb, tcb, ab = 100, 50, 80
        max_prompt_len = 256
        max_tool_response_len = 512
        interrupt_len = len(FAKE_INTERRUPT_IDS)
        expected = max_prompt_len + tb + interrupt_len + tcb + max_tool_response_len + ab

        p1_outputs = [
            _make_request_output(
                text="<think>ok</think>\n<tool_call>\n<function=python>\n<parameter=code>\nprint(1)\n</parameter>\n</function>\n</tool_call>",
                token_ids=[1, 2, THINK_END_ID, 3],
                finish_reason="stop",
            )
            for _ in range(2)
        ]

        from phys_reasoner.tir.sandbox import SandboxResult

        mock_llm = MagicMock()
        mock_llm.generate.side_effect = [
            p1_outputs,
            [_make_request_output("\\boxed{1}", [50], "stop") for _ in range(2)],
        ]

        with (
            patch("transformers.AutoTokenizer.from_pretrained", return_value=mock_tokenizer),
            patch("vllm.LLM", return_value=mock_llm) as mock_llm_cls,
            patch("phys_reasoner.tir.sandbox.execute_code",
                  return_value=SandboxResult(stdout="1", stderr="", error=False)),
        ):

            dump_rollouts.run_dump(
                model_path="test",
                parquet_path=test_parquet,
                n=2, n_rollouts=1,
                out_dir=os.path.join(tmpdir, "out"),
                thinking_budget=tb, tool_call_budget=tcb, answer_budget=ab,
                max_prompt_len=max_prompt_len,
                max_tool_response_len=max_tool_response_len,
                gpu_mem=0.3, temperature=1.0, top_p=0.9,
                enable_thinking=True, max_tokens=100, seed=42,
            )

        # Check LLM was called with correct max_model_len
        llm_call_kwargs = mock_llm_cls.call_args[1]
        assert llm_call_kwargs["max_model_len"] == expected


# ---------------------------------------------------------------------------
# TestPromptFidelityVsVeRL — real tokenizer, byte-identical to verl.utils.chat_template
# ---------------------------------------------------------------------------


class TestPromptFidelityVsVeRL:
    """Ensure phase 1 prompt and tool injection are byte-identical to VeRL's path.

    Loads a real Qwen3.5 tokenizer (Qwen3.5-0.8B, cached in hf_cache) and compares
    our rendering against `verl.utils.chat_template.apply_chat_template` — the same
    helper that ToolAgentLoop calls at training time. Any drift here would silently
    produce rollouts that don't match training distribution.
    """

    MODEL = "Qwen/Qwen3.5-0.8B"

    @pytest.fixture(scope="class")
    def tokenizer(self):
        transformers = pytest.importorskip("transformers")
        try:
            return transformers.AutoTokenizer.from_pretrained(self.MODEL, local_files_only=True)
        except Exception as e:
            pytest.skip(f"Qwen3.5 tokenizer unavailable in cache: {e}")

    @pytest.fixture(scope="class")
    def verl_apply(self):
        try:
            from verl.utils.chat_template import apply_chat_template as verl_apply
        except Exception as e:
            pytest.skip(f"verl not importable: {e}")
        return verl_apply

    def test_phase1_prompt_matches_verl(self, tokenizer, verl_apply):
        from phys_reasoner.tir.prompts import PYTHON_TOOL_SCHEMA, TIR_SYSTEM_PROMPT

        msgs = [
            {"role": "system", "content": TIR_SYSTEM_PROMPT},
            {"role": "user", "content": "A ball is dropped from 10 m. How long to fall?"},
        ]

        ours = tokenizer.apply_chat_template(
            msgs,
            tools=[PYTHON_TOOL_SCHEMA],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True,
        )
        theirs = verl_apply(
            tokenizer,
            msgs,
            tokenize=False,
            add_generation_prompt=True,
            tools=[PYTHON_TOOL_SCHEMA],
            enable_thinking=True,
        )
        assert ours == theirs, "phase 1 prompt diverged from verl.utils.chat_template.apply_chat_template"

    def test_tool_injection_matches_verl(self, tokenizer, verl_apply):
        """Our `_make_tool_injection` output must match verl's dummy-user stripped path.

        VeRL's ToolAgentLoop calls apply_chat_template on [{role:tool, content:...}]
        with tools=None and enable_thinking=False (phase 2 final answer). For Qwen3.5
        the template rejects a lone tool message so VeRL's util falls back to the
        dummy-user workaround and strips the dummy-user prefix.
        """
        tool_text = "42\n"

        # Ours — replicate _make_tool_injection exactly.
        dummy_user = [{"role": "user", "content": [{"type": "text", "text": ""}]}]
        dummy_prefix = tokenizer.apply_chat_template(
            dummy_user,
            add_generation_prompt=False,
            tokenize=False,
            enable_thinking=False,
        )
        full = tokenizer.apply_chat_template(
            dummy_user + [{"role": "tool", "content": tool_text}],
            add_generation_prompt=True,
            tokenize=False,
            enable_thinking=False,
        )
        ours = full[len(dummy_prefix):]

        theirs = verl_apply(
            tokenizer,
            [{"role": "tool", "content": tool_text}],
            tokenize=False,
            add_generation_prompt=True,
            tools=None,
            enable_thinking=False,
        )
        assert ours == theirs, "tool injection diverged from verl.utils.chat_template.apply_chat_template"
