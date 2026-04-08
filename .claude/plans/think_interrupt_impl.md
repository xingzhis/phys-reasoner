# Think-Interrupt Implementation Notes

**Status:** Implemented, unit-tested (15/15 pass), smoke-tested on H200 (job 1470436 — exit 0).

---

## What was implemented

### Problem
Qwen3.5-4B with `enable_thinking=True` can generate arbitrarily long `<think>` traces, exhausting the response budget before emitting a tool call. The think-interrupt caps runaway thinking at a fixed token budget and redirects the model to call the tool.

### Architecture (3 sub-calls per interrupted trajectory)

```
sub-call 1: generate up to thinking_budget tokens
  → if </think> found: natural end, tool call already in output → normal flow
  → if NOT found (runaway): inject interrupt phrase (mask=0) → sub-call 2

sub-call 2: generate up to tool_call_budget tokens (tool call)
  → tool parser extracts <tool_call> → PROCESSING_TOOLS
  → if no tool call found → TERMINATED (case D, reward=0)

[VeRL injects tool response as mask=0 user turn]

sub-call 3: VeRL's normal second assistant turn → final answer
```

### response_mask layout (interrupt case)
```
[1]*thinking_tokens | [0]*interrupt_tokens | [1]*tool_call_tokens | [0]*tool_response_tokens | [1]*answer_tokens
 ≤thinking_budget      len=15 (exact)         ≤tool_call_budget      ≤max_tool_response_length   remainder
```
All mask=0 tokens (interrupt phrase + tool response) are part of `response_mask` and count against `response_length`. This is how VeRL already handles tool responses in multi-turn — same convention.

### Budget identity (the one formula to know)
```
response_length = thinking_budget + 15 + tool_call_budget + max_tool_response_length + answer_budget
```
Where:
- `15` = exact token count of `THINK_INTERRUPT_PHRASE` for Qwen3.5-4B tokenizer (hardcoded in shell, verified)
- `max_tool_response_length` = 512 (existing VeRL config param)
- `answer_budget` = implicit remainder (not a Python param)

Shell derives `MAX_RESPONSE_LEN` from these five components — never set it directly when interrupt is enabled.

---

## Files changed

### `verl/verl/experimental/agent_loop/tool_agent_loop.py`

**Module level:**
- `THINK_INTERRUPT_PHRASE` constant with recompute command in comment

**`__init__`:**
- Reads `self.thinking_budget` and `self.tool_call_budget` from `rollout_config.multi_turn`
- When `thinking_budget is not None`: precomputes `self._interrupt_ids` and `self._think_end_id` from the tokenizer (once, not per trajectory)
- Assert at startup: `thinking_budget + tool_call_budget + len(_interrupt_ids) + max_tool_response_length < response_length`
  - Fails loud at init if budgets don't fit — no silent failures at training time

**`_handle_generating_state`:**
- Sub-call 1: passes `{**sampling_params, "max_tokens": thinking_budget}` when enabled
- Interrupt condition: `len(output.token_ids) >= thinking_budget and self._think_end_id not in output.token_ids`
- If fires: appends `self._interrupt_ids` to `prompt_ids`, `[0]*19` to `response_mask`, `[0.0]*19` to `response_logprobs`
- Sub-call 2: `{**sampling_params, "max_tokens": tool_call_budget}`
- Overwrites `response_ids` with sub-call 2 output (so tool parser sees the tool call, not truncated thinking)
- Appends sub-call 2 tokens to `prompt_ids`, `response_mask`, `response_logprobs`

### `verl/verl/workers/config/rollout.py`

Added two fields to `MultiTurnConfig`:
```python
thinking_budget: Optional[int] = None
tool_call_budget: Optional[int] = None
```

### `scripts/smoke_tir_qwen35.sh` and `scripts/train.sh`

New budget variables (single place to set everything):
```bash
THINKING_BUDGET=...
TOOL_CALL_BUDGET=...
INTERRUPT_LEN=19        # exact for Qwen3.5-4B; recompute cmd in comment
MAX_TOOL_RESPONSE_LEN=512
ANSWER_BUDGET=...
# Derived — never set MAX_RESPONSE_LEN directly when interrupt is enabled:
MAX_RESPONSE_LEN=$((THINKING_BUDGET + INTERRUPT_LEN + TOOL_CALL_BUDGET + MAX_TOOL_RESPONSE_LEN + ANSWER_BUDGET))
```

Both `THINKING_BUDGET` and `TOOL_CALL_BUDGET` default to empty. If unset, `MAX_RESPONSE_LEN` falls back to `${MAX_RESPONSE_LEN:-4096}` (interrupt disabled).

Hydra overrides added:
```bash
${THINKING_BUDGET:++actor_rollout_ref.rollout.multi_turn.thinking_budget=$THINKING_BUDGET}
${TOOL_CALL_BUDGET:++actor_rollout_ref.rollout.multi_turn.tool_call_budget=$TOOL_CALL_BUDGET}
```
(Double `+` = bash `${var:+value}` expansion where `value` starts with Hydra's `+` for new keys.)

`max_tool_response_length` now uses `$MAX_TOOL_RESPONSE_LEN` instead of hardcoded 512.

### `scripts/smoke_tir_interrupt.sbatch`

Smoke test with budgets that force interrupt on every trajectory:
```bash
THINKING_BUDGET=200   # << typical thinking length → always truncated
TOOL_CALL_BUDGET=512
ANSWER_BUDGET=313
# MAX_RESPONSE_LEN = 200+19+512+512+313 = 1556
```
Uses `gpu:h200:1`.

---

## Tests (`tests/test_think_interrupt.py`)

15 tests, all pass. Uses real Qwen3.5-4B tokenizer (loaded via `AutoTokenizer.from_pretrained("Qwen/Qwen3.5-4B")`). Token IDs are real values, not mocked constants.

### `TestThinkInterruptDisabled` (2 tests)
- `test_no_interrupt_when_budget_none`: feature off → single generate call
- `test_sub_call1_uses_original_params_when_disabled`: sampling_params object identity preserved (no copy)

### `TestThinkInterruptEnabled` (11 tests)
- `test_sub_call1_capped_at_thinking_budget`: sub-call 1 `max_tokens == THINKING_BUDGET`
- `test_interrupt_fires_when_truncated`: 2 generate calls when budget hit
- `test_sub_call2_capped_at_tool_call_budget`: sub-call 2 `max_tokens == TOOL_CALL_BUDGET`
- `test_interrupt_ids_appended_with_zero_mask`: mask = `[1]*thinking + [0]*19 + [1]*tool_call`
- `test_response_ids_overwritten_with_sub_call2`: `response_ids` = sub-call 2 tokens (tool parser sees tool call)
- `test_prompt_ids_contain_full_sequence`: full sequence = prompt + thinking + interrupt + tool_call
- `test_no_interrupt_when_think_end_present`: `THINK_END_ID=248069` in output → no interrupt
- `test_no_interrupt_when_output_shorter_than_budget`: short output → no interrupt
- `test_logprobs_padded_with_zeros_for_interrupt`: interrupt positions get `0.0` log_probs
- `test_interrupt_ids_precomputed_not_per_call`: `tokenizer.encode` not called inside the hot path
- `test_case_d_no_tool_call_after_interrupt` **(case D)**: interrupt fires but sub-call 2 has no tool call → `AgentState.TERMINATED`, mask still valid

### `TestAggLossNonContiguousMask` (2 tests)
Verifies VeRL's `agg_loss` handles mask=0 in the middle of a sequence correctly (not just trailing zeros):
- `test_zero_mask_positions_dont_contribute`: 999.0 loss at mask=0 positions → same result as 0.0 there
- `test_token_mean_matches_manual_calculation`: `(2+4+6)/3` with mask `[1,1,0,0,1]`

---

## Key design decisions

| Decision | Rationale |
|---|---|
| Sub-call 1 capped at `thinking_budget` | Stops vLLM from wasting compute on runaway thinking past the budget |
| `_interrupt_ids` precomputed at `__init__` | Not on the per-trajectory hot path |
| Assert at `__init__`, not at call time | Misconfiguration caught at startup, not mid-training |
| `answer_budget` shell-only, not a Python param | Python doesn't need it; VeRL truncates at `response_length` naturally |
| `INTERRUPT_LEN=19` hardcoded in shell | Exact for Qwen3.5-4B; recompute command in comment for other models |
| Tool response counts toward `response_length` | VeRL convention: `response_mask += [0]*tool_response_len`; termination check uses `len(response_mask)` |
| Case D (no tool after interrupt) → TERMINATED | Correct training signal: reward=0 penalizes failure to produce tool call |
| Random injection (ScaleRL 10k–12k) | Not implemented — deferred to v2 |

---

## Known limitations / follow-ups

1. **`VERL_DUMP_DIR` does not produce rollout dumps** — VeRL's dumping requires `trainer.rollout_data_dir` config, not the env var. The env var path seems unused or needs a separate patch. To inspect rollouts, use `dump_rollouts.py` or add `trainer.rollout_data_dir` to the Hydra config.

2. **Edge case: thinking < budget but thinking+tool_call > budget** — sub-call 1 is capped at `thinking_budget`, so if thinking ends naturally at e.g. 11000 tokens and the tool call needs 500 more but budget=12288-500+500... actually: if `</think>` appears before `thinking_budget`, sub-call 1 exits early (vLLM stops at EOS/tool_end), and the full thinking+tool_call is in the output. Only at risk if thinking ends near the budget AND the partial tool_call token count pushes past `thinking_budget`. Acceptable edge case given generous budgets.

3. **Sub-call 3 (answer) max_tokens** — uses full `response_length` from sampling_params. VeRL truncates output at `response_length - len(current_response_mask)`. Answer gets `answer_budget` tokens by construction. Not explicitly capped per-call — minor compute waste possible.

4. **old_log_probs = 0.0 at interrupt positions** — produces `ratio = exp(new_lp)` at those positions, but they are masked out by `response_mask=0` in `agg_loss`. No NaN; no gradient contribution. Verified via `TestAggLossNonContiguousMask`.

5. **Smoke test rewards = 0.0** — expected with `THINKING_BUDGET=200`; model is interrupted before it can reason. Production training uses `THINKING_BUDGET=12288`.
