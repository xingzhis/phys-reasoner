# CoT Think-Interrupt Plan

**Status:** drafted 2026-04-17. Not yet implemented.

**Goal.** Make `COT_BASELINE=1` runs honor `THINKING_BUDGET` the same way TIR runs do. Today the budget is silently ignored because `SingleTurnAgentLoop` does a single `generate()` call. The paper-grade TIR↔CoT ablation needs identical total response length *and* identical interrupt behaviour — otherwise any TIR win could be attributed to the interrupt rather than the tool.

## Confirmed discrepancy

- `scripts/train_async.sh` passes `multi_turn.thinking_budget` as a Hydra override regardless of mode.
- When `COT_BASELINE=1`, it flips to `default_agent_loop=single_turn_agent` + `multi_turn.enable=false`.
- `SingleTurnAgentLoop` (`verl/verl/experimental/agent_loop/single_turn_agent_loop.py:36-84`) does ONE `server_manager.generate()` with raw `sampling_params` and never reads the thinking/tool-call budgets.
- Result: in CoT mode the model generates one blob up to `max_response_length`, and the interrupt-budget knobs are dead weight.

## Design choices (agreed with user)

| Choice | Decision | Rationale |
|---|---|---|
| Where to put the code | **(A)** Modify `SingleTurnAgentLoop` in-place | Mirrors the TIR precedent (the TIR change was done in-place inside `ToolAgentLoop`, not via a new class). Simplest, least bug-prone. Upstream VeRL parity is explicitly not a concern right now. |
| Post-interrupt budget | **(A)** Keep total response length identical to TIR: sub-call-2 caps at `tool_call_budget + max_tool_response_length + answer_budget` (computed as `response_length − thinking_budget − len(interrupt_ids)` so `answer_budget` stays implicit) | Fair ablation — total response tokens are the same between TIR and CoT runs. |
| Interrupt phrase | Reuse existing `"\nOkay, I've thought enough. Time to write my response.\n</think>\n"` | Existing comment in `tool_agent_loop.py:45-48` already notes it was chosen neutrally between `<tool_call>` and direct `\boxed{}`. No tokenizer re-measurement needed. |
| Response-ids composition | **Concatenate** sub-call-2 onto sub-call-1 (NOT overwrite). TIR overwrites so the tool parser sees the tool call; CoT has no parser and the final answer must be in `response_ids`. | Correct mask layout and no loss of the thinking-phase tokens (they are still supervised with mask=1). |
| Scope | Both `scripts/train.sh` and `scripts/train_async.sh`. `scripts/dump_rollouts.py` intentionally untouched — separate session. | User decision. |

## Mask/ids layout (CoT interrupt case)

```
response_mask : [1]*thinking_tokens | [0]*interrupt_tokens | [1]*answer_tokens
                ≤ thinking_budget     len(_interrupt_ids)    ≤ post_interrupt_budget
response_ids  : thinking_tokens   + interrupt_ids         + answer_tokens
```

Natural-exit case (model emits `</think>` before the budget): single generate call, single `[1]*response` mask, no interrupt.

Budget identity (enforced at init):
```
response_length ≥ thinking_budget + len(_interrupt_ids) + 1   (answer must have room for ≥1 token)
```
(Note the TIR identity also includes `tool_call_budget + max_tool_response_length`; CoT doesn't require those because sub-call-2's cap is derived from the remainder.)

## Implementation sketch (`SingleTurnAgentLoop`)

```python
def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.prompt_length = self.rollout_config.prompt_length
    self.response_length = self.rollout_config.response_length

    # Reuse multi_turn.thinking_budget / tool_call_budget (same config surface as ToolAgentLoop).
    # In CoT mode multi_turn.enable=false, but these fields still hold their values.
    self.thinking_budget = self.rollout_config.multi_turn.thinking_budget
    self.tool_call_budget = self.rollout_config.multi_turn.tool_call_budget  # read but not used directly; kept for parity
    if self.thinking_budget is not None:
        self._interrupt_ids = self.tokenizer.encode(THINK_INTERRUPT_PHRASE, add_special_tokens=False)
        self._think_end_id  = self.tokenizer.convert_tokens_to_ids("</think>")
        self._post_interrupt_budget = self.response_length - self.thinking_budget - len(self._interrupt_ids)
        assert self._post_interrupt_budget > 0, (
            f"No room for answer after interrupt: response_length={self.response_length} - "
            f"thinking_budget={self.thinking_budget} - interrupt={len(self._interrupt_ids)} "
            f"= {self._post_interrupt_budget}"
        )
```

`run()` becomes:
1. Tokenize prompt (same as today).
2. Sub-call 1: `generate(max_tokens=thinking_budget)` if feature on, else raw sampling_params.
3. If feature on AND `len(out1.token_ids) >= thinking_budget` AND `_think_end_id not in out1.token_ids`:
   - Extend `response_ids` with `_interrupt_ids`, `response_mask` with `[0]*len(_interrupt_ids)`, `response_logprobs` with `[0.0]*len(_interrupt_ids)` (only if logprobs were returned).
   - Sub-call 2: `generate(max_tokens=_post_interrupt_budget)`. Use `prompt_ids + sub_call_1_tokens + interrupt_ids` as the prompt.
   - Concatenate sub-call-2 tokens / mask=1 / logprobs.
4. Truncate final `response_ids` / `response_mask` / `response_logprobs` to `response_length` (existing code).

Import `THINK_INTERRUPT_PHRASE` from `tool_agent_loop` (single source of truth).

The `request_id` is currently generated with `uuid4().hex`. For the two-call flow we must pass the same request_id to both calls so vLLM's engine sees them as one session (per the TIR code at `tool_agent_loop.py:263` and `:310`). Mirror that.

## Shell changes

### `scripts/train_async.sh`
No change needed. The Hydra overrides for `thinking_budget` / `tool_call_budget` are already emitted in both branches. `single_turn_agent_loop.py` will start reading them once the Python change lands.

### `scripts/train.sh`
Currently hardcodes:
```
actor_rollout_ref.rollout.agent.default_agent_loop=tool_agent
actor_rollout_ref.rollout.multi_turn.enable=true
```
Add a `COT_BASELINE` branch identical to `train_async.sh`:
```
AGENT_LOOP=$([[ "$COT_BASELINE" == "1" ]] && echo single_turn_agent || echo tool_agent)
MULTI_TURN_ENABLE=$([[ "$COT_BASELINE" == "1" ]] && echo false || echo true)
```
and interpolate into the overrides. Leave every other field unchanged.

## Tests — `tests/test_cot_think_interrupt.py`

Mirror structure of `tests/test_think_interrupt.py`. CPU-only, uses real Qwen3.5-4B tokenizer, mocks `server_manager.generate`.

| # | Name | What it checks |
|---|---|---|
| 1 | `test_no_interrupt_when_budget_none` | Feature off → exactly 1 generate call, sampling_params passthrough |
| 2 | `test_interrupt_fires_when_thinking_truncated` | `</think>` absent and `len == budget` → exactly 2 generate calls |
| 3 | `test_no_interrupt_when_think_end_present_in_sub_call_1` | `</think>` token in sub-call-1 output → 1 generate call total |
| 4 | `test_sub_call1_max_tokens_eq_thinking_budget` | Kwarg inspection of first call |
| 5 | `test_sub_call2_max_tokens_eq_post_interrupt_budget` | Kwarg inspection of second call = `response_length − thinking_budget − len(interrupt_ids)` |
| 6 | `test_mask_layout_is_1_0_1` | Exact pattern `[1]*t + [0]*len(interrupt_ids) + [1]*a` |
| 7 | `test_response_ids_concatenated_not_overwritten` | `response_ids == thinking_tokens + interrupt_ids + answer_tokens` (distinguishes from TIR behaviour) |
| 8 | `test_prompt_ids_to_sub_call2_include_interrupt` | Sub-call-2 prompt = original prompt + sub-call-1 tokens + interrupt_ids |
| 9 | `test_logprobs_padded_with_zeros_for_interrupt` | Interrupt positions get 0.0 in `response_logprobs` when logprobs are tracked |
| 10 | `test_logprobs_none_when_disabled` | No logprob padding when model didn't return logprobs |
| 11 | `test_asserts_budget_identity_at_init` | Post-interrupt budget ≤0 → `__init__` raises |
| 12 | `test_same_request_id_across_sub_calls` | Both `server_manager.generate` calls receive the same `request_id` |
| 13 | `test_truncation_to_response_length_after_interrupt` | If sub-call-2 returns more than `post_interrupt_budget`, truncation still caps at `response_length` |

## Smoke tests (2× A100-40 on this host)

Use `scripts/train_smoke_cot_async.sh` (new).

### Smoke parquet
`scripts/build_cot_smoke_parquet.py`:
- Read `data/processed_cot/data/train.parquet`
- Sample 7 rows (varied problem lengths).
- Append one synthetic row with `question = "what is 2+2"` using the existing CoT system prompt from the parquet (copy it from row 0). Reference answer = `4`; data_source = `"smoke"` so the reward fn returns a clean 0/1.
- Total 8 rows. Save to `outputs/smoke_cot_async/${RUN_ID}/smoke8.parquet`.

### Launcher
`scripts/train_smoke_cot_async.sh`:
- Mirrors `scripts/train_smoke_async.sh` (small model Qwen3.5-0.8B, `N_GPUS_ROLLOUT=1 N_GPUS_TRAIN=1 TRAIN_BATCH=4 ROLLOUT_N=4 TOTAL_STEPS=2`).
- Sets `COT_BASELINE=1`.
- Sets tiny budgets so most rows hit interrupt: `THINKING_BUDGET=128 TOOL_CALL_BUDGET=64 ANSWER_BUDGET=128` ⇒ `MAX_RESPONSE_LEN = 128+15+64+512+128 = 847`.
  (Note `TOOL_CALL_BUDGET` and `MAX_TOOL_RESPONSE_LEN` are still in the formula — they buy response length for the post-interrupt answer via option-A budgeting. Value of `TOOL_CALL_BUDGET` in CoT mode is semantically moot but numerically matters because `MAX_RESPONSE_LEN` depends on it.)
- Exports `DUMP_TRAIN_ROLLOUTS=1` and `DUMP_VAL_ROLLOUTS=1` so every rollout lands in `$TRAIN_DIR/rollout_dumps/` and `$TRAIN_DIR/validation_dumps/`.

### Pass criteria (manual spot-check after run)

For **each dumped rollout** (`outputs/physcode_tir/grpo_cot_*/rollout_dumps/*.jsonl`):
1. The `2+2` row → **no interrupt**: decoded response starts with `<think>...`, contains `</think>` before `_interrupt_ids` would fire, mask is all `1`s, ends with `\boxed{4}` or similar.
2. Most long-problem rows → **interrupt fires**: decoded response contains the verbatim phrase `Okay, I've thought enough. Time to write my response.` with `</think>` immediately after; the token range occupied by the phrase has `response_mask=0`; tokens before and after that range have `response_mask=1`; final segment contains `\boxed{...}` (or is at least coherent post-interrupt text, not degenerate repetition).
3. No rollout has a mask longer than `response_length` nor shorter than the decoded response.

For the **TIR regression smoke** (`train_smoke_async.sh` rerun with matching small budgets): existing pass criteria still hold (interrupt fires, `<tool_response>` present, `\boxed{}` present, reward ≥ one trajectory).

I will dump rollouts from both smokes and paste 2–3 decoded examples back for user review before marking the task done.

## Known edge cases to keep in mind

- **`</think>` emitted exactly at budget**: if sub-call-1 returns `len == thinking_budget` AND `</think>` is the last token, the guard `self._think_end_id not in output.token_ids` evaluates False → no interrupt. Correct.
- **Empty sub-call-2**: if model returns 0 tokens after interrupt, mask stays valid (`[1]*t + [0]*i + [1]*0`). No crash. Reward will be 0 (no `\boxed{}`) — correct training signal.
- **Very short prompt, budget larger than natural thinking**: sub-call-1 naturally ends with `</think>` + answer in one go. No interrupt. The "what is 2+2" row exercises this path.
- **Logprobs off (vLLM config)**: no padding needed. Test #10 locks this.
- **Preempted / truncated sub-call-2 by vLLM**: `post_interrupt_budget` is explicit so vLLM won't generate past it. If vLLM truncates due to its own max_model_len, the outer `[: self.response_length]` clamp still holds.

## Rollback

The Python change is additive (new `__init__` fields + branch in `run()`) — revert the single-file edit to `single_turn_agent_loop.py`. The shell addition to `train.sh` is a 4-line diff. No data or checkpoint migration.

## Out of scope

- `scripts/dump_rollouts.py` — user deferred to separate session.
- Random-injection policy (ScaleRL 10k–12k) — not implemented for TIR either; tracked in `.claude/plans/think_interrupt_impl.md` as future work.
- Changing the interrupt phrase or adding a CoT-specific variant.
