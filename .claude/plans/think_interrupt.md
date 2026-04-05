# Plan: Think-Interrupt Patch for VeRL

## Context

During GRPO training with `enable_thinking=True`, Qwen3.5-4B can generate very long `<think>…</think>` blocks on hard physics problems, exhausting the token budget before producing a tool call. This wastes compute and produces zero-reward rollouts with no learning signal.

ScaleRL (Meta) addresses this by forcibly injecting a "time is up" interrupt phrase when thinking hits a budget, teaching the model to generalize to different context lengths. The interrupt is injected with `response_mask=0` (not trained on), so the model sees it as context only.

**Budgets:** thinking = 12,288 tokens; tool-call fallback = 2,048 tokens; final answer = 4,096 tokens.

---

## Files to Change

| File | Change |
|---|---|
| `verl/verl/experimental/agent_loop/tool_agent_loop.py` | Core patch — `__init__` + `_handle_generating_state` |
| `scripts/smoke_tir_qwen35.sh` | Add `THINKING_BUDGET` var + raise `MAX_RESPONSE_LEN` default + pass hydra override |
| `scripts/train.sh` | Same: add `THINKING_BUDGET` var + raise `MAX_RESPONSE_LEN` default + pass hydra override |

After verifying: commit to `physcode` branch of the submodule, then bump the submodule pointer in the main repo.

---

## Change 1 — `tool_agent_loop.py`: `__init__` (line 97)

Add one line at the end of `__init__`, after the existing `self.response_length` assignment:

```python
# Think-interrupt budget (tokens). None = feature disabled (default).
# If the model is still inside <think> when this many tokens are generated,
# an interrupt phrase is injected (mask=0) and generation continues for the tool call.
self.thinking_budget = getattr(self.rollout_config.multi_turn, "thinking_budget", None)
```

---

## Change 2 — `tool_agent_loop.py`: `_handle_generating_state` (line 214)

Insert the block below **after** the existing generate call and its metric/extra_fields updates,
and **before** `agent_data.assistant_turns += 1` (currently line 243).

The insertion point in the existing code looks like:
```python
        if output.routed_experts is not None:
            agent_data.routed_experts = output.routed_experts

        # Check termination conditions          ← INSERT BLOCK BEFORE THIS LINE
        if not ignore_termination and ...
```

**Block to insert:**

```python
        # --- Think-interrupt (ScaleRL-style) ---
        # Detect: model is still inside <think> at the thinking budget.
        # Detection is token-ID based — no decode needed.
        # stop_reason is useless here: VeRL maps both vLLM "stop" and "length"
        # to "completed" (vllm_async_server.py lines 532-536).
        _INTERRUPT_PHRASE = (
            "\nOkay, time is up. Let me stop thinking and formulate a final answer\n</think>\n"
        )
        if self.thinking_budget is not None:
            _think_end_id = self.tokenizer.convert_tokens_to_ids("</think>")
            _thinking_truncated = (
                len(output.token_ids) >= self.thinking_budget
                and _think_end_id not in output.token_ids
            )
            if _thinking_truncated:
                # Inject interrupt with mask=0 (visible as context, not trained on).
                _interrupt_ids = await self.loop.run_in_executor(
                    None,
                    lambda: self.tokenizer.encode(_INTERRUPT_PHRASE, add_special_tokens=False),
                )
                agent_data.prompt_ids += _interrupt_ids
                agent_data.response_mask += [0] * len(_interrupt_ids)
                if agent_data.response_logprobs:
                    agent_data.response_logprobs += [0.0] * len(_interrupt_ids)

                # Sub-call 2: generate the tool call (small budget).
                # Same request_id → same vLLM server → KV cache reuse.
                # sampling_params dict is not mutated by generate() (Ray serializes it).
                _tool_call_params = {**sampling_params, "max_tokens": 2048}
                _output2: TokenOutput = await self.server_manager.generate(
                    request_id=agent_data.request_id,
                    prompt_ids=agent_data.prompt_ids,
                    sampling_params=_tool_call_params,
                    image_data=agent_data.image_data,
                    video_data=agent_data.video_data,
                )
                # Merge sub-call 2 metrics.
                agent_data.metrics["num_preempted"] = (
                    agent_data.metrics.get("num_preempted", 0) + (_output2.num_preempted or 0)
                )
                if _output2.extra_fields.get("max_global_steps"):
                    agent_data.extra_fields["max_global_steps"] = (
                        _output2.extra_fields["max_global_steps"]
                    )
                # Overwrite response_ids with sub-call 2 output so the tool parser
                # (line 263) sees the tool call, not the truncated thinking.
                agent_data.response_ids = _output2.token_ids
                agent_data.prompt_ids += _output2.token_ids
                agent_data.response_mask += [1] * len(_output2.token_ids)
                if _output2.log_probs:
                    agent_data.response_logprobs += _output2.log_probs
        # --- End think-interrupt ---
```

**Why this placement is correct:**
- Both sub-calls count as ONE assistant turn — `assistant_turns += 1` (line 243) fires once after this block.
- `agent_data.response_ids` is overwritten to sub-call 2's tokens so the tool parser on line 263 finds the `<tool_call>` block.
- Termination checks (lines 254-259) run after the increment, as normal.

**Key design facts (do not re-derive):**
- `</think>` is a special token in Qwen3.5 → single integer ID, exact check.
- `stop_reason` cannot distinguish truncation from normal stop (both → `"completed"`).
- `server_manager.generate()` uses `uuid4()` internally per call — same `request_id` is safe.
- `sampling_params` dict is not mutated by `generate()` — Ray serializes before remote call.

---

## Change 3 — `smoke_tir_qwen35.sh` and `train.sh`

Both scripts need the same three changes:

**a) Add `THINKING_BUDGET` env var** (in the configurable params block near `MAX_RESPONSE_LEN`):
```bash
THINKING_BUDGET="${THINKING_BUDGET:-12288}"
```

**b) Raise `MAX_RESPONSE_LEN` default** (currently 4096 in both scripts, too small):
```bash
MAX_RESPONSE_LEN="${MAX_RESPONSE_LEN:-16384}"
```
Update the nearby comment to:
```bash
# TIR response budget (with think-interrupt enabled):
#   thinking(≤12288) + tool_call(2048) + tool_response(512) + answer(4096) + buffer = 16384
```

**c) Pass hydra override** in the `python3 -m verl.trainer.main_ppo` call,
alongside the other `multi_turn.*` args. Use bash conditional expansion so that
`THINKING_BUDGET=""` cleanly disables the feature (no override passed, Python `getattr`
returns `None`):
```bash
    ${THINKING_BUDGET:+actor_rollout_ref.rollout.multi_turn.thinking_budget=$THINKING_BUDGET} \
```

To disable for a run: `THINKING_BUDGET="" bash scripts/train.sh`

---

## Smoke Test Procedure

**Goal:** confirm the interrupt fires, the tool call executes, and a `\boxed{}` answer appears.

```bash
# Step 1: apply dump patch so rollouts are written to disk
cd verl && git apply ../patches/verl_dump_dir.patch && cd ..

# Step 2: run smoke with very low budget (500) to force interrupt on every trajectory
MAX_RESPONSE_LEN=16384 SMOKE_N=2 \
  VERL_DUMP_DIR=outputs/think_interrupt_test \
  THINKING_BUDGET=500 \
  bash scripts/smoke_tir_qwen35.sh

# Step 3: inspect rollouts
grep -c "</think>" outputs/think_interrupt_test/rollout_*.txt    # expect 1 per file
grep "time is up" outputs/think_interrupt_test/rollout_*.txt     # expect match in each
grep "tool_response" outputs/think_interrupt_test/rollout_*.txt  # expect tool executed
grep "boxed" outputs/think_interrupt_test/rollout_*.txt          # expect final answer
```

**Pass criteria:**
1. `</think>` appears exactly once per rollout (interrupt closed it, model didn't reopen)
2. Interrupt phrase present
3. `<tool_response>` present (tool executed after interrupt)
4. `\boxed{}` present in final turn

**Then** re-run with `THINKING_BUDGET=12288` (default) for a normal 2-step smoke to confirm
training still completes without false-positive interrupts on typical problems.

---

## Commit Workflow

```bash
# 1. Revert dump patch before committing (it's a temporary debug tool)
cd verl
git checkout verl/experimental/agent_loop/tool_agent_loop.py verl/trainer/constants_ppo.py

# 2. Commit the think-interrupt patch to physcode branch
git add verl/experimental/agent_loop/tool_agent_loop.py
git commit -m "add think-interrupt: cap thinking at budget, inject ScaleRL-style phrase"
git push fork physcode
cd ..

# 3. Bump submodule pointer in main repo
git add verl
git commit -m "bump verl submodule: think-interrupt patch"
```
