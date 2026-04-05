# Plan: PhysCode — Tool-Integrated Reasoning for Physics via RLVR

**Status:** Active — Week 1 in progress
**Started:** 2026-03-31
**Target completion:** 2026-05-12
**Proposal:** `docs/physcode_proposal_v5.md`

---

## Project Summary

Train Qwen3.5-4B with RLVR to solve physics problems using Tool-Integrated Reasoning (TIR): the model reasons, executes exactly one Python/SymPy code block, receives the output, and reasons to a final \boxed{} answer. Core claim: execution-based reward eliminates the symbolic verification noise (LaTeX FN rates ~68% on expression types) that degrades CoT-GRPO learning, connecting empirically to the RLVεR theoretical framework.

---

## Key Decisions (Frozen)

| Item | Decision |
|---|---|
| Main model | Qwen3.5-4B instruct (thinking OFF for TIR format) |
| Debug model | Qwen3.5-0.8B |
| TIR format | Single code block: `[think]…[code]…[/code][output]…[think]…[answer]\boxed{}` |
| Primary training data | Dr. SCI clean (~65k numerical + expression + MCQ after Goldilocks filter) |
| Supplemental data | 6.8k curated corpus (`candidates_deduped.parquet`) |
| Reward | Binary: R_correct (λ=0 initially; token cost penalty in late ablation) |
| Curriculum | Numerical first → expression + MCQ once stable |
| SFT | Conditional on Stage 0 probe: skip if verifier hit rate ≥15%, else 2–5k TIR demos |
| Goldilocks strategy | B+C — bucket weights + online [0.05, 0.95] filter in GRPO |
| Secondary benchmark | MATH-500 hard subset |

---

## Week 1 — 2026-03-31 to 2026-04-06

**Goal:** Prove the VeRL injection mechanism works; measure LaTeX FN rates; decide on SFT.

### Gating item: VeRL single-block injection
Implement and smoke-test the mid-sequence injection in VeRL rollout:
1. Model generates until `[/code]` stop token
2. Sandbox executes code (30s timeout; subprocess + resource limits)
3. `[output] {result}\n` injected as fixed continuation
4. Model resumes until `[answer]` stop token
5. Final `\boxed{}` extracted and verified

**This must work before any training. Everything else is blocked on this.**

### Stage 0 probe (100-problem TIR zero-shot)
- Run Qwen3.5-4B instruct (thinking OFF) with single-block TIR prompt on 100 problems
- Measure: execution success rate, verifier hit rate, per-type accuracy (numerical/expression/MCQ)
- Decision: if verifier hit rate ≥15% → skip SFT; if <15% → proceed to SFT

### Execution sandbox
- 30s timeout, subprocess isolation, resource limits
- Test on representative physics SymPy expressions

### LaTeX FN rate measurement
- Run rule verifier on training corpus (Dr. SCI + 6.8k), gold-vs-gold round-trip
- Measure per-type FN rate: numerical, expression, equation, MCQ
- This is the baseline for RQ2

### Week 1 outputs
- Working VeRL injection (smoke-tested end-to-end)
- Working execution sandbox
- Stage 0 probe results (execution success rate, verifier hit rate by answer type)
- LaTeX FN rate table by answer type
- SFT decision (go/no-go)

---

## Week 2 — 2026-04-07 to 2026-04-13

**Goal:** SFT if needed; GRPO eval loop instrumented; training data ready.

### Think-interrupt patch for VeRL (MUST DO before first GRPO run)

Implements ScaleRL-style forced thinking truncation. If the model is still inside
`<think>…` at the thinking budget, inject an interrupt phrase (mask=0, not trained on)
and continue generating the tool call. Trains the model to work within token budgets.

**Budgets:** thinking phase = 12,288 tokens; final answer phase = 4,096 tokens.

#### File to patch: `verl/verl/experimental/agent_loop/tool_agent_loop.py`
This is our submodule (`git@github.com:xingzhis/verl.git`, branch `physcode`).
Commit the change directly to the physcode branch — this is a permanent project feature.

#### Changes needed

**`__init__`** — read one new config field (with safe default = None = feature disabled):
```python
self.thinking_budget = getattr(self.rollout_config.multi_turn, "thinking_budget", None)
```

**`_handle_generating_state`** — add after the existing generate block, immediately before
the `assistant_turns += 1` line (~line 257). Detection uses token ID, no decode needed:

```python
INTERRUPT_PHRASE = "\nOkay, time is up. Let me stop thinking and formulate a final answer\n</think>\n"

think_end_id = self.tokenizer.convert_tokens_to_ids("</think>")
thinking_truncated = (
    self.thinking_budget is not None
    and len(output.token_ids) >= self.thinking_budget
    and think_end_id not in output.token_ids  # still inside <think>
)

if thinking_truncated:
    # Inject interrupt with mask=0 (not trained on, same treatment as tool response)
    interrupt_ids = await self.loop.run_in_executor(
        None, lambda: self.tokenizer.encode(INTERRUPT_PHRASE, add_special_tokens=False)
    )
    agent_data.prompt_ids += interrupt_ids
    agent_data.response_mask += [0] * len(interrupt_ids)
    if agent_data.response_logprobs:
        agent_data.response_logprobs += [0.0] * len(interrupt_ids)

    # Sub-call 2: generate just the tool call (small budget)
    # sampling_params is a dict; not mutated by generate() (Ray serializes it)
    tool_call_params = {**sampling_params, "max_tokens": 2048}
    output2: TokenOutput = await self.server_manager.generate(
        request_id=agent_data.request_id,  # same → same vLLM server → KV cache reuse
        prompt_ids=agent_data.prompt_ids,
        sampling_params=tool_call_params,
        image_data=agent_data.image_data,
        video_data=agent_data.video_data,
    )
    # Accumulate metrics from sub-call 2
    agent_data.metrics["num_preempted"] = (
        agent_data.metrics.get("num_preempted", 0) + (output2.num_preempted or 0)
    )
    if output2.extra_fields.get("max_global_steps"):
        agent_data.extra_fields["max_global_steps"] = output2.extra_fields["max_global_steps"]

    # Overwrite response_ids with sub-call 2 output so tool parser sees the tool call
    agent_data.response_ids = output2.token_ids
    agent_data.prompt_ids += output2.token_ids
    agent_data.response_mask += [1] * len(output2.token_ids)
    if output2.log_probs:
        agent_data.response_logprobs += output2.log_probs
```

The `assistant_turns += 1` increment that follows counts both sub-calls as one logical turn —
the turn limit is not affected.

#### Key design decisions (do not change without re-reading the analysis)
- **`stop_reason` is useless**: VeRL maps both vLLM `"stop"` and `"length"` to `"completed"`
  (vllm_async_server.py lines 532–536). Truncation is detected via token ID check instead.
- **`</think>` is a special token** in Qwen3.5 — maps to exactly one integer ID, so
  `think_end_id not in output.token_ids` is a reliable exact check (O(n) int scan, ~µs).
- **Request ID reuse is safe**: `server_manager.generate()` generates a fresh `uuid4()` per
  actual vLLM call (agent_loop.py line 161); `agent_data.request_id` is only for sticky routing.
- **`sampling_params` dict is not mutated**: the vLLM server pops `max_tokens` from its own
  copy (Ray serializes the dict before remote call). Safe to override for sub-call 2.

#### Training script changes
In `smoke_tir_qwen35.sh` (and eventually `train.sh`), add:
```bash
actor_rollout_ref.rollout.multi_turn.thinking_budget=12288
data.max_response_length=16384   # thinking(12288) + tool_call(2048) + tool_resp(512) + answer(4096) + buffer
```
Current `MAX_RESPONSE_LEN=4096` is too small — must increase before enabling the budget.

#### Smoke test to verify the patch

Run with `SMOKE_N=2` on a deliberately hard problem that triggers long thinking.
Set `MAX_RESPONSE_LEN=16384` and `thinking_budget=12288`. Also set `VERL_DUMP_DIR` so you can
inspect the raw rollout and confirm:
1. The interrupt phrase appears in the trajectory with `</think>` before the `<tool_call>`
2. The tool call was executed (tool response present)
3. A `\boxed{}` answer appears in the final turn
4. The trajectory does NOT have `</think>` appearing twice (model didn't re-open thinking)

Commands:
```bash
# Apply dump patch first
cd verl && git apply ../patches/verl_dump_dir.patch && cd ..

# Run smoke with long response budget and thinking budget enabled
MAX_RESPONSE_LEN=16384 SMOKE_N=2 VERL_DUMP_DIR=outputs/think_interrupt_test \
  bash scripts/smoke_tir_qwen35.sh

# Inspect the dumped rollout
ls outputs/think_interrupt_test/
cat outputs/think_interrupt_test/rollout_*.txt | grep -c "</think>"   # expect 1 per rollout
cat outputs/think_interrupt_test/rollout_*.txt | grep "time is up"    # expect at least 1
```

To trigger the interrupt reliably for testing, temporarily set `thinking_budget=500` (very low)
so it fires on every trajectory. Verify the tool call still executes correctly. Then restore to 12288.

#### After verifying: commit and push to physcode branch
```bash
cd verl
git add verl/experimental/agent_loop/tool_agent_loop.py
git commit -m "add think-interrupt: cap thinking at budget, inject ScaleRL-style phrase"
git push fork physcode
cd ..
git add verl && git commit -m "bump verl submodule: think-interrupt patch"
```

#### Stage 0 probe (deferred — not blocking)
`stage0_probe.py` (vLLM manual 2-phase loop) should eventually match VeRL's behavior.
This means adding a `thinking_budget` param that splits phase 1 into think-pass + tool-call-pass,
injecting the same interrupt phrase on truncation. Defer until after the first GRPO run confirms
the VeRL patch is working — it is a diagnostic tool, not a training blocker.

---

### SFT — SKIPPED (decision 2026-04-04)
Native tool-call format (qwen3_coder) + enable_thinking=True produces tool calls in smoke test.
No time for demo curation. Go straight to GRPO. Revisit only if reward signal is flat in Week 3.

### GRPO eval loop instrumentation
- Instrument per-type accuracy logging at checkpoints 500 / 1000 / 2000 gradient steps
- Dev set: 1k stratified holdout balanced by answer type (fixed before any training)
- Needed for RQ2 correlation analysis

### Training data weights
- Add `_train_weight` column to both parquets (Goldilocks B strategy)
- Implement online [0.05, 0.95] filter in VeRL reward wrapper (Goldilocks C strategy)

### (Optional, parallel) Verifier comparison
- Benchmark `desimfj/SCI-Verifier-4B` and `desimfj/SCI-Verifier-8B` vs xVerify-3B-Ib
- Measure FNR by answer type + inference latency
- Decision: if FNR lower and latency acceptable, swap verifier before full GRPO run

### (Optional, parallel) Eval setup
- Set up critpt + any other hard physics benchmarks alongside MATH-500 hard
- Run zero-shot Qwen3.5-4B to get free baselines before training

### Week 2 outputs
- SFT checkpoint (if needed) with ≥30% execution success rate on dev
- Instrumented GRPO eval loop (per-type accuracy at checkpoints)
- Both parquets with `_train_weight` column
- Fixed dev/test splits

---

## Week 3 — 2026-04-14 to 2026-04-20

**Goal:** RL environment verified end-to-end; first 4B GRPO run launched.

Tasks:
- [ ] 0.8B GRPO smoke test on numerical-only curriculum (~subset of Dr. SCI numerical)
- [ ] Debug reward pipeline: execution errors, verifier integration, reward logging
- [ ] Confirm reward signal is not sparse (target: >10% of rollouts correct on numerical)
- [ ] Launch first 4B GRPO run on numerical curriculum once environment is stable
- [ ] (Optional) principia-collection: filter by physics, verifier round-trip on numerical vs math-object split, dedup against existing corpus — only if reward signal is sparse

### Week 3 outputs
- Verified RL environment (reward signal confirmed non-sparse)
- First 4B GRPO learning curve on numerical curriculum

---

## Week 4 — 2026-04-21 to 2026-04-27

**Goal:** Full curriculum running; TIR-GRPO vs CoT-GRPO comparison at checkpoints.

Tasks:
- [ ] Expand 4B GRPO to expression + MCQ types
- [ ] Compare TIR-GRPO vs CoT-GRPO at 500 / 1000 / 2000 gradient step checkpoints
  - Per-type accuracy for both conditions
  - Compute Pearson correlation: per-type LaTeX FN rate vs per-type accuracy gap
- [ ] Start writing related work + methods sections

### Week 4 outputs
- Mid-project comparison table (TIR-GRPO vs CoT-GRPO by answer type + checkpoint)
- First RQ2 correlation plot (FN rate vs accuracy gap)

---

## Week 5 — 2026-04-28 to 2026-05-04

**Goal:** Ablations and OOD eval done; behavioral analysis; results frozen.

Tasks (run training jobs in parallel — GPU-abundant):
- [ ] Cost penalty ablation: λ > 0 vs λ = 0
- [ ] Curriculum ablation: numerical-first vs full mixed from start
- [ ] Scaling ablation: 0.8B vs 4B TIR-GRPO accuracy
- [ ] OOD eval: MATH-500 hard subset + critpt (if set up in Week 2)
- [ ] Behavioral analysis (100–150 sampled outputs per condition):
  - Strategy categorization: direct compute, unit conversion scaffolding, error-free single-shot
  - Failure modes: wrong physics setup / execution error / correct code + wrong interpretation
  - Truncation rate vs CoT baseline (target: below 22%)

### Week 5 outputs
- Ablation table
- OOD result table (MATH-500 hard)
- Behavioral analysis summary

---

## Week 6 — 2026-05-05 to 2026-05-11

**Goal:** Paper written; figures finalized.

Tasks:
- [ ] Freeze all results
- [ ] Finalize figures: per-type accuracy curves, FN rate vs accuracy gap correlation, truncation reduction
- [ ] Complete paper draft
- [ ] If TIR-GRPO does not outperform CoT-GRPO in aggregate: pivot to reward noise analysis framing (see proposal §11)

---

## Fallback Story

If TIR-GRPO does not outperform CoT-GRPO in aggregate accuracy:

> *"We measure how symbolic verifier reward noise varies across physics answer types and show this predicts differential GRPO learning failure at fixed compute budgets. Execution-based reward eliminates the noise on expression and MCQ types but not equation derivations, revealing a principled boundary for when TIR is necessary vs. sufficient for reliable RL training."*

This is a clean empirical contribution independent of aggregate accuracy improvement.

---

## Non-Negotiables

- Do not start GRPO training before VeRL injection is smoke-tested
- Do not skip the Stage 0 probe — it gates the SFT decision
- Do not skip per-type accuracy instrumentation — it is the core of RQ2
- Do not skip CoT-GRPO baseline run — it is the primary comparison
- Do not leave secondary benchmark (MATH-500 hard) unrun

---

## Relationship to Existing Infrastructure

- `src/phys_reasoner/verifier/` — answer verification pipeline (unchanged; used as GRPO reward)
- `data/processed/drsci_physics_clean.parquet` (107,158 rows) — primary training corpus
- `data/processed/candidates_deduped.parquet` (6,866 rows) — supplemental training corpus
- `data/results/rescore_7b_v3.parquet` — authoritative CoT baseline (39.7% pass@1 with xV-7B)
