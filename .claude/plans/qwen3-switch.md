# Qwen3.5 → Qwen3-4B-Thinking Switch — Execution Plan

> **✅ 2026-04-19 status: config work executed.** Branches `switch/qwen3-thinking` exist on both repos with the §2 overrides applied. Local 0.6B end-to-end smoke passed (commit `cb10403`). Perlmutter smoke not yet run with the new stack. This doc is retained as the record of rationale + rollback instructions; live status lives in `.claude/session-notes.md` and `.claude/handoff/perlmutter-onboarding.md`.

**Created:** 2026-04-18.
**Goal:** unblock training throughput (current: 27 min/step on 1 trainer node + 5 rollout nodes; target: ≤5 min/step) by switching to Qwen3-4B-Thinking, enabling flash-attn + Ulysses SP=4 + `use_remove_padding=True`.
**Why:** Qwen3.5 has hybrid GDN/transformer layers that break verl's Ulysses monkey-patch (see `perlmutter_debug/FINDINGS.md` §7). Qwen3 is standard transformer → the retool/dapo recipes work out of the box. Verified by reading `verl/recipe/retool/run_qwen2_7b_dapo.sh`.
**Scope:** infra-only switch. TIR format (single code block), physics reward, data pipeline, prompts — all unchanged. Starting from Thinking (not Base) to skip SFT we don't have time for.

---

## 0. Pre-flight (do once, before branching)

- [ ] Confirm exact HF name for the model. Candidates: `Qwen/Qwen3-4B-Thinking-2507` (most likely) or `Qwen/Qwen3-4B-Thinking`. Check [huggingface.co/Qwen](https://huggingface.co/Qwen) — use whichever is the canonical reasoning variant of Qwen3-4B. Below, references call it `$MODEL_NEW`.
- [ ] Download to `$HF_HOME` = `$ROOT/hf_cache` (SIF has internet on login, not compute):
  ```bash
  ROOT=/nfs/roberts/scratch/pi_sk2433/xs272/phys-reasoner
  HF_HOME=$ROOT/hf_cache apptainer exec --overlay $ROOT/phys-reasoner-overlay-017b.img $ROOT/verl_vllm017.latest.sif \
    huggingface-cli download Qwen/Qwen3-4B-Thinking-2507
  ```
- [ ] Verify the tokenizer loads inside container and note INTERRUPT_LEN for rollback sanity (script line 75 hardcodes 17 for Qwen3.5; different tokenizer → different count, but we're disabling interrupt so this won't matter).
- [ ] Skim [Qwen3-4B-Thinking model card](https://huggingface.co/Qwen) for: chat template differences vs Qwen3.5, whether it auto-emits `<think>` tags, recommended `top_p/temperature`. Adjust §2.3 if needed.

## 1. Branch creation

**Both repos get a branch. Do not revert, do not re-fork.** Your verl fork's 5 custom commits (3 think-interrupt, 1 wandb logger, 1 offload-check skip) stay intact — we disable interrupt via config, keep the other two.

```bash
cd $ROOT
git checkout -b switch/qwen3-thinking
cd verl
git checkout -b switch/qwen3-thinking
cd ..
# confirm both branches
git branch --show-current
git -C verl branch --show-current
```

Commit early, commit often on the branch — if the smoke fails, `git checkout main` everywhere restores the working Qwen3.5 setup.

## 2. Config changes

### 2.1 Edit `scripts/train_async.sh`

All changes are in the Hydra override block (lines ~240–330).

**DELETE these 3 lines** (Qwen3.5-specific; Qwen3 doesn't need them):

```bash
# line ~263
'+actor_rollout_ref.model.override_config={attn_implementation:sdpa}' \
# line ~281
'+actor_rollout_ref.actor.fsdp_config.wrap_policy.transformer_layer_cls_to_wrap=[Qwen3_5DecoderLayer]' \
# line ~284
'+actor_rollout_ref.ref.fsdp_config.wrap_policy.transformer_layer_cls_to_wrap=[Qwen3_5DecoderLayer]' \
```

**ADD these lines** (insert near the `actor` config block, around line 280):

```bash
actor_rollout_ref.actor.ulysses_sequence_parallel_size=${TRAIN_SP:-4} \
actor_rollout_ref.ref.ulysses_sequence_parallel_size=${TRAIN_SP:-4} \
actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True \
actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=${REF_LOG_PROB_MAX_TOKEN_LEN:-81920} \
actor_rollout_ref.rollout.max_num_batched_tokens=$((MAX_PROMPT_LEN + MAX_RESPONSE_LEN)) \
```

Rationale: retool's recipe (`verl/recipe/retool/run_qwen2_7b_dapo.sh` lines 42–62) uses `train_sp=4`, `log_prob_max_token_len_per_gpu = actor_max_token_len_per_gpu × 4`, and explicit `max_num_batched_tokens`. `actor.use_remove_padding=True` is already set at line 261 — don't re-add.

**Change default model** (line ~45):

```bash
MODEL="${MODEL:-Qwen/Qwen3-4B-Thinking-2507}"   # was Qwen/Qwen3.5-4B
```

**Shrink `PPO_MAX_TOKEN_LEN_PER_GPU`** (env var, default at line ~120 in `train_async.sh`):

```bash
# default was 24576; 1024 prompt + 18961 response = 19985 → round to 20480
PPO_MAX_TOKEN_LEN_PER_GPU="${PPO_MAX_TOKEN_LEN_PER_GPU:-20480}"
```

### 2.2 Leave think-interrupt disabled via env

Do **not** set `THINKING_BUDGET` in the smoke sbatch. Line 310 is conditional: `${THINKING_BUDGET:+…}` — empty string means the flag is never added, which means `tool_agent_loop.py` never triggers interrupt. Model will simply respect `max_response_length`.

Later (if you want): port DAPO's `overlong_buffer` math into `src/phys_reasoner/training/reward.py::compute_score`. Soft length pressure via reward, ~30 lines. Reference: `verl/recipe/dapo` reward manager. **Not required for first smoke.**

### 2.3 Edit `scripts/perlmutter/smoke_tir_het.sbatch`

Changes to the `--env` list or the variables passed to `train_async.sh`:

- Add: `--env "PYTORCH_ALLOC_CONF=expandable_segments:True"` (FINDINGS §11.1 item 1 — OOM mitigation).
- Unset or omit: `THINKING_BUDGET`, `TOOL_CALL_BUDGET`, `ANSWER_BUDGET` (no think-interrupt).
- Set: `TRAIN_SP=4`.
- Keep: `N_GPUS_ROLLOUT=4`, `NNODES_ROLLOUT=5`, `N_GPUS_TRAIN=4`, `NNODES_TRAIN=1` initially. After smoke passes, try `NNODES_ROLLOUT=1` (FINDINGS §11.1 item 4 — rollout was 81% idle).
- Keep: `ACTOR_OPT_OFFLOAD=True` initially. Flip to False only if you also go to 2 trainer nodes.
- Confirm `ROLLOUT_ENFORCE_EAGER=True` is set (memory note in `feedback_enforce_eager_degeneration.md` — Qwen3.5 issue, unknown whether Qwen3-Thinking has same CUDA-graph instability; safer to keep True for first smoke, can flip later).

### 2.4 Do NOT touch

- `/opt/phys-extras/` overlay — stays mounted. Harmless. Remove in a separate cleanup pass after Qwen3 is stable.
- `transformers==5.3.0` pin in overlay — stays. Qwen3 works on transformers 4.57.6+ so SIF baseline also works, but leaving the overlay alone means less moving parts.
- `src/phys_reasoner/tir/prompts.py` — the TIR prompt is already model-agnostic (uses `<think>` / tool-call via chat template). Visual-check only.
- `src/phys_reasoner/training/reward.py` — physics verifier, unchanged.
- `verl/` submodule code — no code changes. 5 custom commits stay.

## 3. Smoke test

Submit the same `smoke_tir_het.sbatch` topology as the 2026-04-17 run (het group: 1 trainer + 5 rollout + 1 xverify). Keep `TOTAL_STEPS=10`, `TRAIN_BATCH=128`, `ROLLOUT_N=8` to enable direct comparison vs the 27 min/step baseline.

**Pass criteria:**

| Metric | Previous (Qwen3.5, SDPA, SP=1) | Target (Qwen3, FA, SP=4) | Hard fail |
|---|---|---|---|
| step wall-clock (median over 10 steps) | 1634 s | **≤ 300 s** | ≥ 1200 s |
| step 1 completes | yes | yes | OOM |
| step 2 completes | **no (OOM)** | yes | OOM |
| reward score mean (step 1) | 0.269 | ≥ 0.20 | < 0.05 (model broken) |
| zero-advantage group rate | 22.7% | ≤ 30% | > 50% |
| trainer-rollout throughput ratio | 0.186 active | ≥ 0.5 active | < 0.15 |

If median step ≥ 600 s: **stop and diagnose** before spending more compute. Likely culprits in order: (a) SP=4 not actually active (check launcher logs for `Monkey patch _flash_attention_forward` line), (b) flash-attn backend not picked up (grep for `attn_implementation`), (c) `use_remove_padding` disabled somewhere.

If it OOMs: flip `PPO_MAX_TOKEN_LEN_PER_GPU=16384`, keep `expandable_segments:True`, retry. If still OOM, temporarily enable `ACTOR_PARAM_OFFLOAD=True`.

## 4. Post-smoke tuning (do only if pass)

In order of impact × risk:

1. **Shrink rollout to 1 node.** Rollouter was 81% idle in the previous run. Frees 4 nodes. Config change only.
2. **Try TP=1 on rollout** (currently `tensor_model_parallel_size=N_GPUS_ROLLOUT=4`). For a 4B model, TP=1 across 4 separate vLLM instances usually beats TP=4 on one instance. Compare throughput.
3. **2 trainer nodes + `optimizer_offload=False`.** Recovers PCIe-offload tax. Requires `_ray_bringup.sh` edit (see FINDINGS §8 "Requires `_ray_bringup.sh` edit"). Medium-risk: inter-node NCCL on Slingshot.
4. **`async_training.staleness_threshold=1`.** ~10–15% wall-clock savings if reward curve is stable. Gate on reward not degrading.

## 5. Rollback

If the branch doesn't deliver the target in 2 days of smoke iteration:

```bash
git checkout main
cd verl && git checkout main && cd ..
# Qwen3.5 setup is restored, branch preserved for forensics
```

Nothing on main changes. `/opt/phys-extras/` still valid. Previous smoke sbatch still runs.

## 6. Open items (track, don't block)

- [ ] Chat template for Qwen3-4B-Thinking: does it auto-emit `<think>`, or does the prompt need `enable_thinking=True` like Qwen3.5? First smoke's output inspection will reveal.
- [ ] Port DAPO `overlong_buffer` math to `src/phys_reasoner/training/reward.py::compute_score` (Week 3 task; not blocking smoke).
- [ ] Post-smoke: if trainer-step decomposition is still opaque, sprinkle `time.perf_counter()` prints in `verl/verl/experimental/fully_async_policy/fully_async_trainer.py::_fit_update_weights`. 30-minute job, only if optimization keeps missing.
- [ ] Once Qwen3 is stable: can remove `/opt/phys-extras/` entirely? Verify nothing else depends on transformers 5.3.0 first.
- [ ] Paper framing: "We initialize from Qwen3-4B-Thinking (Qwen's instruction-tuned reasoning variant of Qwen3-4B-Base). Dr.SCI (same model family, Base + SFT + CoT-GRPO) reports 62.7 on GPQA-D; we compare our TIR + RLVR at X." Settle phrasing when results come in.

## 7. Files touched

| File | Change type |
|---|---|
| `scripts/train_async.sh` | delete 3 override lines, add 5 override lines, change MODEL default, change PPO_MAX_TOKEN_LEN_PER_GPU default |
| `scripts/perlmutter/smoke_tir_het.sbatch` | add `PYTORCH_ALLOC_CONF` env, add `TRAIN_SP=4`, unset THINKING_BUDGET/TOOL_CALL_BUDGET/ANSWER_BUDGET |
| `verl/` submodule | **no code changes**; branch only |

## 8. References

- `perlmutter_debug/FINDINGS.md` — prior-run analysis, target numbers, sizing rationale.
- `verl/recipe/retool/run_qwen2_7b_dapo.sh` — config values copied verbatim (SP=4, log_prob_max_token_len × 4, dynamic_bsz).
- `verl/verl/models/transformers/monkey_patch.py` lines 483–491 — why Qwen3.5 can't take flash-attn today.
- `CLAUDE.md` "Required VeRL Overrides for Qwen3.5-4B (GRPO)" section — the overrides this plan is REMOVING.
- `.claude/plans/physcode.md` — parent plan this slots into.
