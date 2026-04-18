# Training Decisions & Strategy Notes

Last updated: 2026-03-30

---

## SFT Stage: Conditional on Stage 0 Probe

**Decision:** Run a 100-problem zero-shot TIR probe (Stage 0) before committing to SFT.

**Threshold:** If verifier hit rate on the TIR zero-shot probe ≥ 15% → skip SFT entirely.
If < 15% → generate 2–5k TIR demonstrations via GPT-4o/Claude (auto-exec filtered) and fine-tune to stabilize format.

**Rationale:**
1. Qwen3.5-4B instruct (thinking OFF) may already produce valid single-block TIR format at useful rates, making SFT unnecessary overhead.
2. SFT is only needed to stabilize code format, SymPy usage, and post-execution reasoning — not to teach physics reasoning.
3. Auto-exec filter on SFT demos ensures only working code examples are used.
4. Papers like DAPO and Dr. GRPO show GRPO works from instruct model directly.

**Fallback:** If GRPO training is unstable (reward stuck near zero), warm-start from SFT checkpoint. Treat as last resort.

**Stage 0 probe details:**
- 100 problems stratified across numerical / expression / MCQ
- Measure: execution success rate, verifier hit rate, per-type accuracy
- Prompt for single-block TIR format: `[think]…[code]…[/code][output]…[think]…[answer]\boxed{}`

---

## Stage 1 data policy — Option A3 (2026-04-14, decided)

**Decision:** For Stage 1, use the **full unfiltered training pool** (drsci_train_split + corpus_train_split, ~108k rows) with uniform sampling. Skip Strategy B offline weighting and skip hard-bank filtering. Compensate for zero-advantage groups with extra steps instead.

**Why not the original Strategy B + DAPO-filter plan (Option A1/A2):**
- VeRL has native DAPO-style `FilterGroupsConfig` (`verl/trainer/config/algorithm.py:43-56`, implementation `verl/recipe/dapo/dapo_ray_trainer.py:240-299`), but **it is NOT wired into the `fully_async_policy` trainer** we use. The async trainer's `_get_samples_from_queue` path ignores filter_groups flags. Porting filter_groups into async would require ~50 lines in `FullyAsyncTrainer._fit_compute_advantage` and interacts non-trivially with `require_batches` and parameter-sync timing.
- Switching to the synchronous trainer would mean abandoning the 3+1 async hardware split.
- Probe homogeneity analysis (Session 8) showed stratum membership explains only 29% (Dr. SCI) / 17% (corpus) of per-problem pass variance, so offline Strategy B weights would be noisy for ~70% of the variance anyway.
- Debugging-cost budget (explicit user priority) favors "no code changes, pay extra rollout compute" over any patch path.

**Mechanism:** Every problem is seen uniformly. GRPO groups where all rollouts are correct or all incorrect contribute exactly zero to the loss under the Dr. GRPO + DAPO-lite config (`use_kl_loss=False`, `kl_ctrl.kl_coef=0.0`) — fully equivalent to DAPO's dynamic-sampling filter, just with extra rollout cost. (Earlier draft of this section worried about KL amplification; that concern applied to conservative-GRPO with `kl_loss_coef=0.001` and is moot under the canonical Dr. GRPO/DAPO setup we're using.)

**Step budget (see "Training step / batch / pool sizing" below):**
- Start at **1500 steps**, measure per-batch zero-advantage-group fraction, extend to 2500–3000 only if plateau not reached. At 1500 steps we cover ~1.8 epochs of the full pool.
- No warmup, no data complication — full 108k pool from step 0 (user preference, 2026-04-14).

**Single-stage training (3-stage plan deprecated):** see "Multi-stage training structure" below for why — under A3 the data churn between stages stops serving a purpose.

**Strategy B / D deferred to future work:**
- Strategy B offline weighting postponed indefinitely; would only return if A3 shows unmanageable zero-advantage waste.
- Strategy D rescoring kept only as an *offline analysis* tool — run on a mid-training checkpoint to produce curriculum-drift figures for the paper. Not used as a training-time mechanism.
- Probe artifacts (`probe_per_problem.parquet`, `probe_summary.txt`) retained for the paper's analysis of initial-model stratification.

---

## Goldilocks Data Selection: Strategy B + D (original plan — deferred to Stage 2+)

**Context:** Pass@k estimates from a small sample tell us per-stratum Goldilocks rates but not per-problem pass rates across 113k rows. Strategy B uses those stratum-level rates to initialize weighted sampling; Strategy D updates per-problem pass rates iteratively using training checkpoints.

### Why not hard-filter to Goldilocks problems only?

- Corpus: hard filter leaves only ~1,200 problems (18% of 6,866) — too small for stable GRPO.
- Dr. SCI: 18% of 107k ≈ 19k problems would survive, but the filter requires per-problem pass rates we don't have for 99% of rows.
- Pass@k estimates from k=8 are noisy near the boundary.
- The Goldilocks zone shifts during training as the model improves — static pre-filtering discards future gradient signal. Strategy D handles this dynamically.

### Strategy B — Probe-based bucket weights (Stage 1 initialization)

Before Stage 1, run 8 TIR rollouts on a stratified probe subset (~2.5k rows) using the base model (Qwen3.5-4B instruct). Compute per-problem pass rate. Aggregate to per-stratum Goldilocks rates (pass_rate ∈ [0.05, 0.95]). Add a `_train_weight` column to the train parquets encoding each problem's stratum weight.

**Stratification dimensions:**
- Dr. SCI: `(inferred_answer_type × extra_info.from × difficulty_bin)` where difficulty_bin = low(0.0) / medium(0.125–0.375) / high(0.5–0.75)
- Corpus: `(primary_answer_type × source)`

Weights are proportional to stratum Goldilocks rate. Problems in strata with very low Goldilocks rate (mostly too hard) are downweighted, not excluded — they enter via the hard bank in later stages.

**Note on old weight table:** An earlier proxy table derived from 600-row pass@k samples has been superseded. The probe rollouts in Step 3 of `data-pipeline.md` produce the authoritative weights.

### Strategy D — Stage rescoring and dataset rebuild

After each training stage (~700 steps), run `rescore_goldilocks.py` with the stage checkpoint on a stratified subset (size TBD, baseline 5k rows; scale up if compute allows). Recompute per-problem pass rates. Update `_train_weight` and the hard bank:

- Problems that were too hard (pass_rate < 0.05 under base model) but now have pass_rate ∈ [0.05, 0.95] graduate from the hard bank into the active training set.
- Problems that have become too easy (pass_rate > 0.95) are downweighted.
- Updated `_train_weight` values take effect for the next stage.

This is how the model's improving capability is exploited: the Goldilocks zone expands as training progresses, and the dataset composition tracks it automatically.

### Why not an online per-batch filter (Strategy C)?

GRPO already handles the degenerate cases without extra engineering: if all k rollouts are correct, group-relative advantage = 1 − 1 = 0 (no gradient); if all wrong, advantage = 0 − 0 = 0. Explicit skip logic would only save the forward pass on those batches. With B+D keeping the dataset clean offline, the frequency of these degenerate batches is low. Strategy C is dropped.

### Hard problem bank

Problems with pass_rate < 0.05 in the initial probe are placed in `data/processed/hard_bank.parquet` rather than discarded. They are re-evaluated at each stage rescore (Strategy D). Once a problem's pass rate rises above 0.05, it enters the active training pool. This is the primary mechanism for the Stage 2+ curriculum expansion.

### Multi-stage training structure (deprecated 2026-04-14 under Option A3)

**Superseded: single-stage training with optional warmup.** The three-stage design was justified by Strategy B+D's data churn between stages (weight rebuilds, hard-bank graduation). Under A3 neither mechanism is in play, so stage boundaries stop serving a purpose — the implicit policy-aware filtering from zero-advantage groups achieves Strategy D's goal automatically.

**New plan: single stage of 1500–3000 steps on the full pool. No warmup, no data churn.**

| Phase | Data | Steps |
|---|---|---|
| main | full 108k pool (drsci_train_split + corpus_train_split) | 1500–3000 |

User decision (2026-04-14): skip the optional warmup to keep the data side as simple as possible. Rely on A3's implicit curriculum (zero-advantage filtering) alone. Dr. SCI's published curriculum benefit comes from their *dynamic* tracker (online reshuffling), not from starting on a medium band — a static 200-step warmup would capture only a fraction of that and isn't worth the parquet/sbatch complexity given no observed pathology.

**For paper analysis only** (not needed for training): run an optional offline rescore on a mid-run checkpoint to produce "curriculum drift" figures — which strata's pass rates improved, which problems graduated from 0-advantage to informative. Doesn't affect training.

### Curriculum — Dr. SCI mitigation strategy (2026-04-14)

**Dr. SCI's method (MiniByte-666 fork):** dynamic difficulty tracker (`difficulty_tracker.py`) + dynamic dataset (`dynamic_difficulty_dataset.py`). Starts with problems in `[0.51, 0.99]` difficulty band; when a problem reaches ≥0.9 accuracy (mastered), replaces it with a random draw from `< 0.51` (harder) pending set. State persists via pickle.

**Our decision:** do not port their tracker; do not do a static warmup either. Rely on A3's implicit policy-aware curriculum alone — problems self-activate when the policy can reach them.

**Why not port the tracker:**
- Active-set replacement requires persistent state threaded through VeRL's async dataloader; non-trivial glue.
- Primary benefit is "don't waste compute on mastered problems"; we've deemed rollout compute cheap.

**Why not even a static warmup:**
- Dr. SCI's benefit is *dynamic* reshuffling; a fixed starting band captures only a small fraction.
- Base model's 14% hit rate on hardest difficulty bucket shows no catastrophic failure mode for the warmup to solve.
- Adds parquet + sbatch complexity without a demonstrated problem.

**Fallback trigger:** if first ~100 smoke steps show high reward variance / unstable gradient on hard types, or dev-set accuracy plateaus early with persistently high zero-advantage fraction, revisit — either static warmup (cheap) or tracker port (expensive).

---

## Algorithm — Dr. GRPO + DAPO-lite (2026-04-14, decided)

**Decision:** train with **Dr. GRPO** (length + std normalization removed) plus the async-compatible subset of DAPO: **clip-higher** (asymmetric PPO clip) and **token-level loss aggregation**, plus mild **overlong-response reward shaping**. Do **not** use DAPO's dynamic sampling (sync-only, already established). Do not adopt GSPO or CISPO for the paper's main method (too new, undervalidated).

**Algorithm comparison:**

| Algorithm | Idea | Async-compat | Maturity | Delta vs GRPO |
|---|---|---|---|---|
| GRPO | group-relative advantage | ✓ | high | baseline |
| Dr. GRPO | remove length & std normalization (bias fixes) | ✓ | medium (2025) | small, principled |
| DAPO full | clip-higher + token-level + dynamic-sampling + overlong shaping | **partial** (dyn-samp is sync-only) | high | large on AIME-style long CoT; we lose 1/4 components on async |
| GSPO | sequence-level IS ratio | ✓ | low (too new) | unknown |
| CISPO | clipped IS + REINFORCE | ✓ | low (MiniMax M1 only) | unknown |

**Rationale for the chosen combination:**
- Dr. GRPO fixes are orthogonal, cheap, and strictly improve GRPO statistical properties (no reason not to take them).
- DAPO's clip-higher and token-level loss drove most of DAPO's accuracy gains in the paper — they're the highest-ROI components and are fully compatible with async.
- Overlong shaping matters for our ~19k response budget with think-interrupt: we want to gently penalize rollouts that burn the budget without producing an answer, without discarding them.
- GSPO/CISPO are both 2025-era and undervalidated; the paper's main contribution is TIR vs CoT-GRPO reward quality, not RL algorithm choice, so we don't want the method to draw reviewer attention.

**Paper framing:** "Dr. GRPO with DAPO-style asymmetric clipping and token-level loss aggregation." Modern, defensible, uncontroversial.

**VeRL flags (verified 2026-04-14 against `verl/recipe/dapo/run_dapo_*.sh` (~15 recipes) and `verl/docs/algo/grpo.md:53-62`):**
```
algorithm.kl_ctrl.kl_coef=0.0                         # DAPO canonical
algorithm.norm_adv_by_std_in_grpo=False               # Dr. GRPO: no std norm
actor_rollout_ref.actor.use_kl_loss=False             # DAPO + Dr. GRPO: KL off
actor_rollout_ref.actor.kl_loss_coef=0.0              # DAPO canonical
actor_rollout_ref.actor.loss_agg_mode=token-mean      # DAPO; VeRL best_practices.rst:184 says this matches Dr. GRPO too
actor_rollout_ref.actor.clip_ratio_low=0.2            # DAPO canonical (all recipes)
actor_rollout_ref.actor.clip_ratio_high=0.28          # DAPO: clip-higher
actor_rollout_ref.actor.clip_ratio_c=10.0             # DAPO canonical
```

**Key correction (2026-04-14):** earlier plan retained `use_kl_loss=True, kl_loss_coef=0.001` from the existing train_async.sh. That was conservative-GRPO, not modern DAPO/Dr. GRPO. All canonical recipes set KL off entirely. With KL off, **zero-advantage groups contribute exactly zero to the loss** — not 90–95% equivalent to filtering, but fully equivalent. The "zero-advantage groups amplify KL regularization" concern from the earlier A3 discussion vanishes.

**Loss aggregation choice (token-mean vs seq-mean-token-sum-norm):** VeRL's Dr. GRPO doc (`docs/algo/grpo.md:59`) prescribes `seq-mean-token-sum-norm` for strict paper faithfulness; `best_practices.rst:184` states `token-mean` matches both Dr. GRPO and DAPO. We use `token-mean` because it matches all DAPO recipes (~15) and the best-practices doc, and is the more common choice in recent long-CoT RL work (DeepSeek-R1, DAPO itself).

**Overlong reward shaping:** add in `src/phys_reasoner/training/reward.py` — if `rollout_token_count > 0.9 * MAX_RESPONSE_LEN` and no `\boxed{}` found, return a small negative score (e.g., -0.3) instead of 0. Avoids encouraging truncation-at-budget behavior.

See `.claude/plans/data-pipeline.md` for historical pipeline spec (Strategy B+D — no longer the active plan).

### Training step / batch / pool sizing (updated 2026-04-14 for Option A3)

**Terminology in VeRL (async GRPO):** one trainer *step* = one optimizer update. Per step the system consumes `ppo_mini_batch_size` prompts × `rollout.n` rollouts per prompt. The dataloader samples with repetition, so the training pool is revisited across epochs within a stage. `trainer.total_epochs=1` disables the epoch cap — step count, not epoch count, is the run budget.

**Active pool for A3:** unfiltered train split — Dr. SCI (~101,729) + corpus (~6,466) = **~108,195 rows**. No offline weighting, no subsampling, multi-part and PHYBench included.

**Data quantity per stage decoupled from step count.** Formula:
```
epochs per stage ≈ (TOTAL_STEPS × ppo_mini_batch_size) / |pool|
```
Target: **1.5–3 epochs per stage**. Large pool keeps per-problem revisit count low; zero-advantage groups cost extra compute but don't hurt coverage.

**Settings for the 4×A100-80G (train) + async rollout layout:**

| Knob | Value | Reason |
|---|---|---|
| `rollout.n` (G) | 8 | Matches probe; standard GRPO group size for small models (SimpleRL, POLARIS) |
| `ppo_mini_batch_size` | 128 | 1024 rollouts/step → low-noise policy gradient |
| `ppo_micro_batch_size_per_gpu` | 2 | Room on 80 GB cards with FSDP-4 + optimizer-offload + grad ckpt |
| `TOTAL_STEPS` per stage | **1500 initial, extend to 3000 if needed** | 1500 ≈ 1.8 epochs. Decide to extend after measuring plateau / 0-adv fraction |
| `save_freq` | 200 | ≥7 checkpoints per stage — generous 48h-job resume points |
| `test_freq` | 100 | Dev-set eval cadence |
| `use_kl_loss`, `kl_loss_coef` | **False, 0.0** | DAPO + Dr. GRPO canonical. Makes A3's zero-advantage groups mathematically equivalent to filtering. |

Hardware is **not** the binding constraint for batch: 4B + FSDP-4 + CPU-offloaded AdamW + grad ckpt leaves plenty of headroom at 20k sequence length.

### Runtime estimate and rollout-node sizing (2026-04-14, calibrated from probe v5)

**Probe calibration (measured):** single-H200 = **~2,400 rollouts/hr** (shard A: 4h 7m for 10k rollouts; shard B matched).

**H200 → A100-80G conversion** for 4B decode at ~20k ctx: ~0.45–0.55× per-GPU (memory-bandwidth ratio 2.4×). In async with per-step weight sync (`trigger_parameter_sync_step=1`), subtract another ~25% overhead.
→ **Effective A100 80G throughput ≈ 800–1,100 rollouts/hr per GPU.**

**Trainer step ceiling (4B + 20k ctx + batch 128 on 4×A100 80G, FSDP-4 + CPU-offload + grad ckpt, micro_bs=2):** estimate 50–80 s/step → 45–70 steps/hr. **Measure in first smoke run before locking rollout-node count.**

**Stage 1 wall-time projections:**

| Rollout nodes | A100s | rollout/hr | Rollout-bound wall time (1500 / 3000 steps) | Trainer-bound floor |
|---|---|---|---|---|
| 3 | 12 | ~11k | ~140 h / ~280 h | — |
| 4 | 16 | ~15k | ~100 h / ~205 h | — |
| **5** | **20** | **~19k** | **~80 h / ~160 h** | ~25 / ~50 h |
| 6 | 24 | ~23k | ~67 h / ~135 h | ~25 / ~50 h |

At ≥5 rollout nodes the trainer becomes the ceiling; adding more rollout nodes past 5 saturates. **Target: 5 rollout nodes.**

**Perlmutter 48h job cap:** Stage 1 at 1500 steps ≈ 2×48h jobs with 5 rollout nodes; 3000 steps ≈ 3–4×48h jobs. Use VeRL's `trainer.resume_mode=auto` to stitch jobs automatically.

**First-run plan:** short (~50-step) smoke on 4×A100 + 1 rollout node to (a) confirm no pathology with zero-advantage groups, (b) measure actual trainer step time, (c) measure actual async rollout throughput on A100. Size the production run from real numbers.

**Reference anchors (published GRPO runs for context):**

| Project | Model | `ppo_mini_batch` | G | Rollouts/step |
|---|---|---|---|---|
| SimpleRL-Zoo | 7B | 128 | 8 | 1024 |
| POLARIS | 4B | 128 | 8 | 1024 |
| DAPO | 32B | 512 | 16 | 8192 |
| DeepSeek-R1-Zero | 7B | 1024 | 64 | 65536 |

---

## Dataset Sizes (pool v2, 2026-04-18)

Pool v2 replaces the earlier 108k training pool with a compute-constrained,
difficulty-filtered subsample. Built by `scripts/build_pool_v2.py`.

**Composition rule (one filter per source, then include):**
- Dr.SCI physics filtered to `difficulty ≥ 0.5` (Qwen3-32B 8-rollout pass-rate
  threshold; matches Dr.SCI's own dynamic-curriculum starting band floor of
  0.51). Pre-filter figure/NaN/unknown-answer-type drops reduce 19,409 →
  18,833 usable rows.
- UGPhysics (EN) — no difficulty labels, no filter.
- PHYSICS (desimfj, cleaned corpus) — no difficulty labels, no filter;
  stratified 80/20 self-split (no public leaderboard to dilute).
- SciBench-RL train — physics subset (excluding atkins/chemmc chemistry), no
  filter. Paired with the official native test split (153 problems from
  disjoint textbooks class/diff/matter) held out unchanged.

**Composition table (`data/processed/pool_v2/{tir,cot}/{train,validation,test}.parquet`):**

| Source | Train | Dev | Test |
|--------|-------|-----|------|
| Dr.SCI physics (≥0.5) | 17,827 | 503 | 503 |
| UGPhysics | 4,980 | 217 | 217 |
| PHYSICS | 502 | 100 | 191 |
| SciBench-RL | 280 | 0 | 153 (native) |
| **Total** | **23,589** | **820** | **1,064** |

At `ppo_mini_batch_size=128` and 200 steps, this yields **1.06 epochs** of
revisit — Rule-A composition (include-all-after-filter, epochs fall out).
Extend to 2–3 epochs (~400–600 steps) if training-step time after the
Qwen3-switch + infra improvements lands in the 5–8 min range.

**Mirror on HF:** `xingzhi0/phys-tir` (TIR system prompt) and `xingzhi0/phys-cot`
(matched CoT-only baseline). Schema: union `extra_info` struct + top-level
`pool` column (`"drsci"` / `"curated"`).

**Contamination check (verified during build):** pairwise MD5-normalized
problem-text intersections across all sources = 0 (including SciBench train ∩
native test = 0, since they're from disjoint textbooks).

**Historical context:** The 2026-04-16 corpus rebuild (`scripts/rebuild_corpus_splits.py`)
dropped PHYSICS/OlympiadBench/SciBench_RL/PHYBench from training to preserve
them as clean external eval. Pool v2 partially reverses that for PHYSICS and
SciBench-RL under the compute-constrained subsample regime; OlympiadBench and
PHYBench remain fully held out. See `docs/eval_plan.md` for eval implications.

---

## Verifier Status (as of 2026-03-30)

All planned verifier work is complete:

| Component | Status |
|-----------|--------|
| Rule verifier (`math_verify_wrapper.py`) | ✅ Done — includes `_sympy_numerical_equiv` numerical substitution tier |
| xVerify integration | ✅ Done — xVerify-7B-I preferred; `local_files_only=True` in sbatch |
| Dr. SCI e-notation fix | ✅ Done |
| MCQ parenthesis normalization | ✅ Done |
| `_is_prose_gold` drop in cleaning | ✅ Done (step 6 in `drsci_clean.py`) |

The plan `floating-percolating-thunder.md` (numerical equiv checker) is **stale/complete**.

---

## TIR Training Decisions (PhysCode, 2026-03-31)

**Reward:** Binary R_correct on final `\boxed{}` answer (λ=0 initially). Token cost penalty added only in late ablation.

**Curriculum:** Numerical problems first (cleanest execution path, highest zero-shot accuracy). Expand to expression + MCQ once training is stable.

**TIR format:** Single code block per trajectory — `[think]…[code]…[/code][output]…[think]…[answer]\boxed{}`. Multi-block explicitly out of scope for MVP.

**Execution sandbox:** 30s timeout, subprocess + resource limits. Timeout distribution should be profiled on dev set before training.

### xVerify in reward function — open infrastructure problem

The rule-only verifier has ~68% FN rate on expression types (confirmed in experiments). This means many correct model answers get reward=0, directly hurting GRPO signal quality for expressions.

**Why xVerify can't be the default in `reward.py` today:**
VeRL calls `compute_score()` per-sample. xVerify-7B needs ~14GB GPU RAM and ~0.5s/call when warm. Loading it inside the reward function is not feasible (cold-load per call, or OOM with rollout model on same GPU).

**Options (in order of practicality):**

| Option | When | Notes |
|--------|------|-------|
| **Rule-only, numerical curriculum** | Smoke test + Phase 1 | Rule verifier FN rate on numerical is low (~5–10%). Start here. |
| **Dedicated reward GPU** | Production run | Run xVerify-7B on a separate A100; call via socket/HTTP from `compute_score()`. VeRL's custom reward function supports this pattern. |
| **Batch post-scoring** | Ablation only | Score rollout batch rule-first; queue expression-type unknowns for xVerify; update rewards before GRPO update step. Requires VeRL reward API change. |
| **xVerify co-located (careful)** | If reward GPU unavailable | Load xVerify-7B once as a module-level singleton in the reward worker process. Only works if VeRL uses a dedicated reward worker process (not same as rollout). |

**Current default in `reward.py`:** `xverify_judge=None` (rule-only). This is correct for the smoke test.

**Action required before full expression-type training:** set up a dedicated reward GPU running `xverify_judge` and wire it into `compute_score`. File an explicit task when starting the production run.

**CoT-GRPO baseline:** Must be run in parallel with TIR-GRPO for comparison (same data, same checkpoints). Required for RQ1 and RQ2.

---

## Immediate Next Steps (as of 2026-04-10)

1. **Think-interrupt patch** — implement in `verl/verl/experimental/agent_loop/tool_agent_loop.py` (spec in `physcode.md` Week 2). Gating item for all training.
2. **Data pipeline** — Steps 0–6 in `.claude/plans/data-pipeline.md`. Each step is a separate session. Steps 0 and 0b can start immediately; Steps 3+ require think-interrupt to be settled.
3. **Stage 1 GRPO run** — numerical curriculum, Strategy B weights, ~700 steps.

---

## Key Numbers for Paper Baselines

| Model | Condition | pass@1 (xV-7B) | Notes |
|-------|-----------|----------------|-------|
| Qwen3.5-4B | Think-ON, full corpus | 39.7% | `rescore_7b_v3.parquet` — authoritative |
| Qwen3.5-4B | No-Think, full corpus | 32.0% | `zero_shot_nothink_xverify_7b.parquet` |
| Qwen3.5-0.8B | No-Think, full corpus | 10.1% | `zero_shot_nothink_08b_xverify_7b.parquet` |
| Qwen3.5-4B | No-Think, corpus sample (pass@8) | 30.3% | `zero_shot_corpus_passk_xv7b.parquet` |
| Qwen3.5-4B | No-Think, Dr. SCI sample (pass@8) | 24.3% | `zero_shot_drsci_sample_xv7b.parquet` (prose-gold filtered) |

Non-truncated accuracy (think-ON): **49.0%** — the ceiling for fixed compute budget.

---

## Sampling Parameters for Thinking-Mode Rollouts (2026-04-13, final)

### The `!!!` Degeneration Problem — Root Cause: CUDA Graph Instability

Qwen3.5-4B with `enable_thinking=True` degenerates into repetitive single-token loops (`!!!...`) in ~96% of rollouts when using vLLM without `enforce_eager=True`. The degeneration appears **exclusively in phase 2** (after tool response injection), never in phase 1 thinking/tool-call generation. It is stochastic (mixed within every problem, not problem-dependent).

**Root cause: CUDA graph capture is numerically unstable with Qwen3.5's hybrid GDN (Gated Delta Network) linear attention layers.** With `enforce_eager=False` (vLLM's default), vLLM captures the forward pass into static CUDA graphs. These graphs produce subtly wrong logits on long-context phase 2 prompts, causing the model to enter absorbing repetition states. VeRL's `RolloutConfig` defaults to `enforce_eager=True`, which is why VeRL rollouts never exhibited this.

Diagnostic history (each row used `enforce_eager=False` unless noted):

| Run | `enforce_eager` | `top_p` | `temp` | `pp` | `!!!` degen |
|-----|-----------------|---------|--------|------|-------------|
| probe_calib_A/B | False | 0.9 | 1.0 | 0 | **96–97%** |
| probe_v2_A_fixed | False | 1.0 | 1.0 | 0 | **97%** |
| probe_fix_pp15 | False | 0.95 | 1.0 | 1.5 | 0% |
| probe_fix_t14 | False | 1.0 | 1.4 | 0 | 0% |
| **eager_test** | **True** | **1.0** | **1.0** | **0** | **0%** |

The `presence_penalty=1.5` and `top_p=0.95` "fixes" were masking the CUDA graph instability by constraining the distribution enough to avoid the degenerate states — not addressing the root cause.

### Decision: `enforce_eager=True`, standard RL sampling (`top_p=1.0`)

**Chosen config** (matching VeRL framework + RL theory):
```
enforce_eager=True
temperature=1.0, top_p=1.0, top_k=-1, presence_penalty=0.0, repetition_penalty=1.0
```

- `enforce_eager=True`: eliminates CUDA graph instability (matches VeRL `RolloutConfig` default)
- `top_p=1.0`: raw policy sampling for on-policy GRPO (no uncorrected IS mismatch)
- No `presence_penalty`: avoids context-dependent distribution modification during training
- Qwen model card params (`top_p=0.95, top_k=20, pp=1.5`) reserved for inference/eval only

### Bug Fixes in `dump_rollouts.py` vs VeRL (2026-04-13)

Four code-level discrepancies were found and fixed:

1. **Missing `enforce_eager=True`** (PRIMARY FIX) — `dump_rollouts.py` used vLLM's default `enforce_eager=False`, enabling CUDA graphs that are unstable with Qwen3.5's GDN attention. VeRL defaults to `enforce_eager=True`. Fixed: `LLM(enforce_eager=True)`.

2. **Missing `<|im_end|>` after `</tool_call>`** — `dump_rollouts.py` stops phase 1 generation at the `</tool_call>` stop string, so the model never emits the `<|im_end|>` token that closes the assistant turn. In VeRL there is no stop string; the model generates past `</tool_call>` and naturally emits `<|im_end|>`. Fixed: `_make_tool_injection` now prepends `<|im_end|>`.

3. **`enable_thinking=False` in tool injection** — `dump_rollouts.py` used `enable_thinking=False` when building the tool response injection, producing a pre-closed empty `<think>\n\n</think>\n\n` block. VeRL passes `enable_thinking=True` (from `apply_chat_template_kwargs`), producing an open `<think>\n` block that lets the model optionally reason before answering. Fixed: injection now uses the same `enable_thinking` flag as phase 1.

4. **`top_p` default** — Changed to 1.0 to match RL theory (raw policy sampling). VeRL framework default is also 1.0; the 0.9 in training scripts was a pre-fix artifact.

### TODO: Update VeRL training scripts before final run

The following VeRL training scripts still use `top_p=0.9` (a pre-fix artifact). Change to `top_p=1.0` for theoretical correctness (raw policy sampling in on-policy GRPO). **Do not edit yet** — preserve reproducibility of existing smoke test results. Apply before the final Stage 1 training run.

| Script | Line | Current | Target |
|--------|------|---------|--------|
| `scripts/smoke_tir_qwen35.sh` | ~207 | `top_p=0.9` | `top_p=1.0` |
| `scripts/smoke_tir.sh` | ~161 | `top_p=0.9` | `top_p=1.0` |
| `scripts/train_async.sh` | ~199 | `top_p=0.9` | `top_p=1.0` |

`enforce_eager=True` is already the VeRL `RolloutConfig` default — no change needed there.

**Open question:** The CUDA graph instability was confirmed on B200 (SM_100, Blackwell). VeRL defaults to `enforce_eager=True` regardless of GPU, suggesting this is a known cross-architecture issue with vLLM + hybrid attention models. Worth testing on H200/A100 if performance matters — CUDA graphs are 2-3x faster, so if they're stable on Ampere/Hopper the training scripts could conditionally enable them.

### vLLM Qwen3.5 Bug: B200-Specific TRTLLM Prefill Corruption (2026-04-13, final diagnosis)

**Bug:** On B200 GPUs (Blackwell SM_100), vLLM with Qwen3.5-4B produces catastrophic output corruption (`!!!` repetition loops in phase 1 thinking). The corruption exhibits two apparent patterns that are actually the same underlying bug:
1. **Single large batch (≥400 prompts)**: ~45% of phase-2 rollouts degenerate immediately
2. **Many sequential small batches (40 each) through same engine**: clean for ~10 chunks, then abrupt 100% degeneration that persists (KV-cache state poisoning pattern)

**Root cause (confirmed):** **TRTLLM prefill attention backend on Blackwell.** vLLM's flashinfer auto-detects TRTLLM for SM_100 and the kernel has known correctness issues with long contexts. Matches [flashinfer#1968](https://github.com/flashinfer-ai/flashinfer/issues/1968), [vllm#35138](https://github.com/vllm-project/vllm/issues/35138), and analogous issues.

**Controlled experiment (B200 vs H200 vs A100/H100, all identical code + params):**

| GPU | Run | Chunks | Degen |
|-----|-----|--------|-------|
| B200 | drift_test_b200 (no fix) | 4 | Chunks 0-5 clean, chunk 10+ drifts to 100% |
| **B200** | **drift_b200_notrtllm (`VLLM_USE_TRTLLM_ATTENTION=0`)** | **4** | **0/160 (0%)** ✓ |
| H200 | drift_test_h200 | 4 | **0/320 (0%)** ✓ |
| **H200** | **probe_v5_A (full scale)** | **25** | **0/10000 (0%)** ✓ |
| RTX 5000 Ada (gpu partition) | drift_test_gpu | 4 | **0/320 (0%)** ✓ |

**The fix:** Set env var `VLLM_USE_TRTLLM_ATTENTION=0` when running on B200, OR run on H200/A100/Ada GPUs which don't use the TRTLLM prefill path.

**Prior hypotheses that were WRONG:**
- ~~`top_p=0.9` too aggressive~~ — same rate at 1.0
- ~~Missing `presence_penalty=1.5`~~ — only masked the bug by reducing "!" probability
- ~~CUDA graph instability (`enforce_eager=False`)~~ — `enforce_eager=True` didn't fix it
- ~~Batch-size threshold at 400~~ — actually the drift happens at any batch size after enough chunks
- ~~FP nondeterminism across batch sizes~~ — would affect all GPUs equally
- ~~Cascade attention corruption (`disable_cascade_attn`)~~ — did not help

**Implications for VeRL training:**
- On A100 (SM_80): completely unaffected — A100 uses FlashAttention 2, not TRTLLM. The planned 3×4×A100 async setup is safe.
- On H200/H100 (SM_90): unaffected — TRTLLM not auto-detected at this SM level.
- On B200 (SM_100): set `VLLM_USE_TRTLLM_ATTENTION=0` as a rollout env var in Hydra config or apptainer exec, OR avoid B200 entirely.
- The `max_num_seqs` cap we considered earlier is **not needed** — there was no real batch-size threshold, just the TRTLLM backend.

**Probe pipeline (2026-04-14):** Use `gpu_h200` partition (or `gpu` partition with RTX 5000 Ada) with default `CHUNK_SIZE=50`. Do not use `gpu_b200` unless `VLLM_USE_TRTLLM_ATTENTION=0` is set.

### Bug Fixes in Probe Pipeline (2026-04-12)

1. **`extract_answer()` now uses last `\boxed{}`** — standard practice (MATH, GSM8K eval). Fixes 0.35% of rollouts incorrectly scored 0.0 when model writes `\boxed{}` in both think block and final answer. Changed in `src/phys_reasoner/verifier/extract.py`.

2. **`dump_rollouts.py` early termination** — when model produces no tool call after phase 1, terminate immediately (no forced phase 2). Matches VeRL's `tool_agent_loop.py` line 361: `return AgentState.TERMINATED`. Previously the probe injected a fake `"(no code block)"` tool response and forced phase 2, causing confusion.

3. **`dump_rollouts.sbatch` env vars** — added missing `PYTHONPATH=/opt/phys-extras/`, `TRITON_CACHE_DIR`, `XDG_CACHE_HOME` (matching `dump_rollouts.sh`). Without these, Qwen3.5 model type and triton caching fail on compute nodes.

### Cluster Notes (2026-04-12)

- **`gpu_rtx6000` (rtx_pro_6000_blackwell) incompatible with this SIF** — flash-attn PTX compiled for older SM arch, Blackwell SM_120 not supported. `cudaErrorUnsupportedPtxVersion`. Avoid this partition.
- **CPU-only scoring jobs**: use `day_amd` partition (easier to get than `day`). `devel` has QOS limits if interactive session running. `gpu_b200` requires `--gres=gpu:1` minimum (QOSMinGRES).
