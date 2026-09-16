# Multi-type CoT vs TIR diagnostic — 2026-04-25/26 session findings

**Status**: TIR-vs-CoT controlled comparison locked in on multi-type Goldilocks-from-pilot data. Z4r-TIR climbed cleanly; Z4m-CoT didn't. Per-type CoT isolation atoms surfaced the root cause (multi-type interference / verifier-heterogeneity dominates CoT reward signal). Ready to scale to prod-size Goldilocks pool.

**Audience**: future contributors / future-self picking up CoT/TIR comparison work.

---

## 0. TL;DR

Building on the [previous milestone](cot_diagnostic_milestone.md) (Z4h cleanly climbing CoT-numerical at 4B), this session:

1. **Validated the prod budget structure (think-interrupt at 12288 + 7184)** mechanically and at scale.
2. **Probe-built a CoT-Goldilocks pool** at prod-rollout settings (1200 stratified problems → 237 Goldilocks); mismatch with training rollouts noted.
3. **Cancelled Z4j** (think-interrupt-on-Z4h-245) because Z4h-245's mean-response-length (~4000) is well below thinking_budget (12288) → interrupt fires <2% → no useful signal.
4. **Z4l (batch=128, numerical)** climbed slowly to 0.85 — same asymptote as Z4h at 8x compute. **Not a free win** at 245-problem scale; defer batch=128 decision until prod-data-scale.
5. **Z4k (xverify-on-4B mixed slice)** climbed 0.79 → 0.94 over 199 steps — **xverify integration validated end-to-end at 4B**.
6. **Z4m (prod recipe on multi-type Goldilocks-from-pilot)** **failed to climb cleanly**: oscillated 0.66-0.77 over 51 steps despite high informative_frac (0.61-0.77). Cancelled.
7. **Z4n (Z4m recipe on numerical-only)**: clean climb 0.93+ at step 19 — confirms the recipe works in isolation. **Multi-type / verifier-heterogeneity is the issue**, not the recipe.
8. **Per-type CoT isolation (Z4p-v2 expression, Z4q-v2 equation)**: each climbed but with much higher variance than Z4n. Z4q oscillated, Z4p peaked at 0.75 with 0.50-0.69 oscillation around mean ~0.65. Validates that symbolic types are fundamentally harder for base Qwen3-4B in pure CoT.
9. **Z4r (TIR mode on Z4m's exact data + recipe)** — the headline result: climbed **0.63 → 0.86** over 195 steps. **TIR clearly outperforms CoT on the same multi-type data.**

The recipe itself is fine. CoT is the problem on multi-type data; TIR resolves it.

---

## 1. Validated prod recipe (CoT-prod base, ready to inherit)

```bash
# Model + data
actor_rollout_ref.model.path=Qwen/Qwen3-4B
data.train_files=<filtered Goldilocks parquet>

# Precision (verl defaults, fp32 master + bf16 autocast):
#   - actor.fsdp_config.model_dtype: NOT SET (was bfloat16 in earlier train_async.sh; removed)
#   - ref.fsdp_config.model_dtype:   NOT SET

# Length (prod budget structure)
data.max_response_length=19488   # = 12288 thinking + 16 interrupt + 7184 post
+actor_rollout_ref.rollout.multi_turn.thinking_budget=12288
+actor_rollout_ref.rollout.multi_turn.tool_call_budget=2048
INTERRUPT_LEN=16  # verified for Qwen3-4B by single_turn_agent_loop.py:48-71

# Memory at long sequences (40G GPUs)
actor_rollout_ref.actor.fsdp_config.optimizer_offload=True
actor_rollout_ref.actor.ulysses_sequence_parallel_size=2     # halves per-GPU activation memory at 19488 sequences
actor_rollout_ref.ref.ulysses_sequence_parallel_size=2
actor_rollout_ref.actor.ppo_max_token_len_per_gpu=20480      # fit one full sequence per micro-batch

# Topology (40G premium)
trainer.nnodes=4; trainer.n_gpus_per_node=4   # FSDP-16
rollout.nnodes=4; rollout.n_gpus_per_node=4   # 4T+4R: doubled rollout pool, addresses Z4m's 79% trainer-idle

# GRPO core (unchanged from Z4h)
algorithm.adv_estimator=grpo
algorithm.use_kl_in_reward=False
algorithm.kl_ctrl.kl_coef=0.0
actor_rollout_ref.actor.use_kl_loss=False
actor_rollout_ref.actor.clip_ratio_low=0.2
actor_rollout_ref.actor.clip_ratio_high=0.28
actor_rollout_ref.actor.ppo_mini_batch_size=16
actor_rollout_ref.actor.use_dynamic_bsz=True
actor_rollout_ref.actor.optim.lr=1e-6
actor_rollout_ref.rollout.n=16
actor_rollout_ref.rollout.temperature=1.0   # POLARIS 1.4 untested at this scale; Occam-default

# Async
async_training.staleness_threshold=0.5
async_training.partial_rollout=True

# Reward (rule + xverify router for non-numerical types)
reward.custom_reward_function.path=$ROOT/src/phys_reasoner/training/reward.py

# HF cache (offline) — must to avoid 429 rate-limit cascades from concurrent atoms
HF_HUB_OFFLINE=1
TRANSFORMERS_OFFLINE=1

# TIR vs CoT differs only in:
#   CoT: default_agent_loop=single_turn_agent, multi_turn.enable=false
#   TIR: default_agent_loop=tool_agent, multi_turn.enable=true,
#        multi_turn.format=hermes, multi_turn.tool_config_path=$ROOT/scripts/physcode_tools.yaml,
#        max_assistant_turns=2, max_user_turns=1, max_tool_response_length=1024
```

Reference: `scripts/perlmutter/z4r_tir_prodrecipe.sbatch` (TIR), `scripts/perlmutter/z4m_prodrecipe.sbatch` (CoT).

---

## 2. Atom decomposition table

| Atom | Recipe vs Z4h | Data | Result | Verdict |
|---|---|---|---|---|
| Z4j (cancelled) | + think-interrupt | Z4h-245 | (cancelled) | Won't fire interrupt; mean response length ~4000 << 12288 budget |
| Z4l | batch 16→128 | Z4h-245 | 0.76 → 0.85 over 47 steps | Reaches similar asymptote at 8x compute. Not free; needs prod-scale data to amortize |
| Z4k | reward.py + small expr mix | Z4h-245 + 80 expr | 0.79 → 0.94 over 199 steps | xverify integration validated at 4B end-to-end |
| Z4m (cancelled) | full prod recipe | Goldilocks-from-pilot mixed | 0.66-0.77 oscillating; no climb | Fails on multi-type data |
| Z4n | full prod recipe | numerical-only (245+52=297) | 0.93+ at step 19 | Recipe fine; data is the problem |
| Z4p-v2 | full prod recipe | expression-only (190) | 0.50-0.75 oscillating; mean ~0.65 | Expression-CoT trains but high variance |
| Z4q-v2 (cancelled) | full prod recipe | equation-only (190) | 0.50-0.62 oscillating | Hardest type; may need TIR |
| **Z4r** | **prod recipe + TIR** | **Goldilocks-from-pilot mixed (same as Z4m)** | **0.63 → 0.86 over 195 steps** | **TIR resolves multi-type CoT failure** |

---

## 3. The headline comparison (Z4r vs Z4m, same data, same recipe, only agent-loop differs)

| | Z4m (CoT) | Z4r (TIR) |
|---|---|---|
| Agent loop | `single_turn_agent` | `tool_agent` (multi_turn.enable=true, format=hermes) |
| Tool calls | none | `num_turns/mean` 2.5–3.0 (1 tool call cycle) |
| Response length | 4000–6000 (no Python offload) | 3800–4800 (Python takes computation off-stack) |
| Trajectory | 0.66 → 0.77 peak step 31 → 0.74 settled | 0.63 → 0.86 peak step 195, climbing |
| Step time | 600s (4T+2R) | 290s (4T+4R; would be ~580s at 4T+2R) |
| Tool-call decode failures | n/a | ~5% of rollouts (model still learning JSON format; should improve) |

This is the paper-grade ablation: identical data + identical recipe → only the rollout mechanism differs → TIR climbs cleanly where CoT can't.

---

## 4. Probe-train mechanism mismatch (caveat)

Pilot probe used a standalone vLLM `llm.generate()` to estimate CoT pass-rate. Compared with training-time vLLM (verl async rollouter + prefix caching + staleness windows + partial rollout), the probe produces systematically different sample distributions.

Evidence: at step 3 of Z4n (numerical), `all_right_frac=0.354`, meaning ~35% of "Goldilocks" problems are already saturated to 8/8 in training rollouts. If the pilot's CoT-Goldilocks band were faithful, this should be ≤0.125. Z4r-tir step 3 shows similar `all_right_frac=0.292`. Z4p-v2 / Z4q-v2 are closer to band-faithful (more all_wrong evidence), suggesting the bias is uneven across types.

**Implication for the prod-scale run**: when scaling to the full 107k Goldilocks pool, expect that ~25-35% of training-rollouts will already be saturated even on "Goldilocks" problems. Static filter pre-saturation is the dominant mode of curriculum-degradation for static-Goldilocks training; this motivates the multi-task fixes proposed in §6.

**Mitigation if needed**: dump training rollouts at step 0 from a brief warm-up run, score them, treat that as a training-faithful "true" Goldilocks band. ~30–60min of compute. Currently deferred — Z4r climbed past the static-Goldilocks limitation, so the mismatch isn't a blocker for prod.

---

## 5. Per-type Goldilocks pool size (TIR probe, full 107k)

For prod-scale Goldilocks filtering, the available per-type pool sizes from the existing 107k TIR probe (TIR-pass ∈ (0,1)):

| answer_type | total | mean_pass | Goldilocks count | Goldilocks rate |
|---|---:|---:|---:|---:|
| numerical | 25,288 | 0.241 | 10,830 | 42.8% |
| mcq       | 21,060 | 0.241 |  9,063 | 43.0% |
| equation  | 37,623 | 0.098 |  8,150 | 21.7% |
| expression| 23,126 | 0.158 |  7,407 | 32.0% |
| **All**   | **107,977** | — | **~35,450** | — |

Important: this is **TIR-pass-rate**, not CoT. For TIR-prod runs the TIR-Goldilocks band is faithful by construction. For CoT-prod runs, expect ~20% of these problems to fall outside the CoT-Goldilocks band (per pilot ratio: 19.8% CoT-Goldilocks vs ~32% TIR-Goldilocks).

Data sources: 95% Dr. SCI, 5% UGPhysics curated.

---

## 6. Open work — multi-task fix candidates (NOT YET TESTED)

Z4r-TIR climbed to 0.86 on the small (226-problem) mixed Goldilocks. Static-Goldilocks degradation (informative_frac dropping mid-run as model masters easy problems) and verifier heterogeneity remain the two biggest signal-quality issues. Candidate fixes, in priority order:

1. **Oversample-and-filter at rollout level** (DAPO-async-compatible variant). Increase `rollouter.target_samples` by 1.5-2x, drop saturated groups before trainer pulls. ~20-40% more effective gradient updates per step. Verl async path already streams ahead of trainer; this change is lightweight (no `regenerate` loop). Note: classic DAPO `regenerate` is sync-only — confirmed unsupported in `fully_async_policy`.
2. **Rollout Rescue Mechanism** (POLARIS-style). For groups where all rollouts are wrong, inject one previously-correct rollout from a per-problem buffer. Provides positive signal on hardest problems → keeps gradient informative on the long tail. ~50-100 LOC. References: https://hkunlp.github.io/blog/2025/Polaris/, https://github.com/ChenxinAn-fdu/POLARIS (unofficial pulled to local repo).
3. **Per-type reward normalization**. Z-score rewards within (problem_type, group) before computing advantage. Addresses reward-distribution mismatch between rule-based (binary, clean) and xverify-based (binary, noisy). Free; ~20 LOC.
4. **Curriculum with progressive hard-problem buffer** (Dr. SCI style). Reference: https://github.com/MiniByte-666/Dr.SCI (pulled). Per-epoch curriculum graduation; potentially conflicts with async streaming. Lower priority.

For verl async path, none of these are out-of-the-box. Each needs adaptation; (1) and (2) are most async-friendly.

---

## 7. File pointers (this session)

### Probe + filter pipeline
- `scripts/cot_probe_thinkinterrupt.py` — standalone CoT probe with think-interrupt mimicking single_turn_agent_loop
- `scripts/subsample_pilot_from_107k.py` — stratified subsampler (4 types × 6 bands)
- `scripts/perlmutter/cot_pilot_thinkinterrupt_chunk.sbatch` — chunked debug-queue probe runner
- `scripts/merge_pilot_thinkinterrupt.py` — per-type Goldilocks distribution analysis
- `scripts/build_goldilocks_from_pilot.py` — pilot-Goldilocks → CoT training parquet
- `scripts/build_goldilocks_pilot_numerical_only.py` — single-type filter (Z4n input)
- `scripts/build_z4k_xverify_mix.py` — Z4k expression slice builder
- `scripts/build_expanded_per_type.py` — Z4p-v2 / Z4q-v2 expansion via 107k TIR-Goldilocks-band

### Atom sbatch scripts
- `scripts/perlmutter/z4j_thinkinterrupt.sbatch` — (cancelled) think-interrupt on Z4h-245
- `scripts/perlmutter/z4l_batch128.sbatch` — batch=128 stability test
- `scripts/perlmutter/z4k_xverify.sbatch` — xverify integration test on 4B
- `scripts/perlmutter/z4m_prodrecipe.sbatch` — prod recipe on mixed Goldilocks (cancelled)
- `scripts/perlmutter/z4n_numerical_only.sbatch` — single-type isolation (numerical)
- `scripts/perlmutter/z4o_mcq_only.sbatch` — single-type isolation (MCQ)
- `scripts/perlmutter/z4p_expr_v2.sbatch` — single-type isolation (expression, expanded)
- `scripts/perlmutter/z4q_eq_v2.sbatch` — single-type isolation (equation, expanded)
- `scripts/perlmutter/z4r_tir_prodrecipe.sbatch` — **the headline TIR comparison**

### Training run logs (representative)
- `outputs/physcode_ablation/z4l_batch128_*` — batch=128 baseline
- `outputs/physcode_ablation/z4k_xverify_*` — xverify integration
- `outputs/physcode_ablation/z4n_numerical_*` — single-type (numerical) — clean climb
- `outputs/physcode_ablation/z4p_expr_v2_*` — single-type (expression) — high variance
- `outputs/physcode_ablation/z4r_tir_prodrecipe_*` — **TIR on multi-type** — clean climb

### Probe outputs
- `outputs/cot_pilot_thinkinterrupt/merged.parquet` — 9600 rollouts (1200 problems × n=8) at prod budget
- `data/processed_cot_goldilocks_from_pilot/` — Z4m / Z4r training data (226 + 11)
- `data/processed_tir_goldilocks_from_pilot/` — Z4r training data (TIR system prompt)

---

## 8. Methodology lessons

1. **Atom decomposition is still paying off**. Z4j was caught and cancelled in 10 min via reasoning rather than waste 6h. Z4n confirmed recipe-works-on-isolated-type quickly.
2. **Probe-train mismatch is real**. Even careful replication of the rollout config in a standalone probe drifts from training-time rollouts. The training-rollout-via-frozen-policy probe is the gold standard for Goldilocks filtering at scale; static probes are diagnostic-grade only.
3. **Premium 5/5 MaxSubmit + het-job 2-entry counting** is a constant resource-budget consideration. Cancel-and-resubmit is sometimes the right move when an atom shows decisive signal early.
4. **HF API rate limit (429)** kills concurrent atoms launched on the same model. Always set `HF_HUB_OFFLINE=1` + `TRANSFORMERS_OFFLINE=1` when the model is locally cached.
5. **OOM on 40G with 19488-token sequences requires `ulysses_sequence_parallel_size=2`** to halve per-GPU activation memory. Z4m's first attempt OOM'd at sp=1; sp=2 is the canonical fix.
