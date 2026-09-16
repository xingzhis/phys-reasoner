# CoT Diagnostic Milestone — 2026-04-24

**Status**: CoT-arm prod-bug investigation closed. Recipe validated at 4B. Ready for scaling and TIR-arm work.

**Audience**: future contributors (incl. future-self) re-entering this codebase. Read top-to-bottom; each section is self-contained but builds on the previous.

---

## 0. TL;DR

Production CoT GRPO at Qwen3-4B was plateauing at reward ~0.62 with grad_norm ~0.05, ppo_kl ~1.3e-3 — across weeks of hparam sweeping. We decomposed the stack into **single-knob atomic experiments** anchored on a known-good 0.5B Z2 baseline, ran ~15 atoms, and identified **three independent prod-blockers**:

1. **`fsdp_config.model_dtype=bfloat16`** stored Adam master weights in bf16 → optimizer precision loss → plateau. Fix: leave it unset (verl default fp32 master + bf16 autocast).
2. **`max_response_length=2048`** truncated 60-80% of base 4B's CoT rollouts → most rollouts gave 0 reward → no signal. Fix: `max_response_length=8192`.
3. **Data difficulty mismatch** — TIR-probe-based filter (used to build Z2's "easy" subset) doesn't transfer to CoT mode → 65% of GRPO groups had 0/16 correct → no informative gradient. Fix: re-probe with multi-rollout CoT and select Goldilocks pass@8 ∈ (0, 1).

**Validated recipe** (Z4h, climbing from 0.70 → 0.87 in 18 training steps): `scripts/perlmutter/z4h_goldilocks.sbatch`.

Throughput knobs (`optimizer_offload=True`, `staleness_threshold=1.0`, `rollout.n=8`, `temperature=1.4`) all individually validated safe at 0.5B. Compose them for prod scale-up.

---

## 1. Strategy

### 1.1 Atom decomposition

The prod stack has dozens of knobs. Hyperparameter sweeps don't decompose them — a swept run combines many changes and a flat result tells you nothing about which knob mattered. Instead:

- **Pick a known-good baseline** that climbs cleanly. We used Z2: `Qwen2.5-0.5B-Instruct` + verl async + filtered numerical physics + rule-based reward. 0.044 → 0.206 reward in 50 training steps.
- **For each suspect prod knob, run a single-knob ablation**: one configuration setting changed from Z2, everything else identical.
- **Naming**: `Z2x` = Z2 + change x (e.g. `Z2b` = +bf16 master). `Z4y` = the same swap propagated to 4B + Z4-specific topology fixes.

### 1.2 Reading the metrics

What looks like "stuck training" can be three different failure modes:

| Symptom | Diagnostic | Example here |
|---|---|---|
| `actor/grad_norm` ~0, `ppo_kl` ~0 | Optimizer not updating | not seen, would mean precision broken |
| `critic/group/all_wrong_frac` high | Data too hard, no GRPO signal | Z4f-v2 (65%) |
| `critic/group/all_right_frac` high | Data too easy, no GRPO signal | Z4g (80%) |
| `response_length/clip_ratio` high | Length truncating before answer | Z4 (85%) |
| `informative_frac` low + reward flat | Goldilocks failure (any of above) | Z4f-v2, Z4g |
| reward+grad_norm dropping smoothly | Working — just slow | Z4h (real) |

The `critic/group/informative_frac` metric (we patched it in via ZVF/group-saturation) is the single most useful diagnostic — it tells you what fraction of GRPO groups have any gradient signal at all. Below ~30%, learning is anemic.

---

## 2. Findings per axis

### 2.1 Precision

Three atoms:

| Atom | Knob change | Reward gstep=199 | Verdict |
|---|---|---|---|
| Z2 | (verl defaults: fp32 master + bf16 autocast) | 0.206 | ✓ baseline |
| Z2b | `actor.fsdp_config.model_dtype=bfloat16` | **0.102** | ✗ plateau |
| Z2c | Z2b + `model.use_fused_kernels=True` | **0.097** | ✗ plateau, fk doesn't fix |

**Finding**: `fsdp_config.model_dtype=bfloat16` stores **master weights** in bf16, losing Adam optimizer-state precision. NOT the same as bf16 autocast (which Z2 already uses by default and is fine). The CLAUDE.md note from 2026-04-22 about `use_fused_kernels` fixing bf16 was at best partial — it doesn't restore learning at 0.5B with our async stack.

**Confound elimination**: prod has `model_dtype=bfloat16` set in `train_async.sh`. Removing this is the single biggest prod fix.

We also explored fp16 end-to-end (sail-sg Precision-RL recipe). The patch already lives in our verl's `dp_actor.py` (legacy worker path) but the engine path used by `fully_async_policy` has hardcoded bf16 autocast and no `ShardedGradScaler`. Porting fp16 to the engine path is unfinished work — we shelved it after confirming Z2's defaults (fp32 master + bf16 autocast) climb cleanly. fp16 e2e is a future optimization, not a blocker.

### 2.2 Length

Two probes + one atom:

| Probe | Setting | Result |
|---|---|---|
| Single-rollout CoT probe at max_tokens=4096 | base Qwen3-4B, temp=1.0, n=1 | 60% truncated at 4096 → length-bound |
| n=8 chunked CoT probe at max_tokens=4096 | same, n=8 | mean pass@8 = 0.19 across 1724 problems |

| Atom | `max_response_length` | clip_ratio at gstep=3 | Verdict |
|---|---|---|---|
| Z4 (orig) | 2048 | 85% | severe |
| Z4f / Z4f-v2 / Z4g / Z4h | 8192 | 3-33% (data-dep) | acceptable |

**Finding**: base Qwen3-4B in CoT mode (no tool delegation) emits ~5000-token reasoning on these physics problems. `max_response_length=2048` truncates ~80% before `\boxed{answer}` → silent zero-reward. Base 4B's natural CoT length isn't model verbosity at start of training — it's intrinsic difficulty of inline math without tool help. (Distinct from TIR mode where the model offloads computation to Python and emits much shorter reasoning.)

**Note**: prod uses `Qwen3-4B-Thinking-2507` and reportedly rare-truncates. We used plain `Qwen/Qwen3-4B` to match prod's later switch but observed it benefits significantly from longer budget.

**Open**: at `max_response_length=8192`, Z4h still has 5-8% clip. Real prod might need 12K-16K. Open follow-up.

### 2.3 Data difficulty (Goldilocks)

The most subtle finding. Z2's "easy" filter was built from a **TIR probe** (with tool support) — pass_rate ≥ 0.75. In CoT mode (no tool), the same problems are dramatically harder. Three atoms:

| Atom | Data filter | Final reward | all_wrong_frac | informative_frac |
|---|---|---|---|---|
| Z4f-v2 | Z2's full 1724 (TIR-pass≥0.75) | flat ~0.30 | 65% | 17% |
| Z4g | pass@1=1 from CoT probe (329) | ceiling ~0.97 | 0% | 20% (all-right) |
| **Z4h** | **pass@8 ∈ (0, 1) from n=8 CoT probe (245)** | **0.70 → 0.82+** | 0-2% | **53-67%** |

**Finding**: Goldilocks band — problems where the base policy succeeds at least once and fails at least once across n=8 rollouts — gives maximal GRPO signal. Boundary filters (pass≥threshold or pass<threshold) miss it.

**Probe code path mismatch caveat**: our CoT probe used `max_tokens=4096`, training uses 8192. So probe pass rates are slightly biased downward (problems that probe truncated would solve in training). The Goldilocks filter is approximate-correct but tends to over-include some problems that will saturate. Acceptable as v1; for prod-aligned, re-probe at max_tokens=8192 + temp=1.4.

**Implementation**:
- Probe: `scripts/cot_probe_base_4b.py` (supports `--n_rollouts`, `--start_idx/--end_idx` for chunking)
- Sbatch: `scripts/perlmutter/cot_probe_4b_z2_n8_chunk.sbatch` (debug qos, 4 chunks of 431 problems × n=8)
- Merge: `scripts/merge_cot_probe_n8.py`
- Filter: `scripts/build_z4h_goldilocks.py`

### 2.4 Throughput knobs

Each tested as Z2-single-knob ablation (all on debug qos, ~30 min, ~50 training steps):

| Atom | Knob change | Final reward (last 5 avg) | Verdict |
|---|---|---|---|
| Z2g | `optimizer_offload=False → True` | 0.265 | ✓ best |
| Z2h | `staleness_threshold=0.5 → 1.0` | 0.218 | ✓ safe |
| Z2i | `rollout.n=16 → 8` | 0.211 | ✓ safe |
| Z2j | n=8 + staleness=1 (combined) | 0.232 | ✓ safe |

All four throughput knobs validated. Compose freely for prod scaling.

### 2.5 Anecdotal / literature-borrowed knobs

Knobs we ship from external recipes, not throughput-driven. Validated cleanly enough to keep but worth tracking as "could-revisit" choices.

| Atom | Knob | Source | Final reward (last 5 avg) | Verdict |
|---|---|---|---|---|
| Z2k | `rollout.temperature=1.0 → 1.4` | POLARIS (Qwen3-4B blog) | 0.181 | ✓ slightly noisier but in band |
| (n/a) | `thinking_budget=12288` (TIR/CoT-with-interrupt) | ScaleRL anecdote | not yet ablated | — |
| (n/a) | total response budget split (think + interrupt + tool_call + tool_response + answer) | mix of POLARIS + ScaleRL + our prod tuning | not yet ablated | — |

These are choices we adopted from external work without a controlled measurement. They're "harmless-enough" defaults but each is a candidate for ablation if the prod-scale recipe under-performs.

### 2.6 Async/sync infrastructure

- **Z2 itself uses `fully_async_policy`** with `staleness_threshold=0.5`. It climbs. So async itself is not broken.
- **xverify-vLLM batched server (`scripts/serve_xverify_vllm.py`)** verified by Z3-v2: clean climb to 0.308 on expression data. The serial HF server (`scripts/serve_xverify.py` from `serve_xverify_perlmutter.sbatch`) gets overwhelmed at scale. Always use the vLLM variant for prod.
- **Engine path vs legacy `dp_actor`**: our async stack uses `verl/workers/engine/fsdp/transformer_impl.py` which lacks the sail-sg fp16 patch. Legacy `dp_actor` has it. If we ever need fp16, port the patch to the engine path.

---

## 3. Validated recipe

The Z4h sbatch (`scripts/perlmutter/z4h_goldilocks.sbatch`) is the canonical 4B recipe. Key settings:

```bash
# Topology: 4 trainer nodes × 4 GPUs (FSDP-16) + 2 rollout nodes × 4 GPUs
trainer.nnodes=4
trainer.n_gpus_per_node=4
rollout.nnodes=2
rollout.n_gpus_per_node=4

# Model
actor_rollout_ref.model.path=Qwen/Qwen3-4B
actor_rollout_ref.model.use_remove_padding=True
actor_rollout_ref.model.enable_gradient_checkpointing=True

# Precision (DELIBERATELY UNSET)
# - actor.fsdp_config.model_dtype: NOT SET → fp32 master weights
# - actor.fsdp_config.dtype: NOT SET → bf16 autocast (verl default)
# DO NOT set model_dtype=bfloat16 — that breaks training.

# Memory at 4B
actor_rollout_ref.actor.fsdp_config.strategy=fsdp2
actor_rollout_ref.actor.fsdp_config.fsdp_size=-1   # full FSDP-16 sharding
actor_rollout_ref.actor.fsdp_config.optimizer_offload=True   # Adam state on CPU
actor_rollout_ref.actor.ulysses_sequence_parallel_size=1

# Length (critical at 4B)
data.max_response_length=8192
data.max_prompt_length=1024
actor_rollout_ref.rollout.max_model_len=9216
actor_rollout_ref.rollout.max_num_batched_tokens=9216
actor_rollout_ref.actor.ppo_max_token_len_per_gpu=12288

# GRPO core
algorithm.adv_estimator=grpo
algorithm.norm_adv_by_std_in_grpo=True
algorithm.use_kl_in_reward=False
algorithm.kl_ctrl.kl_coef=0.0
actor_rollout_ref.actor.use_kl_loss=False
actor_rollout_ref.actor.clip_ratio_low=0.2
actor_rollout_ref.actor.clip_ratio_high=0.28
actor_rollout_ref.actor.ppo_mini_batch_size=16   # small batch; consider 128 for prod
actor_rollout_ref.actor.use_dynamic_bsz=True
actor_rollout_ref.actor.ppo_epochs=1
actor_rollout_ref.actor.optim.lr=1e-6

# Async
async_training.staleness_threshold=0.5   # safe to crank to 1.0 for throughput
async_training.trigger_parameter_sync_step=4
async_training.require_batches=1
async_training.partial_rollout=True

# Rollout
actor_rollout_ref.rollout.name=vllm
actor_rollout_ref.rollout.mode=async
actor_rollout_ref.rollout.n=16   # safe to drop to 8 for throughput
actor_rollout_ref.rollout.temperature=1.0   # 1.4 (POLARIS anecdote) tested safe; choose by training-effect intent, not throughput
actor_rollout_ref.rollout.gpu_memory_utilization=0.7
actor_rollout_ref.rollout.calculate_log_probs=True

# Data (Goldilocks-filtered)
data.train_files=$ROOT/data/processed_cot_z4h_goldilocks/data/train.parquet

# Reward (rule-based for numerical-only)
reward.custom_reward_function.path=$ROOT/src/phys_reasoner/training/reward_polaris.py
```

---

## 4. Open work items (next session)

Mapped to the four axes the user identified:

### 4.1 Data scope (highest priority)

- [ ] Re-probe at `max_tokens=8192, temp=1.4` to align with prod-intent
- [ ] Run probe on full multi-type Dr. SCI dataset (not just numerical-easy)
- [ ] Build Goldilocks filter on full dataset
- [ ] Add xverify path: route expression-type problems to xverify-vllm (server validated by Z3-v2). Use `serve_xverify_vllm_40g.sbatch`.
- [ ] Z4h-prod sbatch: bigger Goldilocks pool + xverify + multi-type → expected larger climb headroom

### 4.2 Harness

- [ ] **Think-interrupt in CoT** — prod uses it, we don't. Single-knob ablation: Z4h + think_interrupt vs Z4h. Need to check whether `single_turn_agent_loop` supports it or if it's only in `tool_agent_loop`. May need to port from TIR side.
- [ ] **TIR arm**: Z4h equivalent in TIR mode. Probe driven by `tool_agent_loop` directly (no rollout-code mismatch). All TIR-specific knobs (thinking_budget, tool_call_budget, answer_budget, max_assistant_turns, sandbox timeout) re-validated.
- [ ] **Larger total budget**: 12K, 16K — let model use longer reasoning if needed.

### 4.3 Speed (compose validated knobs)

- [ ] Submit Z4h-prod with `optimizer_offload=True, staleness=1.0, n=8, temp=1.4` simultaneously
- [ ] Increase `ppo_mini_batch_size=16 → 128` to match prod (need more rollout nodes proportionally)
- [ ] Ablate `ulysses_sequence_parallel_size` at 4B (currently SP=1; SP=2 may give activation memory savings)
- [ ] Ablate `tensor_model_parallel_size` for rollout (currently TP=1)
- [ ] Add rollout nodes: Z4h's bottleneck is roughly balanced gen/update at 47s/60s per step; more rollout nodes mostly hides gen behind update.

### 4.4 Algorithm anecdotes (literature-borrowed defaults to revisit)

These are knobs we ship by default from external work. None are throughput-driven; each is a candidate ablation if the prod-scale recipe under-performs.

- [ ] **ZVF ablation** — we ship our patched verl with `filter_groups.metric=acc + zero-variance mask`. Never ablated. Z2 + ZVF-on/off comparison at 0.5B.
- [ ] **temp=1.4** (POLARIS Qwen3-4B blog) — Z2k showed slightly noisier than temp=1.0 but in band. The training-effect rationale isn't measured at our scale; could be optimal or suboptimal.
- [ ] **thinking_budget=12288** (ScaleRL anecdote) — reused by us for TIR-mode interrupt. Untested at 4B for our data.
- [ ] **DAPO-style filter_groups regenerate** for async — would replace static Goldilocks with dynamic curriculum (regenerate batches that have no informative groups). Currently only ZVF-mask is wired in async; DAPO regenerate is sync-only. **Plausibly novel contribution.**
- [ ] **Hard-pool replay** — keep a buffer of past problems where the model recently failed; resample more often. Cheaper than full DAPO.

### 4.5 Other observations carried forward

- The 0.5B + xverify + expression ablation **already worked** (Z3-v2 climbed to 0.308). Don't re-prove this. Use it as second baseline alongside Z2.
- Prod's `train_async.sh` defaults: bf16 master ✗ (broken), optimizer_offload=True ✓, staleness=0.5 (can be 1.0), TRAIN_SP=2 (not ablated). Strip the bf16 master and run.

---

## 5. Curriculum innovation candidate

Static Goldilocks degrades over training: `informative_frac` drifted 67% → 53% → 42% in Z4h's first 18 training steps as the model mastered the band. Long-run training will exhaust the static set.

**Dynamic Goldilocks proposal**:

1. Maintain a problem pool with current pass-rate estimates (online updates from training rollouts).
2. Each training step, sample problems weighted toward Goldilocks-band (pass ∈ (low, high)).
3. As the model improves, automatically shifts to harder problems (pass-rate drops on previously-easy problems).
4. Optionally retire problems where pass-rate stays at 1.0 across many evaluations.

Implementation requires:
- Tracking per-problem rollout history in async (we have `count/staleness_samples` infra; extending is plausible)
- Sampler hooks in `verl/utils/dataset/rl_dataset.py`
- DAPO's `filter_groups.regenerate` is the synchronous version of this idea

This would be a useful extension of the verl async path AND directly improves the asymptote of training. Worth proposing as a research contribution.

---

## 6. File pointers (for reproducibility)

### Data
- `data/processed_cot_z2_numerical_easy/` — Z2's TIR-derived "easy" filter (1724 problems). Used by Z4f-v2.
- `data/processed_cot_z4d_numerical_short/` — TIR-length-filtered subset (didn't help).
- `data/processed_cot_z4g_cot_solvable/` — pass@1=1 subset (329 problems, too easy).
- `data/processed_cot_z4h_goldilocks/` — pass@8 ∈ (0, 1) subset (245 problems). **Use this.**

### Probe outputs
- `outputs/probe_qwen3_4b_v2_107k/rollouts_scored_trainmatched_107k.parquet` — TIR probe (NOT a CoT proxy).
- `outputs/cot_probe_4b_z2/rollouts_pass1.parquet` — single-rollout CoT probe (deprecated).
- `outputs/cot_probe_4b_z2_n8/rollouts_pass8_all.parquet` — n=8 CoT probe rollouts.
- `outputs/cot_probe_4b_z2_n8/per_problem_pass8.parquet` — per-problem aggregate. **Use this for Goldilocks.**

### Training-run logs (representative)
- `outputs/physcode_ablation/z2_verl_e2e_physics_*` — Z2 baseline
- `outputs/physcode_ablation/z2b_verl_e2e_bf16_*` — bf16 master plateau
- `outputs/physcode_ablation/z4f_verl_e2e_z2recipe_20260424.152836/` — Z4 too-hard data
- `outputs/physcode_ablation/z4g_cot_solvable_*` — Z4 too-easy data
- `outputs/physcode_ablation/z4h_goldilocks_*` — **Validated 4B recipe**

### Sbatch files
- `scripts/perlmutter/z2_verl_e2e_physics.sbatch` — Z2 baseline
- `scripts/perlmutter/z4h_goldilocks.sbatch` — **Use this as 4B template**
- `scripts/perlmutter/z2{b,c,g,h,i,j,k}_*.sbatch` — Z2 ablations (precision + throughput)
- `scripts/perlmutter/serve_xverify_vllm_40g.sbatch` — **Use this xverify, NOT serve_xverify_perlmutter**
- `scripts/perlmutter/cot_probe_4b_z2_n8_chunk.sbatch` — chunked Goldilocks probe template

### Reward functions
- `src/phys_reasoner/training/reward_polaris.py` — minimal boxed-match (rule-based, all answer types fall through to string equality)
- `src/phys_reasoner/training/reward.py` — production rule + xverify-fallback router

---

## 7. Lessons / methodology notes for next time

1. **Atom-decomposition pays.** Each Z2 ablation took 18-30 min on debug qos. Together they answered "is bf16 the bug? is offload the bug? is staleness the bug?" with crisp yes/no. A single sweep with all knobs flipped tells you nothing.

2. **The right baseline matters more than the right experiment.** Z2 climbing cleanly was the anchor. Without it we'd have been chasing ghosts at 4B forever.

3. **Probe-vs-train code path mismatch is real and easy to miss.** Our TIR-probe-based filter didn't transfer to CoT training. Always probe in the exact rollout code path you'll train with — or use `verl`'s existing rollout machinery to generate the probe.

4. **`informative_frac` is the leading indicator.** Reward is the lagging indicator. If informative_frac < 30%, no amount of training fixes it — the data filter is wrong. We'd have saved a week of fp16 patching if we'd watched this metric earlier.

5. **The CLAUDE.md auto-memory note that bf16 was "fixed by use_fused_kernels"** turned out to be at-best-partial. Future-self: trust the auto-memory but verify with a single-knob ablation when it matters.

6. **Optional knobs are ranked-ordered for prod**: most-impactful first.
   - Strip `model_dtype=bfloat16` from `train_async.sh` (huge)
   - Bump `max_response_length=2048 → 8192` (huge)
   - Goldilocks-filter the data (huge for early-step signal)
   - Offload + n=8 + staleness=1 (compose for throughput; all measured-safe)
   - temp=1.4 (POLARIS anecdote) and 12k thinking_budget (ScaleRL anecdote) — borrowed defaults, not throughput; revisit if prod under-performs
   - Think-interrupt for CoT (small; prod has it; not yet ablated)
   - DAPO-style dynamic curriculum (research, large potential)
