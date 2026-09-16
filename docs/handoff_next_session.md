# Handoff — next session pointers

Focus: data-difficulty filtering intervention + optional entropy/temperature
tuning, building on the saturation diagnostics from 2026-04-21.

## What this session concluded

**Training was saturated by data difficulty, not policy sharpness (under test).**
Runs stopped at step_150 on both TIR and CoT, checkpoints preserved
(`latest_checkpointed_iteration.txt=150`). Full diagnostic report landed in
`outputs/saturation_analysis/*/saturation.json` + chat log.

Key numbers to remember (n=64 prompts, n_rollouts=8, temp=1.0 unless noted):

| tag | mean_r | sat_frac | all_wrong | info |
|---|---|---|---|---|
| tir_step50  | 0.508 | 0.594 | 0.312 | 0.406 |
| tir_step100 | 0.477 | 0.578 | 0.312 | 0.422 |
| cot_step50  | 0.537 | 0.641 | 0.281 | 0.359 |
| cot_step100 | 0.551 | 0.594 | 0.281 | 0.406 |

Per-source (step_100, TIR): Dr. SCI mean=0.541 / all_wrong=23.1%, **UGPhysics
mean=0.136 / all_wrong=72.7%**. UGPhysics drives the oversaturation —
never passed through a base-model-difficulty filter.

Persistent all-wrong intersection (step_50 ∩ step_100): 9 Dr.SCI + 8 UGPhysics
per run = 17/64 prompts (≈27% of sample) = target set to drop / down-weight.

Actor trajectory at step_100 (both runs): `ppo_kl ≈ 7e-4`, `grad_norm ≈ 0.02`,
`pg_clipfrac ≈ 5e-4`, `lr = 1e-6`. Phase-1 reward rise 0.49→0.58 in first
50 steps, then plateau — not stuck, data-bounded.

## Perlmutter-proven sbatch jobs (bake these in, don't rewrite)

### Training

| path | purpose |
|---|---|
| `scripts/perlmutter/prod_tir_het_4t_sharedxv.sbatch` | 2-het-group (4 trainer + 6 rollout) TIR prod on premium QoS, reads shared xVerify URL from `outputs/xverify_endpoints/current.url`. **This is the proven topology** — 10 nodes total, fits within premium 5-job cap (1 xverify + 2 het-groups × 2 runs = 5). |
| `scripts/perlmutter/prod_cot_het_4t_sharedxv.sbatch` | Mirror of the TIR prod, flips COT_BASELINE=1. Uses `data/processed_cot/data/train.parquet`. |
| `scripts/perlmutter/serve_xverify_perlmutter.sbatch` | 1-GPU premium xVerify server. Writes URL to `outputs/xverify_endpoints/current.url`. **Submit this first** and wait ~10 min for it to load + advertise before submitting the prods. |
| `scripts/train_async.sh` | Canonical FullyAsyncTrainer driver. `TOTAL_EPOCHS` and `TOTAL_STEPS` env vars both required — verl caps at `min(TOTAL_STEPS*BATCH, len(dataloader)*TOTAL_EPOCHS)`. Currently 552 steps / 3 epochs. |
| `scripts/perlmutter/_ray_bringup.sh` | Ray head + worker bringup for het-group topologies. Supports `TRAIN_WORKER_NODES` for FSDP-16 (4 trainer nodes × 4 GPU). |

### Diagnostic (this session)

| path | purpose |
|---|---|
| `scripts/run_saturation.sh` | Per-checkpoint saturation diagnostic: merge FSDP→HF, sample 64 prompts, n=8 rollouts, score via HTTP xVerify, analyze. Accepts `CKPT_DIR` (FSDP actor dir) OR an HF model name (skips merge). `TEMPERATURE` env overridable. |
| `scripts/run_saturation_all.sh` | 4-GPU parallel orchestrator: fires step_50/100/150 × {TIR,CoT} using slot reuse. **Known bug fixed** in `kill -0` polling for cross-subshell GPU-free barrier (prior version used `wait` which doesn't cross subshells → OOM crashes). |
| `scripts/run_saturation_chunked.sh` | Chunked variant. `TEMP=1.4 bash scripts/run_saturation_chunked.sh` splits 64 prompts across 2 GPUs per combo (4 GPUs total for TIR+CoT). ~2× faster rollout than monolithic. |
| `scripts/run_saturation_temp_test.sh` | One-shot temperature sweep on ckpt_0 (base model). Runs TIR/CoT × {temp=1.0, temp=1.3} in parallel on 4 GPUs. |
| `scripts/analyze_saturation.py` | Group saturation stats from `scored.parquet`: sat_frac + Wilson CI + per-answer-type histogram. |
| `eval/inference/rollout.py` / `rollout.sh` | Canonical rollout harness (TIR + CoT modes). Supports `--start_idx/--end_idx` for cross-GPU sharding. `problem_idx` in outputs uses **original pre-slice indices** so concat across chunks gives unique keys. |
| `eval/scoring/our_verifier.py` | Scores a rollouts parquet via `src/phys_reasoner/verifier/router.py`. **New in this session:** `--xverify_url URL` to use the HTTP client instead of loading xVerify locally. |

## Key quirks and lessons (don't re-discover)

### NERSC / SLURM

- **Premium QoS cap = 5 jobs per user**, het-group components count separately. 1 xverify + 2 het-groups × 2 runs = exactly 5. No headroom — diagnostic jobs must go through `-q shared` (1-GPU, requires `--cpus-per-task=32` per GPU).
- **Interactive QoS = 4h max, 2 submits/user, 1–4 nodes, no queue.** `salloc --nodes=N --qos=interactive --time=04:00:00 --constraint=gpu --account=m2651`. Can run a batch command via `salloc ... srun ... bash -c '...'`.
- **`-q interactive` is salloc-only**, sbatch rejects it.
- **Account is `m2651`** (SLURM auto-appends `_g` for GPU → `m2651_g`). Either works.

### Container / overlay

- Always set `PYTHONNOUSERSITE=1` + `PYTHONPATH=/opt/phys-extras/` + `HOME=/tmp`.
- `--env "HOME=/tmp"` is CRITICAL for vLLM — without it, vLLM tries to write config to `/global/homes/.../.config` (read-only from inside container). Non-fatal but noisy OSError in logs.
- `--env "FLASHINFER_WORKSPACE_BASE=/tmp"` needed for flashinfer (transitively imported by Megatron).
- `PYTHONPATH` should also include `$ROOT/.async-extras/` for numpy 2.x (cupy ABI dependency).

### Async training

- `trainer.total_epochs=1` hardcode was a bug — must pass `TOTAL_EPOCHS` env to `train_async.sh` for multi-epoch runs. Fixed 2026-04-21.
- `disable_custom_all_reduce=True` via `+actor_rollout_ref.rollout.engine_kwargs.vllm.disable_custom_all_reduce=True` is REQUIRED on Perlmutter A100 — env var `VLLM_USE_CUSTOM_ALL_REDUCE=0` is NOT recognized by vLLM 0.17.
- NCCL envs: `NCCL_IB_TIMEOUT=32`, `NCCL_IBEXT_DISABLE=1`, `TORCH_NCCL_ENABLE_MONITORING=0`.
- `NCCL_NVLS_ENABLE=1` was suspected causing crashes but removing it DID NOT fix the issue — the real bug was custom_all_reduce. Leave NCCL_NVLS alone.
- xVerify URL must be in `outputs/xverify_endpoints/current.url` — Ray remote rollout/reward workers do NOT inherit driver env vars, so they fall back to reading this file via `reward.py`. Early bug with `shared.url` crashed because the fallback path is hardcoded to `current.url`.
- Module-global `_XVERIFY_CLIENT` cache in `src/phys_reasoner/training/reward.py:61-95` — pins URL per Ray worker. Flush cache if you rotate xVerify servers.

### Verifier / reward

- `src/phys_reasoner/verifier/router.py::verify_answer` returns verdict ∈ {−1, 0, 1}. **Training treats −1 (unverifiable) as 0.0** (`src/phys_reasoner/training/reward.py:144-145`). Our saturation analysis also maps unverifiable → 0 → same reward space.
- xVerify HTTP server at `scripts/serve_xverify.py`. Health endpoint: `{URL%/judge}/health`.

### GRPO group-saturation metrics (NEW this session)

- **Patch already applied to `verl/verl/trainer/ppo/metric_utils.py`** (commit on local `physcode` branch of the verl submodule — may not be pushed yet; see `docs/patches/verl_grpo_saturation_metrics.patch` for standalone).
- New wandb keys on the next training run:
  - `critic/rewards/std`, `critic/advantages/std`, `critic/score/std`
  - `critic/group/{n, saturated_frac, all_right_frac, all_wrong_frac, informative_frac}`
- Gated on `uid` being in `batch.non_tensor_batch` so non-GRPO runs are unaffected.
- Flows automatically through `FullyAsyncTrainer._fit_collect_metrics` → `MetricsAggregator` → wandb. No config changes needed.
- **Watch `saturated_frac` in the first 20 steps** — if it's already >0.5, the intervention (data filter) wasn't aggressive enough.

## Planned interventions for next run

Priority order based on session findings:

1. **Filter UGPhysics + curated pool by 4B base-model difficulty.** One-off offline pass: run base Qwen3-4B-Thinking-2507 × n=8 over the full UGPhysics+curated set, keep only prompts with reward_mean in [1/8, 7/8]. Rewrite `data/processed_tir/data/train.parquet` and `data/processed_cot/data/train.parquet`. The 32B teacher filter that was applied to Dr.SCI doesn't transfer — 32B ≠ 4B capability.
2. **Enable `algorithm.filter_groups=True`** in `train_async.sh`. verl supports this for the fully-async path (see `verl/verl/experimental/fully_async_policy/shell/dapo_30b_a3b_base_math_fsdp.sh:48-49`). Config: `+algorithm.filter_groups.enable=True +algorithm.filter_groups.metric=acc`.
3. **Verify the `saturated_frac` metric is emitting** on the first 5 steps of the next training run. If it's missing, the patch didn't propagate — check `grep critic/group logs/<new_run>.log`.
4. **Temperature sweep on ckpt_0** (in progress at session end, alloc 51870296 + 51870488). If sat_frac is roughly flat across temp={1.0, 1.3, 1.4}, data saturation hypothesis is reinforced. If temp=1.4 sat_frac drops substantially, consider raising training temperature or adding entropy regularization.

## Resume recipe (if re-using step_150 checkpoints instead of cold start)

```bash
sbatch --export=ALL,EXPERIMENT=grpo_tir_4t_sxv_20260421.083031 \
    scripts/perlmutter/prod_tir_het_4t_sharedxv.sbatch
sbatch --export=ALL,EXPERIMENT=grpo_cot_4t_sxv_20260421.083030 \
    scripts/perlmutter/prod_cot_het_4t_sharedxv.sbatch
```

verl's `resume_mode='auto'` reads `outputs/physcode_tir/<experiment>/latest_checkpointed_iteration.txt=150` and restores model + optimizer + lr_scheduler + rng state. WandB will resume the same run (not start a new curve).

**However:** a fresh-data intervention invalidates resume — optimizer moments are aligned to the old advantage distribution. **Use a new `EXPERIMENT=` tag for the intervention run, don't resume.**

## Repos / branches

- Parent: `https://github.com/xingzhis/phys-reasoner` (this session on `feat/2-trainer-node`).
- verl submodule: `https://github.com/xingzhis/verl` (this session on `physcode`).
- Alt origin: `https://github.com/bwhou1997/phys-reasoner` for solo forks.

## Pending work at session end

- Temp=1.0/1.3/1.4 × TIR/CoT saturation jobs running on alloc 51870296 + 51870488 at EOD 2026-04-21. Results land in `outputs/saturation_analysis/{tir,cot}_ckpt0_temp{1p0,1p3,1p4}/saturation.json`.
- `logs/saturation_temp_test_runner.log` + `logs/saturation_temp14_runner.log` will have the final 2×2 and summary tables.
- Step_150 saturation jobs from earlier orchestrator CRASHED (orchestrator race bug, now fixed). Skip or re-run if needed — step_50→step_100 delta already answered the hypothesis.
