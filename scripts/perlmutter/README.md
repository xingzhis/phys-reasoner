# Perlmutter launch guide

Concise. For the full checklist see [`checklist.md`](checklist.md).

## Files in this folder

| File | Purpose |
|---|---|
| `bootstrap.sh` | Login-node env + import + prefetch + parquet sanity |
| `probe_xverify.py` | Standalone xVerify health + correctness diagnostic |
| `smoke_het.sbatch` | 3-group het-job smoke (5+1+1), short steps, dumps on |
| `smoke_companion.sbatch` | Fallback smoke: 6-node + separate xverify sbatch |
| `prod_het.sbatch` | 3-group het-job production, 1500 steps default |
| `prod_companion.sbatch` | Fallback production, same topology, separate xverify |
| `_ray_bringup.sh` | Shared Ray head/worker srun logic (sourced by sbatch) |
| `checklist.md` | Pre-submit checklist, pass criteria, stop conditions |

## Prerequisites (one-time per cluster)

```bash
# 0. Clone repo + stage SIF/overlay (outside this folder)
sbatch pull_docker.sbatch             # produces the SIF
bash   scripts/setup_overlay.sh       # builds overlay + .async-extras

# 1. Fetch training parquets from HF
HF_TOKEN=hf_... python3 scripts/fetch_dataset.py \
    --repo-id <user>/phys-tir \
    --out-dir data/processed_hf

# 2. Verify environment (CPU-only, safe to re-run)
bash scripts/perlmutter/bootstrap.sh
```

## Fill in cluster-specific values

All sbatch files have `TODO(collab)` comments. Grep to find them:

```bash
grep -n 'TODO(collab)' scripts/perlmutter/*.sbatch
```

At minimum you need: `-A <account>_g`, `-C gpu` (or `-C gpu&a100_80gb`), `-q regular`.

## Pick a variant

**Het (preferred):** single sbatch, dies together, cleanest lifecycle.
Requires Perlmutter to support 3-group heterogeneous jobs and `-q shared` for a 1-GPU slice.

**Companion (fallback):** one 6-node sbatch + one xverify sbatch. More resilient
if het isn't allowed — xverify can be restarted independently.

## Flow

```bash
# SMOKE — full topology, ~10 steps, dumps first 3 train steps to JSONL
sbatch scripts/perlmutter/smoke_het.sbatch         # (or smoke_companion.sbatch)

# After smoke: eyeball rollouts, check pass criteria in checklist.md,
# then run probe_xverify for a quick sanity:
python3 scripts/perlmutter/probe_xverify.py

# RESUME REHEARSAL (before committing to a long production run)
# Submit smoke with TOTAL_STEPS=20 SAVE_FREQ=5, scancel at step 6, resubmit with same EXPERIMENT.

# PRODUCTION — 1500 steps; extend by resubmitting with same EXPERIMENT
sbatch scripts/perlmutter/prod_het.sbatch          # (or prod_companion.sbatch)
sbatch --export=ALL,EXPERIMENT=grpo_prod_20260416.120000 \
       scripts/perlmutter/prod_het.sbatch          # resume after 48h cap
```

## Stop conditions

See [`checklist.md#stop-conditions-for-the-production-run`](checklist.md). In short:
watch `val/score/mean` plateau, `zero_advantage_group_fraction`, and response-length
clip ratio on wandb. Stop resubmitting when any two of the three plateau signals fire.

## GPU class (40G vs 80G)

Default is **A100-40G** (`-C gpu`). If your allocation lands 80G nodes, override the
constraint at submit time and bump the dynamic-bsz budget — no sbatch edits required:

```bash
sbatch --constraint='gpu&a100_80gb' \
       --export=ALL,PPO_MAX_TOKEN_LEN_PER_GPU=36864,VLLM_GPU_MEM_UTIL=0.90 \
       scripts/perlmutter/prod_het.sbatch
```

TODO(collab): verify the exact 80G feature name on Perlmutter (`a100_80gb` is a guess
— could be `hbm80g`, `a100_80g`, etc.). Ask the collaborator at bring-up.

## Learning-rate schedule

Defaults in `train_async.sh`: **constant LR=1e-6 with 20-step linear warmup**. This
matches DAPO, SimpleRL-Zoo, and DeepSeek-R1-Zero canonical recipes — widely validated
for small-model GRPO. Short warmup guards against a big policy move on step 0 when the
advantage-estimator stats are cold.

To run a cosine schedule (LR → LR × min\_lr\_ratio over the full run) pass at submit:

```bash
sbatch --export=ALL,LR_SCHEDULER_TYPE=cosine,LR_MIN_RATIO=0.1 scripts/perlmutter/prod_het.sbatch
```

Only worth switching to cosine if you commit to the full 3k-step run; for 1500 steps
the LR drop is marginal and constant is fine.

## Dynamic batch sizing

Enabled by default (`USE_DYNAMIC_BSZ=1, PPO_MAX_TOKEN_LEN_PER_GPU=24576`). See
[`checklist.md#finding-a-safe-ppo_max_token_len_per_gpu-micro-batch-budget`](checklist.md)
for the tuning procedure. Smoke run logs peak memory per step so you can dial this in
once and lock it.

## Known unknowns for the collaborator

- Does Perlmutter support **3-group hetjobs**? If not → use `*_companion.sbatch`.
- Is **`-q shared --gpus-per-task=1`** the right syntax for a 1-GPU xVerify slot?
- A100-80G constraint name (`gpu&a100_80gb`?) — guessed, verify before submit.
- Any module-load needed before `apptainer exec` (e.g. `module load apptainer`)?

These are listed in `docs/perlmutter-collaborator-questions.md`.
