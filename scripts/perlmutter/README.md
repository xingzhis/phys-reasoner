# Perlmutter launch guide

Concise. See [`checklist.md`](checklist.md) for pass criteria and troubleshooting.

## Files

| File | Purpose |
|---|---|
| `bootstrap.sh` | Login-node env + import + HF prefetch + parquet sanity |
| `probe_xverify.py` | Standalone xVerify health + correctness check |
| `smoke_het.sbatch` | 3-group het smoke (5+1+1), 10 steps, dumps on |
| `smoke_companion.sbatch` | Fallback smoke: 6-node alloc + separate xverify sbatch |
| `prod_het.sbatch` | 3-group het production (TIR, tool-enabled), 3000 steps |
| `prod_companion.sbatch` | Fallback production (TIR), 3000 steps, separate xverify |
| `prod_cot.sbatch` | CoT baseline ablation — same hparams, tool disabled |
| `serve_xverify_perlmutter.sbatch` | xVerify server for the companion variants |
| `_ray_bringup.sh` | Shared Ray head/worker logic (sourced, not executed) |
| `checklist.md` | Pre-submit checklist, pass criteria, stop conditions |

## Defaults (all overridable via `sbatch --export=ALL,KEY=VAL,...`)

| Knob | Default | Notes |
|---|---|---|
| `MODEL` | `Qwen/Qwen3.5-4B` | |
| `TOTAL_STEPS` | `3000` | ~3 epochs over 108k pool; resubmit same `EXPERIMENT` to extend |
| `TRAIN_BATCH` | `128` | `ppo_mini_batch_size` |
| `ROLLOUT_N` | `8` | GRPO group size |
| `SAVE_FREQ` / `TEST_FREQ` | `200` / `100` | 15 ckpts, 30 eval points |
| `LR` | `1e-6` | |
| `LR_SCHEDULER_TYPE` | `constant` | Matches DAPO / SimpleRL / DeepSeek-R1 |
| `LR_WARMUP_STEPS` | `20` | Linear ramp 0 → `LR` over first 20 steps |
| `USE_DYNAMIC_BSZ` | `1` | Dynamic micro-batching |
| `PPO_MAX_TOKEN_LEN_PER_GPU` | `24576` | Tokens/GPU budget; raise to `36864` on 80G |
| `VLLM_GPU_MEM_UTIL` | `0.85` | Raise to `0.90` on 80G |
| `ACTOR_OPT_OFFLOAD` | `True` | CPU-offloaded AdamW |
| Resource | 1 trainer node + 5 rollout nodes + 1 verifier GPU | 4 × A100 per node |

## 1. One-time setup per cluster

```bash
# SIF + overlay (CPU-only; expensive, do once)
sbatch pull_docker.sbatch
bash   scripts/setup_overlay.sh

# Fetch the training parquets — MUST run inside the container (see docs/setup.md §5).
# Two datasets: phys-tir (TIR training + validation) and phys-cot (CoT baseline
# ablation — same rows, system prompt swapped so the model isn't primed to call
# tools). Put HF_TOKEN in .env first (read access is sufficient).
source env.sh
APT() {
  PYTHONNOUSERSITE=1 apptainer exec \
    --overlay "$OVERLAY:ro" --bind /etc/pki:/etc/pki \
    --env "PYTHONNOUSERSITE=1" --env "PYTHONPATH=/opt/phys-extras/" \
    --env "HF_TOKEN=$HF_TOKEN" --env "HF_HOME=$HF_HOME" \
    "$SIF" "$@"
}
APT python3 scripts/fetch_dataset.py --repo-id xingzhi0/phys-tir --out-dir data/processed_hf
APT python3 scripts/fetch_dataset.py --repo-id xingzhi0/phys-cot --out-dir data/processed_cot

# Verify env (login node, CPU-only, idempotent)
bash scripts/perlmutter/bootstrap.sh
```

## 2. Fill cluster-specific TODOs

```bash
grep -n 'TODO(collab)' scripts/perlmutter/*.sbatch
```

At minimum: `-A <account>_g`, `-C gpu` (or `-C gpu&a100_80gb` for 80G), `-q regular`.

## 3. Het vs companion

- **`*_het.sbatch` (preferred):** one sbatch, 3 het-groups, whole job dies together. Requires Perlmutter to support 3-group heterogeneous jobs and a shared-QoS 1-GPU allocation for xVerify.
- **`*_companion.sbatch` (fallback):** one 6-node allocation + a separate xverify sbatch submitted by the companion script. Use if het is not available.

## 4. Flow

```bash
# Smoke — 10 steps, full topology, dumps JSONL rollouts for eyeballing.
sbatch scripts/perlmutter/smoke_het.sbatch

# After smoke: check checklist.md pass criteria, run the xverify probe.
python3 scripts/perlmutter/probe_xverify.py

# Resume rehearsal — scancel mid-run, resubmit with same EXPERIMENT,
# verify verl auto-resumes from the latest checkpoint.
sbatch --export=ALL,TOTAL_STEPS=20,SAVE_FREQ=5 scripts/perlmutter/smoke_het.sbatch
# ... when log shows global_step=6, scancel, then:
sbatch --export=ALL,EXPERIMENT=<prev>,TOTAL_STEPS=20 scripts/perlmutter/smoke_het.sbatch

# Production — 3000 steps. Resubmit with same EXPERIMENT to extend across 48h caps.
sbatch scripts/perlmutter/prod_het.sbatch                           # TIR (main method)
sbatch --export=ALL,EXPERIMENT=<prev> scripts/perlmutter/prod_het.sbatch  # resume
sbatch scripts/perlmutter/prod_cot.sbatch                           # CoT baseline
```

## 5. GPU class (40G default, 80G supported)

```bash
# 80G override — no sbatch edits required
sbatch --constraint='gpu&a100_80gb' \
       --export=ALL,PPO_MAX_TOKEN_LEN_PER_GPU=36864,VLLM_GPU_MEM_UTIL=0.90 \
       scripts/perlmutter/prod_het.sbatch
```

## 6. When to stop

`TOTAL_STEPS=3000` is a **ceiling, not a target** — ~3 epochs over the 108k pool, sized
so plateau lands comfortably inside. Stop earlier by simply not resubmitting once any
two of the three wandb signals fire:

- `val/score/mean` plateau (±0.5% for ≥3 consecutive eval points)
- `actor/zero_advantage_group_fraction > 0.7`
- `response_length/clip_ratio` near 0 with reward ceiling-bound

Full criteria in [`checklist.md` §Stop conditions](checklist.md).

## Known unknowns for the collaborator

- 3-group hetjobs supported? If not → use `*_companion.sbatch`.
- `-q shared --gpus-per-task=1` syntax for a 1-GPU xVerify slot?
- Exact 80G constraint name (`a100_80gb`? `hbm80g`?)
- Any `module load` needed before `apptainer exec`?

Full list: `docs/perlmutter-collaborator-questions.md`.
