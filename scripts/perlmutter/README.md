# Perlmutter launch guide

Concise. See [`checklist.md`](checklist.md) for pass criteria and troubleshooting.

## The two experiments

| Name | Purpose | Dataset | Script family |
|---|---|---|---|
| **TIR** (main) | Single-block tool-integrated reasoning: model reasons, emits one Python/SymPy block, sees output, reasons to `\boxed{}` | `xingzhi0/phys-tir` → `data/processed_tir/` | `*_tir_*.sbatch` |
| **CoT** (baseline) | Pure chain-of-thought, no tool — same rows, same hparams, only the system prompt is swapped | `xingzhi0/phys-cot` → `data/processed_cot/` | `*_cot_*.sbatch` |

The CoT run flips two verl flags (`COT_BASELINE=1` → `default_agent_loop=single_turn_agent`, `multi_turn.enable=false`) and points at the CoT-prompted parquets. Every other hparam is identical so the comparison isolates the tool contribution.

## Files

Naming: `{stage}_{variant}_{topology}.sbatch` where
stage ∈ {smoke, prod}, variant ∈ {tir, cot}, topology ∈ {het, companion}.

| File | Purpose |
|---|---|
| `bootstrap.sh` | Login-node env + import + HF prefetch + parquet sanity |
| `probe_xverify.py` | Standalone xVerify health + correctness check |
| `_ray_bringup.sh` | Shared Ray head/worker logic (sourced, not executed) |
| `smoke_tir_het.sbatch` / `smoke_tir_companion.sbatch` | TIR smoke (10 steps, dumps on) |
| `prod_tir_het.sbatch` / `prod_tir_companion.sbatch` | TIR production (3000-step ceiling) |
| `prod_cot_het.sbatch` / `prod_cot_companion.sbatch` | CoT baseline production (same hparams, tool disabled) |
| `serve_xverify_perlmutter.sbatch` | xVerify server for the `*_companion` variants |
| `checklist.md` | Pre-submit checklist, pass criteria, stop conditions |

**No CoT smoke script on purpose** — smoke tests validate the infrastructure (Ray, xVerify, checkpoints, resume), not the variant. The TIR smoke is enough. If you *need* to smoke-test the CoT path (e.g. after changing `build_cot_parquets.py`), submit `smoke_tir_het.sbatch` with overrides:

```bash
sbatch --export=ALL,COT_BASELINE=1,\
TRAIN_FILES=$ROOT/data/processed_cot/data/train.parquet,\
VAL_FILES=$ROOT/data/processed_cot/data/validation.parquet \
  scripts/perlmutter/smoke_tir_het.sbatch
```

**`*_het` vs `*_companion`:** prefer the het variants (single sbatch, 3 het-groups, dies together). Fall back to companion (one 6-node allocation + a separate xverify sbatch) if the cluster doesn't allow 3-group heterogeneous jobs.

## Defaults (all overridable via `sbatch --export=ALL,KEY=VAL,...`)

| Knob | Default | Notes |
|---|---|---|
| `MODEL` | `Qwen/Qwen3.5-4B` | |
| `TOTAL_STEPS` | `3000` | ~3 epochs over 108k pool; ceiling, not target (see §4) |
| `TRAIN_BATCH` | `128` | `ppo_mini_batch_size` |
| `ROLLOUT_N` | `8` | GRPO group size |
| `SAVE_FREQ` / `TEST_FREQ` | `200` / `100` | 15 ckpts, 30 eval points |
| `LR` / `LR_SCHEDULER_TYPE` / `LR_WARMUP_STEPS` | `1e-6` / `constant` / `20` | Matches DAPO / SimpleRL / DeepSeek-R1 |
| `USE_DYNAMIC_BSZ` / `PPO_MAX_TOKEN_LEN_PER_GPU` | `1` / `24576` | Tokens/GPU; raise to `36864` on 80G |
| `VLLM_GPU_MEM_UTIL` | `0.85` | Raise to `0.90` on 80G |
| `ACTOR_OPT_OFFLOAD` | `True` | CPU AdamW — safe default, ~1–2% end-to-end overhead |
| `EXPERIMENT` (TIR default) | `grpo_tir_<timestamp>` | wandb run name + checkpoint dir |
| `EXPERIMENT` (CoT default) | `grpo_cot_<timestamp>` | separate namespace from TIR |
| Resource | 1 trainer + 5 rollout + 1 verifier GPU | 4 × A100 per node |

## 1. One-time setup per cluster

```bash
# Clone (recursive — pulls the verl submodule)
git clone --recurse-submodules https://github.com/xingzhis/phys-reasoner.git
cd phys-reasoner

# SIF + overlay (CPU-only; expensive, do once)
sbatch pull_docker.sbatch
bash   scripts/setup_overlay.sh

# From here on, wrap every python call in APT() so it runs inside the container
# with the right PYTHONPATH / HF_TOKEN / HF_HOME. Put HF_TOKEN + WANDB_API_KEY
# in .env first (see docs/setup.md §2 for the .env template).
source env.sh
APT() {
  PYTHONNOUSERSITE=1 apptainer exec \
    --overlay "$OVERLAY:ro" --bind /etc/pki:/etc/pki \
    --env "PYTHONNOUSERSITE=1" --env "PYTHONPATH=/opt/phys-extras/" \
    --env "HF_TOKEN=$HF_TOKEN" --env "HF_HOME=$HF_HOME" \
    "$SIF" "$@"
}

# Fetch both datasets: phys-tir (main run) and phys-cot (baseline —
# same rows, only system prompt swapped so the model isn't primed to call tools).
APT python3 scripts/fetch_dataset.py --repo-id xingzhi0/phys-tir --out-dir data/processed_tir
APT python3 scripts/fetch_dataset.py --repo-id xingzhi0/phys-cot --out-dir data/processed_cot

# Verify env (login node, CPU-only, idempotent)
bash scripts/perlmutter/bootstrap.sh
```

## 2. Fill cluster-specific TODOs

```bash
grep -n 'TODO(collab)' scripts/perlmutter/*.sbatch
```

At minimum: `-A <account>_g`, `-C gpu` (or `-C gpu&a100_80gb` for 80G), `-q regular`.

## 3. Flow

### 3a. Smoke (infrastructure validation — TIR only)

```bash
sbatch scripts/perlmutter/smoke_tir_het.sbatch

# After smoke: check checklist.md pass criteria, probe xverify.
APT python3 scripts/perlmutter/probe_xverify.py

# Resume rehearsal — scancel mid-run, resubmit with same EXPERIMENT.
sbatch --export=ALL,TOTAL_STEPS=20,SAVE_FREQ=5 scripts/perlmutter/smoke_tir_het.sbatch
# ... when log shows global_step=6, scancel; then:
sbatch --export=ALL,EXPERIMENT=<prev>,TOTAL_STEPS=20 scripts/perlmutter/smoke_tir_het.sbatch
```

### 3b. Production — TIR (main) + CoT (baseline)

Both runs use a 3000-step ceiling. Resubmit with the SAME `EXPERIMENT` across 48h caps.

```bash
# TIR (main run)
sbatch scripts/perlmutter/prod_tir_het.sbatch
sbatch --export=ALL,EXPERIMENT=<prev> scripts/perlmutter/prod_tir_het.sbatch     # resume

# CoT (baseline, run in parallel — same hparams, no tool)
sbatch scripts/perlmutter/prod_cot_het.sbatch
sbatch --export=ALL,EXPERIMENT=<prev> scripts/perlmutter/prod_cot_het.sbatch     # resume

# 80G override (no sbatch edits needed):
sbatch --constraint='gpu&a100_80gb' \
       --export=ALL,PPO_MAX_TOKEN_LEN_PER_GPU=36864,VLLM_GPU_MEM_UTIL=0.90 \
       scripts/perlmutter/prod_tir_het.sbatch
```

If the cluster doesn't allow 3-group het jobs, use the `_companion` variants instead (one 6-node allocation + a separate xverify sbatch kicked off by the script). Same flags.

**About `<prev>`:** `EXPERIMENT` defaults to `grpo_tir_<timestamp>` (TIR) or `grpo_cot_<timestamp>` (CoT) — if you don't override it, the sbatch auto-generates one at launch. That string is the wandb run name AND the checkpoint directory under `outputs/physcode_tir/<EXPERIMENT>/`. To find a previous run's value, check:

- the sbatch log banner (`[driver] launching ... (EXPERIMENT=grpo_tir_...)`) in `logs/`
- `ls -t outputs/physcode_tir/ | head -3`
- the wandb run name

Pass that string as `EXPERIMENT=<prev>` to resume — verl's `resume_mode=auto` picks up the latest checkpoint in that dir automatically.

## 4. When to stop

`TOTAL_STEPS=3000` is a **ceiling, not a target**. Stop resubmitting once any two of these wandb signals fire:

- `val/score/mean` plateau (±0.5% for ≥3 consecutive eval points)
- `actor/zero_advantage_group_fraction > 0.7`
- `response_length/clip_ratio` near 0 with reward ceiling-bound

Apply to TIR and CoT independently — they may converge at different step counts.

Full criteria in [`checklist.md` §Stop conditions](checklist.md).

## Known unknowns for the collaborator

See `docs/perlmutter-collaborator-questions.md`. Quick version:

- 3-group hetjobs supported? If not → use `*_companion.sbatch`.
- `-q shared --gpus-per-task=1` syntax for a 1-GPU xVerify slot?
- Exact 80G constraint name (`a100_80gb`? `hbm80g`?)
- Any `module load` needed before `apptainer exec`?
