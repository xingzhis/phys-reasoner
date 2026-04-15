# Setup Guide

**Prerequisites:** SLURM cluster with Apptainer, GPU nodes, ~300 GB scratch, login-node internet access.

## 1. Clone

```bash
git clone --recurse-submodules https://github.com/xingzhis/phys-reasoner.git
cd phys-reasoner
```

## 2. Configure

```bash
cp .env.example .env
```

Edit `.env`:
```bash
export SBATCH_PARTITION=gpu      # GPU partition name
export SBATCH_QOS=               # QOS if required, else leave blank
export SBATCH_ACCOUNT=           # account if required, else leave blank
export HF_TOKEN=hf_...           # required for step 5 (private dataset) and gated models
export WANDB_API_KEY=...         # required for training logs (LOGGERS=console,wandb)
# export HF_HOME=/path/to/hf_cache   # default: $ROOT/hf_cache
```

Source before every session:
```bash
source env.sh
```

## 3. Pull container and create overlay

```bash
sbatch pull_docker.sbatch
```

If your CPU and GPU partitions differ:
```bash
sbatch --partition=<cpu-partition> pull_docker.sbatch
```

Verify when done:
```bash
ls -lh verl_vllm017.latest.sif phys-reasoner-overlay-017.img
```

## 4. Install packages

```bash
bash scripts/setup_overlay.sh
```

This installs everything the project needs in one shot:

1. `phys-reasoner[dev]` and `verl` into the overlay
2. `transformers==5.3.0` + `huggingface_hub==1.8.0` + `flash-linear-attention` into
   `/opt/phys-extras/` (avoids the OverlayFS whiteout pitfall on RHEL 8 — see CLAUDE.md)
3. `numpy>=2.1` into `$ROOT/.async-extras/` (only the async training launchers
   prepend this to `PYTHONPATH`; the sync path keeps numpy 1.26 from the SIF base)
4. Smoke checks both paths and runs `e2fsck` to mark the overlay clean

Re-runnable. If you change `pyproject.toml` deps, just run it again.

## 5. Fetch dataset

The merged train / validation / test parquets live on the HF Hub at
[`xingzhi0/phys-tir`](https://huggingface.co/datasets/xingzhi0/phys-tir)

```bash
source env.sh
# rememebr to edit this in .env: export HF_TOKEN=hf_...   # your HF token (read access to the repo is sufficient)

PYTHONNOUSERSITE=1 apptainer exec \
  --overlay "$OVERLAY:ro" --bind /etc/pki:/etc/pki \
  --env "PYTHONNOUSERSITE=1" \
  --env "PYTHONPATH=/opt/phys-extras/" \
  --env "HF_TOKEN=$HF_TOKEN" \
  --env "HF_HOME=$HF_HOME" \
  "$SIF" \
  python3 scripts/fetch_dataset.py --repo-id xingzhi0/phys-tir --out-dir data/processed_hf
```

This writes `data/processed_hf/data/{train,validation,test}.parquet`. The defaults
in `scripts/train_async.sh` already point at these paths. See the dataset README
on the Hub for schema (union `extra_info` struct, `pool` column for provenance)
and source attribution.

## 6. Run tests

```bash
sbatch scripts/run_tests.sbatch
```

Check `logs/granger-tests_<jobid>.log` — all tests should pass.

## 7. Smoke test

```bash
sbatch scripts/smoke_tir_qwen35.sbatch
```

Check `logs/smoke_qwen35_<jobid>.log` for no OOM/CUDA errors and `rollout/tool_response_len > 0`.

For A100 40 GB, lower GPU memory utilization:
```bash
sbatch --export=ALL,VLLM_GPU_MEM_UTIL=0.35 scripts/smoke_tir_qwen35.sbatch
```

For Perlmutter, to request 80 GB A100 nodes specifically:
```bash
sbatch -C "gpu&hbm80g" scripts/smoke_tir_qwen35.sbatch
```

## 8. Async path validation and multi-node rehearsal

Steps 1–7 cover the **synchronous** (colocate) path. The **async** path adds
one validation script and two rehearsal sbatch recipes — all driven by the
files installed in step 4.

### 8a. Bootstrap validator

`scripts/bootstrap_perlmutter.sh` is verify-only (no installs — that's step 4's
job). It runs:

1. env.sh sanity, presence of SIF / overlay / `.async-extras`
2. Container import check (vllm, transformers, verl, ray, qwen3_5, numpy 2.x)
3. `snapshot_download` pre-fetch of Qwen3.5-4B + xVerify-7B-I (skip with `SKIP_PREFETCH=1`)
4. Required parquets present
5. Optional 1-GPU async smoke (auto-skipped if `nvidia-smi` is unavailable)

```bash
# CPU-only checks: runs anywhere
bash scripts/bootstrap_perlmutter.sh

# Full pipeline incl. 1-GPU async smoke:
srun --gres=gpu:1 --time=30:00 --pty bash scripts/bootstrap_perlmutter.sh
```

### 8b. Multi-node async rehearsal

```bash
sbatch scripts/train_async_2node.sbatch
```

### 8c. Checkpoint → kill → resume rehearsal

Required before any unattended multi-day run on a cluster you can't shell into.
Submit twice with the same `EXPERIMENT` name and non-zero `SAVE_FREQ`:

```bash
sbatch --export=ALL,EXPERIMENT=resume_test_v1,SAVE_FREQ=2,TOTAL_STEPS=6 \
       scripts/train_async_2node.sbatch
# wait until global_step >= 2, then:  scancel <jobid>
sbatch --export=ALL,EXPERIMENT=resume_test_v1,SAVE_FREQ=2,TOTAL_STEPS=6 \
       scripts/train_async_2node.sbatch
python3 scripts/verify_resume.py logs/train_async_2node_<job1>.log \
                                 logs/train_async_2node_<job2>.log
```

verl's `trainer.resume_mode=auto` (default) picks up the latest checkpoint in
`outputs/<project>/<EXPERIMENT>/` whenever the same `EXPERIMENT` name is reused.
