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
# export HF_HOME=/path/to/hf_cache   # default: $ROOT/hf_cache
# export WANDB_API_KEY=xxxx
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

## 5. Prepare datasets

```bash
bash scripts/prepare_data.sh     # 6.8k curated corpus → data/processed/corpus_train.parquet
bash scripts/prepare_drsci.sh    # Dr. SCI ~108k      → data/processed/drsci_train.parquet
```

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
