#!/bin/bash
#SBATCH --job-name=granger-in-vivo
#SBATCH --partition=gpu
#SBATCH --qos=qos_nmi
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=60G
#SBATCH --gres=gpu:a100:1
#SBATCH --time=4:00:00
#SBATCH --output=logs/%x_%j.log
#SBATCH --error=logs/%x_%j.err

# Full zero-shot run with thinking OFF (non-thinking mode).
# Runs all samples in one job — outputs are short (~1-2k tokens) so no chunking needed.

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
# Load machine-local overrides if present
[ -f "$ROOT/.env" ] && source "$ROOT/.env"
SIF=$ROOT/verl_vllm017.latest.sif
OVERLAY=$ROOT/phys-reasoner-overlay-017.img

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=8

echo "Job:    $SLURM_JOB_ID  ($SLURM_JOB_NAME)"
echo "Node:   $SLURMD_NODENAME"
echo "GPU:    $CUDA_VISIBLE_DEVICES"
echo "Mode:   thinking=OFF, all samples, max_new_tokens=8192"

apptainer exec --nv \
  --overlay "$OVERLAY:ro" \
  --bind /etc/pki:/etc/pki \
  --env "PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin" \
  --env "PYTHONNOUSERSITE=1" \
  "$SIF" \
  python -u "$ROOT/scripts/run_zero_shot_nothink.py" \
    --input                  "$ROOT/data/processed/candidates_deduped.parquet" \
    --n_per_tier             10000 \
    --max_new_tokens         8192 \
    --gpu_memory_utilization 0.9 \
    --output                 "$ROOT/data/results/zero_shot_nothink.parquet"

echo "Done: $?"
