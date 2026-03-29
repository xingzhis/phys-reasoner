#!/bin/bash
#SBATCH --job-name=drsci-zs-sample
#SBATCH --partition=gpu
#SBATCH --qos=qos_nmi
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --gres=gpu:a100:1
#SBATCH --time=2:00:00
#SBATCH --output=logs/%x_%j.log
#SBATCH --error=logs/%x_%j.err

# Zero-shot pass@1 inference on the Dr. SCI Goldilocks sample (Phase 3b).
#
# Prerequisite: drsci_goldilocks_sample.parquet must exist.
#   Build it with: python scripts/analyze_drsci_goldilocks.py --save_sample
#
# Override model via sbatch --export:
#   sbatch --export=ALL,ZS_MODEL=Qwen/Qwen3.5-4B scripts/zero_shot_drsci_sample.sbatch

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
source "$ROOT/env.sh"

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=8

# Default model: 0.8B (fast profiling); override with ZS_MODEL env var
ZS_MODEL="${ZS_MODEL:-Qwen/Qwen3.5-0.8B}"
ZS_OUTPUT="${ZS_OUTPUT:-$ROOT/data/results/zero_shot_drsci_sample.parquet}"
ZS_INPUT="${ZS_INPUT:-$ROOT/data/processed/drsci_goldilocks_sample.parquet}"

echo "Job:    $SLURM_JOB_ID  ($SLURM_JOB_NAME)"
echo "Node:   $SLURMD_NODENAME"
echo "GPU:    $CUDA_VISIBLE_DEVICES"
echo "Model:  $ZS_MODEL"
echo "Input:  $ZS_INPUT"
echo "Output: $ZS_OUTPUT"

if [ ! -f "$ZS_INPUT" ]; then
    echo "ERROR: $ZS_INPUT not found."
    echo "  Build it with: python scripts/analyze_drsci_goldilocks.py --save_sample"
    exit 1
fi

apptainer exec --nv \
  --overlay "$OVERLAY:ro" \
  --no-home \
  --bind /etc/pki:/etc/pki \
  --env "PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin" \
  --env "HF_HOME=$HF_HOME" \
  --env "PYTHONNOUSERSITE=1" \
  "$SIF" \
  python -u "$ROOT/scripts/run_zero_shot_drsci.py" \
    --input  "$ZS_INPUT" \
    --model  "$ZS_MODEL" \
    --max_new_tokens 8192 \
    --gpu_memory_utilization 0.9 \
    --output "$ZS_OUTPUT"

echo "Done: $?"
