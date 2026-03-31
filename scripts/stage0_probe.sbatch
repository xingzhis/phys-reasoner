#!/bin/bash
#SBATCH --job-name=stage0_probe
#SBATCH --partition=gpu
#SBATCH --qos=qos_nmi
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:a100:1
#SBATCH --time=2:00:00
#SBATCH --output=logs/%x_%j.log
#SBATCH --error=logs/%x_%j.err

# Stage 0 TIR probe — measures zero-shot TIR hit rate before deciding whether SFT is needed.
#
# Usage (defaults run on drsci_physics_clean.parquet, 100 samples):
#   sbatch scripts/stage0_probe.sbatch
#
# Override defaults via --export:
#   sbatch --export=ALL,S0_PARQUET=data/processed/candidates_deduped.parquet,S0_N=200 scripts/stage0_probe.sbatch

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
source "$ROOT/env.sh"

# Defaults (can be overridden via --export)
S0_PARQUET="${S0_PARQUET:-$ROOT/data/processed/drsci_physics_clean.parquet}"
S0_N="${S0_N:-100}"
S0_OUTPUT="${S0_OUTPUT:-$ROOT/data/results/stage0_probe.parquet}"

# Model: HF model ID — auto-downloads to HF_HOME if not cached.
# Pre-download on login node if compute node has no internet:
#   huggingface-cli download Qwen/Qwen3.5-4B
MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3.5-4B}"

mkdir -p "$(dirname "$S0_OUTPUT")" logs

export PYTHONUNBUFFERED=1
export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=8
export HF_HOME="$HF_HOME"
export HF_DATASETS_OFFLINE=1

echo "Job:     $SLURM_JOB_ID  ($SLURM_JOB_NAME)"
echo "Node:    $SLURMD_NODENAME"
echo "GPU:     $CUDA_VISIBLE_DEVICES"
echo "Model:   $MODEL_PATH"
echo "Parquet: $S0_PARQUET"
echo "N:       $S0_N"
echo "Output:  $S0_OUTPUT"

apptainer exec --nv \
  --overlay "$OVERLAY:ro" \
  --no-home \
  --bind /etc/pki:/etc/pki \
  --env "PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin" \
  --env "PYTHONNOUSERSITE=1" \
  --env "HF_HOME=$HF_HOME" \
  --env "HF_DATASETS_OFFLINE=1" \
  "$SIF" \
  python -u -m phys_reasoner.eval.stage0_probe \
    --model   "$MODEL_PATH" \
    --parquet "$S0_PARQUET" \
    --n_samples "$S0_N" \
    --output  "$S0_OUTPUT"

echo "Exit: $?"
