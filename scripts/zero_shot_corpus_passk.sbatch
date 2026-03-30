#!/bin/bash
#SBATCH --job-name=corpus-zs-passk
#SBATCH --partition=gpu
#SBATCH --qos=qos_nmi
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --gres=gpu:h200:1
#SBATCH --time=4:00:00
#SBATCH --output=logs/%x_%j.log
#SBATCH --error=logs/%x_%j.err

# Zero-shot pass@k on a stratified sample of the 6.8k training corpus (Phase 3b).
#
# Override via sbatch --export:
#   sbatch --export=ALL,ZS_MODEL=Qwen/Qwen3.5-4B scripts/zero_shot_corpus_passk.sbatch

ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
source "$ROOT/env.sh"

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=8

ZS_MODEL="${ZS_MODEL:-Qwen/Qwen3.5-4B}"
ZS_N_SAMPLES="${ZS_N_SAMPLES:-8}"
ZS_N_PER_STRATUM="${ZS_N_PER_STRATUM:-20}"
ZS_OUTPUT="${ZS_OUTPUT:-$ROOT/data/results/zero_shot_corpus_passk.parquet}"
ZS_INPUT="${ZS_INPUT:-$ROOT/data/processed/candidates_deduped.parquet}"

echo "Job:    $SLURM_JOB_ID  ($SLURM_JOB_NAME)"
echo "Node:   $SLURMD_NODENAME"
echo "GPU:    $CUDA_VISIBLE_DEVICES"
echo "Model:  $ZS_MODEL"
echo "Input:  $ZS_INPUT"
echo "Output: $ZS_OUTPUT"
echo "n_per_stratum: $ZS_N_PER_STRATUM  n_samples: $ZS_N_SAMPLES"

if [ ! -f "$ZS_INPUT" ]; then
    echo "ERROR: $ZS_INPUT not found."
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
  python -u "$ROOT/scripts/run_zero_shot_corpus_passk.py" \
    --input              "$ZS_INPUT" \
    --model              "$ZS_MODEL" \
    --n_samples          "$ZS_N_SAMPLES" \
    --n_per_stratum      "$ZS_N_PER_STRATUM" \
    --max_new_tokens     8192 \
    --gpu_memory_utilization 0.9 \
    --output             "$ZS_OUTPUT"

echo "Done: $?"
