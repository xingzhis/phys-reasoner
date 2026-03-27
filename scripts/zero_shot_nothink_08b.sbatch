#!/bin/bash
#SBATCH --job-name=granger-nothink-08b
#SBATCH --partition=gpu
#SBATCH --qos=qos_nmi
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=60G
#SBATCH --gres=gpu:h200:1
#SBATCH --time=4:00:00
#SBATCH --output=logs/%x_%j.log
#SBATCH --error=logs/%x_%j.err

# Zero-shot no-think run for Qwen3.5-0.8B, followed by xVerify-7B rescoring.
# Outputs:
#   data/results/zero_shot_nothink_08b.parquet          (inference + rule-tier scores)
#   data/results/zero_shot_nothink_08b_xverify_7b.parquet  (+ xVerify-7B scores)

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
# Load machine-local overrides if present
[ -f "$ROOT/.env" ] && source "$ROOT/.env"
SIF=$ROOT/verl_vllm017.latest.sif
OVERLAY=$ROOT/phys-reasoner-overlay-017.img

export PYTHONUNBUFFERED=1
export PYTHONNOUSERSITE=1
export HF_HOME=${HF_HOME:-$ROOT/hf_cache}
export OMP_NUM_THREADS=8

echo "Job:    $SLURM_JOB_ID  ($SLURM_JOB_NAME)"
echo "Node:   $SLURMD_NODENAME"
echo "GPU:    $CUDA_VISIBLE_DEVICES"

# --- Step 1: Inference ---
echo "=== Step 1: Inference (Qwen3.5-0.8B, thinking OFF) ==="
apptainer exec --nv \
  --overlay "$OVERLAY:ro" \
  --no-home \
  --bind /etc/pki:/etc/pki \
  --env "PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin" \
  --env "HF_HOME=$HF_HOME" \
  --env "PYTHONNOUSERSITE=1" \
  "$SIF" \
  python -u "$ROOT/scripts/run_zero_shot_nothink.py" \
    --model                  Qwen/Qwen3.5-0.8B \
    --input                  "$ROOT/data/processed/candidates_deduped.parquet" \
    --n_per_tier             10000 \
    --max_new_tokens         8192 \
    --gpu_memory_utilization 0.9 \
    --output                 "$ROOT/data/results/zero_shot_nothink_08b.parquet"

if [ $? -ne 0 ]; then
    echo "Inference failed — aborting."
    exit 1
fi

# --- Step 2: xVerify-7B rescore ---
echo "=== Step 2: xVerify-7B rescore ==="
apptainer exec --nv \
  --overlay "$OVERLAY:ro" \
  --no-home \
  --bind /etc/pki:/etc/pki \
  --env "PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin" \
  --env "HF_HOME=$HF_HOME" \
  --env "PYTHONNOUSERSITE=1" \
  "$SIF" \
  python -u "$ROOT/scripts/rescore_xverify.py" \
    --chunks "$ROOT/data/results/zero_shot_nothink_08b.parquet" \
    --orig   "$ROOT/data/processed/candidates_deduped.parquet" \
    --model  IAAR-Shanghai/xVerify-7B-I \
    --output "$ROOT/data/results/zero_shot_nothink_08b_xverify_7b.parquet"

echo "Done: $?"
