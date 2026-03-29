#!/bin/bash
#SBATCH --job-name=rescore-nothink
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

# Rescore zero-shot nothink results with xVerify.
# Usage:
#   sbatch --export=ALL,XV_MODEL=IAAR-Shanghai/xVerify-3B-Ib scripts/rescore_nothink_xverify.sbatch

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
source "$ROOT/env.sh"

XV_MODEL=${XV_MODEL:-IAAR-Shanghai/xVerify-3B-Ib}
XV_OUTPUT=${XV_OUTPUT:-$ROOT/data/results/zero_shot_nothink_xverify.parquet}

export PYTHONUNBUFFERED=1
export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=8

echo "Job:    $SLURM_JOB_ID  ($SLURM_JOB_NAME)"
echo "Node:   $SLURMD_NODENAME"
echo "GPU:    $CUDA_VISIBLE_DEVICES"
echo "Model:  $XV_MODEL"
echo "Output: $XV_OUTPUT"

apptainer exec --nv \
  --overlay "$OVERLAY:ro" \
  --no-home \
  --bind /etc/pki:/etc/pki \
  --env "PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin" \
  --env "PYTHONNOUSERSITE=1" \
  "$SIF" \
  python -u "$ROOT/scripts/rescore_xverify.py" \
    --chunks "$ROOT/data/results/zero_shot_nothink.parquet" \
    --orig   "$ROOT/data/processed/candidates_deduped.parquet" \
    --model  "$XV_MODEL" \
    --output "$XV_OUTPUT"

echo "Done: $?"
