#!/bin/bash
#SBATCH --job-name=rescore_rerun0
#SBATCH --partition=gpu
#SBATCH --qos=qos_nmi
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --gres=gpu:a100:1
#SBATCH --time=1:00:00
#SBATCH --output=logs/%x_%j.log
#SBATCH --error=logs/%x_%j.err

# Rescore rerun chunk0 (48 OlympiadBench samples) with both xVerify models.
# Submit:
#   sbatch scripts/rescore_rerun0_xverify.sbatch

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
source "$ROOT/env.sh"

export PYTHONUNBUFFERED=1
export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=8

echo "Job:    $SLURM_JOB_ID  ($SLURM_JOB_NAME)"
echo "Node:   $SLURMD_NODENAME"
echo "GPU:    $CUDA_VISIBLE_DEVICES"

echo "--- Rescoring with xVerify-3B-Ib ---"
apptainer exec --nv \
  --overlay "$OVERLAY:ro" \
  --no-home \
  --bind /etc/pki:/etc/pki \
  --env "PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin" \
  --env "PYTHONNOUSERSITE=1" \
  "$SIF" \
  python -u "$ROOT/scripts/rescore_xverify.py" \
    --chunks "$ROOT/data/results/zero_shot_rerun_chunk0.parquet" \
    --orig   "$ROOT/data/processed/candidates_deduped.parquet" \
    --model  IAAR-Shanghai/xVerify-3B-Ib \
    --output "$ROOT/data/results/rescore_rerun0_3b.parquet"

echo "--- Rescoring with xVerify-7B-I ---"
apptainer exec --nv \
  --overlay "$OVERLAY:ro" \
  --no-home \
  --bind /etc/pki:/etc/pki \
  --env "PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin" \
  --env "PYTHONNOUSERSITE=1" \
  "$SIF" \
  python -u "$ROOT/scripts/rescore_xverify.py" \
    --chunks "$ROOT/data/results/zero_shot_rerun_chunk0.parquet" \
    --orig   "$ROOT/data/processed/candidates_deduped.parquet" \
    --model  IAAR-Shanghai/xVerify-7B-I \
    --output "$ROOT/data/results/rescore_rerun0_7b.parquet"

echo "Done: $?"
