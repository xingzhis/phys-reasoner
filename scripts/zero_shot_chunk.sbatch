#!/bin/bash
#SBATCH --job-name=granger
#SBATCH --partition=gpu
#SBATCH --qos=qos_nmi
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=60G
#SBATCH --gres=gpu:a100:1
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x_%j.log
#SBATCH --error=logs/%x_%j.err

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
# Load machine-local overrides if present
[ -f "$ROOT/.env" ] && source "$ROOT/.env"
SIF=$ROOT/verl_vllm017.latest.sif
OVERLAY=$ROOT/phys-reasoner-overlay-017.img

# Required: CHUNK_ID (0-indexed), N_CHUNKS — passed via sbatch --export
if [ -z "$CHUNK_ID" ] || [ -z "$N_CHUNKS" ]; then
    echo "Error: CHUNK_ID and N_CHUNKS must be set via sbatch --export"
    exit 1
fi

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=8

echo "Job:    $SLURM_JOB_ID  ($SLURM_JOB_NAME)"
echo "Node:   $SLURMD_NODENAME"
echo "GPU:    $CUDA_VISIBLE_DEVICES"
echo "Chunk:  $CHUNK_ID / $N_CHUNKS"

apptainer exec --nv \
  --overlay "$OVERLAY:ro" \
  --bind /etc/pki:/etc/pki \
  --env "PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin" \
  --env "PYTHONNOUSERSITE=1" \
  "$SIF" \
  python -u "$ROOT/scripts/run_zero_shot.py" \
    --input    "$ROOT/data/processed/candidates_deduped.parquet" \
    --n_per_tier 10000 \
    --max_new_tokens 32768 \
    --chunk_id  "$CHUNK_ID" \
    --n_chunks  "$N_CHUNKS" \
    --output   "$ROOT/data/results/zero_shot_chunk${CHUNK_ID}.parquet"

echo "Done: $?"