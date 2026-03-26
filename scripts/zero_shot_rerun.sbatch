#!/bin/bash
#SBATCH --job-name=granger-rerun
#SBATCH --partition=gpu
#SBATCH --qos=qos_nmi
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --gres=gpu:a100:1
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x_%j.log
#SBATCH --error=logs/%x_%j.err

# Re-runs truncated samples in parallel chunks.
# Submit 8 jobs:
#   for i in $(seq 0 7); do
#     sbatch --export=ALL,CHUNK_ID=$i,N_CHUNKS=8 scripts/zero_shot_rerun.sbatch
#   done
# Then merge with: python scripts/merge_rerun_chunks.py

ROOT=/gpfs/radev/scratch/krishnaswamy_smita/xs272/phys-reasoner
SIF=$ROOT/verl_vllm017.latest.sif
OVERLAY=$ROOT/phys-reasoner-overlay-017.img

if [ -z "$CHUNK_ID" ] || [ -z "$N_CHUNKS" ]; then
    echo "Error: CHUNK_ID and N_CHUNKS must be set via sbatch --export"
    exit 1
fi

export PYTHONUNBUFFERED=1
export HF_HOME=$ROOT/hf_cache
export OMP_NUM_THREADS=8

echo "Job:    $SLURM_JOB_ID  ($SLURM_JOB_NAME)"
echo "Node:   $SLURMD_NODENAME"
echo "GPU:    $CUDA_VISIBLE_DEVICES"
echo "Chunk:  $CHUNK_ID / $N_CHUNKS"
echo "Max tokens: 81920 (Qwen3.5 complex-problem recommendation)"

apptainer exec --nv \
  --overlay "$OVERLAY:ro" \
  --bind /etc/pki:/etc/pki \
  --env "PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin" \
  --env "HF_HOME=$HF_HOME" \
  --env "PYTHONNOUSERSITE=1" \
  "$SIF" \
  python -u "$ROOT/scripts/run_zero_shot_rerun.py" \
    --chunks_glob "$ROOT/data/results/zero_shot_chunk*.parquet" \
    --candidates  "$ROOT/data/processed/candidates_deduped.parquet" \
    --model       Qwen/Qwen3.5-4B \
    --max_new_tokens 81920 \
    --chunk_id    "$CHUNK_ID" \
    --n_chunks    "$N_CHUNKS" \
    --output      "$ROOT/data/results/zero_shot_rerun_chunk${CHUNK_ID}.parquet" \
    --merged_output "$ROOT/data/results/zero_shot_merged.parquet"

echo "Done: $?"
