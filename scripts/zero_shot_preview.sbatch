#!/bin/bash
#SBATCH --job-name=granger-preview
#SBATCH --partition=gpu
#SBATCH --qos=qos_nmi
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=80G
#SBATCH --gres=gpu:a100:1
#SBATCH --time=6:00:00
#SBATCH --output=logs/%x_%j.log
#SBATCH --error=logs/%x_%j.err

ROOT=/gpfs/radev/scratch/krishnaswamy_smita/xs272/phys-reasoner
SIF=$ROOT/verl_vllm017.latest.sif
OVERLAY=$ROOT/phys-reasoner-overlay-017.img

export PYTHONUNBUFFERED=1
export HF_HOME=$ROOT/hf_cache
export OMP_NUM_THREADS=12

echo "Job:    $SLURM_JOB_ID  ($SLURM_JOB_NAME)"
echo "Node:   $SLURMD_NODENAME"
echo "GPU:    $CUDA_VISIBLE_DEVICES"

apptainer exec --nv \
  --overlay "$OVERLAY:ro" \
  --bind /etc/pki:/etc/pki \
  --env "PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin" \
  --env "HF_HOME=$HF_HOME" \
  --env "PYTHONNOUSERSITE=1" \
  "$SIF" \
  python -u "$ROOT/scripts/run_zero_shot.py" \
    --input  "$ROOT/data/processed/candidates_deduped.parquet" \
    --n_per_tier 5 \
    --output "$ROOT/data/results/zero_shot_preview.parquet"

echo "Done: $?"