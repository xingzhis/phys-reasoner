#!/bin/bash
#SBATCH --job-name=granger-diag
#SBATCH --partition=gpu
#SBATCH --qos=qos_nmi
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --gres=gpu:a100:1
#SBATCH --time=0:30:00
#SBATCH --output=logs/%x_%j.log
#SBATCH --error=logs/%x_%j.err

# Diagnostic job: observe natural EOS token distribution.
# Use --max_samples N for a quick check (default 10).
# Increase --time and remove --max_samples for the full 226-row run (~2.5h).

ROOT=/gpfs/radev/scratch/krishnaswamy_smita/xs272/phys-reasoner
SIF=$ROOT/verl_vllm017.latest.sif
OVERLAY=$ROOT/phys-reasoner-overlay-017.img

export PYTHONUNBUFFERED=1
export HF_HOME=$ROOT/hf_cache
export OMP_NUM_THREADS=8

echo "Job:  $SLURM_JOB_ID  ($SLURM_JOB_NAME)"
echo "Node: $SLURMD_NODENAME"
echo "GPU:  $CUDA_VISIBLE_DEVICES"

apptainer exec --nv \
  --overlay "$OVERLAY:ro" \
  --bind /etc/pki:/etc/pki \
  --env "PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin" \
  --env "HF_HOME=$HF_HOME" \
  --env "PYTHONNOUSERSITE=1" \
  "$SIF" \
  python -u "$ROOT/scripts/run_zero_shot_diag.py" \
    --input  "$ROOT/data/processed/candidates_deduped.parquet" \
    --n_per_tier 3 \
    --max_samples 10 \
    --output "$ROOT/data/results/zero_shot_diag.parquet"

echo "Done: $?"