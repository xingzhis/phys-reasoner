#!/bin/bash
#SBATCH --job-name=granger-tests
# Partition and QOS are not set here — set SBATCH_PARTITION and SBATCH_QOS
# in your .env file (copied from .env.example). env.sh exports them so sbatch
# picks them up automatically. Override at submit time if needed:
#   sbatch --partition=<name> --qos=<name> scripts/run_tests.sbatch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=1:00:00
#SBATCH --output=logs/%x_%j.log
#SBATCH --error=logs/%x_%j.err

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
source "$ROOT/env.sh"

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4

echo "Job:    $SLURM_JOB_ID  ($SLURM_JOB_NAME)"
echo "Node:   $SLURMD_NODENAME"
echo "GPU:    $CUDA_VISIBLE_DEVICES"

apptainer exec --nv \
  --overlay "$OVERLAY:ro" \
  --bind /etc/pki:/etc/pki \
  --env "PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin" \
  --env "PYTHONNOUSERSITE=1" \
  "$SIF" \
  python -m pytest "$ROOT/tests/" -v --tb=short 2>&1

echo "Done: $?"
