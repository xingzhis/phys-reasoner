#!/bin/bash
#SBATCH --job-name=physcode_train
# Partition and QOS are not set here — set SBATCH_PARTITION and SBATCH_QOS
# in your .env file (copied from .env.example). env.sh exports them so sbatch
# picks them up automatically. Override at submit time if needed:
#   sbatch --partition=<name> --qos=<name> scripts/train.sbatch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x_%j.log
#SBATCH --error=logs/%x_%j.err

ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
mkdir -p "$ROOT/logs"

echo "Job: $SLURM_JOB_ID on $SLURMD_NODENAME"
unset SIF OVERLAY
source "$ROOT/env.sh"

bash "$ROOT/scripts/train.sh"
