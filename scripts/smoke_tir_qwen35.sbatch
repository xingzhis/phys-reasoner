#!/bin/bash
#SBATCH --job-name=smoke_qwen35
#SBATCH --partition=gpu
#SBATCH --qos=qos_nmi
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --gres=gpu:h100:1
#SBATCH --time=1:00:00
#SBATCH --output=logs/%x_%j.log
#SBATCH --error=logs/%x_%j.err

ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
mkdir -p "$ROOT/logs"

echo "Job: $SLURM_JOB_ID on $SLURMD_NODENAME"
bash "$ROOT/scripts/smoke_tir_qwen35.sh"
