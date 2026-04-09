#!/bin/bash
#SBATCH --job-name=serve_xverify
#SBATCH --partition=test
#SBATCH --nodelist=misha00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=1-00:00:00
#SBATCH --output=logs/%x_%j.log
#SBATCH --error=logs/%x_%j.err

ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
mkdir -p "$ROOT/logs"

echo "Job: $SLURM_JOB_ID on $SLURMD_NODENAME"

unset SIF OVERLAY
source "$ROOT/env.sh"

PORT="${PORT:-8765}"
XVERIFY_MODEL="${XVERIFY_MODEL:-IAAR-Shanghai/xVerify-7B-I}"

PORT="$PORT" XVERIFY_MODEL="$XVERIFY_MODEL" bash "$ROOT/scripts/serve_xverify.sh"
