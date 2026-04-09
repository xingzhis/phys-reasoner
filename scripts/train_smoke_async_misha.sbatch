#!/bin/bash
#SBATCH --job-name=train_smoke_async
#SBATCH --partition=test
#SBATCH --nodelist=misha02
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --gres=gpu:2
#SBATCH --time=1-00:00:00
#SBATCH --output=logs/%x_%j.log
#SBATCH --error=logs/%x_%j.err

ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
mkdir -p "$ROOT/logs"

echo "Job: $SLURM_JOB_ID on $SLURMD_NODENAME"

unset SIF OVERLAY
source "$ROOT/env.sh"

XVERIFY_URL=http://misha00:8765/judge \
ACTOR_OPT_OFFLOAD=True \
THINKING_BUDGET=12288 \
TOOL_CALL_BUDGET=2048 \
ANSWER_BUDGET=4096 \
LOGGERS=console,wandb \
N_GPUS_ROLLOUT=1 \
N_GPUS_TRAIN=1 \
bash "$ROOT/scripts/train_smoke_async.sh"
