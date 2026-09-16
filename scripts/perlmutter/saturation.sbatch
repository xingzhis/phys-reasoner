#!/bin/bash
# saturation.sbatch — 1-GPU diagnostic job: merge a training checkpoint,
# generate N_PROMPTS x N_ROLLOUTS rollouts at training budgets, score with
# the running xVerify server, and compute GRPO group saturation stats.
#
# Submit (one per checkpoint to analyze):
#   sbatch --export=ALL,CKPT_DIR=/abs/.../global_step_100/actor,MODE=tir \
#          scripts/perlmutter/saturation.sbatch
#
#   sbatch --export=ALL,CKPT_DIR=/abs/.../global_step_100/actor,MODE=cot \
#          scripts/perlmutter/saturation.sbatch
#
# Optional overrides: N_PROMPTS (default 64), N_ROLLOUTS (default 8),
# OUT_DIR (default outputs/saturation_analysis/<derived-tag>).
#
# Prereq: xVerify server running (current.url present + healthy).

#SBATCH --job-name=saturation
#SBATCH --output=logs/%x_%j.log
#SBATCH --error=logs/%x_%j.err
#SBATCH -A m2651
#SBATCH -C gpu&hbm80g
# Shared QoS: fractional 1-GPU allocation. Doesn't count against the
# 5-job premium cap that the prod runs + xverify already fill. 48h max.
#SBATCH -q shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
# NERSC gpu_shared_ss11 mandates 32 CPU cores per GPU requested.
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-task=1
#SBATCH --time=02:00:00

set -euo pipefail
ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/../.." && pwd)}"
cd "$ROOT"

: "${CKPT_DIR:?CKPT_DIR env var is required}"
: "${MODE:=tir}"

echo "=== saturation.sbatch ==="
echo "  Job       : $SLURM_JOB_ID"
echo "  Node      : $(hostname)"
echo "  CKPT_DIR  : $CKPT_DIR"
echo "  MODE      : $MODE"
echo "  N_PROMPTS : ${N_PROMPTS:-64}"
echo "  N_ROLLOUTS: ${N_ROLLOUTS:-8}"

export CKPT_DIR MODE
bash "$ROOT/scripts/run_saturation.sh"
