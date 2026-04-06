#!/bin/bash
#SBATCH --job-name=smoke_tir_interrupt
#SBATCH --partition=gpu
#SBATCH --qos=qos_nmi
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --gres=gpu:h200:1
#SBATCH --time=1:00:00
#SBATCH --output=logs/%x_%j.log
#SBATCH --error=logs/%x_%j.err

ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
mkdir -p "$ROOT/logs"

echo "Job: $SLURM_JOB_ID on $SLURMD_NODENAME"

# Small budgets force the interrupt on every trajectory (THINKING_BUDGET=200 << typical thinking).
# MAX_RESPONSE_LEN is derived from the five components — do not set it directly here.
# Pass criteria:
#   1. Interrupt phrase "time is up" present in rollout
#   2. <tool_response> present (tool executed after interrupt)
#   3. \boxed{} present in final answer turn
THINKING_BUDGET=200 \
TOOL_CALL_BUDGET=512 \
ANSWER_BUDGET=313 \
SMOKE_N=2 \
VERL_DUMP_DIR="$ROOT/outputs/think_interrupt_test_${SLURM_JOB_ID}" \
bash "$ROOT/scripts/smoke_tir_qwen35.sh"
# MAX_RESPONSE_LEN = 200 + 15 + 512 + 512 + 313 = 1552
