#!/bin/bash
#SBATCH --job-name=train_smoke_interrupt
# Partition and QOS are not set here — set SBATCH_PARTITION and SBATCH_QOS
# in your .env file (copied from .env.example). env.sh exports them so sbatch
# picks them up automatically. Override at submit time if needed:
#   sbatch --partition=<name> --qos=<name> scripts/train_smoke_interrupt.sbatch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --gres=gpu:1
#SBATCH --time=2:00:00
#SBATCH --output=logs/%x_%j.log
#SBATCH --error=logs/%x_%j.err

# Validates that train.sh works correctly with production think-interrupt budgets.
# Uses the same smoke parquet and 2-step run as train_smoke.sbatch, but adds the
# real budget values so MAX_RESPONSE_LEN is derived from the formula (18959 tokens):
#   MAX_RESPONSE_LEN = THINKING_BUDGET + INTERRUPT_LEN + TOOL_CALL_BUDGET
#                    + MAX_TOOL_RESPONSE_LEN + ANSWER_BUDGET
#                    = 12288 + 15 + 2048 + 512 + 4096 = 18959
#
# Pass criteria (inspect logs after job completes):
#   1. "time is up" present in at least one rollout → interrupt fired
#   2. <tool_response> present in same rollout → tool executed after interrupt
#   3. \boxed{} present in final answer turn → full TIR cycle complete
#   4. critic/score/mean > 0.0 → reward signal non-zero
#
# Usage:
#   sbatch scripts/train_smoke_interrupt.sbatch

ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
mkdir -p "$ROOT/logs"

echo "Job: $SLURM_JOB_ID on $SLURMD_NODENAME"

unset SIF OVERLAY
source "$ROOT/env.sh"

MAX_TOOL_TURNS="${MAX_TOOL_TURNS:-1}"

# Re-stamp smoke parquet with correct system prompt (same logic as train_smoke.sbatch).
BASE_PARQUET="${BASE_PARQUET:-$ROOT/data/processed/smoke.parquet}"
SMOKE_DATA="$ROOT/outputs/rollout_dumps/train_smoke_interrupt_${SLURM_JOB_ID}/smoke_turns${MAX_TOOL_TURNS}.parquet"
mkdir -p "$(dirname "$SMOKE_DATA")"

echo "Re-stamping system prompt for MAX_TOOL_TURNS=$MAX_TOOL_TURNS → $SMOKE_DATA"
PYTHONNOUSERSITE=1 apptainer exec \
  --overlay "$OVERLAY:ro" --no-home \
  --bind /etc/pki:/etc/pki \
  --env "PYTHONNOUSERSITE=1" \
  --env "PYTHONPATH=/opt/phys-extras/" \
  "$SIF" \
  python3 - <<PY "$BASE_PARQUET" "$SMOKE_DATA" "$MAX_TOOL_TURNS"
import sys
sys.path.insert(0, "$ROOT/src")
import pandas as pd
from phys_reasoner.tir.prompts import make_system_prompt

src, dst, max_tool_turns = sys.argv[1], sys.argv[2], int(sys.argv[3])
sys_prompt = make_system_prompt(max_tool_calls=max_tool_turns)
df = pd.read_parquet(src)
def _restamp(msgs):
    msgs = list(msgs)
    if msgs and msgs[0]["role"] == "system":
        msgs[0] = dict(msgs[0], content=sys_prompt)
    return msgs
df["prompt"] = df["prompt"].apply(_restamp)
df.to_parquet(dst, index=False)
print(f"Wrote {len(df)} rows to {dst} (max_tool_calls={max_tool_turns})")
PY

# Production think-interrupt budgets (from .claude/plans/physcode.md).
# INTERRUPT_LEN=15 is hardcoded in train.sh (exact for Qwen3.5-4B tokenizer).
# MAX_RESPONSE_LEN is derived inside train.sh: 12288+15+2048+512+4096 = 18959.
TRAIN_BATCH=2 \
ROLLOUT_N=8 \
TOTAL_STEPS=2 \
MAX_TOOL_TURNS="$MAX_TOOL_TURNS" \
THINKING_BUDGET=12288 \
TOOL_CALL_BUDGET=2048 \
ANSWER_BUDGET=4096 \
TRAIN_FILES="$SMOKE_DATA" \
VAL_FILES="$SMOKE_DATA" \
VERL_DUMP_DIR="$ROOT/outputs/rollout_dumps/train_smoke_interrupt_${SLURM_JOB_ID}" \
bash "$ROOT/scripts/train.sh"
