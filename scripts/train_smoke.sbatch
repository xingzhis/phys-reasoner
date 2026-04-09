#!/bin/bash
#SBATCH --job-name=train_smoke
# Partition and QOS are not set here — set SBATCH_PARTITION and SBATCH_QOS
# in your .env file (copied from .env.example). env.sh exports them so sbatch
# picks them up automatically. Override at submit time if needed:
#   sbatch --partition=<name> --qos=<name> scripts/train_smoke.sbatch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --gres=gpu:1
#SBATCH --time=1:00:00
#SBATCH --output=logs/%x_%j.log
#SBATCH --error=logs/%x_%j.err

ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
mkdir -p "$ROOT/logs"

echo "Job: $SLURM_JOB_ID on $SLURMD_NODENAME"

unset SIF OVERLAY
source "$ROOT/env.sh"

# Tool-call rounds per trajectory. Override with MAX_TOOL_TURNS=2 sbatch ... to test
# iterative TIR (error-recovery mode). Default=1 matches the original smoke test.
MAX_TOOL_TURNS="${MAX_TOOL_TURNS:-1}"

# Smoke parquet written by smoke_tir_qwen35.sh — 2 pre-sampled rows.
# Run that script first if this file is missing.
BASE_PARQUET="${BASE_PARQUET:-$ROOT/data/processed/smoke.parquet}"
SMOKE_DATA="$ROOT/outputs/rollout_dumps/train_smoke_${SLURM_JOB_ID}/smoke_turns${MAX_TOOL_TURNS}.parquet"
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

TRAIN_BATCH=2 \
ROLLOUT_N=8 \
TOTAL_STEPS=2 \
MAX_TOOL_TURNS="$MAX_TOOL_TURNS" \
TRAIN_FILES="$SMOKE_DATA" \
VAL_FILES="$SMOKE_DATA" \
VERL_DUMP_DIR="$ROOT/outputs/rollout_dumps/train_smoke_${SLURM_JOB_ID}" \
bash "$ROOT/scripts/train.sh"
