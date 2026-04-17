#!/usr/bin/env bash
# train_smoke_cot_async.sh — interactive smoke for the CoT think-interrupt path.
#
# Companion to scripts/train_smoke_async.sh (TIR) — same machine topology
# (2x A100-40GB, 1 GPU rollout + 1 GPU train), same small budgets, but
# COT_BASELINE=1. Validates that the CoT single_turn_agent path now honours
# thinking_budget the same way TIR does.
#
# Smoke parquet (built on demand from the local CoT dataset):
#   - N long rows sampled from data/processed_cot/data/train.parquet
#     → expected to hit the tight THINKING_BUDGET → interrupt fires
#   - 1 synthetic "what is 2+2?" row
#     → expected to finish naturally inside THINKING_BUDGET, no interrupt
#
# Pass criteria (inspect TRAIN_DIR/rollout_dumps/*.jsonl and decoded outputs):
#   1. interrupt phrase "Okay, I've thought enough. Time to write my response."
#      appears verbatim in at least one long-row rollout
#   2. response_mask has a zero-run of length == len(_interrupt_ids) at the
#      correct offset for those rollouts
#   3. the 2+2 row's rollout has response_mask all-1s and contains \boxed{4}
#      (or close) with no interrupt phrase
#   4. one parameter sync completes (check train.log)
#
# Usage:
#   bash scripts/train_smoke_cot_async.sh
#   THINKING_BUDGET=64 bash scripts/train_smoke_cot_async.sh   # force interrupt harder
#   N_LONG=3 bash scripts/train_smoke_cot_async.sh             # smaller smoke

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
unset SIF OVERLAY
source "$ROOT/env.sh"

STAMP="$(date +%Y%m%d.%H%M%S)"
RUN_ID="train_smoke_cot_async_${STAMP}"

MODEL="${MODEL:-Qwen/Qwen3.5-0.8B}"

# Tiny budgets → most long rows hit interrupt.
# MAX_RESPONSE_LEN = 128 + 15 + 64 + 512 + 128 = 847 tokens.
THINKING_BUDGET="${THINKING_BUDGET:-128}"
TOOL_CALL_BUDGET="${TOOL_CALL_BUDGET:-64}"
ANSWER_BUDGET="${ANSWER_BUDGET:-128}"

N_LONG="${N_LONG:-7}"
CoT_SRC="${CoT_SRC:-$ROOT/data/processed_cot/data/train.parquet}"
SMOKE_DATA="$ROOT/outputs/rollout_dumps/${RUN_ID}/smoke_cot.parquet"
mkdir -p "$(dirname "$SMOKE_DATA")"

if [[ ! -f "$CoT_SRC" ]]; then
    echo "ERROR: CoT source parquet missing: $CoT_SRC" >&2
    exit 1
fi

echo "Building CoT smoke parquet: $SMOKE_DATA ($N_LONG long rows + 1 short)"
PYTHONNOUSERSITE=1 apptainer exec \
  --overlay "$OVERLAY:ro" --no-home \
  --bind /etc/pki:/etc/pki \
  --env "PYTHONNOUSERSITE=1" \
  --env "PYTHONPATH=/opt/phys-extras/" \
  "$SIF" \
  python3 "$ROOT/scripts/build_cot_smoke_parquet.py" \
      --src "$CoT_SRC" --dst "$SMOKE_DATA" --n "$N_LONG"

# Launch the async training run in CoT mode.
#   - TRAIN_BATCH=4, ROLLOUT_N=2, TOTAL_STEPS=2 → 8 total rollouts per step
#     (we want multiple n samples per prompt so GRPO has a group, but keep it
#     small on the 40GB A100)
#   - N_GPUS_ROLLOUT=1, N_GPUS_TRAIN=1 → 1+1 split on the 2-GPU node
#   - COT_BASELINE=1 → single_turn_agent + multi_turn.enable=false
#   - DUMP_TRAIN_ROLLOUTS=1 + DUMP_VAL_ROLLOUTS=1 → every rollout lands on disk
COT_BASELINE=1 \
MODEL="$MODEL" \
TRAIN_BATCH=4 \
ROLLOUT_N=2 \
TOTAL_STEPS=2 \
THINKING_BUDGET="$THINKING_BUDGET" \
TOOL_CALL_BUDGET="$TOOL_CALL_BUDGET" \
ANSWER_BUDGET="$ANSWER_BUDGET" \
N_GPUS_ROLLOUT=1 \
N_GPUS_TRAIN=1 \
NNODES_ROLLOUT=1 \
NNODES_TRAIN=1 \
VLLM_GPU_MEM_UTIL=0.8 \
TRAIN_FILES="$SMOKE_DATA" \
VAL_FILES="$SMOKE_DATA" \
DUMP_TRAIN_ROLLOUTS=1 \
DUMP_VAL_ROLLOUTS=1 \
VERL_DUMP_DIR="$ROOT/outputs/rollout_dumps/${RUN_ID}" \
WANDB_PROJECT="${WANDB_PROJECT:-physcode_tir_smoke}" \
EXPERIMENT="smoke_cot_${STAMP}" \
bash "$ROOT/scripts/train_async.sh"
