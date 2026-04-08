#!/usr/bin/env bash
# train_smoke_async.sh — interactive smoke test for the fully_async_policy path.
#
# Mirrors scripts/train_smoke_interrupt.sbatch but for the disaggregated
# (rollout + trainer on separate GPUs) path, tuned for a 2x A100-40GB node:
#   - Qwen3.5-0.8B (HF id; cached via HF_HOME under env.sh)
#   - 1 GPU for rollout, 1 GPU for trainer
#   - Small think-interrupt budgets so max_response_length stays tiny
#   - ppo_mini_batch_size=4, rollout.n=4, 2 training steps → 8 total rollouts
#   - On-policy mode (trigger=1, staleness=0, partial_rollout=False) — the
#     simplest fully_async setting, equivalent to colocate-sync except that
#     rollout and train no longer fight for the same GPU memory.
#
# Pass criteria (inspect $TRAIN_DIR/train.log after the run completes):
#   1. "time is up" present in at least one rollout → interrupt fired
#   2. <tool_response> present in same rollout → tool executed after interrupt
#   3. \boxed{} present in final answer turn → full TIR cycle complete
#   4. critic/score/mean > 0.0 → reward signal non-zero
#   5. one parameter sync completes → "parameter sync" / "sync finished" log line
#
# Usage (interactive, on a 2-GPU node):
#   bash scripts/train_smoke_async.sh
#
# Environment overrides:
#   MAX_TOOL_TURNS=2 bash scripts/train_smoke_async.sh
#   THINKING_BUDGET=1024 bash scripts/train_smoke_async.sh
#   MODEL=Qwen/Qwen3.5-4B bash scripts/train_smoke_async.sh   # scale up test

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
unset SIF OVERLAY
source "$ROOT/env.sh"

STAMP="$(date +%Y%m%d.%H%M%S)"
RUN_ID="train_smoke_async_${STAMP}"

MAX_TOOL_TURNS="${MAX_TOOL_TURNS:-1}"
MODEL="${MODEL:-Qwen/Qwen3.5-0.8B}"

# Smoke-sized budgets (tiny so MAX_RESPONSE_LEN stays around ~1.5k tokens).
# Promote to the production values (12288 / 2048 / 4096) after the async path
# is validated end-to-end.
THINKING_BUDGET="${THINKING_BUDGET:-512}"
TOOL_CALL_BUDGET="${TOOL_CALL_BUDGET:-256}"
ANSWER_BUDGET="${ANSWER_BUDGET:-256}"

# Re-stamp the smoke parquet with the correct system prompt for this MAX_TOOL_TURNS,
# same as scripts/train_smoke_interrupt.sbatch. The base parquet is whatever the
# most recent make_smoke_parquet.sh run produced; update BASE_PARQUET if you
# regenerate it.
BASE_PARQUET="${BASE_PARQUET:-$ROOT/outputs/smoke_tir_qwen35_20260408.104819/smoke16.parquet}"
SMOKE_DATA="$ROOT/outputs/rollout_dumps/${RUN_ID}/smoke_turns${MAX_TOOL_TURNS}.parquet"
mkdir -p "$(dirname "$SMOKE_DATA")"

if [[ ! -f "$BASE_PARQUET" ]]; then
    echo "ERROR: BASE_PARQUET not found: $BASE_PARQUET" >&2
    echo "       Run scripts/make_smoke_parquet.sh first, or set BASE_PARQUET to an existing smoke parquet." >&2
    exit 1
fi

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

# Launch the async training run.
#   - TRAIN_BATCH=4, ROLLOUT_N=4, TOTAL_STEPS=2 → 8 total rollouts, 2 param syncs
#   - N_GPUS_ROLLOUT=1, N_GPUS_TRAIN=1 → 1+1 split on the 2-GPU A100 node
#   - VLLM_GPU_MEM_UTIL=0.8 is safe because the rollout GPU is dedicated
MODEL="$MODEL" \
TRAIN_BATCH=4 \
ROLLOUT_N=4 \
TOTAL_STEPS=2 \
MAX_TOOL_TURNS="$MAX_TOOL_TURNS" \
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
VERL_DUMP_DIR="$ROOT/outputs/rollout_dumps/${RUN_ID}" \
WANDB_PROJECT="${WANDB_PROJECT:-physcode_tir_smoke}" \
bash "$ROOT/scripts/train_async.sh"
