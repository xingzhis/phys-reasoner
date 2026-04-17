#!/usr/bin/env bash
# dump_rollouts.sh — Generate TIR rollout transcripts for trajectory inspection.
#
# Runs the same 2-phase vLLM loop as VeRL's ToolAgentLoop (qwen3_coder format,
# enable_thinking=True for phase 1 / False for phase 2, tool schema injected)
# and saves one txt file per rollout + a rollouts.parquet.
#
# Think-interrupt: set THINKING_BUDGET to enable. When enabled, TOOL_CALL_BUDGET
# and ANSWER_BUDGET are also required. Budget identity (matches VeRL):
#   response_length = THINKING_BUDGET + INTERRUPT_LEN + TOOL_CALL_BUDGET
#                   + MAX_TOOL_RESPONSE_LEN + ANSWER_BUDGET
#
# Usage:
#   bash scripts/dump_rollouts.sh                   # 4 problems, 1 rollout, 0.6B
#   N=8 N_ROLLOUTS=8 bash scripts/dump_rollouts.sh  # 8 problems x 8 rollouts
#   N=-1 bash scripts/dump_rollouts.sh              # all rows, parquet order (eval mode)
#   DUMP_TXT=1 bash scripts/dump_rollouts.sh        # also emit per-rollout txt files
#   MODEL=Qwen/Qwen3.5-4B bash scripts/dump_rollouts.sh
#   PARQUET=data/processed/drsci_train.parquet bash scripts/dump_rollouts.sh
#
#   # With think-interrupt:
#   MODEL=Qwen/Qwen3.5-4B THINKING_BUDGET=12288 TOOL_CALL_BUDGET=2048 \
#     ANSWER_BUDGET=4096 bash scripts/dump_rollouts.sh

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
source "$ROOT/env.sh"

MODEL="${MODEL:-Qwen/Qwen3-0.6B}"
PARQUET="${PARQUET:-data/processed/corpus_train.parquet}"
N="${N:-4}"
N_ROLLOUTS="${N_ROLLOUTS:-1}"
SEED="${SEED:-42}"
GPU_MEM="${GPU_MEM:-0.38}"
TEMPERATURE="${TEMPERATURE:-1.0}"
TOP_P="${TOP_P:-1.0}"
MAX_TOKENS="${MAX_TOKENS:-8192}"
MAX_PROMPT_LEN="${MAX_PROMPT_LEN:-1024}"
MAX_TOOL_RESPONSE_LEN="${MAX_TOOL_RESPONSE_LEN:-512}"
DUMP_TXT="${DUMP_TXT:-}"

# Think-interrupt budgets (all three required together, or all empty)
THINKING_BUDGET="${THINKING_BUDGET:-}"
TOOL_CALL_BUDGET="${TOOL_CALL_BUDGET:-}"
ANSWER_BUDGET="${ANSWER_BUDGET:-}"

# Sharding / chunking (all optional)
START_IDX="${START_IDX:-}"
END_IDX="${END_IDX:-}"
CHUNK_SIZE="${CHUNK_SIZE:-}"

# Resolve relative parquet path
[[ "$PARQUET" != /* ]] && PARQUET="$ROOT/$PARQUET"

if [[ -n "${OUT_DIR:-}" ]]; then
    [[ "$OUT_DIR" != /* ]] && OUT_DIR="$ROOT/$OUT_DIR"
else
    TIMESTAMP=$(date +%Y%m%d.%H%M%S)
    OUT_DIR="$ROOT/outputs/rollouts_${TIMESTAMP}"
fi
mkdir -p "$OUT_DIR"

echo "=== dump_rollouts: TIR trajectory inspection ==="
echo "  model       : $MODEL"
echo "  parquet     : $PARQUET  (n=$N, n_rollouts=$N_ROLLOUTS)"
echo "  gpu_mem     : $GPU_MEM"
echo "  output      : $OUT_DIR"
if [[ -n "$THINKING_BUDGET" ]]; then
    echo "  interrupt   : thinking=$THINKING_BUDGET tool_call=$TOOL_CALL_BUDGET answer=$ANSWER_BUDGET"
else
    echo "  max_tokens  : $MAX_TOKENS (no think-interrupt)"
fi

if [[ "${PYTORCH_CUDA_ALLOC_CONF:-}" == *"expandable_segments:True"* ]]; then
    unset PYTORCH_CUDA_ALLOC_CONF
fi

PYTHONNOUSERSITE=1 apptainer exec --nv \
  --overlay "$OVERLAY:ro" \
  --no-home \
  --bind /etc/pki:/etc/pki \
  --env "PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin" \
  --env "PYTHONNOUSERSITE=1" \
  --env "PYTHONPATH=/opt/phys-extras/" \
  --env "PYTHONUNBUFFERED=1" \
  --env "HF_HOME=$HF_HOME" \
  --env "HF_DATASETS_OFFLINE=1" \
  --env "XDG_CACHE_HOME=/tmp/.cache" \
  --env "TRITON_CACHE_DIR=/tmp/.cache/triton" \
  --env "FLASHINFER_WORKSPACE_BASE=/tmp" \
  --env "MPLCONFIGDIR=/tmp/.cache/matplotlib" \
  "$SIF" \
  python3 "$ROOT/scripts/dump_rollouts.py" \
    --model "$MODEL" \
    --parquet "$PARQUET" \
    --n "$N" \
    --n_rollouts "$N_ROLLOUTS" \
    --out_dir "$OUT_DIR" \
    --seed "$SEED" \
    --gpu_mem "$GPU_MEM" \
    --max_tokens "$MAX_TOKENS" \
    --temperature "$TEMPERATURE" \
    --top_p "$TOP_P" \
    --max_prompt_len "$MAX_PROMPT_LEN" \
    --max_tool_response_len "$MAX_TOOL_RESPONSE_LEN" \
    --enable_thinking \
    ${THINKING_BUDGET:+--thinking_budget "$THINKING_BUDGET"} \
    ${TOOL_CALL_BUDGET:+--tool_call_budget "$TOOL_CALL_BUDGET"} \
    ${ANSWER_BUDGET:+--answer_budget "$ANSWER_BUDGET"} \
    ${START_IDX:+--start_idx "$START_IDX"} \
    ${END_IDX:+--end_idx "$END_IDX"} \
    ${CHUNK_SIZE:+--chunk_size "$CHUNK_SIZE"} \
    ${DUMP_TXT:+--dump_txt} \
  2>&1 | tee "$OUT_DIR/run.log"

echo ""
echo "Rollout files:"
ls "$OUT_DIR"/*.txt 2>/dev/null || true
echo ""
echo "Parquet:"
ls "$OUT_DIR"/*.parquet 2>/dev/null || true
