#!/usr/bin/env bash
# dump_rollouts.sh — Generate TIR rollout transcripts for trajectory inspection.
#
# Runs the same 2-phase vLLM loop as VeRL's ToolAgentLoop (hermes format,
# enable_thinking=True for phase 1 / False for phase 2, tool schema injected)
# and saves one txt file per rollout.
#
# Each txt shows: problem → phase1 reasoning+tool_call → extracted code →
# sandbox stdout → phase2 final answer → boxed answer + verifier score.
#
# Usage:
#   bash scripts/dump_rollouts.sh                   # 4 rollouts, 0.6B, corpus
#   N=8 bash scripts/dump_rollouts.sh               # 8 rollouts
#   MODEL=Qwen/Qwen3.5-4B bash scripts/dump_rollouts.sh
#   PARQUET=data/processed/drsci_train.parquet N=4 bash scripts/dump_rollouts.sh

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
source "$ROOT/env.sh"

MODEL="${MODEL:-Qwen/Qwen3-0.6B}"
PARQUET="${PARQUET:-data/processed/corpus_train.parquet}"
N="${N:-4}"
SEED="${SEED:-42}"
GPU_MEM="${GPU_MEM:-0.38}"
MAX_TOKENS="${MAX_TOKENS:-8192}"
TEMPERATURE="${TEMPERATURE:-1.0}"
TOP_P="${TOP_P:-0.9}"

# Resolve relative parquet path
[[ "$PARQUET" != /* ]] && PARQUET="$ROOT/$PARQUET"

TIMESTAMP=$(date +%Y%m%d.%H%M%S)
OUT_DIR="$ROOT/outputs/rollouts_${TIMESTAMP}"
mkdir -p "$OUT_DIR"

echo "=== dump_rollouts: TIR trajectory inspection ==="
echo "  model   : $MODEL"
echo "  parquet : $PARQUET  (n=$N)"
echo "  gpu_mem : $GPU_MEM"
echo "  output  : $OUT_DIR"

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
  "$SIF" \
  python3 "$ROOT/scripts/dump_rollouts.py" \
    --model "$MODEL" \
    --parquet "$PARQUET" \
    --n "$N" \
    --out_dir "$OUT_DIR" \
    --seed "$SEED" \
    --gpu_mem "$GPU_MEM" \
    --max_tokens "$MAX_TOKENS" \
    --temperature "$TEMPERATURE" \
    --top_p "$TOP_P" \
    --enable_thinking \
  2>&1 | tee "$OUT_DIR/run.log"

echo ""
echo "Rollout files:"
ls "$OUT_DIR"/*.txt 2>/dev/null || true
