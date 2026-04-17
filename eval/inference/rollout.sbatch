#!/bin/bash
#SBATCH --job-name=dump_rollouts
# Partition / QOS / gres are NOT set here. Control them via env.sh (SBATCH_PARTITION)
# or override at submit time, e.g.:
#   sbatch --partition=gpu_b200 --gres=gpu:b200:1 scripts/dump_rollouts.sbatch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=8:00:00
#SBATCH --output=logs/%x_%j.log
#SBATCH --error=logs/%x_%j.err

# Configurable via --export:
#   MODEL             — HF model ID (default: Qwen/Qwen3.5-4B)
#   N                 — number of problems to sample, <=0 for all (default: 4)
#   N_ROLLOUTS        — rollouts per problem (default: 1)
#   PARQUET           — parquet file, or "smoke" for a built-in trivial 1-problem
#                       parquet useful for format/thinking smoke tests (default: corpus_train.parquet)
#   GPU_MEM           — vLLM gpu_memory_utilization (default: 0.7)
#   MAX_TOKENS        — max response tokens when think-interrupt disabled (default: 8192)
#   THINKING_BUDGET   — phase 1 token cap; enables think-interrupt when set
#   TOOL_CALL_BUDGET  — phase 1b cap (required with THINKING_BUDGET)
#   ANSWER_BUDGET     — phase 2 cap (required with THINKING_BUDGET)
#   START_IDX, END_IDX — shard slice for parallel chunks
#   CHUNK_SIZE        — problems per flush chunk (crash safety)
#   SEED              — random seed (default: 42)
#   ENABLE_THINKING   — set to 1 to enable <think> in phase 1 (default: 1)
#
# Usage:
#   sbatch scripts/dump_rollouts.sbatch
#   sbatch --partition=gpu_b200 --gres=gpu:b200:1 \
#     --export=ALL,N=-1,N_ROLLOUTS=4,PARQUET=data/processed/probe_subset.parquet,\
#THINKING_BUDGET=12288,TOOL_CALL_BUDGET=2048,ANSWER_BUDGET=4096,START_IDX=0,END_IDX=1250 \
#     scripts/dump_rollouts.sbatch

ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
source "$ROOT/env.sh"

MODEL="${MODEL:-Qwen/Qwen3.5-4B}"
N="${N:-4}"
N_ROLLOUTS="${N_ROLLOUTS:-1}"
PARQUET="${PARQUET:-$ROOT/data/processed/corpus_train.parquet}"
GPU_MEM="${GPU_MEM:-0.7}"
MAX_TOKENS="${MAX_TOKENS:-8192}"
SEED="${SEED:-42}"
ENABLE_THINKING="${ENABLE_THINKING:-1}"  # 1=enable <think> before tool call (recommended); 0 to suppress

# Think-interrupt budgets (all three required together, or all empty)
THINKING_BUDGET="${THINKING_BUDGET:-}"
TOOL_CALL_BUDGET="${TOOL_CALL_BUDGET:-}"
ANSWER_BUDGET="${ANSWER_BUDGET:-}"

# Sampling params (optional — defaults match dump_rollouts.py)
TOP_K="${TOP_K:-}"
PRESENCE_PENALTY="${PRESENCE_PENALTY:-}"

# Sharding / chunking (optional)
START_IDX="${START_IDX:-}"
END_IDX="${END_IDX:-}"
CHUNK_SIZE="${CHUNK_SIZE:-}"

# Resolve relative parquet path
[[ "$PARQUET" != /* && "$PARQUET" != "smoke" ]] && PARQUET="$ROOT/$PARQUET"

if [[ -n "${OUT_DIR:-}" ]]; then
    [[ "$OUT_DIR" != /* ]] && OUT_DIR="$ROOT/$OUT_DIR"
else
    TIMESTAMP=$(date +%Y%m%d.%H%M%S)
    OUT_DIR="$ROOT/outputs/rollouts_${TIMESTAMP}_${SLURM_JOB_ID}"
fi
mkdir -p "$OUT_DIR" "$ROOT/logs"

export PYTHONUNBUFFERED=1
export PYTHONNOUSERSITE=1

# PARQUET=smoke: generate a trivial one-problem parquet for format/thinking smoke tests.
# The problem is intentionally simple (ball drop) so the model finishes well within token budget.
if [ "$PARQUET" = "smoke" ]; then
    PARQUET="$OUT_DIR/smoke.parquet"
    PYTHONNOUSERSITE=1 apptainer exec \
      --overlay "$OVERLAY:ro" --no-home \
      --bind /etc/pki:/etc/pki \
      --env "PYTHONNOUSERSITE=1" \
      "$SIF" python3 - <<'PY' "$PARQUET"
import sys, pandas as pd
from phys_reasoner.tir.prompts import TIR_SYSTEM_PROMPT
path = sys.argv[1]
pd.DataFrame([{
    "data_source": "smoke",
    "prompt": [
        {"role": "system", "content": TIR_SYSTEM_PROMPT},
        {"role": "user",   "content": "A ball is dropped from rest. After t = 3 s, how far has it fallen? Use g = 9.8 m/s²."},
    ],
    "reward_model": {"ground_truth": "44.1", "style": "rule"},
    "extra_info": {"answer_type": "numerical", "unit": "m", "problem":
        "A ball is dropped from rest. After t = 3 s, how far has it fallen? Use g = 9.8 m/s²."},
}]).to_parquet(path, index=False)
print(f"Wrote smoke parquet → {path}")
PY
fi

echo "Job:        $SLURM_JOB_ID  ($SLURM_JOB_NAME)"
echo "Node:       $SLURMD_NODENAME"
echo "GPU:        $CUDA_VISIBLE_DEVICES"
echo "Model:      $MODEL"
echo "Parquet:    $PARQUET"
echo "N:          $N  n_rollouts=$N_ROLLOUTS  seed=$SEED"
if [[ -n "$THINKING_BUDGET" ]]; then
    echo "Interrupt:  thinking=$THINKING_BUDGET tool_call=$TOOL_CALL_BUDGET answer=$ANSWER_BUDGET"
else
    echo "Max tokens: $MAX_TOKENS (no think-interrupt)"
fi
echo "gpu_mem=$GPU_MEM  enable_thinking=$ENABLE_THINKING"
[[ -n "$START_IDX$END_IDX" ]] && echo "Shard:      [$START_IDX:$END_IDX)"
[[ -n "$CHUNK_SIZE" ]] && echo "chunk_size: $CHUNK_SIZE"
echo "Output:     $OUT_DIR"

apptainer exec --nv \
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
  ${VLLM_USE_TRTLLM_ATTENTION:+--env "VLLM_USE_TRTLLM_ATTENTION=$VLLM_USE_TRTLLM_ATTENTION"} \
  "$SIF" \
  python3 "$ROOT/scripts/dump_rollouts.py" \
    --model    "$MODEL" \
    --parquet  "$PARQUET" \
    --n        "$N" \
    --n_rollouts "$N_ROLLOUTS" \
    --out_dir  "$OUT_DIR" \
    --seed     "$SEED" \
    --gpu_mem  "$GPU_MEM" \
    --max_tokens "$MAX_TOKENS" \
    --temperature "${TEMPERATURE:-1.0}" \
    --top_p    "${TOP_P:-1.0}" \
    ${TOP_K:+--top_k "$TOP_K"} \
    ${PRESENCE_PENALTY:+--presence_penalty "$PRESENCE_PENALTY"} \
    $( [ "$ENABLE_THINKING" = "1" ] && echo "--enable_thinking" ) \
    ${THINKING_BUDGET:+--thinking_budget "$THINKING_BUDGET"} \
    ${TOOL_CALL_BUDGET:+--tool_call_budget "$TOOL_CALL_BUDGET"} \
    ${ANSWER_BUDGET:+--answer_budget "$ANSWER_BUDGET"} \
    ${START_IDX:+--start_idx "$START_IDX"} \
    ${END_IDX:+--end_idx "$END_IDX"} \
    ${CHUNK_SIZE:+--chunk_size "$CHUNK_SIZE"}

echo "Exit: $?"
echo "Rollouts saved to: $OUT_DIR"
