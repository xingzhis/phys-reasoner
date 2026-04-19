#!/usr/bin/env bash
# run_eval.sh — wrapper that runs eval/run_eval.py inside the apptainer container.
#
# Required env: BENCHMARK (one of: pool_v2_{drsci,ugphysics,physics,scibench},
#                                  olympiad_oe_to_physics)
#               MODEL     (HF id or local path)
# Optional env: MODE (tir|cot, default tir), N, N_ROLLOUTS, GPU_MEM,
#               THINKING_BUDGET, TOOL_CALL_BUDGET, ANSWER_BUDGET, OUT_DIR,
#               NO_XVERIFY (=1 to disable xVerify in our_verifier scorer)
#
# Usage:
#   BENCHMARK=olympiad_oe_to_physics MODEL=Qwen/Qwen3.5-4B \
#     THINKING_BUDGET=12288 TOOL_CALL_BUDGET=2048 ANSWER_BUDGET=4096 \
#     bash eval/run_eval.sh
#
#   BENCHMARK=pool_v2_drsci MODEL=Qwen/Qwen3.5-4B MODE=cot \
#     THINKING_BUDGET=12288 TOOL_CALL_BUDGET=2048 ANSWER_BUDGET=4096 \
#     bash eval/run_eval.sh

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd -P)"
# env.sh unsets CUDA_VISIBLE_DEVICES (Ray placement). Preserve any caller-set
# pinning so the sharded launcher can place each rollout on its own GPU.
_saved_cuda="${CUDA_VISIBLE_DEVICES:-}"
source "$ROOT/env.sh"
if [ -n "$_saved_cuda" ]; then
    export CUDA_VISIBLE_DEVICES="$_saved_cuda"
fi
unset _saved_cuda

BENCHMARK="${BENCHMARK:?BENCHMARK env var required}"
MODEL="${MODEL:?MODEL env var required}"
MODE="${MODE:-tir}"
N="${N:--1}"
N_ROLLOUTS="${N_ROLLOUTS:-1}"
SEED="${SEED:-42}"
GPU_MEM="${GPU_MEM:-0.7}"
TEMPERATURE="${TEMPERATURE:-1.0}"
TOP_P="${TOP_P:-1.0}"
MAX_TOKENS="${MAX_TOKENS:-8192}"
MAX_PROMPT_LEN="${MAX_PROMPT_LEN:-1024}"
MAX_TOOL_RESPONSE_LEN="${MAX_TOOL_RESPONSE_LEN:-1024}"

THINKING_BUDGET="${THINKING_BUDGET:-}"
TOOL_CALL_BUDGET="${TOOL_CALL_BUDGET:-}"
ANSWER_BUDGET="${ANSWER_BUDGET:-}"
START_IDX="${START_IDX:-}"
END_IDX="${END_IDX:-}"
CHUNK_SIZE="${CHUNK_SIZE:-}"
DUMP_TXT="${DUMP_TXT:-}"

XVERIFY_MODEL="${XVERIFY_MODEL:-IAAR-Shanghai/xVerify-7B-I}"
XVERIFY_DEVICE="${XVERIFY_DEVICE:-cuda}"
NO_XVERIFY="${NO_XVERIFY:-}"

# Skip-stage flags (any non-empty value enables). Used by the sharded launcher
# to split the pipeline into pre-materialize → parallel shard rollouts → score.
SKIP_LOAD="${SKIP_LOAD:-}"
SKIP_ROLLOUT="${SKIP_ROLLOUT:-}"
SKIP_SCORE="${SKIP_SCORE:-}"
FORCE_LOAD="${FORCE_LOAD:-}"
FORCE_ROLLOUT="${FORCE_ROLLOUT:-}"

OUT_DIR_FLAG=""
if [[ -n "${OUT_DIR:-}" ]]; then
    [[ "$OUT_DIR" != /* ]] && OUT_DIR="$ROOT/$OUT_DIR"
    OUT_DIR_FLAG="--out_dir $OUT_DIR"
fi

echo "=== run_eval: $BENCHMARK [$MODE] @ $MODEL"

if [[ "${PYTORCH_CUDA_ALLOC_CONF:-}" == *"expandable_segments:True"* ]]; then
    unset PYTORCH_CUDA_ALLOC_CONF
fi

# Bind the worktree itself so its files (and the symlinks under it pointing at
# data/, hf_cache/, outputs/ in the sibling repo) resolve inside the container.
# Apptainer auto-binds $PWD only when it equals the user's home, which is not
# always true for git worktrees — bind explicitly to be safe.
PYTHONNOUSERSITE=1 apptainer exec --nv \
  --overlay "$OVERLAY:ro" \
  --no-home \
  --bind /etc/pki:/etc/pki \
  --bind "$ROOT" \
  --bind "$(realpath "$ROOT/data")" \
  --bind "$(realpath "$ROOT/hf_cache")" \
  --bind "$(realpath "$ROOT/outputs")" \
  --pwd "$ROOT" \
  --env "PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin" \
  --env "PYTHONNOUSERSITE=1" \
  --env "PYTHONPATH=$ROOT:$ROOT/src:$ROOT/eval/_pkgs:/opt/phys-extras/" \
  --env "PYTHONUNBUFFERED=1" \
  --env "HF_HOME=$HF_HOME" \
  --env "HF_DATASETS_OFFLINE=1" \
  --env "XDG_CACHE_HOME=/tmp/.cache" \
  --env "XDG_CONFIG_HOME=/tmp/.config" \
  --env "VLLM_NO_USAGE_STATS=1" \
  --env "TRITON_CACHE_DIR=/tmp/.cache/triton" \
  --env "FLASHINFER_WORKSPACE_BASE=/tmp" \
  --env "MPLCONFIGDIR=/tmp/.cache/matplotlib" \
  ${CUDA_VISIBLE_DEVICES:+--env "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"} \
  "$SIF" \
  python3 "$ROOT/eval/run_eval.py" \
    --benchmark "$BENCHMARK" \
    --mode "$MODE" \
    --model "$MODEL" \
    --n "$N" \
    --n_rollouts "$N_ROLLOUTS" \
    --seed "$SEED" \
    --gpu_mem "$GPU_MEM" \
    --temperature "$TEMPERATURE" \
    --top_p "$TOP_P" \
    --top_k "${TOP_K:--1}" \
    --repetition_penalty "${REPETITION_PENALTY:-1.0}" \
    --max_tokens "$MAX_TOKENS" \
    --max_prompt_len "$MAX_PROMPT_LEN" \
    --max_tool_response_len "$MAX_TOOL_RESPONSE_LEN" \
    --xverify_model "$XVERIFY_MODEL" \
    --xverify_device "$XVERIFY_DEVICE" \
    ${THINKING_BUDGET:+--thinking_budget "$THINKING_BUDGET"} \
    ${TOOL_CALL_BUDGET:+--tool_call_budget "$TOOL_CALL_BUDGET"} \
    ${ANSWER_BUDGET:+--answer_budget "$ANSWER_BUDGET"} \
    ${START_IDX:+--start_idx "$START_IDX"} \
    ${END_IDX:+--end_idx "$END_IDX"} \
    ${CHUNK_SIZE:+--chunk_size "$CHUNK_SIZE"} \
    ${DUMP_TXT:+--dump_txt} \
    ${NO_XVERIFY:+--no_xverify} \
    ${SKIP_LOAD:+--skip_load} \
    ${SKIP_ROLLOUT:+--skip_rollout} \
    ${SKIP_SCORE:+--skip_score} \
    ${FORCE_LOAD:+--force_load} \
    ${FORCE_ROLLOUT:+--force_rollout} \
    $OUT_DIR_FLAG
