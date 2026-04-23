#!/bin/bash
# smoke_tir_debug_extxv.sbatch — debug-qos VeRL smoke, points at an EXTERNAL
# vLLM xverify server (instead of launching its own).
#
# Purpose: verify that the new vLLM-backed xverify server works correctly with
# VeRL's training reward pipeline. 2 het groups (trainer 1x4 + rollout 3x4),
# 30-min cap, 1 training step is sufficient to exercise the reward path.
#
# Submit:
#   # XVERIFY_URL must be set; default reads it from outputs/xverify_endpoints/vllm.url
#   sbatch --export=ALL,EXTERNAL_XVERIFY_URL=http://nidXXXXXX:8765/judge \
#     scripts/perlmutter/smoke_tir_debug_extxv.sbatch

#SBATCH --job-name=smoke_tir_debug_extxv
#SBATCH --output=logs/%x_%j.log
#SBATCH --error=logs/%x_%j.err

# --- het-group 0: TRAINER (1 × 4 GPU) ---
#SBATCH -A m2651
#SBATCH -C gpu&hbm80g
#SBATCH -q debug
#SBATCH -t 00:30:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-node=4

#SBATCH hetjob
# --- het-group 1: ROLLOUT (3 × 4 GPU) ---
#SBATCH -A m2651
#SBATCH -C gpu&hbm80g
#SBATCH -q debug
#SBATCH -t 00:30:00
#SBATCH -N 3
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-node=4

set -euo pipefail
ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/../.." && pwd)}"
cd "$ROOT"
mkdir -p logs outputs/xverify_endpoints

unset SIF OVERLAY
source "$ROOT/env.sh"

echo "=== smoke_tir_debug_extxv.sbatch ==="
echo "  Trainer : $SLURM_JOB_NODELIST_HET_GROUP_0"
echo "  Rollout : $SLURM_JOB_NODELIST_HET_GROUP_1"

# --- External xverify URL discovery ---
EXTERNAL_XVERIFY_URL="${EXTERNAL_XVERIFY_URL:-}"
if [[ -z "$EXTERNAL_XVERIFY_URL" ]]; then
    # Fall back to vllm.url rendezvous file
    if [[ -f "$ROOT/outputs/xverify_endpoints/vllm.url" ]]; then
        EXTERNAL_XVERIFY_URL=$(head -n1 "$ROOT/outputs/xverify_endpoints/vllm.url")
    fi
fi
[[ -z "$EXTERNAL_XVERIFY_URL" ]] && { echo "ERROR: EXTERNAL_XVERIFY_URL not set and vllm.url missing" >&2; exit 1; }
echo "[xverify] using external server: $EXTERNAL_XVERIFY_URL"

# Probe the server before kicking off training
HEALTH_URL="${EXTERNAL_XVERIFY_URL%/judge}/health"
if ! curl -sf --max-time 5 "$HEALTH_URL" | grep -q '"ready":[[:space:]]*true'; then
    echo "ERROR: xverify server health probe failed at $HEALTH_URL" >&2
    exit 1
fi
echo "[xverify] health check OK"

# Write the URL into the rendezvous file too — VeRL reward.py falls back to
# this if XVERIFY_URL env is unset, and the smoke uses both pathways.
{
    echo "$EXTERNAL_XVERIFY_URL"
    echo "# external_server"
    echo "# job_id=$SLURM_JOB_ID"
    echo "# written_at=$(date -Iseconds)"
} > "$ROOT/outputs/xverify_endpoints/current.url"

XVERIFY_URL="$EXTERNAL_XVERIFY_URL"
export XVERIFY_URL
echo "[xverify] URL=$XVERIFY_URL"

APT_CMD=(
    apptainer exec --nv
    --overlay "$OVERLAY:ro" --no-home --bind /etc/pki:/etc/pki
    --env "PYTHONNOUSERSITE=1" --env "PYTHONUNBUFFERED=1"
    --env "PYTHONPATH=$ROOT/.async-extras:/opt/phys-extras/"
    --env "HF_HOME=$HF_HOME" --env "XDG_CACHE_HOME=/tmp/.cache"
    --env "FLASHINFER_WORKSPACE_BASE=/tmp" --env "HOME=/tmp"
    --env "TRITON_CACHE_DIR=/tmp/.cache/triton"
    --env "MPLCONFIGDIR=/tmp/.cache/matplotlib"
    --env "PHYS_REQUIRE_XVERIFY=1"
    --env "PYTORCH_ALLOC_CONF=expandable_segments:True"
    --env "NCCL_IB_TIMEOUT=32"
    --env "NCCL_IBEXT_DISABLE=1"
    --env "TORCH_NCCL_ENABLE_MONITORING=0"
    "$SIF"
)
export APT_CMD

HEAD_NODE=$(scontrol show hostnames "$SLURM_JOB_NODELIST_HET_GROUP_0" | head -n1)
WORKER_NODES=( $(scontrol show hostnames "$SLURM_JOB_NODELIST_HET_GROUP_1") )
export HEAD_NODE WORKER_NODES

# Resource knobs MUST match the #SBATCH directives above
export N_GPUS_TRAIN="${N_GPUS_TRAIN:-4}"
export N_GPUS_ROLLOUT="${N_GPUS_ROLLOUT:-4}"
export NNODES_TRAIN="${NNODES_TRAIN:-1}"
export NNODES_ROLLOUT="${NNODES_ROLLOUT:-3}"
export HET_SRUN_HEAD="--het-group=0" HET_SRUN_WORKER="--het-group=1"
source "$ROOT/scripts/perlmutter/_ray_bringup.sh"

cleanup() {
    local rc=$?
    type ray_teardown &>/dev/null && ray_teardown || true
    return $rc
}
trap cleanup EXIT

# ----- smoke overrides -----
export MODEL="${MODEL:-Qwen/Qwen3-4B-Thinking-2507}"
export TRAIN_BATCH="${TRAIN_BATCH:-128}"
export ROLLOUT_N="${ROLLOUT_N:-8}"
export TOTAL_STEPS="${TOTAL_STEPS:-2}"   # 1-2 steps is enough to exercise reward
export SAVE_FREQ="${SAVE_FREQ:-99}"
export TEST_FREQ="${TEST_FREQ:-99}"
export THINKING_BUDGET="${THINKING_BUDGET:-}"
export TOOL_CALL_BUDGET="${TOOL_CALL_BUDGET:-}"
export ANSWER_BUDGET="${ANSWER_BUDGET:-}"
export VLLM_GPU_MEM_UTIL="${VLLM_GPU_MEM_UTIL:-0.85}"
export USE_DYNAMIC_BSZ="${USE_DYNAMIC_BSZ:-1}"
export PPO_MAX_TOKEN_LEN_PER_GPU="${PPO_MAX_TOKEN_LEN_PER_GPU:-20480}"
export ACTOR_OPT_OFFLOAD="${ACTOR_OPT_OFFLOAD:-True}"
export TRAIN_SP="${TRAIN_SP:-2}"
export STALENESS="${STALENESS:-1}"
export DUMP_TRAIN_ROLLOUTS=1
export DUMP_VAL_ROLLOUTS=0   # skip val to save time on debug

export TRAIN_FILES="${TRAIN_FILES:-$ROOT/data/processed_tir/data/train.parquet}"
export VAL_FILES="${VAL_FILES:-$ROOT/data/processed_tir/data/validation.parquet}"

export WANDB_PROJECT="${WANDB_PROJECT:-physcode_tir_smoke_extxv}"
export LOGGERS="${LOGGERS:-console}"   # skip wandb on a 30-min smoke
export EXPERIMENT="${EXPERIMENT:-smoke_tir_debug_extxv_$(date +%Y%m%d.%H%M%S)}"

echo "[driver] launching train_async.sh (EXPERIMENT=$EXPERIMENT)"
srun --overlap --het-group=0 --nodes=1 --ntasks=1 -w "$HEAD_NODE" \
     bash "$ROOT/scripts/train_async.sh"
DRIVER_EXIT=$?
echo "[driver] exited with $DRIVER_EXIT"
exit $DRIVER_EXIT
