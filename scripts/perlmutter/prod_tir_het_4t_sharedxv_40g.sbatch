#!/bin/bash
# prod_tir_het_4t_sharedxv_40g.sbatch — TIR prod on A100 40GB (plan #1 first submission).
#
# Same as prod_tir_het_4t_sharedxv.sbatch EXCEPT:
#   - constraint:        gpu&hbm80g → gpu&hbm40g
#   - ACTOR_OPT_OFFLOAD: False      → True   (~32 GB AdamW state → CPU)
#   - VLLM_GPU_MEM_UTIL: 0.85       → 0.72   (leave ~9 GB for CheckpointEngineWorker
#                                              + 2 GB weight-sync buffer + margin)
#
# Memory knobs validated via smoke_cot_het_40g.sbatch on gpu_debug (2T+2R, 4 nodes):
#   v1 @ VLLM=0.80, PPO_MAX=20480: OOM on weight-sync (CUDA graphs push vLLM
#       actual memory to 34.5 GB of 40 GB, <1 GB left for 2 GB sync buffer).
#   v2 @ VLLM=0.65, PPO_MAX=16384: PASSED, step=279.5s at smoke scale.
#   v3 @ VLLM=0.72, PPO_MAX=20480: PASSED, step=274.1s — final config.
#
# TIR differs from CoT only in rollout agent loop (tool_agent vs single_turn_agent)
# and data parquet (processed_tir vs processed_cot). Trainer/rollout memory
# footprint is identical — same model, same FSDP, same max_response_length.
# CoT smoke therefore validates TIR memory-fit. Tool response injection (1024-char
# cap) is char-count, not GPU-bound.
#
# Rationale for switching to 40G: hbm80g pool has 256 nodes with 0 idle and
# ~7-25h premium queue wait. hbm40g has 1408 nodes with zero pending premium
# jobs ahead — expected wait ~1–4h.
#
# ETA: ~3.8 min/step × 417 steps (3 epochs over 17,827-row dataset) ≈ 26.5h
# + init/val overhead ≈ 27–28h. Fits 30h walltime with ~2h margin.

#SBATCH --job-name=physcode_prod_tir_het_4t_sxv_40g
#SBATCH --output=logs/%x_%j.log
#SBATCH --error=logs/%x_%j.err

# =============================================================================
# HET-GROUP 0: TRAINER 4×4 (FSDP-16) on 40G
# =============================================================================
#SBATCH -A m2651
#SBATCH -C gpu&hbm40g
#SBATCH -q premium
#SBATCH -t 30:00:00
#SBATCH -N 4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-node=4

#SBATCH hetjob
# =============================================================================
# HET-GROUP 1: ROLLOUT 6×4 on 40G
# =============================================================================
#SBATCH -A m2651
#SBATCH -C gpu&hbm40g
#SBATCH -q premium
#SBATCH -t 30:00:00
#SBATCH -N 6
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-node=4

set -euo pipefail
ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/../.." && pwd)}"
cd "$ROOT"
mkdir -p logs outputs/xverify_endpoints

unset SIF OVERLAY
source "$ROOT/env.sh"
[[ -f "$SIF" && -f "$OVERLAY" ]] || { echo "ERROR: SIF/OVERLAY missing" >&2; exit 1; }

echo "=== prod_tir_het_4t_sharedxv_40g.sbatch (TIR prod, 40G premium) ==="
echo "  Job       : $SLURM_JOB_ID"
echo "  Trainer   : $SLURM_JOB_NODELIST_HET_GROUP_0  (FSDP-16 on 40G)"
echo "  Rollout   : $SLURM_JOB_NODELIST_HET_GROUP_1"
echo "  xVerify   : shared (external, read from outputs/xverify_endpoints/current.url)"
echo "  SIF       : $SIF"
echo "  OVERLAY   : $OVERLAY"
echo "  HF_HOME   : $HF_HOME"

APT_CMD=(
    apptainer exec --nv
    --overlay "$OVERLAY:ro"
    --no-home
    --bind /etc/pki:/etc/pki
    --env "PYTHONNOUSERSITE=1"
    --env "PYTHONUNBUFFERED=1"
    --env "PYTHONPATH=$ROOT/.async-extras:/opt/phys-extras/"
    --env "HF_HOME=$HF_HOME"
    --env "XDG_CACHE_HOME=/tmp/.cache"
    --env "FLASHINFER_WORKSPACE_BASE=/tmp"
    --env "HOME=/tmp"
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

TRAIN_NODES=( $(scontrol show hostnames "$SLURM_JOB_NODELIST_HET_GROUP_0") )
HEAD_NODE="${TRAIN_NODES[0]}"
TRAIN_WORKER_NODES=( "${TRAIN_NODES[@]:1}" )
WORKER_NODES=( $(scontrol show hostnames "$SLURM_JOB_NODELIST_HET_GROUP_1") )
export HEAD_NODE TRAIN_WORKER_NODES WORKER_NODES

echo "[topology] HEAD=$HEAD_NODE  TRAIN_WORKERS=${TRAIN_WORKER_NODES[*]}  ROLLOUT=${WORKER_NODES[*]}"
(( ${#TRAIN_WORKER_NODES[@]} == 3 )) || echo "WARN: expected 3 trainer workers, got ${#TRAIN_WORKER_NODES[@]}"
(( ${#WORKER_NODES[@]} == 6 )) || echo "WARN: expected 6 rollout workers, got ${#WORKER_NODES[@]}"

export N_GPUS_TRAIN="${N_GPUS_TRAIN:-4}"
export N_GPUS_ROLLOUT="${N_GPUS_ROLLOUT:-4}"
export NNODES_TRAIN="${NNODES_TRAIN:-4}"
export NNODES_ROLLOUT="${NNODES_ROLLOUT:-6}"
export HET_SRUN_HEAD="--het-group=0"
export HET_SRUN_WORKER="--het-group=1"
source "$ROOT/scripts/perlmutter/_ray_bringup.sh"

cleanup() {
    local rc=$?
    echo "[cleanup] exit=$rc — tearing down Ray"
    type ray_teardown &>/dev/null && ray_teardown || true
    return $rc
}
trap cleanup EXIT

URL_FILE="${SHARED_URL_FILE:-$ROOT/outputs/xverify_endpoints/current.url}"
XVERIFY_WAIT_SECS="${XVERIFY_WAIT_SECS:-3600}"
echo "[xverify] polling shared URL file $URL_FILE (timeout ${XVERIFY_WAIT_SECS}s) ..."
deadline=$(( $(date +%s) + XVERIFY_WAIT_SECS ))
while (( $(date +%s) < deadline )); do
    [[ -f "$URL_FILE" ]] && { XVERIFY_URL=$(head -n1 "$URL_FILE"); break; }
    sleep 15
done
[[ -z "${XVERIFY_URL:-}" ]] && { echo "ERROR: shared xverify URL never appeared — is serve_xverify_perlmutter.sbatch running?" >&2; exit 1; }
echo "[xverify] shared URL = $XVERIFY_URL"
health_url="${XVERIFY_URL%/judge}/health"
hc_deadline=$(( $(date +%s) + 300 ))
while (( $(date +%s) < hc_deadline )); do
    curl -sf --max-time 5 "$health_url" >/dev/null 2>&1 && { echo "[xverify] healthy"; break; }
    sleep 10
done
export XVERIFY_URL

# ---- production knobs (same model/batch/budgets/temperature as 80G prod) ----
export MODEL="${MODEL:-Qwen/Qwen3-4B}"
export TRAIN_TEMPERATURE="${TRAIN_TEMPERATURE:-1.4}"
export VAL_TEMPERATURE="${VAL_TEMPERATURE:-1.0}"

export TRAIN_BATCH="${TRAIN_BATCH:-128}"
export ROLLOUT_N="${ROLLOUT_N:-8}"
export TOTAL_STEPS="${TOTAL_STEPS:-552}"
export TOTAL_EPOCHS="${TOTAL_EPOCHS:-3}"
export SAVE_FREQ="${SAVE_FREQ:-50}"
export TEST_FREQ="${TEST_FREQ:-50}"

# Prod budgets — MAX_RESPONSE_LEN = 12288 + 16 + 2048 + 1024 + 4096 = 19,472.
export THINKING_BUDGET="${THINKING_BUDGET:-12288}"
export TOOL_CALL_BUDGET="${TOOL_CALL_BUDGET:-2048}"
export ANSWER_BUDGET="${ANSWER_BUDGET:-4096}"

# 40G memory knobs (validated via smoke v3 — see header):
#   VLLM_GPU_MEM_UTIL=0.72   — vLLM peaks at ~31 GB actual (incl CUDA graphs),
#                              leaving ~9 GB for CheckpointEngineWorker + 2 GB
#                              weight-sync buffer + margin
#   PPO_MAX_TOKEN_LEN_PER_GPU=20480 — same as 80G prod (confirmed safe on 40G)
#   ACTOR_OPT_OFFLOAD=True   — 32 GB AdamW state would not fit on 40G trainer GPU
export VLLM_GPU_MEM_UTIL="${VLLM_GPU_MEM_UTIL:-0.72}"
export USE_DYNAMIC_BSZ="${USE_DYNAMIC_BSZ:-1}"
export PPO_MAX_TOKEN_LEN_PER_GPU="${PPO_MAX_TOKEN_LEN_PER_GPU:-20480}"
export ACTOR_OPT_OFFLOAD="${ACTOR_OPT_OFFLOAD:-True}"

export TRAIN_SP="${TRAIN_SP:-2}"
export STALENESS="${STALENESS:-1}"

# LR bumped 1e-6 → 3e-6 (2026-04-22): prior 150-step run with LR=1e-6 showed
# flat reward + ppo_kl ≈ 8e-4 per step. After filtering saturated prompts, a
# larger LR is safe and needed to produce visible policy movement in ~200 steps.
export LR="${LR:-3e-6}"

# Post-filter parquets (Qwen3-4B probe at temp=1.4, kept 1-7/8 informative).
# Set via --export to point at filtered files; defaults to original 17k.
export TRAIN_FILES="${TRAIN_FILES:-$ROOT/data/processed_tir/data/train.parquet}"
export VAL_FILES="${VAL_FILES:-$ROOT/data/processed_tir/data/validation.parquet}"

export WANDB_PROJECT="${WANDB_PROJECT:-physcode_tir}"
export LOGGERS="${LOGGERS:-console,wandb}"
export EXPERIMENT="${EXPERIMENT:-grpo_tir_qwen3b_t14_lr3e6_40g_$(date +%Y%m%d.%H%M%S)}"

echo "[driver] launching train_async.sh on $HEAD_NODE (EXPERIMENT=$EXPERIMENT, FSDP=$((NNODES_TRAIN*N_GPUS_TRAIN)))"
srun --overlap --het-group=0 --nodes=1 --ntasks=1 -w "$HEAD_NODE" \
     bash "$ROOT/scripts/train_async.sh"
DRIVER_EXIT=$?
echo "[driver] exited with $DRIVER_EXIT"
exit $DRIVER_EXIT
