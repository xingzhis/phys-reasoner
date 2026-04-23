#!/bin/bash
# prod_tir_dapo_sync_40g.sbatch — sync DAPO trainer (recipe.dapo.main_dapo) variant.
#
# Why this exists: our async path (verl.experimental.fully_async_policy) does
# NOT implement filter_groups. The sync DAPO trainer (recipe.dapo.main_dapo)
# DOES — and saturated-group filtering is what POLARIS/DAPO/ScaleRL all use to
# keep gradient signal alive on plateaus. See research notes.
#
# Topology: hybrid_engine=True, single homogeneous allocation. Trainer + rollout
# share the same N nodes (vLLM uses sleep_mode between phases). No separate
# rollout het group → simpler than the async path.
#
# Reads xverify URL from outputs/xverify_endpoints/current.url, written by a
# companion serve_xverify_vllm_40g.sbatch that must be running.
#
# Submit:
#   sbatch --export=ALL,\
#       TRAIN_FILES=$PWD/data/processed_tir_filtered/data/train.parquet,\
#       VAL_FILES=$PWD/data/processed_tir_filtered/data/validation.parquet \
#       scripts/perlmutter/prod_tir_dapo_sync_40g.sbatch

#SBATCH --job-name=physcode_prod_tir_dapo_sync_40g
#SBATCH --output=logs/%x_%j.log
#SBATCH --error=logs/%x_%j.err

# Single homogeneous allocation. Sync DAPO uses hybrid_engine — rollout shares
# trainer GPUs. 4 nodes × 4 GPU = FSDP-16, same shape as our async trainer.
#SBATCH -A m2651
#SBATCH -C gpu&hbm40g
#SBATCH -q premium
#SBATCH -t 06:00:00
#SBATCH -N 4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-node=4

set -euo pipefail
ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/../.." && pwd)}"
cd "$ROOT"
mkdir -p logs

unset SIF OVERLAY
source "$ROOT/env.sh"
[[ -f "$SIF" && -f "$OVERLAY" ]] || { echo "ERROR: SIF/OVERLAY missing" >&2; exit 1; }

echo "=== prod_tir_dapo_sync_40g.sbatch ==="
echo "  Job       : $SLURM_JOB_ID"
echo "  Nodes     : $SLURM_JOB_NODELIST  (single homogeneous; hybrid_engine)"
echo "  SIF       : $SIF"
echo "  OVERLAY   : $OVERLAY"

NODES=( $(scontrol show hostnames "$SLURM_JOB_NODELIST") )
HEAD_NODE="${NODES[0]}"
WORKER_NODES=( "${NODES[@]:1}" )
export HEAD_NODE WORKER_NODES

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

# Ray: head + N-1 workers, all in one allocation (no het). Reuse existing helper.
export N_GPUS_TRAIN="${N_GPUS_TRAIN:-4}"
export N_GPUS_ROLLOUT="${N_GPUS_TRAIN}"   # hybrid: same GPUs
export NNODES_TRAIN="${SLURM_JOB_NUM_NODES}"
export NNODES_ROLLOUT="$NNODES_TRAIN"
export HET_SRUN_HEAD=""
export HET_SRUN_WORKER=""
source "$ROOT/scripts/perlmutter/_ray_bringup.sh"

cleanup() {
    local rc=$?
    type ray_teardown &>/dev/null && ray_teardown || true
    return $rc
}
trap cleanup EXIT

# --- xverify discovery (external, from companion serve_xverify_vllm_40g.sbatch) ---
URL_FILE="${SHARED_URL_FILE:-$ROOT/outputs/xverify_endpoints/current.url}"
XVERIFY_WAIT_SECS="${XVERIFY_WAIT_SECS:-3600}"
echo "[xverify] polling $URL_FILE (timeout ${XVERIFY_WAIT_SECS}s) ..."
deadline=$(( $(date +%s) + XVERIFY_WAIT_SECS ))
while (( $(date +%s) < deadline )); do
    [[ -f "$URL_FILE" ]] && { XVERIFY_URL=$(head -n1 "$URL_FILE"); break; }
    sleep 15
done
[[ -z "${XVERIFY_URL:-}" ]] && { echo "ERROR: xverify URL never appeared in $URL_FILE" >&2; exit 1; }
echo "[xverify] using $XVERIFY_URL"
health_url="${XVERIFY_URL%/judge}/health"
hc_deadline=$(( $(date +%s) + 300 ))
while (( $(date +%s) < hc_deadline )); do
    curl -sf --max-time 5 "$health_url" >/dev/null 2>&1 && { echo "[xverify] healthy"; break; }
    sleep 10
done
export XVERIFY_URL

# --- POLARIS-aligned hparams (per research report) ---
export MODEL="${MODEL:-Qwen/Qwen3-4B}"
export TRAIN_TEMPERATURE="${TRAIN_TEMPERATURE:-1.4}"
export VAL_TEMPERATURE="${VAL_TEMPERATURE:-1.0}"

export TRAIN_BATCH="${TRAIN_BATCH:-128}"
export GEN_BATCH="${GEN_BATCH:-$((TRAIN_BATCH * 3))}"
export PPO_MINI_BATCH="${PPO_MINI_BATCH:-32}"
export ROLLOUT_N="${ROLLOUT_N:-8}"
export TOTAL_STEPS="${TOTAL_STEPS:-200}"
export TOTAL_EPOCHS="${TOTAL_EPOCHS:-3}"
export SAVE_FREQ="${SAVE_FREQ:-50}"
export TEST_FREQ="${TEST_FREQ:-50}"

# Prod budgets (same as async)
export THINKING_BUDGET="${THINKING_BUDGET:-12288}"
export TOOL_CALL_BUDGET="${TOOL_CALL_BUDGET:-2048}"
export ANSWER_BUDGET="${ANSWER_BUDGET:-4096}"

# Memory
export VLLM_GPU_MEM_UTIL="${VLLM_GPU_MEM_UTIL:-0.72}"
export DYN_BSZ_FLAG="${DYN_BSZ_FLAG:-True}"
export PPO_MAX_TOKEN_LEN_PER_GPU="${PPO_MAX_TOKEN_LEN_PER_GPU:-20480}"
export ACTOR_OPT_OFFLOAD="${ACTOR_OPT_OFFLOAD:-True}"

# POLARIS-aligned: LR=1e-6, KL=0.001, filter_groups on
export LR="${LR:-1e-6}"
export LR_WARMUP_STEPS="${LR_WARMUP_STEPS:-10}"
export KL_COEF="${KL_COEF:-0.001}"
export FILTER_GROUPS_ENABLE="${FILTER_GROUPS_ENABLE:-true}"
export FILTER_GROUPS_METRIC="${FILTER_GROUPS_METRIC:-acc}"
export FILTER_GROUPS_MAX_GENS="${FILTER_GROUPS_MAX_GENS:-10}"

export TRAIN_SP="${TRAIN_SP:-2}"

export TRAIN_FILES="${TRAIN_FILES:-$ROOT/data/processed_tir/data/train.parquet}"
export VAL_FILES="${VAL_FILES:-$ROOT/data/processed_tir/data/validation.parquet}"

export WANDB_PROJECT="${WANDB_PROJECT:-physcode_tir}"
export LOGGERS="${LOGGERS:-console,wandb}"
export EXPERIMENT="${EXPERIMENT:-grpo_tir_dapo_sync_polaris_$(date +%Y%m%d.%H%M%S)}"

# Resource knobs visible to train_dapo_sync.sh
export NNODES="$SLURM_JOB_NUM_NODES"

echo "[driver] launching train_dapo_sync.sh on $HEAD_NODE (EXPERIMENT=$EXPERIMENT)"
srun --overlap --nodes=1 --ntasks=1 -w "$HEAD_NODE" \
     bash "$ROOT/scripts/train_dapo_sync.sh"
DRIVER_EXIT=$?
echo "[driver] exited with $DRIVER_EXIT"
exit $DRIVER_EXIT
