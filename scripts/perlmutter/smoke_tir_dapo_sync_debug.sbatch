#!/bin/bash
# smoke_tir_dapo_sync_debug.sbatch — debug-qos sync DAPO smoke test.
#
# Same shape as prod_tir_dapo_sync_40g.sbatch but on debug qos with 30-min cap.
# Uses TOTAL_STEPS=2 to verify infrastructure (config validity, model load, first
# step, reward call) before consuming premium time. Reuses external xverify from
# the running serve_xverify_vllm_40g.sbatch (current.url).

#SBATCH --job-name=smoke_tir_dapo_sync_debug
#SBATCH --output=logs/%x_%j.log
#SBATCH --error=logs/%x_%j.err

#SBATCH -A m2651
#SBATCH -C gpu&hbm40g
#SBATCH -q debug
#SBATCH -t 00:30:00
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

echo "=== smoke_tir_dapo_sync_debug.sbatch ==="
echo "  Job   : $SLURM_JOB_ID"
echo "  Nodes : $SLURM_JOB_NODELIST"

NODES=( $(scontrol show hostnames "$SLURM_JOB_NODELIST") )
HEAD_NODE="${NODES[0]}"
WORKER_NODES=( "${NODES[@]:1}" )
export HEAD_NODE WORKER_NODES

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

export N_GPUS_TRAIN="${N_GPUS_TRAIN:-4}"
export N_GPUS_ROLLOUT="$N_GPUS_TRAIN"
export NNODES_TRAIN="$SLURM_JOB_NUM_NODES"
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

URL_FILE="${SHARED_URL_FILE:-$ROOT/outputs/xverify_endpoints/current.url}"
echo "[xverify] polling $URL_FILE ..."
deadline=$(( $(date +%s) + 600 ))
while (( $(date +%s) < deadline )); do
    [[ -f "$URL_FILE" ]] && { XVERIFY_URL=$(head -n1 "$URL_FILE"); break; }
    sleep 10
done
[[ -z "${XVERIFY_URL:-}" ]] && { echo "ERROR: xverify URL never appeared" >&2; exit 1; }
echo "[xverify] $XVERIFY_URL"
export XVERIFY_URL

# Smoke overrides — 2 steps, no checkpoint, no val
export MODEL="${MODEL:-Qwen/Qwen3-4B}"
export TRAIN_BATCH=128
export GEN_BATCH=$((TRAIN_BATCH * 3))
export PPO_MINI_BATCH=32
export ROLLOUT_N=8
export TOTAL_STEPS=2
export TOTAL_EPOCHS=1
export SAVE_FREQ=99
export TEST_FREQ=99
export VAL_BEFORE_TRAIN=false
export THINKING_BUDGET=12288
export TOOL_CALL_BUDGET=2048
export ANSWER_BUDGET=4096
export VLLM_GPU_MEM_UTIL=0.72
export DYN_BSZ_FLAG=True
export PPO_MAX_TOKEN_LEN_PER_GPU=20480
export ACTOR_OPT_OFFLOAD=True
export LR=1e-6
export LR_WARMUP_STEPS=10
export KL_COEF=0.001
export FILTER_GROUPS_ENABLE=true
export FILTER_GROUPS_METRIC=acc
export FILTER_GROUPS_MAX_GENS=10
export TRAIN_SP=2
export TRAIN_TEMPERATURE=1.4
export VAL_TEMPERATURE=1.0

export TRAIN_FILES="${TRAIN_FILES:-$ROOT/data/processed_tir_filtered/data/train.parquet}"
export VAL_FILES="${VAL_FILES:-$ROOT/data/processed_tir_filtered/data/validation.parquet}"

export WANDB_PROJECT="${WANDB_PROJECT:-physcode_tir_smoke}"
export LOGGERS="console"   # skip wandb for 30-min smoke
export EXPERIMENT="smoke_tir_dapo_sync_debug_$(date +%Y%m%d.%H%M%S)"

export NNODES="$SLURM_JOB_NUM_NODES"

echo "[driver] launching train_dapo_sync.sh on $HEAD_NODE (EXPERIMENT=$EXPERIMENT)"
srun --overlap --nodes=1 --ntasks=1 -w "$HEAD_NODE" \
     bash "$ROOT/scripts/train_dapo_sync.sh"
DRIVER_EXIT=$?
echo "[driver] exited with $DRIVER_EXIT"
exit $DRIVER_EXIT
