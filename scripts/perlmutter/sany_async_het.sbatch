#!/bin/bash
# sany_async_het.sbatch — Qwen3-0.6B + DeepScaleR ASYNC sanity.
# Small-scale test of our async path: 1 trainer node + 1 rollout node (4 GPU each).
# Tests whether our async pipeline can train at all with a small model + clean
# rule-based reward on well-behaved data.

#SBATCH --job-name=sany_async_qwen06b
#SBATCH --output=logs/%x_%j.log
#SBATCH --error=logs/%x_%j.err

# HET-GROUP 0: TRAINER 1×4 (FSDP-4)
#SBATCH -A m2651
#SBATCH -C gpu&hbm40g
#SBATCH -q premium
#SBATCH -t 02:00:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-node=4

#SBATCH hetjob
# HET-GROUP 1: ROLLOUT 1×4
#SBATCH -A m2651
#SBATCH -C gpu&hbm40g
#SBATCH -q premium
#SBATCH -t 02:00:00
#SBATCH -N 1
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

echo "=== sany_async_het.sbatch (0.6B + DeepScaleR async, 1t+1r) ==="
echo "  Job       : $SLURM_JOB_ID"
echo "  Trainer   : $SLURM_JOB_NODELIST_HET_GROUP_0  (FSDP-4 on 40G)"
echo "  Rollout   : $SLURM_JOB_NODELIST_HET_GROUP_1  (4 vLLM engines TP=1)"

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
    --env "PHYS_REQUIRE_XVERIFY=0"
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

export N_GPUS_TRAIN=4
export N_GPUS_ROLLOUT=4
export NNODES_TRAIN=1
export NNODES_ROLLOUT=1
export HET_SRUN_HEAD="--het-group=0"
export HET_SRUN_WORKER="--het-group=1"
source "$ROOT/scripts/perlmutter/_ray_bringup.sh"

cleanup() {
    local rc=$?
    type ray_teardown &>/dev/null && ray_teardown || true
    return $rc
}
trap cleanup EXIT

# --- SAN-Y async knobs ---
export MODEL="Qwen/Qwen3-0.6B"
export TRAIN_FILES="$ROOT/data/deepscaler/train.parquet"
export VAL_FILES="$ROOT/data/deepscaler/train.parquet"
export REWARD_FN_PATH="$ROOT/src/phys_reasoner/training/reward_polaris.py"
export REWARD_FN_NAME=compute_score

# POLARIS-aligned recipe on async:
export TRAIN_SP=1
export STALENESS=0
export ROLLOUT_TP=1        # 4 parallel vLLM engines on rollout node
export FILTER_GROUPS_ENABLE=true
export FILTER_GROUPS_METRIC=acc
export NORM_ADV_BY_STD=True
export USE_ROLLOUT_LOG_PROBS=False   # POLARIS-faithful (now default but explicit)

export COT_BASELINE=1       # no TIR
export TRAIN_TEMPERATURE=1.4
export VAL_TEMPERATURE=1.0

# Small scale + reasonable response budget
export TRAIN_BATCH=64
export ROLLOUT_N=8
export THINKING_BUDGET=6144
export TOOL_CALL_BUDGET=1024
export ANSWER_BUDGET=1024
export PPO_MAX_TOKEN_LEN_PER_GPU=10240
export VLLM_GPU_MEM_UTIL=0.7
export ACTOR_OPT_OFFLOAD=True

export LR=1e-6
export TOTAL_STEPS=60
export SAVE_FREQ=-1
export TEST_FREQ=-1

export WANDB_PROJECT=physcode_polaris_anchor
export LOGGERS="console,wandb"
export EXPERIMENT="${EXPERIMENT:-sany_async_0.6b_deepscaler_$(date +%Y%m%d.%H%M%S)}"

echo "[driver] launching train_async.sh on $HEAD_NODE (EXPERIMENT=$EXPERIMENT)"
srun --overlap --het-group=0 --nodes=1 --ntasks=1 -w "$HEAD_NODE" \
     bash "$ROOT/scripts/train_async.sh"
DRIVER_EXIT=$?
echo "[driver] exited with $DRIVER_EXIT"
exit $DRIVER_EXIT
