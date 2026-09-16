#!/bin/bash
# smoke_tir_het_4t.sbatch — Phase-4 *backup* smoke on 4-trainer-node (FSDP-16).
#
# This is a contingency smoke for the scenario where 2T+3R doesn't deliver a
# step time ≤5 min at prod budget (see plan-multi-trainer-node.md + smoke B/C
# measurements). Only submit this if the 2T result mandates more iron.
#
# Topology:
#   het-group 0  TRAINER   4 nodes × 4 A100   Ray head + 3 trainer workers (FSDP-16)
#   het-group 1  ROLLOUT   8 nodes × 4 A100   safety margin for 4T step rate
#                                             (6R minimum to keep up, 8R for tail
#                                             latency margin)
#   het-group 2  XVERIFY   1 × 1 A100
#
# Total: 13 compute nodes + 1 xverify node. Budget-expanded per user direction;
# prior 2T+3R smoke showed 2T+offload=True only 1.08× faster than 1T, insufficient
# to fit 3 epochs in 48h. This smoke tests FSDP-16 + offload=False together.
#
# Expected risks not yet observed:
#   - FSDP-16 cross-node all-reduce scaling on Slingshot is unknown. Beyond
#     FSDP-8, gains may be smaller than linear. If step time doesn't drop
#     meaningfully vs 2T, revert (no prod-4T variant created yet).
#   - NCCL timeouts more likely at FSDP-16; NCCL_IB_TIMEOUT may need 64+.
#
# Submit with prod budgets:
#   sbatch --export=ALL,THINKING_BUDGET=12288,TOOL_CALL_BUDGET=2048,ANSWER_BUDGET=4096 \
#     scripts/perlmutter/smoke_tir_het_4t.sbatch

#SBATCH --job-name=physcode_smoke_tir_het_4t
#SBATCH --output=logs/%x_%j.log
#SBATCH --error=logs/%x_%j.err

# --- het-group 0: TRAINER (4 × 4 GPU = FSDP-16) ---
#SBATCH -A m2651
#SBATCH -C gpu&hbm80g
#SBATCH -q premium
#SBATCH -t 04:00:00
#SBATCH -N 4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-node=4

#SBATCH hetjob
# --- het-group 1: ROLLOUT (8 × 4 GPU — scaled for aggressive trainer step rate) ---
# At 4T+offload=False, projected step time drops toward ~3 min. Rollout must
# produce 1024 trajectories inside the ~6 min staleness window; 6R covers ~4
# min, 8R gives ~2× safety margin for tail-latency robustness.
#SBATCH -A m2651
#SBATCH -C gpu&hbm80g
#SBATCH -q premium
#SBATCH -t 04:00:00
#SBATCH -N 8
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-node=4

#SBATCH hetjob
# --- het-group 2: XVERIFY (1 × 1 GPU, full A100 node on premium) ---
#SBATCH -A m2651
#SBATCH -C gpu&hbm80g
#SBATCH -q premium
#SBATCH -t 04:00:00
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-task=1

set -euo pipefail
ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/../.." && pwd)}"
cd "$ROOT"
mkdir -p logs outputs/xverify_endpoints

unset SIF OVERLAY
source "$ROOT/env.sh"

echo "=== smoke_tir_het_4t.sbatch ==="
echo "  Trainer : $SLURM_JOB_NODELIST_HET_GROUP_0  (FSDP-16)"
echo "  Rollout : $SLURM_JOB_NODELIST_HET_GROUP_1"
echo "  xVerify : $SLURM_JOB_NODELIST_HET_GROUP_2"

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
    # FSDP-16 cross-node NCCL: consider raising NCCL_IB_TIMEOUT=64 if this
    # smoke hits timeouts during trainer-trainer init.
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
XVERIFY_NODE=$(scontrol show hostnames "$SLURM_JOB_NODELIST_HET_GROUP_2" | head -n1)
export HEAD_NODE TRAIN_WORKER_NODES WORKER_NODES XVERIFY_NODE

echo "[topology] HEAD=$HEAD_NODE  TRAIN_WORKERS=${TRAIN_WORKER_NODES[*]}  ROLLOUT=${WORKER_NODES[*]}"

rm -f "$ROOT/outputs/xverify_endpoints/current.url"
echo "[xverify] launching on $XVERIFY_NODE ..."
srun --het-group=2 --nodes=1 --ntasks=1 \
     --output="logs/%x_%j.xverify.log" \
     bash "$ROOT/scripts/serve_xverify.sh" &
XVERIFY_PID=$!

export N_GPUS_TRAIN="${N_GPUS_TRAIN:-4}"
export N_GPUS_ROLLOUT="${N_GPUS_ROLLOUT:-4}"
export NNODES_TRAIN="${NNODES_TRAIN:-4}"
export NNODES_ROLLOUT="${NNODES_ROLLOUT:-6}"
export HET_SRUN_HEAD="--het-group=0" HET_SRUN_WORKER="--het-group=1"
source "$ROOT/scripts/perlmutter/_ray_bringup.sh"

cleanup() {
    local rc=$?
    type ray_teardown &>/dev/null && ray_teardown || true
    kill "$XVERIFY_PID" 2>/dev/null || true
    return $rc
}
trap cleanup EXIT

URL_FILE="$ROOT/outputs/xverify_endpoints/current.url"
XVERIFY_WAIT_SECS="${XVERIFY_WAIT_SECS:-900}"
deadline=$(( $(date +%s) + XVERIFY_WAIT_SECS ))
while (( $(date +%s) < deadline )); do
    [[ -f "$URL_FILE" ]] && { XVERIFY_URL=$(head -n1 "$URL_FILE"); break; }
    sleep 10
done
[[ -z "${XVERIFY_URL:-}" ]] && { echo "ERROR: xverify not up" >&2; exit 1; }
export XVERIFY_URL
echo "[xverify] URL=$XVERIFY_URL"

# ----- smoke overrides -----
export MODEL="${MODEL:-Qwen/Qwen3-4B-Thinking-2507}"
export TRAIN_BATCH="${TRAIN_BATCH:-128}"
export ROLLOUT_N="${ROLLOUT_N:-8}"
export TOTAL_STEPS="${TOTAL_STEPS:-10}"
export SAVE_FREQ="${SAVE_FREQ:-5}"
export TEST_FREQ="${TEST_FREQ:-5}"
# Budget defaults empty — pass prod budgets via --export when submitting.
export THINKING_BUDGET="${THINKING_BUDGET:-}"
export TOOL_CALL_BUDGET="${TOOL_CALL_BUDGET:-}"
export ANSWER_BUDGET="${ANSWER_BUDGET:-}"
export VLLM_GPU_MEM_UTIL="${VLLM_GPU_MEM_UTIL:-0.85}"
export USE_DYNAMIC_BSZ="${USE_DYNAMIC_BSZ:-1}"
export PPO_MAX_TOKEN_LEN_PER_GPU="${PPO_MAX_TOKEN_LEN_PER_GPU:-20480}"
# First FSDP-16 smoke: keep offload on (memory-safe baseline). Flip to False
# via --export once this smoke is green for the full trainer speedup claim.
export ACTOR_OPT_OFFLOAD="${ACTOR_OPT_OFFLOAD:-True}"
# TRAIN_SP=2 leaves 8-way DP on 16 trainer GPUs. SP=4 is plausible later but
# validate FSDP-16 first without adding variables.
export TRAIN_SP="${TRAIN_SP:-2}"
export STALENESS="${STALENESS:-1}"
export DUMP_TRAIN_ROLLOUTS=1
export DUMP_VAL_ROLLOUTS=1

export TRAIN_FILES="${TRAIN_FILES:-$ROOT/data/processed_tir/data/train.parquet}"
export VAL_FILES="${VAL_FILES:-$ROOT/data/processed_tir/data/validation.parquet}"

export WANDB_PROJECT="${WANDB_PROJECT:-physcode_tir_smoke}"
export LOGGERS="${LOGGERS:-console,wandb}"
export EXPERIMENT="${EXPERIMENT:-smoke_tir_het_4t_$(date +%Y%m%d.%H%M%S)}"

echo "[driver] launching train_async.sh (EXPERIMENT=$EXPERIMENT, FSDP=$((NNODES_TRAIN*N_GPUS_TRAIN)))"
srun --overlap --het-group=0 --nodes=1 --ntasks=1 -w "$HEAD_NODE" \
     bash "$ROOT/scripts/train_async.sh"
DRIVER_EXIT=$?
echo "[driver] exited with $DRIVER_EXIT"
exit $DRIVER_EXIT
