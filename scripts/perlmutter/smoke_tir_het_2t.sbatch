#!/bin/bash
# smoke_tir_het_2t.sbatch — Phase-4 multi-trainer-node smoke on Perlmutter.
#
# Same shape as smoke_tir_het.sbatch but with TRAINER expanded to 2 nodes × 4 GPU
# (FSDP=8). See .claude/handoff/plan-multi-trainer-node.md and
# perlmutter_debug/FINDINGS.md §12 Tier-A lever #2.
#
# First-run policy (per plan-multi-trainer-node.md):
#   - Keep ACTOR_OPT_OFFLOAD=True on the first smoke (memory-safe).
#   - Retry with ACTOR_OPT_OFFLOAD=False once the FSDP-8 topology is green.
#   - Don't change other structural knobs (TRAIN_SP, STALENESS) in the first run.
#
# Submit with prod budgets on:
#   sbatch --export=ALL,THINKING_BUDGET=12288,TOOL_CALL_BUDGET=2048,ANSWER_BUDGET=4096 \
#     scripts/perlmutter/smoke_tir_het_2t.sbatch
#
# Rollback: the single-trainer-node path in smoke_tir_het.sbatch is untouched.

#SBATCH --job-name=physcode_smoke_tir_het_2t
#SBATCH --output=logs/%x_%j.log
#SBATCH --error=logs/%x_%j.err

# --- het-group 0: TRAINER (2 × 4 GPU = FSDP-8) ---
# Perlmutter canonical values (see scripts/perlmutter/README.md §2).
#SBATCH -A m2651
#SBATCH -C gpu&hbm80g
#SBATCH -q premium
#SBATCH -t 04:00:00
#SBATCH -N 2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-node=4

#SBATCH hetjob
# --- het-group 1: ROLLOUT (3 × 4 GPU for smoke; prod budget measured sufficient) ---
#SBATCH -A m2651
#SBATCH -C gpu&hbm80g
#SBATCH -q premium
#SBATCH -t 04:00:00
#SBATCH -N 3
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-node=4

#SBATCH hetjob
# --- het-group 2: XVERIFY (1 × 1 GPU) ---
#SBATCH -A m2651
#SBATCH -C gpu&hbm80g
# NOTE: m2651 lacks access to '-q shared'; xverify reserves a whole 4×A100 node.
# See scripts/perlmutter/README.md "xverify GPU waste".
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

echo "=== smoke_tir_het_2t.sbatch ==="
echo "  Trainer : $SLURM_JOB_NODELIST_HET_GROUP_0  (FSDP-8)"
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
    # NCCL tuning — see smoke_tir_het.sbatch for rationale. Cross-node FSDP-8
    # adds a new all-reduce path on Slingshot; IB_TIMEOUT bump is especially
    # relevant here. If NCCL init hangs across trainer nodes, try raising to 64.
    --env "NCCL_IB_TIMEOUT=32"
    --env "NCCL_IBEXT_DISABLE=1"
    --env "TORCH_NCCL_ENABLE_MONITORING=0"
    # vLLM custom all-reduce disabled at the verl config layer (train_async.sh).
    "$SIF"
)
export APT_CMD

# Trainer node 0 = Ray head + FullyAsyncTrainer actor (pinned via `trainer_node`).
# Trainer nodes 1..N-1 = additional Ray workers in same het-group, also tagged
# `trainer_node: 1` so FSDP placement spans the full trainer pool.
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

# Resource knobs MUST match the #SBATCH directives above.
export N_GPUS_TRAIN="${N_GPUS_TRAIN:-4}"
export N_GPUS_ROLLOUT="${N_GPUS_ROLLOUT:-4}"
export NNODES_TRAIN="${NNODES_TRAIN:-2}"
export NNODES_ROLLOUT="${NNODES_ROLLOUT:-3}"
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
# Pass prod budgets via --export when submitting; default (empty) keeps the
# no-interrupt path for cold-start smokes (matches smoke_tir_het.sbatch default).
export THINKING_BUDGET="${THINKING_BUDGET:-}"
export TOOL_CALL_BUDGET="${TOOL_CALL_BUDGET:-}"
export ANSWER_BUDGET="${ANSWER_BUDGET:-}"
export VLLM_GPU_MEM_UTIL="${VLLM_GPU_MEM_UTIL:-0.85}"
export USE_DYNAMIC_BSZ="${USE_DYNAMIC_BSZ:-1}"
export PPO_MAX_TOKEN_LEN_PER_GPU="${PPO_MAX_TOKEN_LEN_PER_GPU:-20480}"
# First FSDP-8 smoke: keep optimizer_offload=True (memory-safe baseline).
# FINDINGS §12 Tier-A #2 predicts FSDP-8 can drop offload for another 1.3-1.5×;
# flip this to False via --export once this smoke is green.
export ACTOR_OPT_OFFLOAD="${ACTOR_OPT_OFFLOAD:-True}"
# TRAIN_SP=2 kept from 1T config. At FSDP-8 we could try SP=4 (DP=2) later,
# but first validate the plain FSDP-8 path without adding variables.
export TRAIN_SP="${TRAIN_SP:-2}"
export STALENESS="${STALENESS:-1}"
export DUMP_TRAIN_ROLLOUTS=1
export DUMP_VAL_ROLLOUTS=1

export TRAIN_FILES="${TRAIN_FILES:-$ROOT/data/processed_tir/data/train.parquet}"
export VAL_FILES="${VAL_FILES:-$ROOT/data/processed_tir/data/validation.parquet}"

export WANDB_PROJECT="${WANDB_PROJECT:-physcode_tir_smoke}"
export LOGGERS="${LOGGERS:-console,wandb}"
export EXPERIMENT="${EXPERIMENT:-smoke_tir_het_2t_$(date +%Y%m%d.%H%M%S)}"

echo "[driver] launching train_async.sh (EXPERIMENT=$EXPERIMENT, FSDP=$((NNODES_TRAIN*N_GPUS_TRAIN)))"
srun --overlap --het-group=0 --nodes=1 --ntasks=1 -w "$HEAD_NODE" \
     bash "$ROOT/scripts/train_async.sh"
DRIVER_EXIT=$?
echo "[driver] exited with $DRIVER_EXIT"
exit $DRIVER_EXIT
