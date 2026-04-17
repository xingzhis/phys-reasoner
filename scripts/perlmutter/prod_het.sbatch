#!/bin/bash
# prod_het.sbatch — Production GRPO on Perlmutter, 3-group heterogeneous job.
#
# Topology (dies together; any preemption tears the whole job down):
#   het-group 0  TRAINER   1 node × 4 A100   Ray head + FSDP actor/ref
#   het-group 1  ROLLOUT   5 nodes × 4 A100  Ray workers + vLLM engines
#   het-group 2  XVERIFY   1 × 1 A100       reward server (shared-QoS)
#
# Submit:
#   sbatch scripts/perlmutter/prod_het.sbatch
# Resume (same EXPERIMENT → verl auto-resumes from latest checkpoint):
#   sbatch --export=ALL,EXPERIMENT=<prev> scripts/perlmutter/prod_het.sbatch
#
# All 'TODO(collab)' markers are cluster-specific values the collaborator must fill.
# Grep '^# TODO(collab)' to find every one.

#SBATCH --job-name=physcode_prod_het
#SBATCH --output=logs/%x_%j.log
#SBATCH --error=logs/%x_%j.err

# =============================================================================
# HET-GROUP 0: TRAINER (Ray head) — 1 node × 4 GPU
# =============================================================================
# TODO(collab): NERSC project account with GPU allocation, e.g. m1234_g
#SBATCH -A TODO_ACCOUNT_g
# TODO(collab): -C gpu = A100-40G. For 80G: -C 'gpu&a100_80gb' (confirm exact feature name).
#SBATCH -C gpu
# TODO(collab): 'regular' (48h cap) | 'premium' (faster, 2x cost) — confirm budget policy.
#SBATCH -q regular
#SBATCH -t 48:00:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-node=4

#SBATCH hetjob
# =============================================================================
# HET-GROUP 1: ROLLOUT (Ray workers) — 5 nodes × 4 GPU
# =============================================================================
# TODO(collab): same account as group 0
#SBATCH -A TODO_ACCOUNT_g
# TODO(collab): match group 0 GPU class (both 40G or both 80G; mixing is fine in theory but untested)
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH -t 48:00:00
#SBATCH -N 5
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-node=4

#SBATCH hetjob
# =============================================================================
# HET-GROUP 2: XVERIFY (reward server) — 1 × 1 GPU (shared QoS)
# =============================================================================
# TODO(collab): same account
#SBATCH -A TODO_ACCOUNT_g
# TODO(collab): 7B fits on 40G; either class fine
#SBATCH -C gpu
# TODO(collab): Perlmutter shared-GPU QoS. NERSC typically supports '-q shared' with
#               --gpus-per-task=1 for fractional-node allocation. CONFIRM before submit —
#               if not available, use the companion variant (prod_companion.sbatch).
#SBATCH -q shared
#SBATCH -t 48:00:00
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
[[ -f "$SIF" ]]     || { echo "ERROR: SIF not found: $SIF" >&2; exit 1; }
[[ -f "$OVERLAY" ]] || { echo "ERROR: OVERLAY not found: $OVERLAY" >&2; exit 1; }

echo "=== prod_het.sbatch ==="
echo "  Job       : $SLURM_JOB_ID"
echo "  Trainer   : $SLURM_JOB_NODELIST_HET_GROUP_0"
echo "  Rollout   : $SLURM_JOB_NODELIST_HET_GROUP_1"
echo "  xVerify   : $SLURM_JOB_NODELIST_HET_GROUP_2"
echo "  SIF       : $SIF"
echo "  OVERLAY   : $OVERLAY"
echo "  HF_HOME   : $HF_HOME"

# -----------------------------------------------------------------------------
# Apptainer prefix used everywhere.
# -----------------------------------------------------------------------------
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
    "$SIF"
)
export APT_CMD

# -----------------------------------------------------------------------------
# Resolve node roles from het-group env vars.
# -----------------------------------------------------------------------------
HEAD_NODE=$(scontrol show hostnames "$SLURM_JOB_NODELIST_HET_GROUP_0" | head -n1)
WORKER_NODES=( $(scontrol show hostnames "$SLURM_JOB_NODELIST_HET_GROUP_1") )
XVERIFY_NODE=$(scontrol show hostnames "$SLURM_JOB_NODELIST_HET_GROUP_2" | head -n1)
export HEAD_NODE WORKER_NODES XVERIFY_NODE

(( ${#WORKER_NODES[@]} == 5 )) || echo "WARN: expected 5 rollout workers, got ${#WORKER_NODES[@]}"

# -----------------------------------------------------------------------------
# Launch xVerify in het-group 2.
# Writes rendezvous URL to outputs/xverify_endpoints/current.url.
# -----------------------------------------------------------------------------
rm -f "$ROOT/outputs/xverify_endpoints/current.url"
echo "[xverify] launching on $XVERIFY_NODE ..."
srun --het-group=2 --nodes=1 --ntasks=1 \
     --output="logs/%x_%j.xverify.log" \
     bash "$ROOT/scripts/serve_xverify.sh" &
XVERIFY_PID=$!

# -----------------------------------------------------------------------------
# Ray cluster bring-up (trainer = head, rollout = workers).
# These defaults MUST match the #SBATCH --gpus-per-node / -N directives above —
# if you override one via --export=ALL,N_GPUS_ROLLOUT=2 you must ALSO edit the
# SBATCH header, otherwise Ray sees a different GPU count than Slurm allocated.
# -----------------------------------------------------------------------------
export N_GPUS_TRAIN="${N_GPUS_TRAIN:-4}"
export N_GPUS_ROLLOUT="${N_GPUS_ROLLOUT:-4}"
export NNODES_TRAIN="${NNODES_TRAIN:-1}"
export NNODES_ROLLOUT="${NNODES_ROLLOUT:-5}"
export HET_SRUN_HEAD="--het-group=0"
export HET_SRUN_WORKER="--het-group=1"
source "$ROOT/scripts/perlmutter/_ray_bringup.sh"

# -----------------------------------------------------------------------------
# Cleanup trap. Guarded — ray_teardown is only defined by _ray_bringup.sh after
# a successful source; if bring-up failed early this trap fires with it undefined.
# -----------------------------------------------------------------------------
cleanup() {
    local rc=$?
    echo "[cleanup] exit=$rc — tearing down Ray + xVerify"
    type ray_teardown &>/dev/null && ray_teardown || true
    kill "$XVERIFY_PID" 2>/dev/null || true
    return $rc
}
trap cleanup EXIT

# -----------------------------------------------------------------------------
# Poll for xVerify URL (server takes ~1-3 min to load the 7B model).
# -----------------------------------------------------------------------------
URL_FILE="$ROOT/outputs/xverify_endpoints/current.url"
# 900s covers cold HF cache + xVerify-7B load on a fresh node; het-group 2 starts
# simultaneously with the others so there's no queue time to absorb.
XVERIFY_WAIT_SECS="${XVERIFY_WAIT_SECS:-900}"
echo "[xverify] polling $URL_FILE (timeout ${XVERIFY_WAIT_SECS}s) ..."
deadline=$(( $(date +%s) + XVERIFY_WAIT_SECS ))
while (( $(date +%s) < deadline )); do
    [[ -f "$URL_FILE" ]] && { XVERIFY_URL=$(head -n1 "$URL_FILE"); break; }
    sleep 10
done
[[ -z "${XVERIFY_URL:-}" ]] && { echo "ERROR: xverify did not come up" >&2; exit 1; }
echo "[xverify] URL = $XVERIFY_URL"
# Health check — fail loudly if reachable but not ready.
health_url="${XVERIFY_URL%/judge}/health"
hc_deadline=$(( $(date +%s) + 300 ))
while (( $(date +%s) < hc_deadline )); do
    curl -sf --max-time 5 "$health_url" >/dev/null 2>&1 && { echo "[xverify] healthy"; break; }
    sleep 10
done
export XVERIFY_URL

# -----------------------------------------------------------------------------
# Production knobs — override via sbatch --export=ALL,KEY=VAL to tweak.
# N_GPUS_* / NNODES_* are already exported above (before Ray bring-up) so that
# both Ray and verl see the same values; do NOT re-assign them here.
# -----------------------------------------------------------------------------
export MODEL="${MODEL:-Qwen/Qwen3.5-4B}"

export TRAIN_BATCH="${TRAIN_BATCH:-128}"            # ppo_mini_batch_size
export ROLLOUT_N="${ROLLOUT_N:-8}"                  # GRPO group size
export TOTAL_STEPS="${TOTAL_STEPS:-3000}"           # ~3 epochs over 108k pool
export SAVE_FREQ="${SAVE_FREQ:-200}"                # 15 checkpoints across the run
export TEST_FREQ="${TEST_FREQ:-100}"                # 30 eval points across the run

# Sequence budgets (match probe).
export THINKING_BUDGET="${THINKING_BUDGET:-12288}"
export TOOL_CALL_BUDGET="${TOOL_CALL_BUDGET:-2048}"
export ANSWER_BUDGET="${ANSWER_BUDGET:-4096}"

# vLLM memory util. Rollout GPU dedicated → safe at 0.85; bump to 0.90 on 80G.
export VLLM_GPU_MEM_UTIL="${VLLM_GPU_MEM_UTIL:-0.85}"

# Trainer memory. Use dynamic batch sizing (see Q10 in checklist.md).
# PPO_MICRO_BS_PER_GPU is ignored when USE_DYNAMIC_BSZ=1 below.
export USE_DYNAMIC_BSZ="${USE_DYNAMIC_BSZ:-1}"
export PPO_MAX_TOKEN_LEN_PER_GPU="${PPO_MAX_TOKEN_LEN_PER_GPU:-24576}"
export ACTOR_OPT_OFFLOAD="${ACTOR_OPT_OFFLOAD:-True}"

# Data — collaborator will set these from fetch_dataset.py output.
# TODO(collab): point TRAIN_FILES and VAL_FILES at the Perlmutter-local paths.
export TRAIN_FILES="${TRAIN_FILES:-$ROOT/data/processed_tir/data/train.parquet}"
export VAL_FILES="${VAL_FILES:-$ROOT/data/processed_tir/data/validation.parquet}"

# Logging.
export WANDB_PROJECT="${WANDB_PROJECT:-physcode_tir}"
export LOGGERS="${LOGGERS:-console,wandb}"
export EXPERIMENT="${EXPERIMENT:-grpo_prod_$(date +%Y%m%d.%H%M%S)}"

# -----------------------------------------------------------------------------
# Launch driver on trainer node. --overlap shares the HEAD task slot with the
# backgrounded `ray start --head --block` srun.
# -----------------------------------------------------------------------------
echo "[driver] launching train_async.sh on $HEAD_NODE (EXPERIMENT=$EXPERIMENT)"
srun --overlap --het-group=0 --nodes=1 --ntasks=1 -w "$HEAD_NODE" \
     bash "$ROOT/scripts/train_async.sh"
DRIVER_EXIT=$?
echo "[driver] exited with $DRIVER_EXIT"
exit $DRIVER_EXIT
