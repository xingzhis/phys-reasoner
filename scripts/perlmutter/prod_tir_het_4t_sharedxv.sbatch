#!/bin/bash
# prod_tir_het_4t_sharedxv.sbatch — TIR prod on premium QoS under the 5-job cap.
#
# Topology (2 het-groups, NO xverify here — uses shared external xverify):
#   het-group 0  TRAINER   4 nodes × 4 A100   FSDP-16 (Ray head + 3 workers)
#   het-group 1  ROLLOUT   6 nodes × 4 A100
#
# Reads XVERIFY_URL from outputs/xverify_endpoints/current.url (the default
# rendezvous path that verl's reward.py falls back to). A separately-submitted
# serve_xverify_perlmutter.sbatch writes to that path by default, so no
# --export=URL_FILE override is needed on the xverify submission either.
# This layout fits two prod runs + one shared xverify into premium's 5-job
# cap (1+2+2=5).
#
# Submit order (recommended):
#   rm -f outputs/xverify_endpoints/current.url   # clear any stale rendezvous
#   sbatch scripts/perlmutter/serve_xverify_perlmutter.sbatch
#   # wait ~10 min for xverify to RUNNING + write current.url + load model
#   sbatch scripts/perlmutter/prod_tir_het_4t_sharedxv.sbatch
#   sbatch scripts/perlmutter/prod_cot_het_4t_sharedxv.sbatch
#
# Rollback: the standalone prod_tir_het_4t.sbatch (with its own xverify) is
# untouched — switch back by cancelling these and submitting that instead.

#SBATCH --job-name=physcode_prod_tir_het_4t_sxv
#SBATCH --output=logs/%x_%j.log
#SBATCH --error=logs/%x_%j.err

# =============================================================================
# HET-GROUP 0: TRAINER 4×4 (FSDP-16)
# =============================================================================
#SBATCH -A m2651
#SBATCH -C gpu&hbm80g
#SBATCH -q premium
#SBATCH -t 48:00:00
#SBATCH -N 4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-node=4

#SBATCH hetjob
# =============================================================================
# HET-GROUP 1: ROLLOUT 6×4
# =============================================================================
#SBATCH -A m2651
#SBATCH -C gpu&hbm80g
#SBATCH -q premium
#SBATCH -t 48:00:00
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

echo "=== prod_tir_het_4t_sharedxv.sbatch ==="
echo "  Job       : $SLURM_JOB_ID"
echo "  Trainer   : $SLURM_JOB_NODELIST_HET_GROUP_0  (FSDP-16)"
echo "  Rollout   : $SLURM_JOB_NODELIST_HET_GROUP_1"
echo "  xVerify   : shared (external, read from outputs/xverify_endpoints/shared.url)"
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

# ---- topology resolution (4 trainer nodes: 1 head + 3 workers) ----
TRAIN_NODES=( $(scontrol show hostnames "$SLURM_JOB_NODELIST_HET_GROUP_0") )
HEAD_NODE="${TRAIN_NODES[0]}"
TRAIN_WORKER_NODES=( "${TRAIN_NODES[@]:1}" )
WORKER_NODES=( $(scontrol show hostnames "$SLURM_JOB_NODELIST_HET_GROUP_1") )
export HEAD_NODE TRAIN_WORKER_NODES WORKER_NODES

echo "[topology] HEAD=$HEAD_NODE  TRAIN_WORKERS=${TRAIN_WORKER_NODES[*]}  ROLLOUT=${WORKER_NODES[*]}"
(( ${#TRAIN_WORKER_NODES[@]} == 3 )) || echo "WARN: expected 3 trainer workers, got ${#TRAIN_WORKER_NODES[@]}"
(( ${#WORKER_NODES[@]} == 6 )) || echo "WARN: expected 6 rollout workers, got ${#WORKER_NODES[@]}"

# ---- Ray cluster bring-up (no xverify here, shared externally) ----
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

# ---- poll the SHARED xverify URL file (written by serve_xverify_perlmutter) ----
# Use the default `current.url` path so Ray remote rollout/reward workers,
# whose apptainer env doesn't include XVERIFY_URL (propagation only reaches the
# driver, not remote nodes), can discover the URL via reward.py's default
# fallback. Writing to `shared.url` and relying on env-var hand-off broke on
# job 51846557 — workers read the stale current.url and cascade-failed.
URL_FILE="${SHARED_URL_FILE:-$ROOT/outputs/xverify_endpoints/current.url}"
# Extended wait window — shared xverify may also be queued; premium 1-GPU jobs
# usually start in <10 min but 1 h gives us margin if the QoS is busier.
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

# ---- production knobs (identical to prod_tir_het_4t.sbatch, minus xverify) ----
export MODEL="${MODEL:-Qwen/Qwen3-4B-Thinking-2507}"

export TRAIN_BATCH="${TRAIN_BATCH:-128}"
export ROLLOUT_N="${ROLLOUT_N:-8}"
# 3 epochs over 23,589-row filtered pool. BOTH TOTAL_STEPS and TOTAL_EPOCHS
# must be set — verl caps by min(TOTAL_STEPS*BATCH, len(dataloader)*TOTAL_EPOCHS).
export TOTAL_STEPS="${TOTAL_STEPS:-552}"
export TOTAL_EPOCHS="${TOTAL_EPOCHS:-3}"
export SAVE_FREQ="${SAVE_FREQ:-50}"
export TEST_FREQ="${TEST_FREQ:-50}"

# Prod budgets — MAX_RESPONSE_LEN = 12288 + 16 + 2048 + 1024 + 4096 = 19,472.
export THINKING_BUDGET="${THINKING_BUDGET:-12288}"
export TOOL_CALL_BUDGET="${TOOL_CALL_BUDGET:-2048}"
export ANSWER_BUDGET="${ANSWER_BUDGET:-4096}"

export VLLM_GPU_MEM_UTIL="${VLLM_GPU_MEM_UTIL:-0.85}"
export USE_DYNAMIC_BSZ="${USE_DYNAMIC_BSZ:-1}"
export PPO_MAX_TOKEN_LEN_PER_GPU="${PPO_MAX_TOKEN_LEN_PER_GPU:-20480}"

# Tier-A #2 full claim — validated on smoke_tir_het_4t (51819857, 2.06× speedup).
export ACTOR_OPT_OFFLOAD="${ACTOR_OPT_OFFLOAD:-False}"

export TRAIN_SP="${TRAIN_SP:-2}"
export STALENESS="${STALENESS:-1}"

export TRAIN_FILES="${TRAIN_FILES:-$ROOT/data/processed_tir/data/train.parquet}"
export VAL_FILES="${VAL_FILES:-$ROOT/data/processed_tir/data/validation.parquet}"

export WANDB_PROJECT="${WANDB_PROJECT:-physcode_tir}"
export LOGGERS="${LOGGERS:-console,wandb}"
# Distinct _sxv_ tag so this run doesn't collide with the pending regular-queue
# submission (51822832) if it ever starts, and disambiguates on wandb.
export EXPERIMENT="${EXPERIMENT:-grpo_tir_4t_sxv_$(date +%Y%m%d.%H%M%S)}"

echo "[driver] launching train_async.sh on $HEAD_NODE (EXPERIMENT=$EXPERIMENT, FSDP=$((NNODES_TRAIN*N_GPUS_TRAIN)))"
srun --overlap --het-group=0 --nodes=1 --ntasks=1 -w "$HEAD_NODE" \
     bash "$ROOT/scripts/train_async.sh"
DRIVER_EXIT=$?
echo "[driver] exited with $DRIVER_EXIT"
exit $DRIVER_EXIT
