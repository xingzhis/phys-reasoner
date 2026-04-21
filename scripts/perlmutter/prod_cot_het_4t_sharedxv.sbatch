#!/bin/bash
# prod_cot_het_4t_sharedxv.sbatch — CoT prod ablation on premium, 5-job cap.
#
# Identical to prod_tir_het_4t_sharedxv.sbatch EXCEPT COT_BASELINE=1 which flips:
#   - actor_rollout_ref.rollout.agent.default_agent_loop = single_turn_agent
#   - actor_rollout_ref.rollout.multi_turn.enable        = false
# (via scripts/train_async.sh). Uses data/processed_cot/ (identical rows,
# CoT-only system prompt) so the ablation is faithful.
#
# Relies on the same shared external xverify at outputs/xverify_endpoints/shared.url.
# See prod_tir_het_4t_sharedxv.sbatch for the submit order.

#SBATCH --job-name=physcode_prod_cot_het_4t_sxv
#SBATCH --output=logs/%x_%j.log
#SBATCH --error=logs/%x_%j.err

# --- het-group 0: TRAINER 4×4 (FSDP-16) ---
#SBATCH -A m2651
#SBATCH -C gpu&hbm80g
#SBATCH -q premium
#SBATCH -t 48:00:00
#SBATCH -N 4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-node=4

#SBATCH hetjob
# --- het-group 1: ROLLOUT 6×4 ---
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

echo "=== prod_cot_het_4t_sharedxv.sbatch (CoT-GRPO baseline, no tool) ==="
echo "  Job     : $SLURM_JOB_ID"
echo "  Trainer : $SLURM_JOB_NODELIST_HET_GROUP_0  (FSDP-16)"
echo "  Rollout : $SLURM_JOB_NODELIST_HET_GROUP_1"
echo "  xVerify : shared (external, outputs/xverify_endpoints/shared.url)"

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

TRAIN_NODES=( $(scontrol show hostnames "$SLURM_JOB_NODELIST_HET_GROUP_0") )
HEAD_NODE="${TRAIN_NODES[0]}"
TRAIN_WORKER_NODES=( "${TRAIN_NODES[@]:1}" )
WORKER_NODES=( $(scontrol show hostnames "$SLURM_JOB_NODELIST_HET_GROUP_1") )
export HEAD_NODE TRAIN_WORKER_NODES WORKER_NODES

echo "[topology] HEAD=$HEAD_NODE  TRAIN_WORKERS=${TRAIN_WORKER_NODES[*]}  ROLLOUT=${WORKER_NODES[*]}"

export N_GPUS_TRAIN="${N_GPUS_TRAIN:-4}"
export N_GPUS_ROLLOUT="${N_GPUS_ROLLOUT:-4}"
export NNODES_TRAIN="${NNODES_TRAIN:-4}"
export NNODES_ROLLOUT="${NNODES_ROLLOUT:-6}"
export HET_SRUN_HEAD="--het-group=0"
export HET_SRUN_WORKER="--het-group=1"
source "$ROOT/scripts/perlmutter/_ray_bringup.sh"

cleanup() {
    local rc=$?
    type ray_teardown &>/dev/null && ray_teardown || true
    return $rc
}
trap cleanup EXIT

# Use default current.url path (Ray workers' fallback target — env-var hand-off
# doesn't reach remote workers). See prod_tir_het_4t_sharedxv.sbatch for detail.
URL_FILE="${SHARED_URL_FILE:-$ROOT/outputs/xverify_endpoints/current.url}"
XVERIFY_WAIT_SECS="${XVERIFY_WAIT_SECS:-3600}"
echo "[xverify] polling shared URL file $URL_FILE (timeout ${XVERIFY_WAIT_SECS}s) ..."
deadline=$(( $(date +%s) + XVERIFY_WAIT_SECS ))
while (( $(date +%s) < deadline )); do
    [[ -f "$URL_FILE" ]] && { XVERIFY_URL=$(head -n1 "$URL_FILE"); break; }
    sleep 15
done
[[ -z "${XVERIFY_URL:-}" ]] && { echo "ERROR: shared xverify URL never appeared" >&2; exit 1; }
echo "[xverify] shared URL = $XVERIFY_URL"
health_url="${XVERIFY_URL%/judge}/health"
hc_deadline=$(( $(date +%s) + 300 ))
while (( $(date +%s) < hc_deadline )); do
    curl -sf --max-time 5 "$health_url" >/dev/null 2>&1 && { echo "[xverify] healthy"; break; }
    sleep 10
done
export XVERIFY_URL

# ---- CoT: flip the single_turn_agent path ----
export COT_BASELINE=1
export MODEL="${MODEL:-Qwen/Qwen3-4B-Thinking-2507}"

export TRAIN_BATCH="${TRAIN_BATCH:-128}"
export ROLLOUT_N="${ROLLOUT_N:-8}"
export TOTAL_STEPS="${TOTAL_STEPS:-552}"
export TOTAL_EPOCHS="${TOTAL_EPOCHS:-3}"
export SAVE_FREQ="${SAVE_FREQ:-50}"
export TEST_FREQ="${TEST_FREQ:-50}"

# Identical budgets to TIR — single_turn_agent_loop.py validates that both modes
# reach MAX_RESPONSE_LEN=19472 (CoT's post_interrupt absorbs TIR's tool+answer
# share), making the ablation fair on total response length.
export THINKING_BUDGET="${THINKING_BUDGET:-12288}"
export TOOL_CALL_BUDGET="${TOOL_CALL_BUDGET:-2048}"
export ANSWER_BUDGET="${ANSWER_BUDGET:-4096}"

export VLLM_GPU_MEM_UTIL="${VLLM_GPU_MEM_UTIL:-0.85}"
export USE_DYNAMIC_BSZ="${USE_DYNAMIC_BSZ:-1}"
export PPO_MAX_TOKEN_LEN_PER_GPU="${PPO_MAX_TOKEN_LEN_PER_GPU:-20480}"
export ACTOR_OPT_OFFLOAD="${ACTOR_OPT_OFFLOAD:-False}"

export TRAIN_SP="${TRAIN_SP:-2}"
export STALENESS="${STALENESS:-1}"

export TRAIN_FILES="${TRAIN_FILES:-$ROOT/data/processed_cot/data/train.parquet}"
export VAL_FILES="${VAL_FILES:-$ROOT/data/processed_cot/data/validation.parquet}"

export WANDB_PROJECT="${WANDB_PROJECT:-physcode_tir}"
export LOGGERS="${LOGGERS:-console,wandb}"
# Distinct _sxv_ tag avoids collision with the pending regular-queue
# submission (51822835) and disambiguates on wandb.
export EXPERIMENT="${EXPERIMENT:-grpo_cot_4t_sxv_$(date +%Y%m%d.%H%M%S)}"

echo "[driver] launching train_async.sh COT_BASELINE=1 (EXPERIMENT=$EXPERIMENT, FSDP=$((NNODES_TRAIN*N_GPUS_TRAIN)))"
srun --overlap --het-group=0 --nodes=1 --ntasks=1 -w "$HEAD_NODE" \
     bash "$ROOT/scripts/train_async.sh"
DRIVER_EXIT=$?
echo "[driver] exited with $DRIVER_EXIT"
exit $DRIVER_EXIT
