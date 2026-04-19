#!/bin/bash
# smoke_tir_het.sbatch — Full-topology smoke on Perlmutter (3-group het, ~10 steps).
#
# Same shape as prod_tir_het.sbatch but:
#   - short step budget (TOTAL_STEPS=10)
#   - SAVE_FREQ=5 so you can scancel after step 6 and test auto-resume
#   - dumps first ~3 training steps to JSONL for rollout inspection
#   - WANDB project suffixed '_smoke' so prod charts stay clean
#
# Pass criteria: see scripts/perlmutter/checklist.md.
#
# Submit:
#   sbatch scripts/perlmutter/smoke_tir_het.sbatch
# Resume test (after scancel mid-run):
#   sbatch --export=ALL,EXPERIMENT=<same-as-prev> scripts/perlmutter/smoke_tir_het.sbatch

#SBATCH --job-name=physcode_smoke_tir_het
#SBATCH --output=logs/%x_%j.log
#SBATCH --error=logs/%x_%j.err

# --- het-group 0: TRAINER (1 × 4 GPU) ---
# TODO(collab): account/queue/constraint — same as prod_tir_het.sbatch
#SBATCH -A TODO_ACCOUNT_g
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH -t 04:00:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-node=4

#SBATCH hetjob
# --- het-group 1: ROLLOUT (5 × 4 GPU) ---
#SBATCH -A TODO_ACCOUNT_g
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH -t 04:00:00
#SBATCH -N 5
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-node=4

#SBATCH hetjob
# --- het-group 2: XVERIFY (1 × 1 GPU) ---
#SBATCH -A TODO_ACCOUNT_g
#SBATCH -C gpu
# TODO(collab): '-q shared' for fractional allocation if Perlmutter supports it
#SBATCH -q shared
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

echo "=== smoke_tir_het.sbatch ==="
echo "  Trainer : $SLURM_JOB_NODELIST_HET_GROUP_0"
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
    # NCCL tuning for inter-node Ray/FSDP on Perlmutter Slingshot. Adapted
    # from Dr.SCI's verl launch (FINDINGS §5) minus NCCL_IB_HCA=mlx5 which
    # is Azure-fabric-specific. Worst case: ignored. NVLS_ENABLE helps
    # NCCL pick NVLink switch when present; IBEXT_DISABLE silences a known
    # warning on Slingshot; ENABLE_MONITORING=0 suppresses a watchdog
    # thread that periodically prints to stderr; IB_TIMEOUT bumps the
    # default 18 to handle congested-fabric retries.
    --env "NCCL_IB_TIMEOUT=32"
    --env "NCCL_NVLS_ENABLE=1"
    --env "NCCL_IBEXT_DISABLE=1"
    --env "TORCH_NCCL_ENABLE_MONITORING=0"
    "$SIF"
)
export APT_CMD

HEAD_NODE=$(scontrol show hostnames "$SLURM_JOB_NODELIST_HET_GROUP_0" | head -n1)
WORKER_NODES=( $(scontrol show hostnames "$SLURM_JOB_NODELIST_HET_GROUP_1") )
XVERIFY_NODE=$(scontrol show hostnames "$SLURM_JOB_NODELIST_HET_GROUP_2" | head -n1)
export HEAD_NODE WORKER_NODES XVERIFY_NODE

rm -f "$ROOT/outputs/xverify_endpoints/current.url"
echo "[xverify] launching on $XVERIFY_NODE ..."
srun --het-group=2 --nodes=1 --ntasks=1 \
     --output="logs/%x_%j.xverify.log" \
     bash "$ROOT/scripts/serve_xverify.sh" &
XVERIFY_PID=$!

# Resource knobs MUST match the #SBATCH directives above; env overrides must
# also edit the SBATCH header or Ray/Slurm will disagree on allocated GPUs.
export N_GPUS_TRAIN="${N_GPUS_TRAIN:-4}"
export N_GPUS_ROLLOUT="${N_GPUS_ROLLOUT:-4}"
export NNODES_TRAIN="${NNODES_TRAIN:-1}"
export NNODES_ROLLOUT="${NNODES_ROLLOUT:-5}"
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

# ----- smoke overrides (resource knobs already exported above) -----
export MODEL="${MODEL:-Qwen/Qwen3-4B-Thinking-2507}"
export TRAIN_BATCH="${TRAIN_BATCH:-128}"
export ROLLOUT_N="${ROLLOUT_N:-8}"
export TOTAL_STEPS="${TOTAL_STEPS:-10}"
export SAVE_FREQ="${SAVE_FREQ:-5}"
export TEST_FREQ="${TEST_FREQ:-5}"
# Think-interrupt disabled for Qwen3 switch smoke: empty budgets → train_async.sh
# omits the thinking_budget Hydra key (${VAR:+...}), which short-circuits the
# interrupt path inside verl's tool_agent_loop (verified: thinking_budget=None
# bypasses max_tokens override and interrupt injection entirely).
export THINKING_BUDGET="${THINKING_BUDGET:-}"
export TOOL_CALL_BUDGET="${TOOL_CALL_BUDGET:-}"
export ANSWER_BUDGET="${ANSWER_BUDGET:-}"
export VLLM_GPU_MEM_UTIL="${VLLM_GPU_MEM_UTIL:-0.85}"
export USE_DYNAMIC_BSZ="${USE_DYNAMIC_BSZ:-1}"
export PPO_MAX_TOKEN_LEN_PER_GPU="${PPO_MAX_TOKEN_LEN_PER_GPU:-20480}"
export ACTOR_OPT_OFFLOAD="${ACTOR_OPT_OFFLOAD:-True}"
# Qwen3 enablers (blocked on Qwen3.5 by GDN + flash-attn monkey-patch):
#   TRAIN_SP=2 → Ulysses sequence parallelism over 2 of 4 trainer GPUs,
#   DP=2 remaining for FSDP sharding. See qwen3-switch.md §2 (safer than SP=4
#   first-try because SP=4 on 4 GPUs gives DP=1 and no FSDP shard).
#   STALENESS=1 → rollout of step t+1 overlaps trainer update of step t,
#   saving ~3 min wall-clock/step. use_rollout_log_probs + DAPO clip already
#   cover the IS correction.
export TRAIN_SP="${TRAIN_SP:-2}"
export STALENESS="${STALENESS:-1}"
export DUMP_TRAIN_ROLLOUTS=1
export DUMP_VAL_ROLLOUTS=1

export TRAIN_FILES="${TRAIN_FILES:-$ROOT/data/processed_tir/data/train.parquet}"
export VAL_FILES="${VAL_FILES:-$ROOT/data/processed_tir/data/validation.parquet}"

export WANDB_PROJECT="${WANDB_PROJECT:-physcode_tir_smoke}"
export LOGGERS="${LOGGERS:-console,wandb}"
export EXPERIMENT="${EXPERIMENT:-smoke_tir_het_$(date +%Y%m%d.%H%M%S)}"

echo "[driver] launching train_async.sh (EXPERIMENT=$EXPERIMENT)"
srun --overlap --het-group=0 --nodes=1 --ntasks=1 -w "$HEAD_NODE" \
     bash "$ROOT/scripts/train_async.sh"
DRIVER_EXIT=$?
echo "[driver] exited with $DRIVER_EXIT"
exit $DRIVER_EXIT
