#!/bin/bash
# smoke_tir_companion.sbatch — Full-topology smoke, FALLBACK variant (no het jobs).
#
# Mirrors prod_tir_companion.sbatch but with short step budget and dumps enabled.
# Pass criteria: see scripts/perlmutter/checklist.md.
#
# Submit:
#   sbatch scripts/perlmutter/smoke_tir_companion.sbatch

#SBATCH --job-name=physcode_smoke_tir_companion
#SBATCH --output=logs/%x_%j.log
#SBATCH --error=logs/%x_%j.err

# TODO(collab): fill — same as prod_tir_companion.sbatch
#SBATCH -A TODO_ACCOUNT_g
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH -t 04:00:00
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

echo "=== smoke_tir_companion.sbatch ==="
echo "  Nodelist : $SLURM_JOB_NODELIST"

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
    "$SIF"
)
export APT_CMD

XVERIFY_JOBID=""
if [[ -z "${XVERIFY_URL:-}" ]]; then
    rm -f "$ROOT/outputs/xverify_endpoints/current.url"
    XVERIFY_JOBID=$(sbatch --parsable "$ROOT/scripts/perlmutter/serve_xverify_perlmutter.sbatch" 2>&1 || echo "")
    [[ "$XVERIFY_JOBID" =~ ^[0-9]+$ ]] || { echo "ERROR: xverify sbatch failed: $XVERIFY_JOBID" >&2; exit 1; }
    echo "[xverify] companion job: $XVERIFY_JOBID"
fi

cleanup_outer() {
    local rc=$?
    ray_teardown 2>/dev/null || true
    [[ -n "$XVERIFY_JOBID" ]] && scancel "$XVERIFY_JOBID" 2>/dev/null || true
    return $rc
}
trap cleanup_outer EXIT

NODES=( $(scontrol show hostnames "$SLURM_JOB_NODELIST") )
(( ${#NODES[@]} >= 6 )) || { echo "ERROR: need >=6 nodes, got ${#NODES[@]}" >&2; exit 1; }
echo "[topo] allocated ${#NODES[@]} nodes; using 1 trainer + $((${#NODES[@]} - 1)) rollout"
HEAD_NODE="${NODES[0]}"
WORKER_NODES=( "${NODES[@]:1}" )
export HEAD_NODE WORKER_NODES

export N_GPUS_TRAIN="${N_GPUS_TRAIN:-4}"
export N_GPUS_ROLLOUT="${N_GPUS_ROLLOUT:-4}"
export NNODES_TRAIN="${NNODES_TRAIN:-1}"
export NNODES_ROLLOUT="${NNODES_ROLLOUT:-$(( ${#NODES[@]} - 1 ))}"
export HET_SRUN_HEAD="" HET_SRUN_WORKER=""
source "$ROOT/scripts/perlmutter/_ray_bringup.sh"

URL_FILE="$ROOT/outputs/xverify_endpoints/current.url"
deadline=$(( $(date +%s) + 1800 ))
while (( $(date +%s) < deadline )); do
    [[ -f "$URL_FILE" ]] && { XVERIFY_URL=$(head -n1 "$URL_FILE"); break; }
    sleep 10
done
[[ -z "${XVERIFY_URL:-}" ]] && { echo "ERROR: xverify not up" >&2; exit 1; }
export XVERIFY_URL
echo "[xverify] URL=$XVERIFY_URL"

# ----- smoke knobs (resource knobs already exported above) -----
export MODEL="${MODEL:-Qwen/Qwen3.5-4B}"
export TRAIN_BATCH="${TRAIN_BATCH:-128}"
export ROLLOUT_N="${ROLLOUT_N:-8}"
export TOTAL_STEPS="${TOTAL_STEPS:-10}"
export SAVE_FREQ="${SAVE_FREQ:-5}"
export TEST_FREQ="${TEST_FREQ:-5}"
export THINKING_BUDGET="${THINKING_BUDGET:-12288}"
export TOOL_CALL_BUDGET="${TOOL_CALL_BUDGET:-2048}"
export ANSWER_BUDGET="${ANSWER_BUDGET:-4096}"
export VLLM_GPU_MEM_UTIL="${VLLM_GPU_MEM_UTIL:-0.85}"
export USE_DYNAMIC_BSZ="${USE_DYNAMIC_BSZ:-1}"
export PPO_MAX_TOKEN_LEN_PER_GPU="${PPO_MAX_TOKEN_LEN_PER_GPU:-24576}"
export ACTOR_OPT_OFFLOAD="${ACTOR_OPT_OFFLOAD:-True}"
export DUMP_TRAIN_ROLLOUTS=1
export DUMP_VAL_ROLLOUTS=1

export TRAIN_FILES="${TRAIN_FILES:-$ROOT/data/processed_tir/data/train.parquet}"
export VAL_FILES="${VAL_FILES:-$ROOT/data/processed_tir/data/validation.parquet}"

export WANDB_PROJECT="${WANDB_PROJECT:-physcode_tir_smoke}"
export LOGGERS="${LOGGERS:-console,wandb}"
export EXPERIMENT="${EXPERIMENT:-smoke_tir_companion_$(date +%Y%m%d.%H%M%S)}"

echo "[driver] launching train_async.sh (EXPERIMENT=$EXPERIMENT)"
srun --overlap --nodes=1 --ntasks=1 -w "$HEAD_NODE" \
     bash "$ROOT/scripts/train_async.sh"
DRIVER_EXIT=$?
echo "[driver] exited with $DRIVER_EXIT"
exit $DRIVER_EXIT
