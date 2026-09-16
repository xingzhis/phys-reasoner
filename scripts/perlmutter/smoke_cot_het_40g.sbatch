#!/bin/bash
# smoke_cot_het_40g.sbatch — memory-fit smoke for the CoT prod path on A100 40GB.
#
# Goal: in ~1h, validate that the prod CoT training config runs without OOM on
# hbm40g nodes. If it passes we swap the pending prod_cot_het_4t_sharedxv.sbatch
# to 40GB for a ~10x faster queue (premium hbm40g has 0 pending jobs ahead).
#
# Topology (minimal stress test, not a perf test):
#   het-group 0  TRAINER   2 nodes × 4 A100-40G   FSDP-8, Ulysses-SP=2
#   het-group 1  ROLLOUT   2 nodes × 4 A100-40G   8 vLLM engines
# xVerify: reused from the currently-running 80G xverify job (51874940).
#          current.url already points at it — if that job dies, smoke OOM
#          check still runs; xverify only matters for reward accuracy, not
#          memory fit.
#
# 40G-specific memory knobs vs prod_cot_het_4t_sharedxv.sbatch:
#   ACTOR_OPT_OFFLOAD      False → True   (~32 GB optimizer state to CPU)
#   VLLM_GPU_MEM_UTIL      0.85  → 0.80   (leave headroom: KV budget is half)
#   PPO_MAX_TOKEN_LEN_PER_GPU  20480 → 16384  (caps activation peak)
#   TRAIN_BATCH            128   → 64    (halves concurrent rollouts, speeds smoke)
#   TOTAL_STEPS            552   → 3     (3 iters is enough to hit peak memory)

#SBATCH --job-name=physcode_smoke_cot_40g
#SBATCH --output=logs/%x_%j.log
#SBATCH --error=logs/%x_%j.err

# --- het-group 0: TRAINER (2 × 4 GPU = FSDP-8) ---
# gpu_debug: 30min hard cap, 8-node cap, fast-start queue. Avoids the
# 5-job premium cap (already full). FSDP-8 is a memory-tightness upper
# bound vs prod FSDP-16 (more bytes per rank), so if smoke fits, prod fits.
#SBATCH -A m2651
#SBATCH -C gpu&hbm40g
#SBATCH -q debug
#SBATCH -t 00:30:00
#SBATCH -N 2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-node=4

#SBATCH hetjob
# --- het-group 1: ROLLOUT (2 × 4 GPU) ---
#SBATCH -A m2651
#SBATCH -C gpu&hbm40g
#SBATCH -q debug
#SBATCH -t 00:30:00
#SBATCH -N 2
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

echo "=== smoke_cot_het_40g.sbatch (40G memory-fit check for CoT prod path) ==="
echo "  Job     : $SLURM_JOB_ID"
echo "  Trainer : $SLURM_JOB_NODELIST_HET_GROUP_0  (FSDP-8 on 40G)"
echo "  Rollout : $SLURM_JOB_NODELIST_HET_GROUP_1  (8 vLLM engines on 40G)"
echo "  xVerify : reused from existing current.url"

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
export NNODES_TRAIN="${NNODES_TRAIN:-2}"
export NNODES_ROLLOUT="${NNODES_ROLLOUT:-2}"
export HET_SRUN_HEAD="--het-group=0"
export HET_SRUN_WORKER="--het-group=1"
source "$ROOT/scripts/perlmutter/_ray_bringup.sh"

cleanup() {
    local rc=$?
    type ray_teardown &>/dev/null && ray_teardown || true
    return $rc
}
trap cleanup EXIT

# Use existing xverify (currently running 80G xverify_server job 51874940 with
# ~20h TIME_LEFT at submit time). Any reachable xverify works for a memory-fit
# smoke; we're not measuring reward accuracy here.
URL_FILE="${SHARED_URL_FILE:-$ROOT/outputs/xverify_endpoints/current.url}"
XVERIFY_WAIT_SECS="${XVERIFY_WAIT_SECS:-120}"
echo "[xverify] polling $URL_FILE (timeout ${XVERIFY_WAIT_SECS}s) ..."
deadline=$(( $(date +%s) + XVERIFY_WAIT_SECS ))
while (( $(date +%s) < deadline )); do
    [[ -f "$URL_FILE" ]] && { XVERIFY_URL=$(head -n1 "$URL_FILE"); break; }
    sleep 5
done
[[ -z "${XVERIFY_URL:-}" ]] && { echo "ERROR: xverify URL missing" >&2; exit 1; }
echo "[xverify] URL=$XVERIFY_URL"
health_url="${XVERIFY_URL%/judge}/health"
hc_deadline=$(( $(date +%s) + 120 ))
while (( $(date +%s) < hc_deadline )); do
    curl -sf --max-time 5 "$health_url" >/dev/null 2>&1 && { echo "[xverify] healthy"; break; }
    sleep 5
done
export XVERIFY_URL

# ---- CoT path ----
export COT_BASELINE=1
export MODEL="${MODEL:-Qwen/Qwen3-4B}"
export TRAIN_TEMPERATURE="${TRAIN_TEMPERATURE:-1.4}"
export VAL_TEMPERATURE="${VAL_TEMPERATURE:-1.0}"

# Smoke sizing — under the 30min debug walltime. Peak memory is set by
# max_response_len (unchanged), not batch; 1 step × TRAIN_BATCH=32 × ROLLOUT_N=8
# fully exercises rollout KV cache, reward compute, FSDP forward/backward/opt.
export TRAIN_BATCH="${TRAIN_BATCH:-32}"
export ROLLOUT_N="${ROLLOUT_N:-8}"
export TOTAL_STEPS="${TOTAL_STEPS:-1}"
export TOTAL_EPOCHS="${TOTAL_EPOCHS:-1}"
export SAVE_FREQ="${SAVE_FREQ:--1}"
export TEST_FREQ="${TEST_FREQ:--1}"
export VAL_BEFORE_TRAIN="${VAL_BEFORE_TRAIN:-false}"

# Prod budgets — same as prod_cot_het_4t_sharedxv.sbatch so we hit the real
# max_response_len (19472). This is the only memory stressor that matters.
export THINKING_BUDGET="${THINKING_BUDGET:-12288}"
export TOOL_CALL_BUDGET="${TOOL_CALL_BUDGET:-2048}"
export ANSWER_BUDGET="${ANSWER_BUDGET:-4096}"

# 40G memory knobs.
# v1 @ 0.80: OOM on weight-sync (vLLM CUDA graphs push actual use to 34.5 GB,
#            <1 GB left for CheckpointEngineWorker buffer).
# v2 @ 0.65: PASSED (step=1 timing_s/step=279.53s). Safe baseline.
# v3 @ 0.72 + PPO_MAX=20480: max-throughput probe — combines more vLLM KV cache
#            with prod-value activation packing. If v3 OOMs on rollout, revert
#            to 0.65; if on trainer, revert PPO_MAX to 16384.
export VLLM_GPU_MEM_UTIL="${VLLM_GPU_MEM_UTIL:-0.72}"
export USE_DYNAMIC_BSZ="${USE_DYNAMIC_BSZ:-1}"
export PPO_MAX_TOKEN_LEN_PER_GPU="${PPO_MAX_TOKEN_LEN_PER_GPU:-20480}"
export ACTOR_OPT_OFFLOAD="${ACTOR_OPT_OFFLOAD:-True}"

export TRAIN_SP="${TRAIN_SP:-2}"
export STALENESS="${STALENESS:-1}"

export TRAIN_FILES="${TRAIN_FILES:-$ROOT/data/processed_cot/data/train.parquet}"
export VAL_FILES="${VAL_FILES:-$ROOT/data/processed_cot/data/validation.parquet}"

export WANDB_PROJECT="${WANDB_PROJECT:-physcode_tir_smoke}"
export LOGGERS="${LOGGERS:-console,wandb}"
export EXPERIMENT="${EXPERIMENT:-smoke_cot_40g_$(date +%Y%m%d.%H%M%S)}"

echo "[driver] launching train_async.sh COT_BASELINE=1 (EXPERIMENT=$EXPERIMENT, FSDP=$((NNODES_TRAIN*N_GPUS_TRAIN)))"
srun --overlap --het-group=0 --nodes=1 --ntasks=1 -w "$HEAD_NODE" \
     bash "$ROOT/scripts/train_async.sh"
DRIVER_EXIT=$?
echo "[driver] exited with $DRIVER_EXIT"
exit $DRIVER_EXIT
