#!/bin/bash
# prod_tir_het_2t.sbatch — Production GRPO on Perlmutter (Phase-4 multi-trainer).
#
# Topology (dies together; any preemption tears the whole job down):
#   het-group 0  TRAINER   2 nodes × 4 A100   Ray head + trainer worker, FSDP-8
#   het-group 1  ROLLOUT   3 nodes × 4 A100   Ray workers + vLLM engines (TP=4)
#   het-group 2  XVERIFY   1 × 1 A100        reward server
#
# Uses the same knobs as prod_tir_het.sbatch but with trainer scaled to 2 nodes
# (FSDP-8). See .claude/handoff/plan-multi-trainer-node.md and FINDINGS §12
# Tier-A lever #2 (estimated 1.7–1.9× trainer speedup vs FSDP-4).
#
# Rollout default is 3 nodes (matches smoke_tir_het measured-adequate config at
# prod budget). If 2T update_actor drops below rollout wall-clock, bump to 6 by
# editing `-N 3` → `-N 6` in het-group-1 below (no other changes needed).
#
# First submission policy (per plan-multi-trainer-node.md): keep
# ACTOR_OPT_OFFLOAD=True for the first FSDP-8 run; flip to False once memory
# headroom is confirmed (potential extra 1.2–1.4× via FINDINGS Tier-B #9).
#
# Submit:
#   sbatch scripts/perlmutter/prod_tir_het_2t.sbatch
# Resume (same EXPERIMENT → verl auto-resumes):
#   sbatch --export=ALL,EXPERIMENT=<prev> scripts/perlmutter/prod_tir_het_2t.sbatch
#
# Rollback: the single-trainer-node variant prod_tir_het.sbatch is untouched.

#SBATCH --job-name=physcode_prod_tir_het_2t
#SBATCH --output=logs/%x_%j.log
#SBATCH --error=logs/%x_%j.err

# =============================================================================
# HET-GROUP 0: TRAINER (Ray head + 1 trainer worker) — 2 nodes × 4 GPU (FSDP-8)
# =============================================================================
#SBATCH -A m2651
#SBATCH -C gpu&hbm80g
#SBATCH -q premium
#SBATCH -t 48:00:00
#SBATCH -N 2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-node=4

#SBATCH hetjob
# =============================================================================
# HET-GROUP 1: ROLLOUT — 3 nodes × 4 GPU (bump to 6 if rollout becomes bottleneck)
# =============================================================================
#SBATCH -A m2651
#SBATCH -C gpu&hbm80g
#SBATCH -q premium
#SBATCH -t 48:00:00
#SBATCH -N 3
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-node=4

#SBATCH hetjob
# =============================================================================
# HET-GROUP 2: XVERIFY — 1 × 1 GPU (reserves a whole 4×A100 on m2651)
# =============================================================================
#SBATCH -A m2651
#SBATCH -C gpu&hbm80g
#SBATCH -q premium
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

echo "=== prod_tir_het_2t.sbatch ==="
echo "  Job       : $SLURM_JOB_ID"
echo "  Trainer   : $SLURM_JOB_NODELIST_HET_GROUP_0  (FSDP-8)"
echo "  Rollout   : $SLURM_JOB_NODELIST_HET_GROUP_1"
echo "  xVerify   : $SLURM_JOB_NODELIST_HET_GROUP_2"
echo "  SIF       : $SIF"
echo "  OVERLAY   : $OVERLAY"
echo "  HF_HOME   : $HF_HOME"

# -----------------------------------------------------------------------------
# Apptainer prefix.
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
    --env "PYTORCH_ALLOC_CONF=expandable_segments:True"
    # NCCL tuning for Perlmutter Slingshot. Cross-node FSDP-8 adds inter-node
    # all-reduce; NCCL_IB_TIMEOUT=32 is the current value; raise to 64 only if
    # timeouts appear during trainer-to-trainer NCCL init.
    --env "NCCL_IB_TIMEOUT=32"
    --env "NCCL_IBEXT_DISABLE=1"
    --env "TORCH_NCCL_ENABLE_MONITORING=0"
    # vLLM's custom all-reduce disabled via verl config (train_async.sh).
    "$SIF"
)
export APT_CMD

# -----------------------------------------------------------------------------
# Resolve node roles. Trainer het-group splits into HEAD (Ray head) + WORKERS.
# -----------------------------------------------------------------------------
TRAIN_NODES=( $(scontrol show hostnames "$SLURM_JOB_NODELIST_HET_GROUP_0") )
HEAD_NODE="${TRAIN_NODES[0]}"
TRAIN_WORKER_NODES=( "${TRAIN_NODES[@]:1}" )
WORKER_NODES=( $(scontrol show hostnames "$SLURM_JOB_NODELIST_HET_GROUP_1") )
XVERIFY_NODE=$(scontrol show hostnames "$SLURM_JOB_NODELIST_HET_GROUP_2" | head -n1)
export HEAD_NODE TRAIN_WORKER_NODES WORKER_NODES XVERIFY_NODE

echo "[topology] HEAD=$HEAD_NODE  TRAIN_WORKERS=${TRAIN_WORKER_NODES[*]}  ROLLOUT=${WORKER_NODES[*]}"
(( ${#TRAIN_WORKER_NODES[@]} == 1 )) || echo "WARN: expected 1 trainer worker, got ${#TRAIN_WORKER_NODES[@]}"
(( ${#WORKER_NODES[@]} == 3 )) || echo "WARN: expected 3 rollout workers, got ${#WORKER_NODES[@]}"

# -----------------------------------------------------------------------------
# Launch xVerify.
# -----------------------------------------------------------------------------
rm -f "$ROOT/outputs/xverify_endpoints/current.url"
echo "[xverify] launching on $XVERIFY_NODE ..."
srun --het-group=2 --nodes=1 --ntasks=1 \
     --output="logs/%x_%j.xverify.log" \
     bash "$ROOT/scripts/serve_xverify.sh" &
XVERIFY_PID=$!

# -----------------------------------------------------------------------------
# Ray cluster bring-up (FSDP-8 trainer pool + rollout workers).
# -----------------------------------------------------------------------------
export N_GPUS_TRAIN="${N_GPUS_TRAIN:-4}"
export N_GPUS_ROLLOUT="${N_GPUS_ROLLOUT:-4}"
export NNODES_TRAIN="${NNODES_TRAIN:-2}"
export NNODES_ROLLOUT="${NNODES_ROLLOUT:-3}"
export HET_SRUN_HEAD="--het-group=0"
export HET_SRUN_WORKER="--het-group=1"
source "$ROOT/scripts/perlmutter/_ray_bringup.sh"

cleanup() {
    local rc=$?
    echo "[cleanup] exit=$rc — tearing down Ray + xVerify"
    type ray_teardown &>/dev/null && ray_teardown || true
    kill "$XVERIFY_PID" 2>/dev/null || true
    return $rc
}
trap cleanup EXIT

# -----------------------------------------------------------------------------
# Poll for xVerify URL.
# -----------------------------------------------------------------------------
URL_FILE="$ROOT/outputs/xverify_endpoints/current.url"
XVERIFY_WAIT_SECS="${XVERIFY_WAIT_SECS:-900}"
echo "[xverify] polling $URL_FILE (timeout ${XVERIFY_WAIT_SECS}s) ..."
deadline=$(( $(date +%s) + XVERIFY_WAIT_SECS ))
while (( $(date +%s) < deadline )); do
    [[ -f "$URL_FILE" ]] && { XVERIFY_URL=$(head -n1 "$URL_FILE"); break; }
    sleep 10
done
[[ -z "${XVERIFY_URL:-}" ]] && { echo "ERROR: xverify did not come up" >&2; exit 1; }
echo "[xverify] URL = $XVERIFY_URL"
health_url="${XVERIFY_URL%/judge}/health"
hc_deadline=$(( $(date +%s) + 300 ))
while (( $(date +%s) < hc_deadline )); do
    curl -sf --max-time 5 "$health_url" >/dev/null 2>&1 && { echo "[xverify] healthy"; break; }
    sleep 10
done
export XVERIFY_URL

# -----------------------------------------------------------------------------
# Production knobs. Same defaults as prod_tir_het.sbatch.
# -----------------------------------------------------------------------------
export MODEL="${MODEL:-Qwen/Qwen3-4B-Thinking-2507}"

export TRAIN_BATCH="${TRAIN_BATCH:-128}"
export ROLLOUT_N="${ROLLOUT_N:-8}"
# 3 epochs over 23,589-row filtered pool: 23589 / 128 ≈ 184 steps/epoch × 3.
export TOTAL_STEPS="${TOTAL_STEPS:-552}"
export SAVE_FREQ="${SAVE_FREQ:-50}"
export TEST_FREQ="${TEST_FREQ:-50}"

# Prod sequence budgets — think-interrupt enabled. MAX_RESPONSE_LEN = 19,472.
export THINKING_BUDGET="${THINKING_BUDGET:-12288}"
export TOOL_CALL_BUDGET="${TOOL_CALL_BUDGET:-2048}"
export ANSWER_BUDGET="${ANSWER_BUDGET:-4096}"

export VLLM_GPU_MEM_UTIL="${VLLM_GPU_MEM_UTIL:-0.85}"
export USE_DYNAMIC_BSZ="${USE_DYNAMIC_BSZ:-1}"
export PPO_MAX_TOKEN_LEN_PER_GPU="${PPO_MAX_TOKEN_LEN_PER_GPU:-20480}"

# ACTOR_OPT_OFFLOAD policy for Phase-4:
#   - First FSDP-8 run: keep True (memory-safe baseline, same as 1T prod).
#   - Once green: flip to False via --export for the full Tier-A #2 speedup
#     (combined 1T offload-on → 2T offload-off = ~1.8-2.5× trainer).
export ACTOR_OPT_OFFLOAD="${ACTOR_OPT_OFFLOAD:-True}"

# Qwen3 knobs (unchanged from 1T prod). TRAIN_SP=2 at FSDP-8 leaves 4-way DP.
# Could try SP=4 (DP=2) later but stick with known-good SP=2 on first 2T prod.
export TRAIN_SP="${TRAIN_SP:-2}"
export STALENESS="${STALENESS:-1}"

export TRAIN_FILES="${TRAIN_FILES:-$ROOT/data/processed_tir/data/train.parquet}"
export VAL_FILES="${VAL_FILES:-$ROOT/data/processed_tir/data/validation.parquet}"

export WANDB_PROJECT="${WANDB_PROJECT:-physcode_tir}"
export LOGGERS="${LOGGERS:-console,wandb}"
export EXPERIMENT="${EXPERIMENT:-grpo_tir_2t_$(date +%Y%m%d.%H%M%S)}"

# -----------------------------------------------------------------------------
# Launch driver on trainer head. FSDP-8 spans HEAD + TRAIN_WORKER_NODES.
# -----------------------------------------------------------------------------
echo "[driver] launching train_async.sh on $HEAD_NODE (EXPERIMENT=$EXPERIMENT, FSDP=$((NNODES_TRAIN*N_GPUS_TRAIN)))"
srun --overlap --het-group=0 --nodes=1 --ntasks=1 -w "$HEAD_NODE" \
     bash "$ROOT/scripts/train_async.sh"
DRIVER_EXIT=$?
echo "[driver] exited with $DRIVER_EXIT"
exit $DRIVER_EXIT
