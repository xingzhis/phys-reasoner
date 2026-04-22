#!/bin/bash
# prod_tir_het_4t.sbatch — Production GRPO on Perlmutter (Phase-4 4T+6R+offload=False).
#
# Topology:
#   het-group 0  TRAINER   4 nodes × 4 A100   Ray head + 3 trainer workers (FSDP-16)
#   het-group 1  ROLLOUT   6 nodes × 4 A100   ~2.9× rollout margin at measured step time
#   het-group 2  XVERIFY   1 × 1 A100         reward server (full A100 node on m2651)
#
# Total: 11 compute nodes. Regular QoS (1× charge) per user direction — ok for
# 48h jobs because m2651's regular cap also allows 48h and wait times are
# typically 2-12h for 11-node jobs vs <1h on premium.
#
# Speedup validated on smoke_tir_het_4t (job 51819857): step 2 steady-state
# 4.04 min/step (vs 1T baseline 8.05 min) = 2.06× speedup on update_actor.
# 3 epochs (552 steps) × 4.04 min = 37 h run-time + ~3 h overhead → ~40 h total,
# 8 h margin under 48 h wall.
#
# Submit:
#   sbatch scripts/perlmutter/prod_tir_het_4t.sbatch
# Resume (same EXPERIMENT):
#   sbatch --export=ALL,EXPERIMENT=<prev> scripts/perlmutter/prod_tir_het_4t.sbatch

#SBATCH --job-name=physcode_prod_tir_het_4t
#SBATCH --output=logs/%x_%j.log
#SBATCH --error=logs/%x_%j.err

# =============================================================================
# HET-GROUP 0: TRAINER (Ray head + 3 trainer workers) — 4 nodes × 4 GPU (FSDP-16)
# =============================================================================
#SBATCH -A m2651
#SBATCH -C gpu&hbm80g
#SBATCH -q regular
#SBATCH -t 48:00:00
#SBATCH -N 4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-node=4

#SBATCH hetjob
# =============================================================================
# HET-GROUP 1: ROLLOUT — 6 nodes × 4 GPU
# =============================================================================
#SBATCH -A m2651
#SBATCH -C gpu&hbm80g
#SBATCH -q regular
#SBATCH -t 48:00:00
#SBATCH -N 4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-node=4

#SBATCH hetjob
# =============================================================================
# HET-GROUP 2: XVERIFY — 1 × 1 GPU (reserves whole A100 node on premium/regular)
# =============================================================================
#SBATCH -A m2651
#SBATCH -C gpu&hbm80g
#SBATCH -q regular
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
[[ -f "$SIF" && -f "$OVERLAY" ]] || { echo "ERROR: SIF/OVERLAY missing" >&2; exit 1; }

echo "=== prod_tir_het_4t.sbatch ==="
echo "  Job       : $SLURM_JOB_ID"
echo "  Trainer   : $SLURM_JOB_NODELIST_HET_GROUP_0  (FSDP-16)"
echo "  Rollout   : $SLURM_JOB_NODELIST_HET_GROUP_1"
echo "  xVerify   : $SLURM_JOB_NODELIST_HET_GROUP_2"
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
    # Perlmutter Slingshot NCCL tuning. IB_TIMEOUT=32 survived FSDP-16 init in
    # smoke D; raise to 64 here only if this prod run hits timeouts.
    --env "NCCL_IB_TIMEOUT=32"
    --env "NCCL_IBEXT_DISABLE=1"
    --env "TORCH_NCCL_ENABLE_MONITORING=0"
    # vLLM custom all-reduce disabled via verl config (train_async.sh).
    "$SIF"
)
export APT_CMD

# ---- topology resolution (4 trainer nodes: 1 head + 3 workers) ----
TRAIN_NODES=( $(scontrol show hostnames "$SLURM_JOB_NODELIST_HET_GROUP_0") )
HEAD_NODE="${TRAIN_NODES[0]}"
TRAIN_WORKER_NODES=( "${TRAIN_NODES[@]:1}" )
WORKER_NODES=( $(scontrol show hostnames "$SLURM_JOB_NODELIST_HET_GROUP_1") )
XVERIFY_NODE=$(scontrol show hostnames "$SLURM_JOB_NODELIST_HET_GROUP_2" | head -n1)
export HEAD_NODE TRAIN_WORKER_NODES WORKER_NODES XVERIFY_NODE

echo "[topology] HEAD=$HEAD_NODE  TRAIN_WORKERS=${TRAIN_WORKER_NODES[*]}  ROLLOUT=${WORKER_NODES[*]}"
(( ${#TRAIN_WORKER_NODES[@]} == 3 )) || echo "WARN: expected 3 trainer workers, got ${#TRAIN_WORKER_NODES[@]}"
(( ${#WORKER_NODES[@]} == 6 )) || echo "WARN: expected 6 rollout workers, got ${#WORKER_NODES[@]}"

# ---- launch xVerify ----
# Per-JOBID URL file so parallel prod runs (TIR + CoT) don't race on
# outputs/xverify_endpoints/current.url and end up cross-wired.
URL_FILE="$ROOT/outputs/xverify_endpoints/${SLURM_JOB_ID}.url"
export URL_FILE
rm -f "$URL_FILE"
echo "[xverify] launching on $XVERIFY_NODE (URL_FILE=$URL_FILE) ..."
srun --het-group=2 --nodes=1 --ntasks=1 \
     --output="logs/%x_%j.xverify.log" \
     --export=ALL,URL_FILE="$URL_FILE" \
     bash "$ROOT/scripts/serve_xverify.sh" &
XVERIFY_PID=$!

# ---- Ray cluster bring-up (FSDP-16 trainer pool + 6 rollout workers) ----
export N_GPUS_TRAIN="${N_GPUS_TRAIN:-4}"
export N_GPUS_ROLLOUT="${N_GPUS_ROLLOUT:-4}"
export NNODES_TRAIN="${NNODES_TRAIN:-4}"
export NNODES_ROLLOUT="${NNODES_ROLLOUT:-4}"
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

# ---- xVerify URL rendezvous (per-JOBID URL_FILE set above) ----
XVERIFY_WAIT_SECS="${XVERIFY_WAIT_SECS:-900}"
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

# ---- production knobs (all validated by smoke_tir_het_4t, job 51819857) ----
export MODEL="${MODEL:-Qwen/Qwen3-4B}"
# Match the SXV variant — base Qwen3-4B + train-temp 1.4 + val-temp 1.0
export TRAIN_TEMPERATURE="${TRAIN_TEMPERATURE:-1.4}"
export VAL_TEMPERATURE="${VAL_TEMPERATURE:-1.0}"

export TRAIN_BATCH="${TRAIN_BATCH:-128}"
export ROLLOUT_N="${ROLLOUT_N:-8}"
# 3 epochs over 23,589-row filtered pool: 23589 / 128 ≈ 184 steps/epoch × 3.
# NOTE: verl caps by min(TOTAL_STEPS * BATCH, len(dataloader) * TOTAL_EPOCHS) —
# BOTH must be set so the cap binds correctly. TOTAL_EPOCHS=3 is the binding
# knob here; TOTAL_STEPS is a safety ceiling slightly above the epoch math.
export TOTAL_STEPS="${TOTAL_STEPS:-552}"
export TOTAL_EPOCHS="${TOTAL_EPOCHS:-3}"
export SAVE_FREQ="${SAVE_FREQ:-50}"
export TEST_FREQ="${TEST_FREQ:-50}"

# Prod sequence budgets — think-interrupt enabled. MAX_RESPONSE_LEN = 19,472.
export THINKING_BUDGET="${THINKING_BUDGET:-12288}"
export TOOL_CALL_BUDGET="${TOOL_CALL_BUDGET:-2048}"
export ANSWER_BUDGET="${ANSWER_BUDGET:-4096}"

export VLLM_GPU_MEM_UTIL="${VLLM_GPU_MEM_UTIL:-0.85}"
export USE_DYNAMIC_BSZ="${USE_DYNAMIC_BSZ:-1}"
export PPO_MAX_TOKEN_LEN_PER_GPU="${PPO_MAX_TOKEN_LEN_PER_GPU:-20480}"

# Tier-A #2 full claim: offload=False + FSDP-16. Validated by smoke 51819857
# (2.06× update_actor speedup vs 1T baseline).
export ACTOR_OPT_OFFLOAD="${ACTOR_OPT_OFFLOAD:-False}"

# Qwen3 switch knobs. TRAIN_SP=2 at FSDP-16 leaves 8-way DP. Could try SP=4 (DP=4)
# later for another ~1.1-1.3× but not validated yet — keep 2 for stability.
export TRAIN_SP="${TRAIN_SP:-2}"
export STALENESS="${STALENESS:-1}"

export TRAIN_FILES="${TRAIN_FILES:-$ROOT/data/processed_tir/data/train.parquet}"
export VAL_FILES="${VAL_FILES:-$ROOT/data/processed_tir/data/validation.parquet}"

export WANDB_PROJECT="${WANDB_PROJECT:-physcode_tir}"
export LOGGERS="${LOGGERS:-console,wandb}"
export EXPERIMENT="${EXPERIMENT:-grpo_tir_qwen3b_t14_reg_$(date +%Y%m%d.%H%M%S)}"

# ---- launch driver on trainer head ----
echo "[driver] launching train_async.sh on $HEAD_NODE (EXPERIMENT=$EXPERIMENT, FSDP=$((NNODES_TRAIN*N_GPUS_TRAIN)))"
srun --overlap --het-group=0 --nodes=1 --ntasks=1 -w "$HEAD_NODE" \
     bash "$ROOT/scripts/train_async.sh"
DRIVER_EXIT=$?
echo "[driver] exited with $DRIVER_EXIT"
exit $DRIVER_EXIT
