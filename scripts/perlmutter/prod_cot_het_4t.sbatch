#!/bin/bash
# prod_cot_het_4t.sbatch — CoT-GRPO baseline for paper ablation (Phase-4 topology).
#
# Identical to prod_tir_het_4t.sbatch EXCEPT for COT_BASELINE=1, which flips:
#   - actor_rollout_ref.rollout.agent.default_agent_loop = single_turn_agent
#   - actor_rollout_ref.rollout.multi_turn.enable        = false
# (see scripts/train_async.sh:200-205)
#
# All hparams (LR, batch, clips, epochs, budgets, FSDP) match the TIR run so the
# comparison isolates the tool contribution. Uses the CoT-prompted dataset variant
# (data/processed_cot/data/), which has identical rows to TIR — only the system
# prompt is swapped.
#
# Submit:
#   sbatch scripts/perlmutter/prod_cot_het_4t.sbatch
# Resume:
#   sbatch --export=ALL,EXPERIMENT=<prev> scripts/perlmutter/prod_cot_het_4t.sbatch

#SBATCH --job-name=physcode_prod_cot_het_4t
#SBATCH --output=logs/%x_%j.log
#SBATCH --error=logs/%x_%j.err

# --- het-group 0: TRAINER 4×4 (FSDP-16) ---
#SBATCH -A m2651
#SBATCH -C gpu&hbm80g
#SBATCH -q regular
#SBATCH -t 48:00:00
#SBATCH -N 4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-node=4

#SBATCH hetjob
# --- het-group 1: ROLLOUT 6×4 ---
#SBATCH -A m2651
#SBATCH -C gpu&hbm80g
#SBATCH -q regular
#SBATCH -t 48:00:00
#SBATCH -N 4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-node=4

#SBATCH hetjob
# --- het-group 2: XVERIFY 1×1 ---
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

echo "=== prod_cot_het_4t.sbatch (CoT-GRPO baseline, no tool) ==="
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
    --env "NCCL_IB_TIMEOUT=32"
    --env "NCCL_IBEXT_DISABLE=1"
    --env "TORCH_NCCL_ENABLE_MONITORING=0"
    "$SIF"
)
export APT_CMD

# ---- topology resolution ----
TRAIN_NODES=( $(scontrol show hostnames "$SLURM_JOB_NODELIST_HET_GROUP_0") )
HEAD_NODE="${TRAIN_NODES[0]}"
TRAIN_WORKER_NODES=( "${TRAIN_NODES[@]:1}" )
WORKER_NODES=( $(scontrol show hostnames "$SLURM_JOB_NODELIST_HET_GROUP_1") )
XVERIFY_NODE=$(scontrol show hostnames "$SLURM_JOB_NODELIST_HET_GROUP_2" | head -n1)
export HEAD_NODE TRAIN_WORKER_NODES WORKER_NODES XVERIFY_NODE

echo "[topology] HEAD=$HEAD_NODE  TRAIN_WORKERS=${TRAIN_WORKER_NODES[*]}  ROLLOUT=${WORKER_NODES[*]}"

# ---- launch xVerify ----
# Per-JOBID URL file so parallel prod runs (TIR + CoT) don't race on the
# shared rendezvous path. See scripts/serve_xverify.sh for the env contract.
URL_FILE="$ROOT/outputs/xverify_endpoints/${SLURM_JOB_ID}.url"
export URL_FILE
rm -f "$URL_FILE"
echo "[xverify] launching on $XVERIFY_NODE (URL_FILE=$URL_FILE) ..."
srun --het-group=2 --nodes=1 --ntasks=1 \
     --output="logs/%x_%j.xverify.log" \
     --export=ALL,URL_FILE="$URL_FILE" \
     bash "$ROOT/scripts/serve_xverify.sh" &
XVERIFY_PID=$!

# ---- Ray bring-up ----
export N_GPUS_TRAIN="${N_GPUS_TRAIN:-4}"
export N_GPUS_ROLLOUT="${N_GPUS_ROLLOUT:-4}"
export NNODES_TRAIN="${NNODES_TRAIN:-4}"
export NNODES_ROLLOUT="${NNODES_ROLLOUT:-4}"
export HET_SRUN_HEAD="--het-group=0"
export HET_SRUN_WORKER="--het-group=1"
source "$ROOT/scripts/perlmutter/_ray_bringup.sh"

cleanup() {
    local rc=$?
    type ray_teardown &>/dev/null && ray_teardown || true
    kill "$XVERIFY_PID" 2>/dev/null || true
    return $rc
}
trap cleanup EXIT

# URL_FILE (per-JOBID) was set above before launching xVerify.
XVERIFY_WAIT_SECS="${XVERIFY_WAIT_SECS:-900}"
deadline=$(( $(date +%s) + XVERIFY_WAIT_SECS ))
while (( $(date +%s) < deadline )); do
    [[ -f "$URL_FILE" ]] && { XVERIFY_URL=$(head -n1 "$URL_FILE"); break; }
    sleep 10
done
[[ -z "${XVERIFY_URL:-}" ]] && { echo "ERROR: xverify not up" >&2; exit 1; }
export XVERIFY_URL
echo "[xverify] URL=$XVERIFY_URL"

# ---- CoT baseline: COT_BASELINE=1 flips train_async.sh to single_turn_agent ----
export COT_BASELINE=1
export MODEL="${MODEL:-Qwen/Qwen3-4B}"
export TRAIN_TEMPERATURE="${TRAIN_TEMPERATURE:-1.4}"
export VAL_TEMPERATURE="${VAL_TEMPERATURE:-1.0}"
export TRAIN_BATCH="${TRAIN_BATCH:-128}"
export ROLLOUT_N="${ROLLOUT_N:-8}"
# 3 epochs over filtered pool — both TOTAL_STEPS and TOTAL_EPOCHS must agree
# because verl caps by min(TOTAL_STEPS * BATCH, len(dataloader) * TOTAL_EPOCHS).
export TOTAL_STEPS="${TOTAL_STEPS:-552}"
export TOTAL_EPOCHS="${TOTAL_EPOCHS:-3}"
export SAVE_FREQ="${SAVE_FREQ:-50}"
export TEST_FREQ="${TEST_FREQ:-50}"
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
# Distinct experiment name so wandb + outputs don't collide with TIR run.
export EXPERIMENT="${EXPERIMENT:-grpo_cot_qwen3b_t14_reg_$(date +%Y%m%d.%H%M%S)}"

echo "[driver] launching train_async.sh COT_BASELINE=1 (EXPERIMENT=$EXPERIMENT, FSDP=$((NNODES_TRAIN*N_GPUS_TRAIN)))"
srun --overlap --het-group=0 --nodes=1 --ntasks=1 -w "$HEAD_NODE" \
     bash "$ROOT/scripts/train_async.sh"
DRIVER_EXIT=$?
echo "[driver] exited with $DRIVER_EXIT"
exit $DRIVER_EXIT
