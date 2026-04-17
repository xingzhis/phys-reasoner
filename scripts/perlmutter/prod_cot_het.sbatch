#!/bin/bash
# prod_cot_het.sbatch — CoT-GRPO baseline for the TIR paper ablation.
#
# Identical to prod_tir_het.sbatch EXCEPT for COT_BASELINE=1, which flips:
#   - actor_rollout_ref.rollout.agent.default_agent_loop = single_turn_agent
#   - actor_rollout_ref.rollout.multi_turn.enable        = false
#
# All other hparams (LR, batch, clips, steps, dataset) are identical so the
# comparison with the main TIR run isolates the tool contribution.
#
# Submit (AFTER filling TODO(collab) markers — identical to prod_tir_het.sbatch):
#   sbatch scripts/perlmutter/prod_cot_het.sbatch
# Resume:
#   sbatch --export=ALL,EXPERIMENT=<prev> scripts/perlmutter/prod_cot_het.sbatch
#
# Uses the CoT-prompted dataset variant (data/processed_cot/data/), produced
# by scripts/build_cot_parquets.py and published to HF as <user>/phys-cot.
# Identical rows to the TIR parquets, only the system prompt is swapped.

#SBATCH --job-name=physcode_prod_cot_het
#SBATCH --output=logs/%x_%j.log
#SBATCH --error=logs/%x_%j.err

# --- het-group 0: TRAINER ---
# TODO(collab): same as prod_tir_het.sbatch
#SBATCH -A TODO_ACCOUNT_g
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH -t 48:00:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-node=4

#SBATCH hetjob
# --- het-group 1: ROLLOUT ---
#SBATCH -A TODO_ACCOUNT_g
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH -t 48:00:00
#SBATCH -N 5
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-node=4

#SBATCH hetjob
# --- het-group 2: XVERIFY ---
#SBATCH -A TODO_ACCOUNT_g
#SBATCH -C gpu
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
[[ -f "$SIF" && -f "$OVERLAY" ]] || { echo "ERROR: SIF/OVERLAY missing" >&2; exit 1; }

echo "=== prod_cot_het.sbatch (CoT-GRPO baseline, no tool) ==="
echo "  Trainer   : $SLURM_JOB_NODELIST_HET_GROUP_0"
echo "  Rollout   : $SLURM_JOB_NODELIST_HET_GROUP_1"
echo "  xVerify   : $SLURM_JOB_NODELIST_HET_GROUP_2"

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

export N_GPUS_TRAIN="${N_GPUS_TRAIN:-4}"
export N_GPUS_ROLLOUT="${N_GPUS_ROLLOUT:-4}"
export NNODES_TRAIN="${NNODES_TRAIN:-1}"
export NNODES_ROLLOUT="${NNODES_ROLLOUT:-5}"
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

# ----- CoT baseline: identical hparams to prod_tir_het.sbatch, just no tool -----
export COT_BASELINE=1
export MODEL="${MODEL:-Qwen/Qwen3.5-4B}"
export TRAIN_BATCH="${TRAIN_BATCH:-128}"
export ROLLOUT_N="${ROLLOUT_N:-8}"
export TOTAL_STEPS="${TOTAL_STEPS:-3000}"
export SAVE_FREQ="${SAVE_FREQ:-200}"
export TEST_FREQ="${TEST_FREQ:-100}"
export THINKING_BUDGET="${THINKING_BUDGET:-12288}"
export TOOL_CALL_BUDGET="${TOOL_CALL_BUDGET:-2048}"
export ANSWER_BUDGET="${ANSWER_BUDGET:-4096}"
export VLLM_GPU_MEM_UTIL="${VLLM_GPU_MEM_UTIL:-0.85}"
export USE_DYNAMIC_BSZ="${USE_DYNAMIC_BSZ:-1}"
export PPO_MAX_TOKEN_LEN_PER_GPU="${PPO_MAX_TOKEN_LEN_PER_GPU:-24576}"
export ACTOR_OPT_OFFLOAD="${ACTOR_OPT_OFFLOAD:-True}"
export TRAIN_FILES="${TRAIN_FILES:-$ROOT/data/processed_cot/data/train.parquet}"
export VAL_FILES="${VAL_FILES:-$ROOT/data/processed_cot/data/validation.parquet}"
export WANDB_PROJECT="${WANDB_PROJECT:-physcode_tir}"
export LOGGERS="${LOGGERS:-console,wandb}"
# Distinct experiment name so wandb + outputs/ don't collide with TIR run.
export EXPERIMENT="${EXPERIMENT:-grpo_cot_$(date +%Y%m%d.%H%M%S)}"

echo "[driver] launching train_async.sh COT_BASELINE=1 (EXPERIMENT=$EXPERIMENT)"
srun --overlap --het-group=0 --nodes=1 --ntasks=1 -w "$HEAD_NODE" \
     bash "$ROOT/scripts/train_async.sh"
DRIVER_EXIT=$?
echo "[driver] exited with $DRIVER_EXIT"
exit $DRIVER_EXIT
