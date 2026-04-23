#!/bin/bash
# prod_tir_het_4t_vllmxv_40g.sbatch — TIR prod with co-located vLLM xverify.
#
# Variant of prod_tir_het_4t_sharedxv_40g.sbatch:
#   - drops the dependency on a separately-submitted xverify server
#   - adds het-group 2 (1 node × 4 GPU) running scripts/serve_xverify_vllm.py
#     with TP=4 (continuous batching, ~150-400 req/s vs ~5 req/s serial)
#   - xverify lives exactly as long as training; one premium slot total
#
# All trainer + rollout knobs are inherited unchanged from the *_sharedxv_*.
#
# Submit:
#   sbatch --export=ALL,\
#       TRAIN_FILES=$PWD/data/processed_tir_filtered/data/train.parquet,\
#       VAL_FILES=$PWD/data/processed_tir_filtered/data/validation.parquet \
#       scripts/perlmutter/prod_tir_het_4t_vllmxv_40g.sbatch

#SBATCH --job-name=physcode_prod_tir_4t_vllmxv_40g
#SBATCH --output=logs/%x_%j.log
#SBATCH --error=logs/%x_%j.err

# =============================================================================
# HET-GROUP 0: TRAINER 4×4 (FSDP-16) on 40G
# =============================================================================
#SBATCH -A m2651
#SBATCH -C gpu&hbm40g
#SBATCH -q premium
#SBATCH -t 30:00:00
#SBATCH -N 4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-node=4

#SBATCH hetjob
# =============================================================================
# HET-GROUP 1: ROLLOUT 6×4 on 40G
# =============================================================================
#SBATCH -A m2651
#SBATCH -C gpu&hbm40g
#SBATCH -q premium
#SBATCH -t 30:00:00
#SBATCH -N 6
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-node=4

#SBATCH hetjob
# =============================================================================
# HET-GROUP 2: XVERIFY 1×4 (vLLM TP=4) on 40G
# =============================================================================
#SBATCH -A m2651
#SBATCH -C gpu&hbm40g
#SBATCH -q premium
#SBATCH -t 30:00:00
#SBATCH -N 1
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

echo "=== prod_tir_het_4t_vllmxv_40g.sbatch (TIR prod, co-located vLLM xverify) ==="
echo "  Job       : $SLURM_JOB_ID"
echo "  Trainer   : $SLURM_JOB_NODELIST_HET_GROUP_0  (FSDP-16 on 40G)"
echo "  Rollout   : $SLURM_JOB_NODELIST_HET_GROUP_1"
echo "  xVerify   : $SLURM_JOB_NODELIST_HET_GROUP_2  (vLLM TP=4 on 40G)"
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

TRAIN_NODES=( $(scontrol show hostnames "$SLURM_JOB_NODELIST_HET_GROUP_0") )
HEAD_NODE="${TRAIN_NODES[0]}"
TRAIN_WORKER_NODES=( "${TRAIN_NODES[@]:1}" )
WORKER_NODES=( $(scontrol show hostnames "$SLURM_JOB_NODELIST_HET_GROUP_1") )
XVERIFY_NODE=$(scontrol show hostnames "$SLURM_JOB_NODELIST_HET_GROUP_2" | head -n1)
export HEAD_NODE TRAIN_WORKER_NODES WORKER_NODES XVERIFY_NODE

echo "[topology] HEAD=$HEAD_NODE  TRAIN_WORKERS=${TRAIN_WORKER_NODES[*]}  ROLLOUT=${WORKER_NODES[*]}  XVERIFY=$XVERIFY_NODE"
(( ${#TRAIN_WORKER_NODES[@]} == 3 )) || echo "WARN: expected 3 trainer workers, got ${#TRAIN_WORKER_NODES[@]}"
(( ${#WORKER_NODES[@]} == 6 )) || echo "WARN: expected 6 rollout workers, got ${#WORKER_NODES[@]}"

# ---- launch vLLM xverify (TP=4) on het-group 2 ----
# Use the canonical current.url location: VeRL's _discover_xverify_url() falls
# back to this path when XVERIFY_URL env isn't propagated to Ray workers (which
# happens whenever the rollouter / agent-loop workers spawn on rollout nodes —
# they don't inherit the trainer-shell env). Writing to a per-job file caused
# the rollouter to fail at its first reward call (verified in 51905206 logs).
URL_FILE="${XVERIFY_URL_FILE:-$ROOT/outputs/xverify_endpoints/current.url}"
rm -f "$URL_FILE"
echo "[xverify] launching vLLM TP=4 server on $XVERIFY_NODE ..."
srun --het-group=2 --nodes=1 --ntasks=1 \
     --output="logs/%x_%j.xverify.log" \
     bash -c "
        source $ROOT/env.sh
        export PYTHONNOUSERSITE=1
        HOST=\$(hostname)
        URL=http://\$HOST:8765/judge
        # Atomic rendezvous publish once /health reports ready
        ( for _ in \$(seq 1 240); do
            sleep 5
            if curl -sf --max-time 5 http://\$HOST:8765/health 2>/dev/null | grep -q '\"ready\":[[:space:]]*true'; then
                tmp='$URL_FILE.tmp.\$\$'
                {
                    echo \$URL
                    echo '# host='\$HOST
                    echo '# port=8765'
                    echo '# tp_size=4'
                    echo '# backend=vllm-async'
                    echo '# job_id=$SLURM_JOB_ID'
                    echo '# written_at='\$(date -Iseconds)
                } > \$tmp
                mv \$tmp '$URL_FILE'
                echo '[xverify] healthy, wrote $URL_FILE'
                break
            fi
        done ) &
        apptainer exec --nv \
          --overlay '$OVERLAY:ro' --no-home --bind /etc/pki:/etc/pki \
          --env PYTHONNOUSERSITE=1 --env PYTHONUNBUFFERED=1 \
          --env PYTHONPATH=$ROOT/src:/opt/phys-extras/ \
          --env HF_HOME=$HF_HOME --env HF_HUB_OFFLINE=1 --env TRANSFORMERS_OFFLINE=1 \
          --env XDG_CACHE_HOME=/tmp/.cache --env HOME=/tmp \
          --env FLASHINFER_WORKSPACE_BASE=/tmp --env TRITON_CACHE_DIR=/tmp/.cache/triton \
          --env VLLM_USE_V1=1 \
          '$SIF' \
          python3 $ROOT/scripts/serve_xverify_vllm.py \
            --host 0.0.0.0 --port 8765 \
            --model IAAR-Shanghai/xVerify-7B-I \
            --tp_size 4 --gpu_mem 0.85 --max_model_len 4096
     " &
XVERIFY_PID=$!

export N_GPUS_TRAIN="${N_GPUS_TRAIN:-4}"
export N_GPUS_ROLLOUT="${N_GPUS_ROLLOUT:-4}"
export NNODES_TRAIN="${NNODES_TRAIN:-4}"
export NNODES_ROLLOUT="${NNODES_ROLLOUT:-6}"
export HET_SRUN_HEAD="--het-group=0"
export HET_SRUN_WORKER="--het-group=1"
source "$ROOT/scripts/perlmutter/_ray_bringup.sh"

cleanup() {
    local rc=$?
    echo "[cleanup] exit=$rc — tearing down Ray + xverify"
    type ray_teardown &>/dev/null && ray_teardown || true
    kill $XVERIFY_PID 2>/dev/null || true
    rm -f "$URL_FILE"
    return $rc
}
trap cleanup EXIT

# Poll our own xverify URL_FILE
XVERIFY_WAIT_SECS="${XVERIFY_WAIT_SECS:-1800}"
echo "[xverify] polling $URL_FILE (timeout ${XVERIFY_WAIT_SECS}s) ..."
deadline=$(( $(date +%s) + XVERIFY_WAIT_SECS ))
while (( $(date +%s) < deadline )); do
    [[ -f "$URL_FILE" ]] && { XVERIFY_URL=$(head -n1 "$URL_FILE"); break; }
    sleep 10
done
[[ -z "${XVERIFY_URL:-}" ]] && { echo "ERROR: xverify never came up" >&2; exit 1; }
echo "[xverify] URL = $XVERIFY_URL"
export XVERIFY_URL

# ---- production knobs (same model/batch/budgets/temperature as 80G prod) ----
export MODEL="${MODEL:-Qwen/Qwen3-4B}"
export TRAIN_TEMPERATURE="${TRAIN_TEMPERATURE:-1.4}"
export VAL_TEMPERATURE="${VAL_TEMPERATURE:-1.0}"

export TRAIN_BATCH="${TRAIN_BATCH:-128}"
export ROLLOUT_N="${ROLLOUT_N:-8}"
export TOTAL_STEPS="${TOTAL_STEPS:-552}"
export TOTAL_EPOCHS="${TOTAL_EPOCHS:-3}"
export SAVE_FREQ="${SAVE_FREQ:-50}"
export TEST_FREQ="${TEST_FREQ:-50}"

# Prod budgets — MAX_RESPONSE_LEN = 12288 + 16 + 2048 + 1024 + 4096 = 19,472.
export THINKING_BUDGET="${THINKING_BUDGET:-12288}"
export TOOL_CALL_BUDGET="${TOOL_CALL_BUDGET:-2048}"
export ANSWER_BUDGET="${ANSWER_BUDGET:-4096}"

# 40G memory knobs (validated via smoke v3)
export VLLM_GPU_MEM_UTIL="${VLLM_GPU_MEM_UTIL:-0.72}"
export USE_DYNAMIC_BSZ="${USE_DYNAMIC_BSZ:-1}"
export PPO_MAX_TOKEN_LEN_PER_GPU="${PPO_MAX_TOKEN_LEN_PER_GPU:-20480}"
export ACTOR_OPT_OFFLOAD="${ACTOR_OPT_OFFLOAD:-True}"

export TRAIN_SP="${TRAIN_SP:-2}"
export STALENESS="${STALENESS:-1}"
export LR="${LR:-3e-6}"

# Post-filter parquets (Qwen3-4B probe at temp=1.4, kept 1-7/8 informative).
export TRAIN_FILES="${TRAIN_FILES:-$ROOT/data/processed_tir/data/train.parquet}"
export VAL_FILES="${VAL_FILES:-$ROOT/data/processed_tir/data/validation.parquet}"

export WANDB_PROJECT="${WANDB_PROJECT:-physcode_tir}"
export LOGGERS="${LOGGERS:-console,wandb}"
export EXPERIMENT="${EXPERIMENT:-grpo_tir_qwen3b_t14_lr3e6_40g_vllmxv_$(date +%Y%m%d.%H%M%S)}"

echo "[driver] launching train_async.sh on $HEAD_NODE (EXPERIMENT=$EXPERIMENT, FSDP=$((NNODES_TRAIN*N_GPUS_TRAIN)))"
srun --overlap --het-group=0 --nodes=1 --ntasks=1 -w "$HEAD_NODE" \
     bash "$ROOT/scripts/train_async.sh"
DRIVER_EXIT=$?
echo "[driver] exited with $DRIVER_EXIT"
exit $DRIVER_EXIT
