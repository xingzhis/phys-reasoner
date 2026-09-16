#!/bin/bash
# probe_qwen3_4b_chunk.sbatch — 10-node premium probe chunk for Qwen3-4B TIR.
#
# Submit 5 copies (CHUNK_ID=0..4) to fill the 5-slot premium cap. Each chunk
# processes 17827/5 = 3566 prompts × n=8 on its own 10-node hbm40g allocation.
# Per chunk: 28,528 rollouts / 40 GPUs = 713 rollouts/GPU → ~48 min compute.
#
# Submit:
#   for C in 0 1 2 3 4; do
#     sbatch --export=ALL,CHUNK_ID=$C,N_CHUNKS=5 scripts/perlmutter/probe_qwen3_4b_chunk.sbatch
#   done
#
# Output: outputs/probe_qwen3_4b/fleet/chunk_${CHUNK_ID}/node_${NODE_ID}/gpu_${GPU_ID}/
# Uses eval/inference/rollout.py in TIR mode with training-matched budgets.
# Scoring is deferred to a post-step (score_probe_rollouts.py + xverify).

#SBATCH --job-name=probe_qwen3_4b_chunk
#SBATCH --output=logs/%x_%j.log
#SBATCH --error=logs/%x_%j.err
#SBATCH -A m2651
#SBATCH -C gpu&hbm40g
#SBATCH -q premium
#SBATCH -t 01:30:00
#SBATCH -N 10
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-node=4

set -euo pipefail
ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/../.." && pwd)}"
cd "$ROOT"

: "${CHUNK_ID:?CHUNK_ID (0..N_CHUNKS-1) required}"
: "${N_CHUNKS:?N_CHUNKS required (5 in standard fleet)}"

unset SIF OVERLAY
source "$ROOT/env.sh"
[[ -f "$SIF" && -f "$OVERLAY" ]] || { echo "ERROR: SIF/OVERLAY missing" >&2; exit 1; }

mkdir -p logs

echo "=== probe_qwen3_4b_chunk CHUNK=$CHUNK_ID/$N_CHUNKS ==="
echo "  Job    : $SLURM_JOB_ID"
echo "  Nodes  : $SLURM_JOB_NODELIST"
echo "  Start  : $(date -Iseconds)"

# ---- config ----
TRAIN_PARQUET="${TRAIN_PARQUET:-$ROOT/data/processed_tir/data/train.parquet}"
MODEL="${MODEL:-Qwen/Qwen3-4B}"
N_ROLLOUTS="${N_ROLLOUTS:-8}"
TEMPERATURE="${TEMPERATURE:-1.4}"
SEED_BASE="${SEED_BASE:-42}"

THINKING_BUDGET=12288
TOOL_CALL_BUDGET=2048
ANSWER_BUDGET=4096
MAX_PROMPT_LEN=1024
MODE="tir"

# Determine chunk's [start, end) over full parquet
source <(PYTHONNOUSERSITE=1 apptainer exec --overlay "$OVERLAY:ro" \
  --bind /etc/pki:/etc/pki --env "PYTHONPATH=/opt/phys-extras/" "$SIF" python3 -c "
import pyarrow.parquet as pq
n = pq.read_metadata('$TRAIN_PARQUET').num_rows
per = (n + $N_CHUNKS - 1) // $N_CHUNKS
s = $CHUNK_ID * per
e = min(s + per, n)
print(f'CHUNK_START={s}')
print(f'CHUNK_END={e}')
print(f'CHUNK_N={e-s}')
print(f'PARQUET_N={n}')
")

echo "[probe] chunk rows [$CHUNK_START, $CHUNK_END) = $CHUNK_N prompts (of $PARQUET_N total)"
echo "[probe] expected rollouts this chunk: $((CHUNK_N * N_ROLLOUTS))"

OUT_BASE="$ROOT/outputs/probe_qwen3_4b/fleet/chunk_$(printf '%02d' $CHUNK_ID)"
mkdir -p "$OUT_BASE"

# Resolve node list (10 nodes)
NODES=( $(scontrol show hostnames "$SLURM_JOB_NODELIST") )
N_NODES=${#NODES[@]}
echo "[probe] $N_NODES nodes: ${NODES[*]}"

# APT_CMD_STR removed — apptainer is invoked inline below to pin
# CUDA_VISIBLE_DEVICES per slot (critical on NERSC where SLURM's --export doesn't
# reliably propagate per-srun-task GPU pinning into containerized children).

# Split chunk across 40 GPUs (10 nodes × 4 GPUs)
TOTAL_GPUS=$(( N_NODES * 4 ))
PER_GPU=$(( (CHUNK_N + TOTAL_GPUS - 1) / TOTAL_GPUS ))
echo "[probe] split: $CHUNK_N prompts / $TOTAL_GPUS GPUs = $PER_GPU prompts per GPU"

T_LAUNCH=$(date +%s)
launched=0
for ((node_i=0; node_i<N_NODES; node_i++)); do
  NODE="${NODES[$node_i]}"
  for gpu in 0 1 2 3; do
    slot=$(( node_i * 4 + gpu ))
    s=$(( CHUNK_START + slot * PER_GPU ))
    e=$(( s + PER_GPU ))
    (( e > CHUNK_END )) && e=$CHUNK_END
    (( s >= CHUNK_END )) && continue

    OUT_DIR="$OUT_BASE/slot_$(printf '%03d' $slot)"
    mkdir -p "$OUT_DIR"

    # Run this GPU's shard via srun to the specific node, pinning CUDA_VISIBLE_DEVICES
    # through apptainer's --env (SLURM --export doesn't reliably propagate to
    # child processes inside the container — earlier attempts showed all 4
    # vLLM engines landed on GPU 0 sharing memory).
    # Stagger engine init within a node: gpu=0 starts immediately, gpu=k waits
    # k*20s inside the container. All srun's still fire in parallel from the
    # parent (slots on different nodes do not race each other — only the 4
    # engines on the same GPU node race for KV cache memory at vLLM init).
    STAGGER=$(( gpu * 20 ))
    srun --nodes=1 --ntasks=1 --nodelist="$NODE" --overlap --gpus-per-task=0 \
      bash -c "sleep $STAGGER; apptainer exec --nv --overlay '$OVERLAY:ro' --no-home --bind /etc/pki:/etc/pki \
        --env 'CUDA_VISIBLE_DEVICES=$gpu' \
        --env 'PYTHONNOUSERSITE=1' --env 'PYTHONUNBUFFERED=1' \
        --env 'PYTHONPATH=$ROOT/src:$ROOT/.async-extras:/opt/phys-extras/' \
        --env 'HF_HOME=$HF_HOME' --env 'HF_HUB_OFFLINE=1' --env 'TRANSFORMERS_OFFLINE=1' \
        --env 'XDG_CACHE_HOME=/tmp/.cache' --env 'HOME=/tmp' \
        --env 'FLASHINFER_WORKSPACE_BASE=/tmp' --env 'TRITON_CACHE_DIR=/tmp/.cache/triton' \
        --env 'MPLCONFIGDIR=/tmp/.cache/matplotlib' \
        --env 'PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True' \
        --env 'VLLM_USE_V1=1' \
        --env 'VLLM_MAX_NUM_SEQS=32' --env 'VLLM_MAX_NUM_BATCHED_TOKENS=8192' \
        '$SIF' python3 -u eval/inference/rollout.py \
        --mode $MODE --model '$MODEL' \
        --parquet '$TRAIN_PARQUET' \
        --n -1 --n_rollouts $N_ROLLOUTS \
        --start_idx $s --end_idx $e \
        --seed $((SEED_BASE + slot)) \
        --temperature $TEMPERATURE --top_p 1.0 \
        --max_prompt_len $MAX_PROMPT_LEN \
        --thinking_budget $THINKING_BUDGET \
        --tool_call_budget $TOOL_CALL_BUDGET \
        --answer_budget $ANSWER_BUDGET \
        --gpu_mem 0.85 \
        --chunk_size 25 \
        --out_dir '$OUT_DIR'" \
      > "logs/${SLURM_JOB_NAME}_${SLURM_JOB_ID}_chunk${CHUNK_ID}_slot${slot}.log" 2>&1 &
    launched=$((launched+1))
  done
done

echo "[probe] launched $launched worker srun processes"
RC=0
wait || RC=$?
T_END=$(date +%s)
ELAPSED=$(( T_END - T_LAUNCH ))

TOTAL_ROLLOUTS=$(( CHUNK_N * N_ROLLOUTS ))
echo "[probe-metric] chunk=$CHUNK_ID total_rollouts=$TOTAL_ROLLOUTS elapsed_seconds=$ELAPSED exit=$RC"
exit $RC
