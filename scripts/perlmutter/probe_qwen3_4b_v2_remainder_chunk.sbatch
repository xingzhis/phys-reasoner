#!/bin/bash
# probe_qwen3_4b_v2_remainder_chunk.sbatch
#
# Second probe pass against the older HF revision xingzhi0/phys-tir @ 39600de261
# (107,977 rows, a superset of the current 17,827-row train.parquet).
# We only roll out the 91,531-row REMAINDER (rows not already covered by the
# first probe), materialized at:
#   data/probe_v2_phys_tir_39600de2/remainder_vs_probe1.parquet
#
# Join key used to compute the remainder: md5(extra_info.problem + '|||' +
# reward_model.ground_truth). Mapping from remainder row index back to the
# original 107k row index lives in:
#   data/probe_v2_phys_tir_39600de2/remainder_to_old_idx.json
#
# Submit 5 copies (CHUNK_ID=0..4) on 10-node hbm40g allocations each:
#   for C in 0 1 2 3 4; do
#     sbatch --export=ALL,CHUNK_ID=$C,N_CHUNKS=5 \
#       scripts/perlmutter/probe_qwen3_4b_v2_remainder_chunk.sbatch
#   done
#
# Throughput extrapolated from last fleet run: 28,528 rollouts finished in
# ~9–16 min on 40 GPUs (median ~15 min). Per chunk here: 18,307 prompts × 8 =
# 146,456 rollouts / 40 GPUs ≈ 3,661 rollouts/GPU → ~77 min worst case.
# Walltime set to 01:45:00 to cover the tail; 1 h is NOT feasible for a single
# chunk, so a partial result is expected if walltime is short — intermediate
# parquet saves happen every 25 prompts per GPU so progress is preserved.
#
# Outputs go to a fresh tree to avoid any chance of overlapping with the first
# probe's files:
#   outputs/probe_qwen3_4b_v2_107k/fleet/chunk_${CHUNK_ID}/slot_${SLOT}/
# Scoring is deferred to a post-step (same pattern as first probe).

#SBATCH --job-name=probe_qwen3_4b_v2
#SBATCH --output=logs/%x_%j.log
#SBATCH --error=logs/%x_%j.err
#SBATCH -A m2651
#SBATCH -C gpu&hbm40g
#SBATCH -q premium
#SBATCH -t 01:45:00
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

echo "=== probe_qwen3_4b_v2_remainder CHUNK=$CHUNK_ID/$N_CHUNKS ==="
echo "  Job    : $SLURM_JOB_ID"
echo "  Nodes  : $SLURM_JOB_NODELIST"
echo "  Start  : $(date -Iseconds)"

# ---- config ----
# NB: TRAIN_PARQUET here points at the REMAINDER parquet, NOT the 107k full
# file. Rollout records' problem_idx field will therefore be in
# [0, 91531) = index into remainder_vs_probe1.parquet. Use
# data/probe_v2_phys_tir_39600de2/remainder_to_old_idx.json to map back to the
# original 107k revision row index if needed.
TRAIN_PARQUET="${TRAIN_PARQUET:-$ROOT/data/probe_v2_phys_tir_39600de2/remainder_vs_probe1.parquet}"
MODEL="${MODEL:-Qwen/Qwen3-4B}"
N_ROLLOUTS="${N_ROLLOUTS:-8}"
TEMPERATURE="${TEMPERATURE:-1.4}"
SEED_BASE="${SEED_BASE:-142}"  # different seed base from probe 1 (42) to avoid identical draws

THINKING_BUDGET=12288
TOOL_CALL_BUDGET=2048
ANSWER_BUDGET=4096
MAX_PROMPT_LEN=1024
MODE="tir"

# Determine chunk's [start, end) over the remainder parquet
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

echo "[probe-v2] chunk rows [$CHUNK_START, $CHUNK_END) = $CHUNK_N prompts (of $PARQUET_N total)"
echo "[probe-v2] expected rollouts this chunk: $((CHUNK_N * N_ROLLOUTS))"

OUT_BASE="$ROOT/outputs/probe_qwen3_4b_v2_107k/fleet/chunk_$(printf '%02d' $CHUNK_ID)"
mkdir -p "$OUT_BASE"

# Resolve node list (10 nodes)
NODES=( $(scontrol show hostnames "$SLURM_JOB_NODELIST") )
N_NODES=${#NODES[@]}
echo "[probe-v2] $N_NODES nodes: ${NODES[*]}"

# Split chunk across 40 GPUs (10 nodes × 4 GPUs)
TOTAL_GPUS=$(( N_NODES * 4 ))
PER_GPU=$(( (CHUNK_N + TOTAL_GPUS - 1) / TOTAL_GPUS ))
echo "[probe-v2] split: $CHUNK_N prompts / $TOTAL_GPUS GPUs = $PER_GPU prompts per GPU"

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

echo "[probe-v2] launched $launched worker srun processes"
RC=0
wait || RC=$?
T_END=$(date +%s)
ELAPSED=$(( T_END - T_LAUNCH ))

TOTAL_ROLLOUTS=$(( CHUNK_N * N_ROLLOUTS ))
echo "[probe-v2-metric] chunk=$CHUNK_ID total_rollouts=$TOTAL_ROLLOUTS elapsed_seconds=$ELAPSED exit=$RC"
exit $RC
