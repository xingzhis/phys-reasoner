#!/bin/bash
# Parallel-chunk train-matched scorer.
#
# Env vars (required):
#   MERGED      absolute path to an input merged parquet
#   CHUNK_ID    zero-padded chunk index (used in out_name + logs)
#   N_CHUNKS    total number of chunks to split MERGED into
#
# Env vars (optional):
#   OUT_NAME    default: rollouts_scored_trainmatched_chunk_${CHUNK_ID}.parquet
#   OUT_DIR     default: outputs/probe_qwen3_4b_v2_107k
#   GPU_MEM     0.85
#   MAX_MODEL_LEN  16384
#   ROW_TIMEOUT    5
#
# Launch via salloc trailing-command. Works with both interactive and debug QoS.
# Each chunk is independent and writes its own {chunk_id}-suffixed output file —
# no overwrite risk.

set -euo pipefail
ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/../.." && pwd)}"
cd "$ROOT"

: "${MERGED:?MERGED parquet path required}"
: "${CHUNK_ID:?CHUNK_ID required}"
: "${N_CHUNKS:?N_CHUNKS required}"

unset SIF OVERLAY
source "$ROOT/env.sh"
[[ -f "$SIF" && -f "$OVERLAY" ]] || { echo "ERROR: SIF/OVERLAY missing" >&2; exit 1; }

OUT_DIR="${OUT_DIR:-$ROOT/outputs/probe_qwen3_4b_v2_107k}"
OUT_NAME="${OUT_NAME:-rollouts_scored_trainmatched_chunk_${CHUNK_ID}.parquet}"
GPU_MEM="${GPU_MEM:-0.85}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-16384}"
ROW_TIMEOUT="${ROW_TIMEOUT:-5}"
XV_MODEL="${XV_MODEL:-IAAR-Shanghai/xVerify-7B-I}"

mkdir -p "$OUT_DIR"

# Compute [start_idx, end_idx) for this chunk over MERGED.
source <(PYTHONNOUSERSITE=1 apptainer exec --overlay "$OVERLAY:ro" \
  --bind /etc/pki:/etc/pki --env "PYTHONPATH=/opt/phys-extras/" "$SIF" python3 -c "
import pyarrow.parquet as pq
n = pq.read_metadata('$MERGED').num_rows
per = (n + $N_CHUNKS - 1) // $N_CHUNKS
s = $CHUNK_ID * per
e = min(s + per, n)
print(f'START_IDX={s}')
print(f'END_IDX={e}')
print(f'CHUNK_N={e-s}')
print(f'TOTAL_N={n}')
")

echo "=== score_train_matched_chunk ==="
echo "  Job     : ${SLURM_JOB_ID:-N/A}"
echo "  Node    : $(hostname)"
echo "  Merged  : $MERGED (total rows: $TOTAL_N)"
echo "  Chunk   : $CHUNK_ID/$N_CHUNKS  slice [$START_IDX:$END_IDX) = $CHUNK_N rows"
echo "  Out     : $OUT_DIR/$OUT_NAME"
echo "  Start   : $(date -Iseconds)"

T0=$(date +%s)
srun --ntasks=1 --nodes=1 --gpus-per-node=1 --cpus-per-task=32 --overlap \
  bash -c "PYTHONNOUSERSITE=1 apptainer exec --nv \
  --overlay '$OVERLAY:ro' --no-home --bind /etc/pki:/etc/pki \
  --env 'PYTHONNOUSERSITE=1' --env 'PYTHONUNBUFFERED=1' \
  --env 'PYTHONPATH=$ROOT/src:/opt/phys-extras/' \
  --env 'HF_HOME=$HF_HOME' --env 'HF_HUB_OFFLINE=1' --env 'TRANSFORMERS_OFFLINE=1' \
  --env 'XDG_CACHE_HOME=/tmp/.cache' --env 'HOME=/tmp' \
  --env 'FLASHINFER_WORKSPACE_BASE=/tmp' \
  --env 'TRITON_CACHE_DIR=/tmp/.cache/triton' \
  --env 'MPLCONFIGDIR=/tmp/.cache/matplotlib' \
  --env 'VLLM_USE_V1=1' \
  '$SIF' \
  python3 -u '$ROOT/scripts/score_train_matched.py' \
    --merged '$MERGED' \
    --out_dir '$OUT_DIR' \
    --out_name '$OUT_NAME' \
    --start_idx $START_IDX --end_idx $END_IDX \
    --xverify_model '$XV_MODEL' \
    --gpu_mem '$GPU_MEM' \
    --max_model_len '$MAX_MODEL_LEN' \
    --row_timeout '$ROW_TIMEOUT'"

RC=$?
echo "[score_train_matched_chunk ${CHUNK_ID}] exit=$RC elapsed=$(($(date +%s)-T0))s"
exit $RC
