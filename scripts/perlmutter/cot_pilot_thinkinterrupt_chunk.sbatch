#!/bin/bash
# Chunked CoT probe with think-interrupt (prod-budget) for the P-pilot.
# Submit with --export=ALL,CHUNK_ID=N,N_CHUNKS=M to pick a chunk.
# Each chunk writes its own parquet, so partial progress survives kills.

#SBATCH --job-name=cot_pilot_ti_chunk
#SBATCH --output=logs/%x_%j.log
#SBATCH --error=logs/%x_%j.err
#SBATCH -A m2651
#SBATCH -C gpu
#SBATCH -q debug
#SBATCH -t 00:30:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-node=4

set -euo pipefail
ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/../.." && pwd)}"
cd "$ROOT"
mkdir -p logs outputs/cot_pilot_thinkinterrupt

unset SIF OVERLAY
source "$ROOT/env.sh"

INPUT="${INPUT:-$ROOT/data/processed_cot_pilot_1k/data/train.parquet}"
MODEL="${MODEL:-Qwen/Qwen3-4B}"
TP="${TP:-4}"
THINKING_BUDGET="${THINKING_BUDGET:-12288}"
POST_INTERRUPT_BUDGET="${POST_INTERRUPT_BUDGET:-7184}"
TEMPERATURE="${TEMPERATURE:-1.0}"
N_ROLLOUTS="${N_ROLLOUTS:-8}"
CHUNK_ID="${CHUNK_ID:?must set CHUNK_ID}"
N_CHUNKS="${N_CHUNKS:?must set N_CHUNKS}"
TOTAL_ROWS="${TOTAL_ROWS:-1200}"

CHUNK_SIZE=$(( (TOTAL_ROWS + N_CHUNKS - 1) / N_CHUNKS ))
START=$(( CHUNK_ID * CHUNK_SIZE ))
END=$(( START + CHUNK_SIZE ))
[[ $END -gt $TOTAL_ROWS ]] && END=$TOTAL_ROWS
OUTPUT="$ROOT/outputs/cot_pilot_thinkinterrupt/chunk_${CHUNK_ID}_of_${N_CHUNKS}.parquet"

echo "=== cot_pilot_thinkinterrupt chunk $CHUNK_ID/$N_CHUNKS rows [$START:$END] ==="

PYTHONNOUSERSITE=1 apptainer exec --nv \
  --overlay "$OVERLAY:ro" --no-home --bind /etc/pki:/etc/pki \
  --env "PYTHONNOUSERSITE=1" --env "PYTHONUNBUFFERED=1" \
  --env "PYTHONPATH=$ROOT/src:/opt/phys-extras/" \
  --env "HF_HOME=$HF_HOME" --env "XDG_CACHE_HOME=/tmp/.cache" --env "HOME=/tmp" \
  --env "FLASHINFER_WORKSPACE_BASE=/tmp" \
  --env "TRITON_CACHE_DIR=/tmp/.cache/triton" \
  --env "VLLM_USE_V1=1" \
  "$SIF" \
  python3 "$ROOT/scripts/cot_probe_thinkinterrupt.py" \
    --input "$INPUT" \
    --output "$OUTPUT" \
    --model "$MODEL" \
    --tp "$TP" \
    --thinking_budget "$THINKING_BUDGET" \
    --post_interrupt_budget "$POST_INTERRUPT_BUDGET" \
    --temperature "$TEMPERATURE" \
    --n_rollouts "$N_ROLLOUTS" \
    --start_idx "$START" --end_idx "$END"
