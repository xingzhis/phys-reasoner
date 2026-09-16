#!/bin/bash
# Quick CoT probe: base Qwen3-4B pass@1 on Z2's numerical_easy set.
# Goal: measure REAL CoT response length so we can build a CoT-length filter
# instead of the TIR-based one that doesn't work.
# No hetjob — just vLLM TP=4 on one node.

#SBATCH --job-name=cot_probe_4b_z2
#SBATCH --output=logs/%x_%j.log
#SBATCH --error=logs/%x_%j.err
#SBATCH -A m2651
#SBATCH -C gpu&hbm40g
#SBATCH -q premium
#SBATCH -t 00:30:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-node=4

set -euo pipefail
ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/../.." && pwd)}"
cd "$ROOT"
mkdir -p logs outputs/cot_probe_4b_z2

unset SIF OVERLAY
source "$ROOT/env.sh"

INPUT="${INPUT:-$ROOT/data/processed_cot_z2_numerical_easy/data/train.parquet}"
OUTPUT="${OUTPUT:-$ROOT/outputs/cot_probe_4b_z2/rollouts_pass1.parquet}"
MODEL="${MODEL:-Qwen/Qwen3-4B}"
TP="${TP:-4}"
MAX_TOKENS="${MAX_TOKENS:-4096}"
TEMP="${TEMP:-1.0}"
SUBSET="${SUBSET:-0}"

echo "=== cot_probe_4b_z2.sbatch ==="
echo "  Input : $INPUT"
echo "  Output: $OUTPUT"
echo "  Model : $MODEL"
echo "  TP    : $TP"

PYTHONNOUSERSITE=1 apptainer exec --nv \
  --overlay "$OVERLAY:ro" --no-home --bind /etc/pki:/etc/pki \
  --env "PYTHONNOUSERSITE=1" \
  --env "PYTHONUNBUFFERED=1" \
  --env "PYTHONPATH=$ROOT/src:/opt/phys-extras/" \
  --env "HF_HOME=$HF_HOME" \
  --env "XDG_CACHE_HOME=/tmp/.cache" --env "HOME=/tmp" \
  --env "FLASHINFER_WORKSPACE_BASE=/tmp" \
  --env "TRITON_CACHE_DIR=/tmp/.cache/triton" \
  --env "VLLM_USE_V1=1" \
  "$SIF" \
  python3 "$ROOT/scripts/cot_probe_base_4b.py" \
    --input "$INPUT" \
    --output "$OUTPUT" \
    --model "$MODEL" \
    --tp "$TP" \
    --max_tokens "$MAX_TOKENS" \
    --temperature "$TEMP" \
    --subset "$SUBSET"
