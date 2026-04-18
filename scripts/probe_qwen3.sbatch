#!/bin/bash
#SBATCH --job-name=probe_qwen3
#SBATCH --partition=gpu_h200
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --gres=gpu:h200:1
#SBATCH --time=1:30:00
#SBATCH --output=logs/%x_%j_%A_%a.log
#SBATCH --error=logs/%x_%j_%A_%a.err
#SBATCH --array=0-1

# Stage-0 probe: zero-shot TIR evaluation on Qwen3-4B-{Thinking,Instruct}-2507.
# Used to pick the starting model for the Qwen3.5 → Qwen3 infra switch.
# Decision input for .claude/plans/qwen3-switch.md §0 and §2.3.
#
# Array layout:
#   idx 0 → Qwen3-4B-Thinking-2507  (enable_thinking=True)
#   idx 1 → Qwen3-4B-Instruct-2507  (enable_thinking=False; no-think only)

set -euo pipefail

ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "$ROOT"
mkdir -p "$ROOT/logs" "$ROOT/data/results"

unset SIF OVERLAY
source "$ROOT/env.sh"

case "${SLURM_ARRAY_TASK_ID:-0}" in
  0)
    MODEL="Qwen/Qwen3-4B-Thinking-2507"
    TAG="thinking"
    ENABLE_THINKING=true
    ;;
  1)
    MODEL="Qwen/Qwen3-4B-Instruct-2507"
    TAG="instruct"
    ENABLE_THINKING=false
    ;;
  *)
    echo "Unknown SLURM_ARRAY_TASK_ID: ${SLURM_ARRAY_TASK_ID}" >&2
    exit 2
    ;;
esac

OUT="$ROOT/data/results/probe_qwen3_${TAG}.parquet"

echo "[probe] job=$SLURM_JOB_ID task=$SLURM_ARRAY_TASK_ID model=$MODEL tag=$TAG enable_thinking=$ENABLE_THINKING"
echo "[probe] node=$SLURMD_NODENAME out=$OUT"
nvidia-smi -L || true

PYTHONNOUSERSITE=1 apptainer exec --nv \
  --overlay "$OVERLAY:ro" \
  --bind /etc/pki:/etc/pki \
  --env "PYTHONNOUSERSITE=1" \
  --env "PYTHONPATH=$ROOT/src:/opt/phys-extras/" \
  --env "HF_HOME=$HF_HOME" \
  --env "HF_HUB_OFFLINE=1" \
  --env "TRANSFORMERS_OFFLINE=1" \
  "$SIF" \
  python3 -m phys_reasoner.eval.stage0_probe \
    --model "$MODEL" \
    --parquet "$ROOT/data/processed/probe_subset.parquet" \
    --n_samples 200 \
    --max_new_tokens 8192 \
    --enable_thinking "$ENABLE_THINKING" \
    --output "$OUT"

echo "[probe] done. out=$OUT"
