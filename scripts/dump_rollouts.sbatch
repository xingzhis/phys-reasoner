#!/bin/bash
#SBATCH --job-name=dump_rollouts
#SBATCH --partition=gpu
#SBATCH --qos=qos_nmi
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:h200:1
#SBATCH --time=1:00:00
#SBATCH --output=logs/%x_%j.log
#SBATCH --error=logs/%x_%j.err

# Configurable via --export:
#   MODEL          — HF model ID (default: Qwen/Qwen3.5-4B)
#   N              — number of rollouts (default: 4)
#   PARQUET        — parquet file, or "smoke" for a built-in trivial 1-problem
#                    parquet useful for format/thinking smoke tests (default: corpus_train.parquet)
#   GPU_MEM        — vLLM gpu_memory_utilization (default: 0.7)
#   MAX_TOKENS     — max response tokens (default: 8192)
#   SEED           — random seed (default: 42)
#   ENABLE_THINKING — set to 1 to enable <think> in phase 1 (default: 0)
#
# Usage:
#   sbatch scripts/dump_rollouts.sbatch
#   sbatch --export=ALL,N=1,PARQUET=smoke,ENABLE_THINKING=0 scripts/dump_rollouts.sbatch
#   sbatch --export=ALL,N=1,PARQUET=smoke,ENABLE_THINKING=1 scripts/dump_rollouts.sbatch

ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
source "$ROOT/env.sh"

MODEL="${MODEL:-Qwen/Qwen3.5-4B}"
N="${N:-4}"
PARQUET="${PARQUET:-$ROOT/data/processed/corpus_train.parquet}"
GPU_MEM="${GPU_MEM:-0.7}"
MAX_TOKENS="${MAX_TOKENS:-8192}"
SEED="${SEED:-42}"
ENABLE_THINKING="${ENABLE_THINKING:-1}"  # 1=enable <think> before tool call (recommended); 0 to suppress

TIMESTAMP=$(date +%Y%m%d.%H%M%S)
OUT_DIR="$ROOT/outputs/rollouts_${TIMESTAMP}_${SLURM_JOB_ID}"
mkdir -p "$OUT_DIR" "$ROOT/logs"

export PYTHONUNBUFFERED=1
export PYTHONNOUSERSITE=1

# PARQUET=smoke: generate a trivial one-problem parquet for format/thinking smoke tests.
# The problem is intentionally simple (ball drop) so the model finishes well within token budget.
if [ "$PARQUET" = "smoke" ]; then
    PARQUET="$OUT_DIR/smoke.parquet"
    PYTHONNOUSERSITE=1 apptainer exec \
      --overlay "$OVERLAY:ro" --no-home \
      --bind /etc/pki:/etc/pki \
      --env "PYTHONNOUSERSITE=1" \
      "$SIF" python3 - <<'PY' "$PARQUET"
import sys, pandas as pd
from phys_reasoner.tir.prompts import TIR_SYSTEM_PROMPT
path = sys.argv[1]
pd.DataFrame([{
    "data_source": "smoke",
    "prompt": [
        {"role": "system", "content": TIR_SYSTEM_PROMPT},
        {"role": "user",   "content": "A ball is dropped from rest. After t = 3 s, how far has it fallen? Use g = 9.8 m/s²."},
    ],
    "reward_model": {"ground_truth": "44.1", "style": "rule"},
    "extra_info": {"answer_type": "numerical", "unit": "m", "problem":
        "A ball is dropped from rest. After t = 3 s, how far has it fallen? Use g = 9.8 m/s²."},
}]).to_parquet(path, index=False)
print(f"Wrote smoke parquet → {path}")
PY
fi

echo "Job:        $SLURM_JOB_ID  ($SLURM_JOB_NAME)"
echo "Node:       $SLURMD_NODENAME"
echo "GPU:        $CUDA_VISIBLE_DEVICES"
echo "Model:      $MODEL"
echo "Parquet:    $PARQUET"
echo "N:          $N  seed=$SEED"
echo "Max tokens: $MAX_TOKENS  gpu_mem=$GPU_MEM  enable_thinking=$ENABLE_THINKING"
echo "Output:     $OUT_DIR"

apptainer exec --nv \
  --overlay "$OVERLAY:ro" \
  --no-home \
  --bind /etc/pki:/etc/pki \
  --env "PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin" \
  --env "PYTHONNOUSERSITE=1" \
  --env "PYTHONUNBUFFERED=1" \
  --env "HF_HOME=$HF_HOME" \
  --env "HF_DATASETS_OFFLINE=1" \
  "$SIF" \
  python3 "$ROOT/scripts/dump_rollouts.py" \
    --model    "$MODEL" \
    --parquet  "$PARQUET" \
    --n        "$N" \
    --out_dir  "$OUT_DIR" \
    --seed     "$SEED" \
    --gpu_mem  "$GPU_MEM" \
    --max_tokens "$MAX_TOKENS" \
    --temperature 1.0 \
    --top_p    0.9 \
    $( [ "$ENABLE_THINKING" = "1" ] && echo "--enable_thinking" )

echo "Exit: $?"
echo "Rollouts saved to: $OUT_DIR"
