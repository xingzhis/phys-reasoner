#!/bin/bash
# Local (non-slurm) runner: fire both Qwen3 probes in background, one per GPU.
#
#   bash scripts/probe_qwen3_local.sh
#
# Logs: logs/probe_qwen3_{thinking,instruct}.log
# Outputs: data/results/probe_qwen3_{thinking,instruct}.parquet
#
# Sampling params match the Qwen3-4B-{Thinking,Instruct}-2507 HF model cards.
set -euo pipefail

ROOT="${ROOT:-$(cd "$(dirname "$0")/.." && pwd -P)}"
cd "$ROOT"
mkdir -p logs data/results

unset SIF OVERLAY
# shellcheck disable=SC1091
source "$ROOT/env.sh"

N_SAMPLES="${N_SAMPLES:-50}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-8192}"
PARQUET="${PARQUET:-$ROOT/data/processed/probe_subset.parquet}"

run_one() {
  local gpu="$1" model="$2" tag="$3" thinking="$4"
  local temp="$5" top_p="$6" top_k="$7" min_p="$8"
  local out="$ROOT/data/results/probe_qwen3_${tag}.parquet"
  local log="$ROOT/logs/probe_qwen3_${tag}.log"

  echo "[probe] gpu=$gpu model=$model tag=$tag thinking=$thinking T=$temp top_p=$top_p top_k=$top_k min_p=$min_p"
  echo "[probe] out=$out log=$log"

  CUDA_VISIBLE_DEVICES="$gpu" PYTHONNOUSERSITE=1 apptainer exec --nv \
    --overlay "$OVERLAY:ro" \
    --bind /etc/pki:/etc/pki \
    --env "PYTHONNOUSERSITE=1" \
    --env "PYTHONPATH=$ROOT/src:/opt/phys-extras/" \
    --env "HF_HOME=$HF_HOME" \
    --env "HF_TOKEN=${HF_TOKEN:-}" \
    "$SIF" \
    python3 -m phys_reasoner.eval.stage0_probe \
      --model "$model" \
      --parquet "$PARQUET" \
      --n_samples "$N_SAMPLES" \
      --max_new_tokens "$MAX_NEW_TOKENS" \
      --enable_thinking "$thinking" \
      --temperature "$temp" --top_p "$top_p" --top_k "$top_k" --min_p "$min_p" \
      --output "$out" \
      >"$log" 2>&1 &
  echo "[probe] pid=$! tag=$tag"
}

# Qwen3-4B-Thinking-2507: T=0.6 top_p=0.95 top_k=20 min_p=0 (HF card)
run_one 0 "Qwen/Qwen3-4B-Thinking-2507" thinking true 0.6 0.95 20 0
# Qwen3-4B-Instruct-2507: T=0.7 top_p=0.8  top_k=20 min_p=0 (HF card)
run_one 1 "Qwen/Qwen3-4B-Instruct-2507" instruct false 0.7 0.8 20 0

wait
echo "[probe] both jobs finished."
