#!/usr/bin/env bash
# Per-benchmark "CoT wins vs TIR" diff dumper. For each benchmark where both
# TIR/<tag> and CoT/<tag> have a scored.parquet, writes per-case txt files to
# outputs/eval/<bench>/diff_cot_wins_<tag>/ for human spot-checking.
#
# Runs purely on CPU (reads parquets, writes txt) — safe to run while the
# training-rollout orchestrator is using both GPUs.
#
# Usage:
#   bash eval/dump_cot_wins.sh                       # tag=train, max=50 cases
#   TAG=qwen MAX=30 bash eval/dump_cot_wins.sh       # different preset or cap
#   DIRECTION=tir_wins bash eval/dump_cot_wins.sh    # inverse direction

set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd -P)"

MODEL="${MODEL:-Qwen/Qwen3.5-4B}"
MODEL_SLUG="$(printf %s "$MODEL" | tr '/' '-' | tr -c '[:alnum:]._-' _)"
TAG="${TAG:-train}"
DIRECTION="${DIRECTION:-cot_wins}"
MAX="${MAX:-50}"

BENCHMARKS=(
    pool_v2_scibench
    pool_v2_physics
    pool_v2_ugphysics
    olympiad_oe_to_physics
    pool_v2_drsci
)

for BENCH in "${BENCHMARKS[@]}"; do
    TIR_DIR="$ROOT/outputs/eval/${BENCH}/tir__${MODEL_SLUG}__${TAG}"
    COT_DIR="$ROOT/outputs/eval/${BENCH}/cot__${MODEL_SLUG}__${TAG}"
    OUT_DIR="$ROOT/outputs/eval/${BENCH}/diff_${DIRECTION}_${TAG}"

    TIR_OK=0; COT_OK=0
    [ -f "$TIR_DIR/scored.parquet" ] && [ -f "$TIR_DIR/rollouts.parquet" ] && TIR_OK=1
    [ -f "$COT_DIR/scored.parquet" ] && [ -f "$COT_DIR/rollouts.parquet" ] && COT_OK=1

    if [ $TIR_OK -eq 0 ] || [ $COT_OK -eq 0 ]; then
        echo "[skip $BENCH] missing scored/rollouts (tir=$TIR_OK cot=$COT_OK)"
        continue
    fi

    echo "[$BENCH] → $OUT_DIR"
    PYTHONNOUSERSITE=1 apptainer exec \
      --overlay "$ROOT/phys-reasoner-overlay-017.img:ro" --no-home \
      --bind /etc/pki:/etc/pki --bind "$ROOT" \
      --bind "$(realpath "$ROOT/outputs")" \
      --pwd "$ROOT" \
      --env PYTHONNOUSERSITE=1 \
      --env "PYTHONPATH=$ROOT:$ROOT/src:$ROOT/eval/_pkgs:/opt/phys-extras/" \
      "$ROOT/verl_vllm017.latest.sif" \
      python3 -m eval.scoring.diff_tir_vs_cot \
        --tir_dir "$TIR_DIR" \
        --cot_dir "$COT_DIR" \
        --out_dir "$OUT_DIR" \
        --direction "$DIRECTION" \
        --max "$MAX"
done

# Summary of cot-wins counts across benchmarks (also per answer_type breakdown)
echo ""
echo "=== Per-benchmark summary (direction=$DIRECTION tag=$TAG) ==="
for BENCH in "${BENCHMARKS[@]}"; do
    M="$ROOT/outputs/eval/${BENCH}/diff_${DIRECTION}_${TAG}/_manifest.txt"
    if [ -f "$M" ]; then
        n_found=$(grep '^n_found' "$M" | awk -F':' '{print $2}' | tr -d ' ')
        echo "--- $BENCH  (n_found=$n_found) ---"
        echo "  types:"
        awk 'NR>3 {print $3}' "$M" | sort | uniq -c | sed 's/^/   /'
    fi
done
