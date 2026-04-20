#!/usr/bin/env bash
# Follow-up sweep covering the external benchmarks that are NOT in the main
# 5-benchmark matrix: PHYBench (1000 rows), ABench Phy_A (400), ABench Phy_B
# (400 = 100×4 parametric). Same 4 (mode, preset) cells as run_all_sweeps.sh:
#
#   tir/train, cot/train, tir/qwen, cot/qwen
#
# Each cell is sharded across both GPUs via run_sharded_sweep.sh (with
# BENCHMARKS env override targeting only the external ones). Outputs land in
# outputs/eval/<bench>/<mode>__<model>__<tag>/ following the main matrix's
# naming convention.
#
# Waits for any currently-running sharded launcher (run_sharded_sweep or
# legacy run_pool_v2_sharded) before starting, so this can be queued while
# the main matrix is still in flight.
#
# Usage:
#   MODEL=Qwen/Qwen3-4B-Thinking-2507 bash eval/run_external_sweeps.sh
#   MODEL=Qwen/Qwen3.5-4B bash eval/run_external_sweeps.sh  # for the paused qwen3.5 baseline

set -uo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd -P)"
LOGDIR="$ROOT/logs"
mkdir -p "$LOGDIR"

MODEL="${MODEL:-Qwen/Qwen3-4B-Thinking-2507}"
MODEL_SLUG="$(printf %s "$MODEL" | tr '/' '-' | tr -c '[:alnum:]._-' _)"
EXT_BENCHMARKS="phybench,abench_phy_a,abench_phy_b"
SUMMARY_FILE="$LOGDIR/external_sweeps_summary.txt"
: > "$SUMMARY_FILE"

echo "[external-orchestrator] polling for any running sharded launcher (~30s ticks)..."
# pgrep treats backslash-pipe `\|` as a literal string under ERE, not
# alternation — use separate pgrep calls to test both names.
while pgrep -af 'run_sharded_sweep'   >/dev/null 2>&1 \
   || pgrep -af 'run_pool_v2_sharded' >/dev/null 2>&1 \
   || pgrep -af 'run_all_sweeps'      >/dev/null 2>&1; do
    sleep 30
done
echo "[external-orchestrator] no sharded launcher running — proceeding"

_run_sweep() {
    local mode="$1"
    local preset="$2"
    local tag="$preset"
    local logfile="$LOGDIR/external_sweep_${mode}_${preset}.log"
    echo "[external-orchestrator] === sweep: mode=$mode preset=$preset tag=$tag ==="
    echo "[external-orchestrator]     log → $logfile"
    MODEL="$MODEL" MODE="$mode" PRESET="$preset" TAG="$tag" \
        BENCHMARKS="$EXT_BENCHMARKS" \
        bash "$ROOT/eval/run_sharded_sweep.sh" > "$logfile" 2>&1
    local rc=$?
    if [ $rc -eq 0 ]; then
        echo "[external-orchestrator] sweep $mode/$preset OK"
    else
        echo "[external-orchestrator] sweep $mode/$preset FAILED (exit $rc) — see $logfile"
    fi
    return $rc
}

[ -z "${SKIP_TIR_TRAIN:-}" ] && _run_sweep tir train
[ -z "${SKIP_COT_TRAIN:-}" ] && _run_sweep cot train
[ -z "${SKIP_TIR_QWEN:-}" ]  && _run_sweep tir qwen
[ -z "${SKIP_COT_QWEN:-}" ]  && _run_sweep cot qwen

# Final summary table
{
    echo "=== EXTERNAL SWEEPS SUMMARY (model=$MODEL) ==="
    echo ""
    for MODE in tir cot; do
        for PRESET in train qwen; do
            TAG="$PRESET"
            echo "--- mode=$MODE preset=$PRESET tag=$TAG ---"
            for BENCH in phybench abench_phy_a abench_phy_b; do
                S="$ROOT/outputs/eval/${BENCH}/${MODE}__${MODEL_SLUG}__${TAG}/scored.summary.txt"
                if [ -f "$S" ]; then
                    pass=$(grep '^pass@1' "$S" | head -1)
                    n=$(grep '^n ' "$S" | head -1 | awk '{print $NF}')
                    echo "  $BENCH (n=$n)  $pass"
                else
                    echo "  $BENCH  MISSING ($S)"
                fi
            done
            echo ""
        done
    done
} | tee "$SUMMARY_FILE"

echo ""
echo "[external-orchestrator] full summary written to $SUMMARY_FILE"
