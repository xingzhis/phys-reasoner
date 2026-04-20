#!/usr/bin/env bash
# Master orchestrator: runs the 4 mode × sampling-preset sweeps for the
# Qwen3.5-4B base zero-shot calibration matrix.
#
# Sweeps (sequential — each uses both GPUs):
#   tir × train  → outputs/eval/<bench>/tir__<model>__train/
#   cot × train  → outputs/eval/<bench>/cot__<model>__train/
#   tir × qwen   → outputs/eval/<bench>/tir__<model>__qwen/
#   cot × qwen   → outputs/eval/<bench>/cot__<model>__qwen/
#
# train preset: temp=1.0 top_p=1.0 top_k=-1 rep_pen=1.0  (matches RL training)
# qwen preset:  temp=0.6 top_p=0.95 top_k=20 rep_pen=1.0 (Qwen team-recommended)
# Budgets: thinking=12288 tool_call=2048 answer=4096 (matches training).
#
# This script can be invoked while a previous TIR-train run is still ongoing —
# it polls until that process exits, then renames its outputs to the new
# convention before queueing the remaining 3 sweeps.
#
# Usage:
#   MODEL=Qwen/Qwen3.5-4B bash eval/run_all_sweeps.sh
#
# Skip a sweep with SKIP_TIR_TRAIN=1 / SKIP_COT_TRAIN=1 / SKIP_TIR_QWEN=1 /
# SKIP_COT_QWEN=1 (any non-empty value).

set -uo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd -P)"
LOGDIR="$ROOT/logs"
mkdir -p "$LOGDIR"

MODEL="${MODEL:-Qwen/Qwen3-4B-Thinking-2507}"
MODEL_SLUG="$(printf %s "$MODEL" | tr '/' '-' | tr -c '[:alnum:]._-' _)"
SUMMARY_FILE="$LOGDIR/all_sweeps_summary.txt"
: > "$SUMMARY_FILE"

# ----- Wait for any currently-running sharded sweep -----------------------------
# pgrep uses ERE by default; older versions of this script used `\|` which on
# pgrep is the literal string "\|" — false negative every time. Use plain `|`
# in -E mode to be explicit, and check both old + new launcher names.
echo "[orchestrator] polling for any running sharded launcher (~30s ticks)..."
while pgrep -af 'run_pool_v2_sharded' >/dev/null 2>&1 \
   || pgrep -af 'run_sharded_sweep'   >/dev/null 2>&1; do
    sleep 30
done
echo "[orchestrator] no sharded launcher running — proceeding"

# ----- Rename TIR-train outputs to consistent naming ---------------------------
# The shakeout (OlympiadBench TIR train) and the prior pool_v2 TIR train run
# used ad-hoc tags; align them to <mode>__<model>__train so the matrix is tidy.
_relabel() {
    local from="$1"
    local to="$2"
    if [ -d "$from" ] && [ ! -e "$to" ]; then
        echo "[orchestrator] rename $(basename "$from") → $(basename "$to")"
        mv "$from" "$to"
    fi
}
TIR_TRAIN_TAG="train"
_relabel "$ROOT/outputs/eval/olympiad_oe_to_physics/tir__${MODEL_SLUG}__shakeout" \
         "$ROOT/outputs/eval/olympiad_oe_to_physics/tir__${MODEL_SLUG}__${TIR_TRAIN_TAG}"
for SRC in scibench physics ugphysics drsci; do
    _relabel "$ROOT/outputs/eval/pool_v2_${SRC}/tir__${MODEL_SLUG}___zeroshot_sharded" \
             "$ROOT/outputs/eval/pool_v2_${SRC}/tir__${MODEL_SLUG}__${TIR_TRAIN_TAG}"
    # Also handle the older non-sharded short-lived run if present
    _relabel "$ROOT/outputs/eval/pool_v2_${SRC}/tir__${MODEL_SLUG}___zeroshot" \
             "$ROOT/outputs/eval/pool_v2_${SRC}/tir__${MODEL_SLUG}__${TIR_TRAIN_TAG}_pre_sharded_attempt"
done

# ----- Sweep runner -------------------------------------------------------------
_run_sweep() {
    local mode="$1"
    local preset="$2"
    local tag="$preset"
    local logfile="$LOGDIR/sweep_${mode}_${preset}.log"
    echo "[orchestrator] === sweep: mode=$mode preset=$preset tag=$tag ==="
    echo "[orchestrator]     log → $logfile"
    MODEL="$MODEL" MODE="$mode" PRESET="$preset" TAG="$tag" \
        bash "$ROOT/eval/run_sharded_sweep.sh" > "$logfile" 2>&1
    local rc=$?
    if [ $rc -eq 0 ]; then
        echo "[orchestrator] sweep $mode/$preset OK"
    else
        echo "[orchestrator] sweep $mode/$preset FAILED (exit $rc) — see $logfile"
    fi
    return $rc
}

# ----- Run the 3 remaining sweeps (TIR-train assumed already done) -------------
[ -z "${SKIP_TIR_TRAIN:-}" ] && [ ! -f "$ROOT/outputs/eval/pool_v2_drsci/tir__${MODEL_SLUG}__${TIR_TRAIN_TAG}/scored.summary.txt" ] \
    && _run_sweep tir train

[ -z "${SKIP_COT_TRAIN:-}" ] && _run_sweep cot train
[ -z "${SKIP_TIR_QWEN:-}" ]  && _run_sweep tir qwen
[ -z "${SKIP_COT_QWEN:-}" ]  && _run_sweep cot qwen

# ----- Final summary table -----------------------------------------------------
{
    echo "=== ALL SWEEPS SUMMARY (model=$MODEL) ==="
    echo ""
    for MODE in tir cot; do
        for PRESET in train qwen; do
            TAG="$PRESET"
            echo "--- mode=$MODE preset=$PRESET tag=$TAG ---"
            for BENCH in pool_v2_scibench pool_v2_physics pool_v2_ugphysics olympiad_oe_to_physics pool_v2_drsci; do
                S="$ROOT/outputs/eval/${BENCH}/${MODE}__${MODEL_SLUG}__${TAG}/scored.summary.txt"
                if [ -f "$S" ]; then
                    pass=$(grep '^pass@1' "$S" | head -1)
                    n=$(grep '^n ' "$S" | head -1 | awk '{print $NF}')
                    echo "  $BENCH ($n rows)  $pass"
                else
                    echo "  $BENCH  MISSING ($S)"
                fi
            done
            echo ""
        done
    done
} | tee "$SUMMARY_FILE"

echo ""
echo "[orchestrator] full summary written to $SUMMARY_FILE"
