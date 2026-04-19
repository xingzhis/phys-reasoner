#!/usr/bin/env bash
# check_smoke.sh — summarize the most recent Perlmutter smoke run.
#
# Reads the newest outputs/physcode_tir_smoke/*/train.log, extracts per-step
# timing lines emitted by the FullyAsyncTrainer print (verl d822b76a), and
# prints a one-screen summary:
#   - each step's timing_s/* breakdown
#   - median step wall over steps 2+
#   - dominant trainer phase
#   - pass/warn/fail verdict vs SMOKE_HANDOFF §5.2 pass criteria
#
# Usage:
#   bash scripts/perlmutter/check_smoke.sh             # most recent run
#   bash scripts/perlmutter/check_smoke.sh path/to/train.log   # specific log
#
# Does NOT submit, cancel, or modify anything. Read-only.

set -uo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

if [[ $# -ge 1 ]]; then
    LOG="$1"
else
    LOG="$(ls -t outputs/physcode_tir_smoke/*/train.log 2>/dev/null | head -1)"
fi

if [[ -z "${LOG:-}" || ! -f "$LOG" ]]; then
    echo "ERROR: no train.log found (tried: outputs/physcode_tir_smoke/*/train.log)" >&2
    echo "Pass an explicit path as \$1 if the log lives elsewhere." >&2
    exit 1
fi

echo "=== check_smoke: $LOG ==="
echo

echo "--- FullyAsyncTrainer timing lines (one per step) ---"
grep "FullyAsyncTrainer\] step=" "$LOG" || {
    echo "(none found — either job did not reach any training step, or it ran on a verl commit without the timing print)"
    echo
    echo "--- last 5 stderr-ish lines ---"
    grep -E "Traceback|Error|OOM|OutOfMemory" "$LOG" | grep -v "vLLMHttpServer pid\|Worker pid\|EngineCore" | tail -5
    exit 0
}
echo

echo "--- per-step wall clock (parsed from timing_s/step=…s) ---"
# Extract "timing_s/step=NN.NNs" → one number per step
steps=$(grep "FullyAsyncTrainer\] step=" "$LOG" \
        | sed -nE 's/.*timing_s\/step=([0-9.]+)s.*/\1/p')
if [[ -z "$steps" ]]; then
    echo "(could not parse timing_s/step from any line — printing raw lines above)"
    exit 0
fi
echo "$steps"
echo

N=$(echo "$steps" | wc -l | awk '{print $1}')
echo "step count: $N"

# Median over steps 2+ (skip step-1 warmup)
if [[ "$N" -ge 2 ]]; then
    med=$(echo "$steps" | tail -n +2 | sort -n | awk '
        { a[NR] = $1 }
        END { if (NR % 2) print a[(NR+1)/2]; else print (a[NR/2] + a[NR/2+1]) / 2 }
    ')
    echo "median step wall (steps 2+): ${med}s"

    # Pass verdict vs SMOKE_HANDOFF §5.2
    awk -v m="$med" 'BEGIN {
        if (m <= 300)      { print "VERDICT: PASS (median ≤ 300s — at Phase-0 target)" }
        else if (m <= 600) { print "VERDICT: WARN (median in 300-600s band — ask user before Phase 1)" }
        else if (m <= 1200){ print "VERDICT: WARN-HIGH (median in 600-1200s — Ulysses SP or flash-attn likely not firing; check open-questions.md §2-3)" }
        else               { print "VERDICT: FAIL (median > 1200s — likely baseline-equivalent; stop and ask)" }
    }'
else
    echo "(only one step — report step-1 wall but do not form a verdict yet)"
fi
echo

echo "--- dominant trainer phase on last step ---"
# Extract all timing_s/KEY=VAL pairs from the last step line
last=$(grep "FullyAsyncTrainer\] step=" "$LOG" | tail -1)
echo "$last" | tr ' ' '\n' | grep -E "^timing_s/[^=]+=[0-9.]+s$" \
    | grep -vE "^timing_s/step=|^timing_s/start_profile=|^timing_s/stop_profile=" \
    | sed -E 's/timing_s\/([^=]+)=([0-9.]+)s/\2\t\1/' \
    | sort -nr | head -5 | awk -F'\t' '{printf "  %-28s %7.2fs\n", $2, $1}'
echo

echo "--- zero-advantage group rate (from rollout dump, if present) ---"
DUMP=$(dirname "$LOG")/rollout_dumps
if [[ -d "$DUMP" ]]; then
    LATEST_JSONL=$(ls -t "$DUMP"/*.jsonl 2>/dev/null | head -1)
    if [[ -n "$LATEST_JSONL" ]]; then
        echo "source: $LATEST_JSONL"
        echo "(to compute zero-adv rate: group rollouts by problem_idx, std(score) < 1e-6 → zero-adv)"
        echo "(parse with python; check_smoke.sh keeps this read-only — see SMOKE_HANDOFF §3.1 for the reference analysis)"
    fi
else
    echo "(no rollout_dumps dir — DUMP_TRAIN_ROLLOUTS was off?)"
fi
echo

echo "=== done ==="
