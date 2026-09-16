#!/usr/bin/env bash
# run_saturation_all.sh — fan out saturation diagnostics across the 4 GPUs
# on a single interactive node. Fires TIR+CoT for step_100 immediately, then
# TIR+CoT for step_150 once those checkpoints finish saving.
#
# Usage (inside a `salloc --nodes 1 --qos interactive --time 04:00:00 -C gpu
# -A m2651` allocation, from the project root):
#
#   bash scripts/run_saturation_all.sh
#
# Outputs land in outputs/saturation_analysis/<tag>/ and log files land in
# logs/saturation_<tag>.log. Writes a final summary table at the end once all
# four jobs have finished.

set -u  # NOT -e: we want to keep going even if one job fails

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
mkdir -p logs

TIR_RUN="outputs/physcode_tir/grpo_tir_4t_sxv_20260421.083031"
COT_RUN="outputs/physcode_tir/grpo_cot_4t_sxv_20260421.083030"

run_one() {
    local gpu="$1" mode="$2" run_dir="$3" step="$4"
    local ckpt="$ROOT/$run_dir/global_step_${step}/actor"
    local tag="${mode}_step${step}"
    local logf="logs/saturation_${tag}.log"
    local outdir="outputs/saturation_analysis/${tag}"

    echo "[$(date +%H:%M:%S)] [GPU $gpu] launching $tag → $logf"
    CUDA_VISIBLE_DEVICES="$gpu" \
    CKPT_DIR="$ckpt" \
    MODE="$mode" \
    OUT_DIR="$outdir" \
    bash "$ROOT/scripts/run_saturation.sh" > "$logf" 2>&1 &
    echo $! > "logs/saturation_${tag}.pid"
}

wait_for_ckpt() {
    local run_dir="$1" step="$2" label="$3"
    local ckpt="$ROOT/$run_dir/global_step_${step}/actor"
    local deadline=$(( $(date +%s) + 7200 ))  # 2h
    # Check that all 16 FSDP shards are present before declaring ready.
    while (( $(date +%s) < deadline )); do
        if [[ -d "$ckpt" ]]; then
            local n_shards
            n_shards=$(ls "$ckpt"/model_world_size_16_rank_*.pt 2>/dev/null | wc -l)
            if (( n_shards == 16 )); then
                echo "[$(date +%H:%M:%S)] [$label] step_${step} ready ($n_shards shards)"
                return 0
            fi
        fi
        sleep 30
    done
    echo "[$(date +%H:%M:%S)] [$label] TIMEOUT waiting for step_${step}" >&2
    return 1
}

summarize() {
    echo ""
    echo "============================================================"
    echo "SUMMARY"
    echo "============================================================"
    for tag in tir_step50 cot_step50 tir_step100 cot_step100 tir_step150 cot_step150; do
        local js="outputs/saturation_analysis/${tag}/saturation.json"
        if [[ -f "$js" ]]; then
            python3 - <<PY
import json
with open("$js") as f:
    d = json.load(f)
print(f"{'$tag':14s}  n_groups={d['n_groups']:4d}  "
      f"mean_reward={d['mean_reward']:.3f}  "
      f"saturated={d['saturated_frac']:.3f}  "
      f"allR={d['all_right_frac']:.3f}  allW={d['all_wrong_frac']:.3f}  "
      f"informative={d['informative_frac']:.3f}")
PY
        else
            echo "$tag: not finished (missing $js)"
        fi
    done
}

# ---- fire step_50 and step_100 immediately (all 4 ckpts on disk) ----
run_one 0 tir "$TIR_RUN" 50
run_one 1 cot "$COT_RUN" 50
run_one 2 tir "$TIR_RUN" 100
run_one 3 cot "$COT_RUN" 100

# ---- wait for step_150 ckpts, then reuse GPUs 0 and 1 as they free up ----
# step_50 jobs finish around the same time TIR-150 ckpt lands (~20 min), so
# GPU 0 is a natural home for TIR-150. CoT-150 waits longer (~60 min) but
# GPU 1's step_50 job is long done by then — plenty of headroom.
# bash builtin `wait` only operates on children of the CURRENT shell, so a
# subshell can't wait on a PID it didn't fork. Poll via kill -0 instead so
# the GPU-freed barrier actually fires before we launch step_150.
gpu_free_wait() {
    local pid="$1"
    while kill -0 "$pid" 2>/dev/null; do sleep 15; done
}

(
    wait_for_ckpt "$TIR_RUN" 150 "TIR"
    if [[ -f logs/saturation_tir_step50.pid ]]; then
        gpu_free_wait "$(cat logs/saturation_tir_step50.pid)"
    fi
    run_one 0 tir "$TIR_RUN" 150
) &
WATCH_TIR_PID=$!
(
    wait_for_ckpt "$COT_RUN" 150 "CoT"
    if [[ -f logs/saturation_cot_step50.pid ]]; then
        gpu_free_wait "$(cat logs/saturation_cot_step50.pid)"
    fi
    run_one 1 cot "$COT_RUN" 150
) &
WATCH_COT_PID=$!

# Wait on watchdogs (they exit after launching or timing out)
wait "$WATCH_TIR_PID" "$WATCH_COT_PID"

# Now wait on all launched saturation jobs
for tag in tir_step50 cot_step50 tir_step100 cot_step100 tir_step150 cot_step150; do
    pidf="logs/saturation_${tag}.pid"
    if [[ -f "$pidf" ]]; then
        pid=$(cat "$pidf")
        echo "[$(date +%H:%M:%S)] waiting on $tag (pid $pid) ..."
        wait "$pid" 2>/dev/null || true
        echo "[$(date +%H:%M:%S)] $tag done"
    fi
done

summarize
