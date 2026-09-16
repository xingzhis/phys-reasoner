#!/usr/bin/env bash
# run_saturation_temp_test.sh — decisive policy-sharpness vs data-saturation test.
#
# Fires 4 parallel saturation jobs on a single interactive node (4 GPUs),
# all using the UNTRAINED base model (ckpt_0 = Qwen3-4B-Thinking-2507):
#   GPU 0 : TIR @ temperature=1.3 (the test)
#   GPU 1 : CoT @ temperature=1.3 (the test)
#   GPU 2 : TIR @ temperature=1.0 (reference)
#   GPU 3 : CoT @ temperature=1.0 (reference)
#
# Same 64-prompt sample (seed=42), same n=8 rollouts, same scoring path as
# scripts/run_saturation_all.sh. No FSDP merge (base model = HF path).
#
# Interpretation:
#   - If ckpt_0 temp=1.3 sat_frac is MUCH lower than our existing
#     ckpt_100 temp=1.0 sat_frac (~58-59%), then training has sharpened the
#     policy and temperature can recover group diversity → policy saturation.
#     Filter-based interventions wouldn't fix it.
#   - If ckpt_0 temp=1.3 sat_frac stays high (~60%+), then the base model
#     already can't find group diversity on these prompts → data saturation,
#     filter-based interventions SHOULD work.
#   - The ckpt_0 temp=1.0 reference tells us what "no training, standard
#     sampling" sat_frac looks like — establishing the training delta and
#     the temperature delta as separable quantities.
#
# Usage (inside a 1-node×4-GPU interactive alloc, from project root):
#   bash scripts/run_saturation_temp_test.sh

set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
mkdir -p logs

BASE_MODEL="Qwen/Qwen3-4B-Thinking-2507"

run_one() {
    local gpu="$1" mode="$2" temp="$3"
    local tag="${mode}_ckpt0_temp${temp/./p}"  # e.g. tir_ckpt0_temp1p3
    local logf="logs/saturation_${tag}.log"
    local outdir="outputs/saturation_analysis/${tag}"

    echo "[$(date +%H:%M:%S)] [GPU $gpu] launching $tag (temp=$temp) → $logf"
    CUDA_VISIBLE_DEVICES="$gpu" \
    CKPT_DIR="$BASE_MODEL" \
    MODE="$mode" \
    TEMPERATURE="$temp" \
    OUT_DIR="$outdir" \
    bash "$ROOT/scripts/run_saturation.sh" > "$logf" 2>&1 &
    echo $! > "logs/saturation_${tag}.pid"
}

summarize() {
    echo ""
    echo "============================================================"
    echo "SUMMARY — ckpt_0 temperature sweep"
    echo "============================================================"
    printf "%-25s %8s %8s %8s %8s %8s %8s\n" "tag" "n_grp" "mean_r" "sat_f" "allR" "allW" "info"
    for tag in tir_ckpt0_temp1p0 cot_ckpt0_temp1p0 tir_ckpt0_temp1p3 cot_ckpt0_temp1p3; do
        local js="outputs/saturation_analysis/${tag}/saturation.json"
        if [[ -f "$js" ]]; then
            python3 -c "
import json
d = json.load(open('$js'))
print(f'{\"$tag\":<25s} {d[\"n_groups\"]:8d} {d[\"mean_reward\"]:8.3f} {d[\"saturated_frac\"]:8.3f} {d[\"all_right_frac\"]:8.3f} {d[\"all_wrong_frac\"]:8.3f} {d[\"informative_frac\"]:8.3f}'
)"
        else
            echo "$tag: not finished"
        fi
    done
    echo ""
    echo "Comparison points (from earlier run):"
    echo "  tir_step100 (temp=1.0) sat_frac=0.578 info=0.422"
    echo "  cot_step100 (temp=1.0) sat_frac=0.594 info=0.406"
}

# Fire all 4 in parallel
run_one 0 tir 1.3
run_one 1 cot 1.3
run_one 2 tir 1.0
run_one 3 cot 1.0

# Wait on all
for tag in tir_ckpt0_temp1p3 cot_ckpt0_temp1p3 tir_ckpt0_temp1p0 cot_ckpt0_temp1p0; do
    pidf="logs/saturation_${tag}.pid"
    if [[ -f "$pidf" ]]; then
        pid=$(cat "$pidf")
        echo "[$(date +%H:%M:%S)] waiting on $tag (pid $pid) ..."
        wait "$pid" 2>/dev/null || true
        echo "[$(date +%H:%M:%S)] $tag done"
    fi
done

summarize
