#!/usr/bin/env bash
# run_saturation_chunked.sh — chunked variant of the saturation pipeline for
# base-model temperature experiments. Splits the 64-prompt sample across
# N_CHUNKS GPUs per combo, runs in parallel, merges the rollout parquets,
# scores once, analyzes once.
#
# Usage (from inside an interactive alloc, project root):
#   TEMP=1.4 bash scripts/run_saturation_chunked.sh
#
# Env overrides:
#   TEMP         default 1.4 (the temperature to test)
#   MODES        default "tir cot" (which modes to run — space-separated)
#   N_PROMPTS    default 64 (MUST match existing ckpt_100 runs for apples-to-apples)
#   N_ROLLOUTS   default 8
#   N_CHUNKS     default 2 (chunks per combo; 2 combos × N_CHUNKS GPUs total)
#   GPU_OFFSET   default 0 (if combined with another job on the same node)
#
# Outputs:
#   outputs/saturation_analysis/{tir,cot}_ckpt0_temp{TEMP}/saturation.json
#   logs/saturation_{tir,cot}_ckpt0_temp{TEMP}_chunk{0,1}.log

set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
mkdir -p logs outputs/saturation_analysis

TEMP="${TEMP:-1.4}"
MODES="${MODES:-tir cot}"
N_PROMPTS="${N_PROMPTS:-64}"
N_ROLLOUTS="${N_ROLLOUTS:-8}"
N_CHUNKS="${N_CHUNKS:-2}"
GPU_OFFSET="${GPU_OFFSET:-0}"
SEED="${SEED:-42}"
BASE_MODEL="${BASE_MODEL:-Qwen/Qwen3-4B-Thinking-2507}"

TEMP_TAG="${TEMP/./p}"  # "1.4" -> "1p4"

# Resolve xVerify URL (same server used by the main interactive run)
URL_FILE="${SHARED_URL_FILE:-$ROOT/outputs/xverify_endpoints/current.url}"
[[ -f "$URL_FILE" ]] || { echo "ERROR: no xverify URL at $URL_FILE" >&2; exit 1; }
XVERIFY_URL=$(head -n1 "$URL_FILE")

_SAVED_CVD="${CUDA_VISIBLE_DEVICES:-}"
unset SIF OVERLAY
source "$ROOT/env.sh"
[[ -n "$_SAVED_CVD" ]] && export CUDA_VISIBLE_DEVICES="$_SAVED_CVD"

APT_RUN_CPU=(
    apptainer exec --overlay "$OVERLAY:ro" --no-home --bind /etc/pki:/etc/pki
    --env "PYTHONNOUSERSITE=1" --env "PYTHONPATH=$ROOT/.async-extras:/opt/phys-extras/"
    --env "HF_HOME=$HF_HOME" --env "XDG_CACHE_HOME=/tmp/.cache" --env "HOME=/tmp"
    --env "TRITON_CACHE_DIR=/tmp/.cache/triton" --env "MPLCONFIGDIR=/tmp/.cache/matplotlib"
    "$SIF"
)

echo "=== run_saturation_chunked.sh ==="
echo "  TEMP          : $TEMP (tag=$TEMP_TAG)"
echo "  MODES         : $MODES"
echo "  N_PROMPTS     : $N_PROMPTS"
echo "  N_CHUNKS      : $N_CHUNKS per combo (GPUs per combo)"
echo "  BASE_MODEL    : $BASE_MODEL"
echo "  XVERIFY_URL   : $XVERIFY_URL"
echo "  GPU_OFFSET    : $GPU_OFFSET"

# ---- Step 1: sample prompts once per mode (seed-stable, matches the earlier runs) ----
for mode in $MODES; do
    TRAIN_PARQUET="$ROOT/data/processed_${mode}/data/train.parquet"
    tag="${mode}_ckpt0_temp${TEMP_TAG}"
    outdir="$ROOT/outputs/saturation_analysis/$tag"
    mkdir -p "$outdir"
    SAMPLE_PARQUET="$outdir/sample_prompts.parquet"
    if [[ -f "$SAMPLE_PARQUET" ]]; then
        echo "[sample $mode] reusing existing sample_prompts.parquet"
    else
        "${APT_RUN_CPU[@]}" python3 - <<EOF
import pandas as pd
df = pd.read_parquet("$TRAIN_PARQUET")
sample = df.sample(n=min($N_PROMPTS, len(df)), random_state=$SEED).reset_index(drop=True)
sample.to_parquet("$SAMPLE_PARQUET", index=False)
print(f"[sample $mode] wrote {len(sample)} prompts -> $SAMPLE_PARQUET")
EOF
    fi
done

# ---- Step 2: fire all chunks in parallel (one per GPU) ----
mode_list=($MODES)
CHUNK_LEN=$(( N_PROMPTS / N_CHUNKS ))
GPU=$GPU_OFFSET
declare -A CHUNK_PIDS

for mode in "${mode_list[@]}"; do
    tag="${mode}_ckpt0_temp${TEMP_TAG}"
    outdir="$ROOT/outputs/saturation_analysis/$tag"
    SAMPLE_PARQUET="$outdir/sample_prompts.parquet"
    for ((c=0; c<N_CHUNKS; c++)); do
        start=$(( c * CHUNK_LEN ))
        end=$(( start + CHUNK_LEN ))
        chunk_dir="$outdir/rollouts/chunk_$c"
        logf="logs/saturation_${tag}_chunk${c}.log"
        echo "[$(date +%H:%M:%S)] GPU $GPU: $mode chunk $c [$start,$end) -> $logf"
        mkdir -p "$chunk_dir"
        (
            CUDA_VISIBLE_DEVICES="$GPU" \
            MODE="$mode" \
            MODEL="$BASE_MODEL" \
            PARQUET="$SAMPLE_PARQUET" \
            N="$N_PROMPTS" \
            N_ROLLOUTS="$N_ROLLOUTS" \
            TEMPERATURE="$TEMP" \
            START_IDX="$start" \
            END_IDX="$end" \
            THINKING_BUDGET=12288 TOOL_CALL_BUDGET=2048 ANSWER_BUDGET=4096 \
            MAX_PROMPT_LEN=1024 MAX_TOOL_RESPONSE_LEN=1024 \
            GPU_MEM=0.80 \
            OUT_DIR="$chunk_dir" \
            bash "$ROOT/eval/inference/rollout.sh"
        ) > "$logf" 2>&1 &
        CHUNK_PIDS["${tag}_c${c}"]=$!
        GPU=$(( GPU + 1 ))
    done
done

# ---- Step 3: wait for all chunks ----
for key in "${!CHUNK_PIDS[@]}"; do
    pid="${CHUNK_PIDS[$key]}"
    echo "[$(date +%H:%M:%S)] waiting on $key (pid $pid) ..."
    wait "$pid" 2>/dev/null || true
    echo "[$(date +%H:%M:%S)] $key done"
done

# ---- Step 4: for each mode, merge chunk parquets, then score and analyze ----
for mode in "${mode_list[@]}"; do
    tag="${mode}_ckpt0_temp${TEMP_TAG}"
    outdir="$ROOT/outputs/saturation_analysis/$tag"
    rollouts_dir="$outdir/rollouts"
    merged_parquet="$rollouts_dir/rollouts.parquet"

    # Merge chunk parquets. Each chunk_dir has its own rollouts.parquet from
    # rollout.sh; concat them into one flat rollouts.parquet that scoring and
    # analysis expect. Problem_idx offsets are already correct because the
    # same sample_prompts.parquet is passed to every chunk, and rollout.py
    # uses START_IDX as the absolute row index.
    "${APT_RUN_CPU[@]}" python3 - <<EOF
import pandas as pd, glob, os
paths = sorted(glob.glob("$rollouts_dir/chunk_*/rollouts.parquet"))
dfs = [pd.read_parquet(p) for p in paths]
merged = pd.concat(dfs, ignore_index=True)
merged.to_parquet("$merged_parquet", index=False)
print(f"[merge $tag] concatenated {len(dfs)} chunks -> {len(merged)} rollouts")
EOF

    # Score + analyze in parallel (both CPU/HTTP-bound, no GPU contention)
    (
        echo "[score $tag] via HTTP xVerify"
        "${APT_RUN_CPU[@]}" python3 "$ROOT/eval/scoring/our_verifier.py" \
            --rollouts "$merged_parquet" \
            --out "$outdir/scored.parquet" \
            --xverify_url "$XVERIFY_URL"
        echo "[analyze $tag]"
        "${APT_RUN_CPU[@]}" python3 "$ROOT/scripts/analyze_saturation.py" \
            --scored "$outdir/scored.parquet" \
            --n_rollouts "$N_ROLLOUTS" \
            --out "$outdir/saturation.json"
    ) > "logs/saturation_${tag}_score.log" 2>&1 &
done
wait

# ---- Step 5: summary ----
echo ""
echo "============================================================"
echo "SUMMARY — ckpt_0 temp=$TEMP (chunked, N_CHUNKS=$N_CHUNKS)"
echo "============================================================"
printf "%-30s %8s %8s %8s %8s %8s %8s\n" "tag" "n_grp" "mean_r" "sat_f" "allR" "allW" "info"
for mode in "${mode_list[@]}"; do
    tag="${mode}_ckpt0_temp${TEMP_TAG}"
    js="outputs/saturation_analysis/$tag/saturation.json"
    if [[ -f "$js" ]]; then
        python3 -c "
import json
d = json.load(open('$js'))
print(f'{\"$tag\":<30s} {d[\"n_groups\"]:8d} {d[\"mean_reward\"]:8.3f} {d[\"saturated_frac\"]:8.3f} {d[\"all_right_frac\"]:8.3f} {d[\"all_wrong_frac\"]:8.3f} {d[\"informative_frac\"]:8.3f}'
)"
    else
        echo "$tag: not finished (missing $js)"
    fi
done
