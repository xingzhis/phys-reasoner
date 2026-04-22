#!/usr/bin/env bash
# run_saturation_qwen3_4b.sh — zero-shot TIR test on Qwen/Qwen3-4B (base, NOT
# the -Thinking-2507 variant). 64-prompt sample (same seed=42 → same prompts
# as the thinking-model runs), 8 rollouts, temp in {1.0, 1.4}.
#
# Layout: 4 GPUs, 2 temps in parallel, 2 chunks per temp.
#   GPU 0 : TIR temp=1.0, prompts 0-31
#   GPU 1 : TIR temp=1.0, prompts 32-63
#   GPU 2 : TIR temp=1.4, prompts 0-31
#   GPU 3 : TIR temp=1.4, prompts 32-63
#
# Writes to: outputs/saturation_analysis/tir_qwen3_4b_temp{1p0,1p4}/
# (separate tag from the -Thinking-2507 runs; no collision).
#
# Usage (inside interactive alloc, 4 GPUs, from project root):
#   bash scripts/run_saturation_qwen3_4b.sh

set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
mkdir -p logs outputs/saturation_analysis

BASE_MODEL="Qwen/Qwen3-4B"
N_PROMPTS="${N_PROMPTS:-64}"
N_ROLLOUTS="${N_ROLLOUTS:-8}"
SEED="${SEED:-42}"
TEMPS=(1.0 1.4)
CHUNKS_PER_TEMP=2
CHUNK_LEN=$(( N_PROMPTS / CHUNKS_PER_TEMP ))
TRAIN_PARQUET="$ROOT/data/processed_tir/data/train.parquet"

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

echo "=== run_saturation_qwen3_4b.sh ==="
echo "  BASE_MODEL    : $BASE_MODEL"
echo "  TEMPS         : ${TEMPS[*]}"
echo "  N_PROMPTS     : $N_PROMPTS"
echo "  CHUNKS/TEMP   : $CHUNKS_PER_TEMP (len=$CHUNK_LEN)"
echo "  XVERIFY_URL   : $XVERIFY_URL"
echo "  TRAIN_PARQUET : $TRAIN_PARQUET"

# ---- Step 1: sample once per temp (each gets its own outdir) ----
for temp in "${TEMPS[@]}"; do
    temp_tag="${temp/./p}"
    tag="tir_qwen3_4b_temp${temp_tag}"
    outdir="$ROOT/outputs/saturation_analysis/$tag"
    mkdir -p "$outdir"
    SAMPLE_PARQUET="$outdir/sample_prompts.parquet"
    if [[ -f "$SAMPLE_PARQUET" ]]; then
        echo "[sample $tag] reusing existing"
    else
        "${APT_RUN_CPU[@]}" python3 - <<EOF
import pandas as pd
df = pd.read_parquet("$TRAIN_PARQUET")
sample = df.sample(n=min($N_PROMPTS, len(df)), random_state=$SEED).reset_index(drop=True)
sample.to_parquet("$SAMPLE_PARQUET", index=False)
print(f"[sample $tag] wrote {len(sample)} prompts")
EOF
    fi
done

# ---- Step 2: fire 4 chunks (2 temps × 2 chunks) on 4 GPUs ----
declare -A CHUNK_PIDS
GPU=0
for temp in "${TEMPS[@]}"; do
    temp_tag="${temp/./p}"
    tag="tir_qwen3_4b_temp${temp_tag}"
    outdir="$ROOT/outputs/saturation_analysis/$tag"
    SAMPLE_PARQUET="$outdir/sample_prompts.parquet"
    for ((c=0; c<CHUNKS_PER_TEMP; c++)); do
        start=$(( c * CHUNK_LEN ))
        end=$(( start + CHUNK_LEN ))
        chunk_dir="$outdir/rollouts/chunk_$c"
        logf="logs/saturation_${tag}_chunk${c}.log"
        mkdir -p "$chunk_dir"
        echo "[$(date +%H:%M:%S)] GPU $GPU: TIR temp=$temp chunk $c [$start,$end) -> $logf"
        (
            CUDA_VISIBLE_DEVICES="$GPU" \
            MODE=tir \
            MODEL="$BASE_MODEL" \
            PARQUET="$SAMPLE_PARQUET" \
            N="$N_PROMPTS" \
            N_ROLLOUTS="$N_ROLLOUTS" \
            TEMPERATURE="$temp" \
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

# ---- Step 4: for each temp, merge chunks + score + analyze ----
for temp in "${TEMPS[@]}"; do
    temp_tag="${temp/./p}"
    tag="tir_qwen3_4b_temp${temp_tag}"
    outdir="$ROOT/outputs/saturation_analysis/$tag"
    rollouts_dir="$outdir/rollouts"
    merged_parquet="$rollouts_dir/rollouts.parquet"

    "${APT_RUN_CPU[@]}" python3 - <<EOF
import pandas as pd, glob, os
paths = sorted(glob.glob("$rollouts_dir/chunk_*/rollouts.parquet"))
dfs = [pd.read_parquet(p) for p in paths]
merged = pd.concat(dfs, ignore_index=True)
merged.to_parquet("$merged_parquet", index=False)
print(f"[merge $tag] {len(dfs)} chunks -> {len(merged)} rollouts")
EOF

    (
        echo "[score $tag]"
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
echo "SUMMARY — Qwen/Qwen3-4B (non-thinking base) TIR zero-shot"
echo "============================================================"
printf "%-30s %8s %8s %8s %8s %8s %8s\n" "tag" "n_grp" "mean_r" "sat_f" "allR" "allW" "info"
for temp in "${TEMPS[@]}"; do
    temp_tag="${temp/./p}"
    tag="tir_qwen3_4b_temp${temp_tag}"
    js="outputs/saturation_analysis/$tag/saturation.json"
    if [[ -f "$js" ]]; then
        python3 -c "
import json
d = json.load(open('$js'))
print(f'{\"$tag\":<30s} {d[\"n_groups\"]:8d} {d[\"mean_reward\"]:8.3f} {d[\"saturated_frac\"]:8.3f} {d[\"all_right_frac\"]:8.3f} {d[\"all_wrong_frac\"]:8.3f} {d[\"informative_frac\"]:8.3f}'
)"
    else
        echo "$tag: not finished"
    fi
done
echo ""
echo "Reference (Qwen3-4B-Thinking-2507, ckpt_0):"
echo "  temp=1.0 sat=0.609 mean=0.512 info=0.391"
echo "  temp=1.4 sat=0.422 mean=0.342 info=0.578"
