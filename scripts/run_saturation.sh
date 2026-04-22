#!/usr/bin/env bash
# run_saturation.sh — one-shot GRPO saturation diagnostic for a training
# checkpoint. Does: merge FSDP → rollout → score → analyze.
#
# Needs a running xVerify server (outputs/xverify_endpoints/current.url);
# reuses it via HTTP so we don't have to load xVerify locally alongside
# Qwen. Bit-identical to training reward (same router.py code path).
#
# Inputs (env):
#   CKPT_DIR   (required)  /path/to/.../global_step_N/actor
#   MODE       tir|cot     default=tir — determines rollout mode + data path
#   N_PROMPTS  default=64  number of training prompts to sample
#   N_ROLLOUTS default=8   match training rollout_n
#   OUT_DIR    default=outputs/saturation_analysis/<tag>
#   XVERIFY_URL  default=<read from outputs/xverify_endpoints/current.url>
#
# Budgets match training (THINKING=12288, TOOL=2048, ANSWER=4096 → MAX=19472).
# Model is merged once to a scratch dir under $OUT_DIR/merged_hf/.

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

CKPT_DIR="${CKPT_DIR:?CKPT_DIR is required — path to .../global_step_N/actor}"
MODE="${MODE:-tir}"
N_PROMPTS="${N_PROMPTS:-64}"
N_ROLLOUTS="${N_ROLLOUTS:-8}"
SEED="${SEED:-42}"

# Derive tag from the checkpoint path
CKPT_STEP=$(basename "$(dirname "$CKPT_DIR")")        # e.g. global_step_100
CKPT_RUN=$(basename "$(dirname "$(dirname "$CKPT_DIR")")")  # e.g. grpo_tir_4t_...
TAG="${MODE}_${CKPT_RUN}_${CKPT_STEP}"
OUT_DIR="${OUT_DIR:-$ROOT/outputs/saturation_analysis/$TAG}"
mkdir -p "$OUT_DIR"

# Resolve xVerify URL
if [[ -z "${XVERIFY_URL:-}" ]]; then
    URL_FILE="$ROOT/outputs/xverify_endpoints/current.url"
    [[ -f "$URL_FILE" ]] || { echo "ERROR: no XVERIFY_URL and $URL_FILE missing" >&2; exit 1; }
    XVERIFY_URL=$(head -n1 "$URL_FILE")
fi

# Data path by mode (both parquets have identical rows, only prompt differs)
if [[ "$MODE" == "tir" ]]; then
    TRAIN_PARQUET="$ROOT/data/processed_tir/data/train.parquet"
else
    TRAIN_PARQUET="$ROOT/data/processed_cot/data/train.parquet"
fi

echo "=== run_saturation.sh ==="
echo "  CKPT_DIR       : $CKPT_DIR"
echo "  MODE           : $MODE"
echo "  N_PROMPTS      : $N_PROMPTS"
echo "  N_ROLLOUTS     : $N_ROLLOUTS"
echo "  TRAIN_PARQUET  : $TRAIN_PARQUET"
echo "  XVERIFY_URL    : $XVERIFY_URL"
echo "  OUT_DIR        : $OUT_DIR"

_SAVED_CVD="${CUDA_VISIBLE_DEVICES:-}"
unset SIF OVERLAY
source "$ROOT/env.sh"
[[ -n "$_SAVED_CVD" ]] && export CUDA_VISIBLE_DEVICES="$_SAVED_CVD"

APT_RUN_CPU=(
    apptainer exec
    --overlay "$OVERLAY:ro" --no-home --bind /etc/pki:/etc/pki
    --env "PYTHONNOUSERSITE=1"
    --env "PYTHONPATH=$ROOT/.async-extras:/opt/phys-extras/"
    --env "HF_HOME=$HF_HOME"
    --env "XDG_CACHE_HOME=/tmp/.cache" --env "HOME=/tmp"
    --env "TRITON_CACHE_DIR=/tmp/.cache/triton"
    --env "MPLCONFIGDIR=/tmp/.cache/matplotlib"
    "$SIF"
)
APT_RUN_GPU=(
    apptainer exec --nv
    --overlay "$OVERLAY:ro" --no-home --bind /etc/pki:/etc/pki
    --env "PYTHONNOUSERSITE=1"
    --env "PYTHONPATH=$ROOT/.async-extras:/opt/phys-extras/"
    --env "HF_HOME=$HF_HOME"
    --env "XDG_CACHE_HOME=/tmp/.cache" --env "HOME=/tmp"
    --env "TRITON_CACHE_DIR=/tmp/.cache/triton"
    --env "FLASHINFER_WORKSPACE_BASE=/tmp"
    --env "MPLCONFIGDIR=/tmp/.cache/matplotlib"
    "$SIF"
)

# ---- STEP 1: merge FSDP shards → HF safetensors (or skip for base-model eval) ----
# When CKPT_DIR doesn't look like an FSDP actor directory (i.e. no
# model_world_size_*.pt shards inside), treat it as an HF-model path or name
# and skip the merge. Enables temperature-sweep experiments on ckpt_0.
if compgen -G "$CKPT_DIR/model_world_size_*.pt" > /dev/null; then
    MERGED_DIR="$OUT_DIR/merged_hf"
    if [[ -f "$MERGED_DIR/config.json" ]] && ls "$MERGED_DIR"/*.safetensors &>/dev/null; then
        echo "[merge] already done → $MERGED_DIR"
    else
        echo "[merge] FSDP → HF ..."
        "${APT_RUN_CPU[@]}" python3 -m verl.model_merger merge \
            --backend fsdp \
            --local_dir "$CKPT_DIR" \
            --target_dir "$MERGED_DIR"
    fi
else
    MERGED_DIR="$CKPT_DIR"
    echo "[merge] skipped — CKPT_DIR is an HF model path/name, no FSDP shards"
fi

# ---- STEP 2: sample prompts from training parquet ----
SAMPLE_PARQUET="$OUT_DIR/sample_prompts.parquet"
"${APT_RUN_CPU[@]}" python3 - <<EOF
import pandas as pd
df = pd.read_parquet("$TRAIN_PARQUET")
sample = df.sample(n=min($N_PROMPTS, len(df)), random_state=$SEED).reset_index(drop=True)
sample.to_parquet("$SAMPLE_PARQUET", index=False)
print(f"[sample] wrote {len(sample)} prompts → $SAMPLE_PARQUET")
EOF

# ---- STEP 3: rollout via vLLM ----
ROLLOUTS_DIR="$OUT_DIR/rollouts"
if [[ -f "$ROLLOUTS_DIR/rollouts.parquet" ]]; then
    echo "[rollout] already done → $ROLLOUTS_DIR/rollouts.parquet"
else
    echo "[rollout] generating $N_PROMPTS x $N_ROLLOUTS rollouts (temp=${TEMPERATURE:-1.0})..."
    MODE="$MODE" \
    MODEL="$MERGED_DIR" \
    PARQUET="$SAMPLE_PARQUET" \
    N="$N_PROMPTS" \
    N_ROLLOUTS="$N_ROLLOUTS" \
    TEMPERATURE="${TEMPERATURE:-1.0}" \
    THINKING_BUDGET=12288 TOOL_CALL_BUDGET=2048 ANSWER_BUDGET=4096 \
    MAX_PROMPT_LEN=1024 MAX_TOOL_RESPONSE_LEN=1024 \
    GPU_MEM=0.80 \
    OUT_DIR="$ROLLOUTS_DIR" \
    bash "$ROOT/eval/inference/rollout.sh"
fi

ROLLOUTS_PARQUET="$ROLLOUTS_DIR/rollouts.parquet"
[[ -f "$ROLLOUTS_PARQUET" ]] || { echo "ERROR: rollout did not produce $ROLLOUTS_PARQUET" >&2; exit 1; }

# ---- STEP 4: score with HTTP xVerify ----
SCORED_PARQUET="$OUT_DIR/scored.parquet"
if [[ -f "$SCORED_PARQUET" ]]; then
    echo "[score] already done → $SCORED_PARQUET"
else
    echo "[score] scoring via HTTP xVerify ..."
    "${APT_RUN_CPU[@]}" python3 "$ROOT/eval/scoring/our_verifier.py" \
        --rollouts "$ROLLOUTS_PARQUET" \
        --out "$SCORED_PARQUET" \
        --xverify_url "$XVERIFY_URL"
fi

# ---- STEP 5: saturation analysis ----
"${APT_RUN_CPU[@]}" python3 "$ROOT/scripts/analyze_saturation.py" \
    --scored "$SCORED_PARQUET" \
    --n_rollouts "$N_ROLLOUTS" \
    --out "$OUT_DIR/saturation.json"

echo ""
echo "=== done: $OUT_DIR ==="
