#!/usr/bin/env bash
# Inside a salloc'd 4-GPU node, launch vLLM xverify server, validate vs old
# server, then run the full scoring pipeline.
#
# Required env: SIF, OVERLAY, ROOT, HF_HOME (set via env.sh)
# Optional: TP_SIZE (default 4), VALIDATION_N (default 200), OLD_URL (existing
#           transformers-backed server for correctness comparison).

set -euo pipefail
ROOT="${ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "$ROOT"
source "$ROOT/env.sh"

PORT="${PORT:-8765}"
TP_SIZE="${TP_SIZE:-4}"
HOST=$(hostname)
NEW_URL="http://${HOST}:${PORT}/judge"
NEW_URL_FILE="${NEW_URL_FILE:-$ROOT/outputs/xverify_endpoints/vllm.url}"
OLD_URL="${OLD_URL:-}"
VALIDATION_N="${VALIDATION_N:-200}"
export PYTHONNOUSERSITE=1
APT_BASE=( apptainer exec --nv
    --overlay "$OVERLAY:ro" --no-home --bind /etc/pki:/etc/pki
    --env "PYTHONNOUSERSITE=1" --env "PYTHONUNBUFFERED=1"
    --env "PYTHONPATH=$ROOT/src:/opt/phys-extras/"
    --env "HF_HOME=$HF_HOME"
    --env "HF_HUB_OFFLINE=1" --env "TRANSFORMERS_OFFLINE=1"
    --env "XDG_CACHE_HOME=/tmp/.cache" --env "HOME=/tmp"
    --env "FLASHINFER_WORKSPACE_BASE=/tmp"
    --env "TRITON_CACHE_DIR=/tmp/.cache/triton"
    --env "VLLM_USE_V1=1" )

mkdir -p "$(dirname "$NEW_URL_FILE")" logs

echo "[run] host=$HOST port=$PORT tp=$TP_SIZE"
echo "[run] NEW_URL=$NEW_URL"
echo "[run] OLD_URL=$OLD_URL"
echo "[run] VALIDATION_N=$VALIDATION_N"

# 1) Launch vLLM xverify server in background
echo "[run] === starting vLLM xverify server ==="
"${APT_BASE[@]}" "$SIF" \
    python3 scripts/serve_xverify_vllm.py \
        --host 0.0.0.0 --port "$PORT" \
        --model IAAR-Shanghai/xVerify-7B-I \
        --tp_size "$TP_SIZE" \
        --gpu_mem 0.85 \
        --max_model_len 4096 \
    > "logs/serve_xverify_vllm_${SLURM_JOB_ID:-0}.log" 2>&1 &
SERVER_PID=$!
echo "[run] server pid=$SERVER_PID"

# Write URL rendezvous (so existing scripts/clients can discover it)
{
    echo "$NEW_URL"
    echo "# host=$HOST"
    echo "# port=$PORT"
    echo "# tp_size=$TP_SIZE"
    echo "# backend=vllm-async"
    echo "# job_id=${SLURM_JOB_ID:-N/A}"
    echo "# written_at=$(date -Iseconds)"
} > "$NEW_URL_FILE"
echo "[run] wrote rendezvous: $NEW_URL_FILE"

# Cleanup on exit
trap 'echo "[run] cleaning up"; rm -f "$NEW_URL_FILE"; kill $SERVER_PID 2>/dev/null || true' EXIT

# 2) Wait for server health
echo "[run] === waiting for server health ==="
for i in $(seq 1 60); do
    sleep 5
    if curl -sf "http://${HOST}:${PORT}/health" 2>/dev/null | grep -q '"ready":[[:space:]]*true'; then
        echo "[run] server healthy after $((i*5))s"
        break
    fi
    if ! kill -0 $SERVER_PID 2>/dev/null; then
        echo "[run] server died — see logs/serve_xverify_vllm_${SLURM_JOB_ID:-0}.log"
        exit 1
    fi
    echo "[run] $((i*5))s waiting ..."
done

if ! curl -sf "http://${HOST}:${PORT}/health" 2>/dev/null | grep -q '"ready":[[:space:]]*true'; then
    echo "[run] FAIL: server never became healthy"
    exit 1
fi

# 3) Validation: side-by-side compare with OLD_URL on $VALIDATION_N rows.
if [[ -n "$OLD_URL" ]]; then
    echo "[run] === validation: comparing new vs old server on $VALIDATION_N rows ==="
    "${APT_BASE[@]}" "$SIF" \
        python3 scripts/compare_servers.py \
            --merged outputs/probe_qwen3_4b/rollouts_merged.parquet \
            --new_url "$NEW_URL" \
            --old_url "$OLD_URL" \
            --max_rows "$VALIDATION_N"
else
    echo "[run] skipping validation (OLD_URL not set)"
fi

# 4) Full scoring + filter via the (now-fast) server
echo "[run] === full scoring against vLLM server ==="
# Point build_filtered_parquets at the new URL
"${APT_BASE[@]}" "$SIF" \
    python3 scripts/build_filtered_parquets.py \
        --root "$ROOT" \
        --fleet outputs/probe_qwen3_4b/fleet \
        --xverify_url_file "$NEW_URL_FILE"

echo "[run] === DONE; keeping server alive until salloc ends (Ctrl-C or scancel to stop) ==="
wait $SERVER_PID
