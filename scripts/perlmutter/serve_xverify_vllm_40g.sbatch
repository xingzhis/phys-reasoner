#!/bin/bash
# serve_xverify_vllm_40g.sbatch — long-lived vLLM xVerify on premium A100 40GB.
#
# Standalone xverify server for shared use across multiple training jobs (TIR,
# CoT, ablations). Replaces serve_xverify_shared_40g.sbatch for the new
# vLLM-async backend.
#
# - Backend: vLLM AsyncLLMEngine with TP=4 → ~150-400 req/s vs serial ~5 req/s
# - QoS: premium (fast start) vs shared (queue is unreliable)
# - Walltime: 30:00:00 to outlive a 30h prod job
# - Writes URL atomically to outputs/xverify_endpoints/current.url ONLY after
#   /health reports ready, so any caller cached URL keeps working until ready
#
# Submit:
#   sbatch scripts/perlmutter/serve_xverify_vllm_40g.sbatch

#SBATCH --job-name=xverify_vllm
#SBATCH --output=logs/%x_%j.log
#SBATCH --error=logs/%x_%j.err
#SBATCH -A m2651
#SBATCH -C gpu&hbm40g
#SBATCH -q premium
#SBATCH -t 30:00:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-node=4

set -euo pipefail
ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/../.." && pwd)}"
cd "$ROOT"
mkdir -p logs outputs/xverify_endpoints

unset SIF OVERLAY
source "$ROOT/env.sh"

PORT="${PORT:-8765}"
TP_SIZE="${TP_SIZE:-4}"
HOST=$(hostname)
URL="http://${HOST}:${PORT}/judge"
URL_FILE="${URL_FILE:-$ROOT/outputs/xverify_endpoints/current.url}"

echo "=== serve_xverify_vllm_40g.sbatch ==="
echo "  Job   : $SLURM_JOB_ID"
echo "  Host  : $HOST"
echo "  Port  : $PORT"
echo "  TP    : $TP_SIZE"
echo "  URL   : $URL"
echo "  RDV   : $URL_FILE"

# Atomic rendezvous publish ONLY after the server is healthy. Any caller that
# already cached an older URL keeps using it until ours is up.
write_url_when_ready() {
    local hc="${URL%/judge}/health"
    for _ in $(seq 1 240); do
        sleep 5
        if curl -sf --max-time 5 "$hc" 2>/dev/null | grep -q '"ready":[[:space:]]*true'; then
            local tmp="$URL_FILE.tmp.$$"
            {
                echo "$URL"
                echo "# host=$HOST"
                echo "# port=$PORT"
                echo "# tp_size=$TP_SIZE"
                echo "# backend=vllm-async"
                echo "# job_id=$SLURM_JOB_ID"
                echo "# written_at=$(date -Iseconds)"
            } > "$tmp"
            mv "$tmp" "$URL_FILE"
            echo "[xverify] healthy, wrote $URL_FILE"
            return 0
        fi
    done
    echo "[xverify] health check never passed within 20 minutes" >&2
    return 1
}
write_url_when_ready &
WRITER_PID=$!

# Cleanup: only remove the URL file if it still points at us (don't clobber
# a replacement server that raced in after we wrote).
cleanup() {
    if [[ -f "$URL_FILE" ]] && [[ "$(head -n1 "$URL_FILE" 2>/dev/null)" == "$URL" ]]; then
        rm -f "$URL_FILE"
        echo "[xverify] cleaned $URL_FILE"
    fi
    kill $WRITER_PID 2>/dev/null || true
}
trap cleanup EXIT

export PYTHONNOUSERSITE=1
apptainer exec --nv \
  --overlay "$OVERLAY:ro" --no-home --bind /etc/pki:/etc/pki \
  --env "PYTHONNOUSERSITE=1" \
  --env "PYTHONUNBUFFERED=1" \
  --env "PYTHONPATH=$ROOT/src:/opt/phys-extras/" \
  --env "HF_HOME=$HF_HOME" \
  --env "HF_HUB_OFFLINE=1" --env "TRANSFORMERS_OFFLINE=1" \
  --env "XDG_CACHE_HOME=/tmp/.cache" --env "HOME=/tmp" \
  --env "FLASHINFER_WORKSPACE_BASE=/tmp" \
  --env "TRITON_CACHE_DIR=/tmp/.cache/triton" \
  --env "VLLM_USE_V1=1" \
  "$SIF" \
  python3 "$ROOT/scripts/serve_xverify_vllm.py" \
    --host 0.0.0.0 --port "$PORT" \
    --model IAAR-Shanghai/xVerify-7B-I \
    --tp_size "$TP_SIZE" \
    --gpu_mem 0.85 \
    --max_model_len 4096
