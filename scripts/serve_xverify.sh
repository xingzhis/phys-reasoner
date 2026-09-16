#!/usr/bin/env bash
# serve_xverify.sh — launch the remote xVerify HTTP service inside apptainer.
#
# Run on a node with one spare GPU (e.g. misha01:4090). The trainer on
# another node sets XVERIFY_URL=http://<this-host>:<port>/judge to route the
# verifier's xVerify fallback to this service.
#
# Usage:
#   bash scripts/serve_xverify.sh                       # defaults: port 8765, xVerify-7B-I
#   PORT=9000 XVERIFY_MODEL=IAAR-Shanghai/xVerify-3B-Ib bash scripts/serve_xverify.sh
#
# Health check from another node:
#   curl http://<host>:8765/health

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
unset SIF OVERLAY
source "$ROOT/env.sh"

PORT="${PORT:-8765}"
HOST_BIND="${HOST_BIND:-0.0.0.0}"
XVERIFY_MODEL="${XVERIFY_MODEL:-IAAR-Shanghai/xVerify-7B-I}"

# Rendezvous: trainer discovers this server by reading URL_FILE. Callers may
# override via env to avoid collisions when multiple prod jobs run concurrently
# (e.g. TIR + CoT parallel). Default stays current.url for back-compat with
# single-job sbatches. See scripts/perlmutter/prod_tir_het_4t.sbatch for the
# per-JOBID pattern.
HOST=$(hostname)
URL="http://${HOST}:${PORT}/judge"
URL_FILE="${URL_FILE:-$ROOT/outputs/xverify_endpoints/current.url}"
mkdir -p "$(dirname "$URL_FILE")"

echo "=== serve_xverify.sh ==="
echo "  bind  : $HOST_BIND:$PORT"
echo "  model : $XVERIFY_MODEL"
echo "  URL   : $URL"
echo "  write : $URL_FILE"
echo "  HF_HOME=$HF_HOME"

TMP="$URL_FILE.tmp.$$"
{
    echo "$URL"
    echo "# host=$HOST"
    echo "# port=$PORT"
    echo "# model=$XVERIFY_MODEL"
    echo "# written_at=$(date -Iseconds)"
    [[ -n "${SLURM_JOB_ID:-}" ]] && echo "# job_id=$SLURM_JOB_ID"
} > "$TMP"
mv "$TMP" "$URL_FILE"

# Only remove the rendezvous file if it still points at us — avoids clobbering
# a replacement server that raced in after we wrote.
cleanup_url() {
    if [[ -f "$URL_FILE" ]] && [[ "$(head -n1 "$URL_FILE" 2>/dev/null)" == "$URL" ]]; then
        rm -f "$URL_FILE"
        echo "[xverify] URL file cleaned"
    fi
}
trap cleanup_url EXIT

PYTHONNOUSERSITE=1 apptainer exec --nv \
  --overlay "$OVERLAY:ro" --no-home \
  --bind /etc/pki:/etc/pki \
  --env "PYTHONNOUSERSITE=1" \
  --env "PYTHONUNBUFFERED=1" \
  --env "PYTHONPATH=$ROOT/src:/opt/phys-extras/" \
  --env "HF_HOME=$HF_HOME" \
  "$SIF" \
  python3 "$ROOT/scripts/serve_xverify.py" \
    --host "$HOST_BIND" --port "$PORT" --model "$XVERIFY_MODEL"
