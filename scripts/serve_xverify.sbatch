#!/bin/bash
# serve_xverify.sbatch — cluster-generic xVerify HTTP reward server.
#
# Replaces scripts/serve_xverify_misha.sbatch (which hardcoded --partition=test
# and --nodelist=misha00 that don't exist on this cluster).
#
# On startup, writes the reachable URL to a rendezvous file:
#   $ROOT/outputs/xverify_endpoints/current.url
# so that other sbatch jobs (e.g. train_async_2node.sbatch) can discover it
# without knowing the node name in advance.
#
# Submission:
#   sbatch scripts/serve_xverify.sbatch
#
# Defaults: 1x any-class GPU (A100 preferred via fallback), 6h wall time.
# Override model or port via env:
#   sbatch --export=ALL,PORT=9000,XVERIFY_MODEL=IAAR-Shanghai/xVerify-3B-Ib \
#          scripts/serve_xverify.sbatch
#
#SBATCH --job-name=xverify_server
#SBATCH --partition=gpu
#SBATCH --qos=qos_nmi
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --gres=gpu:l40s:1
#SBATCH --time=6:00:00
#SBATCH --output=logs/%x_%j.log
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "$ROOT"
mkdir -p logs outputs/xverify_endpoints

unset SIF OVERLAY
source "$ROOT/env.sh"
# Pin to active overlay (env.sh default is stale — 017 vs 017b)
OVERLAY="$ROOT/phys-reasoner-overlay-017b.img"
export OVERLAY

PORT="${PORT:-8765}"
XVERIFY_MODEL="${XVERIFY_MODEL:-IAAR-Shanghai/xVerify-7B-I}"

HOST=$(hostname)
# Use hostname as-is; compute nodes on misha resolve each other via DNS.
URL="http://${HOST}:${PORT}/judge"
URL_FILE="$ROOT/outputs/xverify_endpoints/current.url"

echo "=== serve_xverify.sbatch ==="
echo "  Job   : $SLURM_JOB_ID"
echo "  Host  : $HOST"
echo "  Port  : $PORT"
echo "  Model : $XVERIFY_MODEL"
echo "  URL   : $URL"
echo "  Write : $URL_FILE"

# Write the URL file atomically so readers see the final value.
TMP="$URL_FILE.tmp.$$"
{
    echo "$URL"
    echo "# job_id=$SLURM_JOB_ID"
    echo "# host=$HOST"
    echo "# port=$PORT"
    echo "# model=$XVERIFY_MODEL"
    echo "# written_at=$(date -Iseconds)"
} > "$TMP"
mv "$TMP" "$URL_FILE"

# Ensure the URL file is removed on exit so stale endpoints don't mislead readers.
trap 'rm -f "$URL_FILE"; echo "[xverify] URL file cleaned"' EXIT

# Launch the underlying server script (uses apptainer internally).
PORT="$PORT" XVERIFY_MODEL="$XVERIFY_MODEL" bash "$ROOT/scripts/serve_xverify.sh"
