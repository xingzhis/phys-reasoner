#!/bin/bash
# serve_xverify_perlmutter.sbatch — xVerify reward server for Perlmutter.
#
# Used by prod_tir_companion.sbatch / smoke_tir_companion.sbatch / prod_cot_companion.sbatch when 3-group hetjobs
# aren't available. Writes URL to outputs/xverify_endpoints/current.url for
# cross-job rendezvous.
#
# Submit:
#   sbatch scripts/perlmutter/serve_xverify_perlmutter.sbatch
#
# Overrides:
#   sbatch --export=ALL,PORT=9000,XVERIFY_MODEL=IAAR-Shanghai/xVerify-3B-Ib \
#          scripts/perlmutter/serve_xverify_perlmutter.sbatch

#SBATCH --job-name=xverify_server
#SBATCH --output=logs/%x_%j.log
#SBATCH --error=logs/%x_%j.err

# NERSC project account. Must be the same _g account used by the
#               training sbatch (cross-job rendezvous via shared filesystem works
#               fine across accounts, but node-hour accounting will differ).
#SBATCH -A m2651

# 7B fits on any 40G+ card; '-C gpu' is fine.
#SBATCH -C gpu&hbm80g

# Shared QoS for fractional (1-GPU) allocation. NERSC typically
#               supports '-q shared' with --gpus-per-task=1. If not available,
#               drop to '-q regular' and accept a whole-node allocation (wastes
#               3 GPUs) or bundle into the het variant instead.
#SBATCH -q premium

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-task=1
#SBATCH --time=48:00:00

set -euo pipefail

ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/../.." && pwd)}"
cd "$ROOT"
mkdir -p logs outputs/xverify_endpoints

unset SIF OVERLAY
source "$ROOT/env.sh"

PORT="${PORT:-8765}"
XVERIFY_MODEL="${XVERIFY_MODEL:-IAAR-Shanghai/xVerify-7B-I}"

# On Perlmutter, hostname resolves to the compute-node name which is reachable
# from other nodes on the same fabric. If you see "connection refused" from the
# training job, fall back to `hostname --ip-address` and write the IP instead.
HOST=$(hostname)
URL="http://${HOST}:${PORT}/judge"
# URL_FILE can be overridden via --export=ALL,URL_FILE=... when multiple
# concurrent consumers need distinct rendezvous files (e.g. TIR+CoT prod
# sharing one xverify — use a `shared.url` path in that case).
URL_FILE="${URL_FILE:-$ROOT/outputs/xverify_endpoints/current.url}"

echo "=== serve_xverify_perlmutter.sbatch ==="
echo "  Job   : $SLURM_JOB_ID"
echo "  Host  : $HOST"
echo "  Port  : $PORT"
echo "  Model : $XVERIFY_MODEL"
echo "  URL   : $URL"

# Atomic URL-file write so readers see the final value.
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

# Clean up URL file on exit so stale endpoints don't mislead readers.
trap 'rm -f "$URL_FILE"; echo "[xverify] URL file cleaned"' EXIT

# Delegate to the underlying server runner (uses apptainer internally).
# Pass URL_FILE explicitly so serve_xverify.sh writes to the same path.
PORT="$PORT" XVERIFY_MODEL="$XVERIFY_MODEL" URL_FILE="$URL_FILE" bash "$ROOT/scripts/serve_xverify.sh"
