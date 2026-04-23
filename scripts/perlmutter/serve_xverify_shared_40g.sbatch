#!/bin/bash
# serve_xverify_shared_40g.sbatch — xVerify reward server on gpu_shared 1-GPU.
#
# Runs on gpu_shared QoS (fractional 1-GPU allocation, NOT counted against
# 5-job premium cap). Uses 40G A100. xVerify-7B fits in ~14 GB so 40G is plenty.
#
# Use this when premium cap is saturated by training/probe jobs.
#
# Submit:
#   sbatch scripts/perlmutter/serve_xverify_shared_40g.sbatch

#SBATCH --job-name=xverify_server
#SBATCH --output=logs/%x_%j.log
#SBATCH --error=logs/%x_%j.err
#SBATCH -A m2651
#SBATCH -C gpu&hbm40g
#SBATCH -q shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
# NERSC shared QoS mandates 32 CPU cores per GPU requested.
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-task=1
#SBATCH --time=24:00:00

set -euo pipefail

ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/../.." && pwd)}"
cd "$ROOT"
mkdir -p logs outputs/xverify_endpoints

unset SIF OVERLAY
source "$ROOT/env.sh"

PORT="${PORT:-8765}"
XVERIFY_MODEL="${XVERIFY_MODEL:-IAAR-Shanghai/xVerify-7B-I}"

HOST=$(hostname)
URL="http://${HOST}:${PORT}/judge"
URL_FILE="${URL_FILE:-$ROOT/outputs/xverify_endpoints/current.url}"

echo "=== serve_xverify_shared_40g.sbatch ==="
echo "  Job   : $SLURM_JOB_ID"
echo "  Host  : $HOST"
echo "  Port  : $PORT"
echo "  Model : $XVERIFY_MODEL"
echo "  URL   : $URL"

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

trap 'rm -f "$URL_FILE"; echo "[xverify] URL file cleaned"' EXIT

PORT="$PORT" XVERIFY_MODEL="$XVERIFY_MODEL" URL_FILE="$URL_FILE" bash "$ROOT/scripts/serve_xverify.sh"
