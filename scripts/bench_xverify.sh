#!/usr/bin/env bash
# bench_xverify.sh — drive scripts/bench_xverify.py against a live xVerify
# HTTP service inside apptainer.
#
# Companion to scripts/serve_xverify.sh. Run that on the GPU node first, then
# run this on a node that can reach it (set XVERIFY_HOST accordingly).
#
# Usage:
#   bash scripts/bench_xverify.sh
#   N=300 THREADS=8 XVERIFY_HOST=misha00 PORT=8765 bash scripts/bench_xverify.sh
#   PARQUET=data/processed/drsci_physics_clean.parquet bash scripts/bench_xverify.sh
#
# Env vars:
#   XVERIFY_HOST  hostname of the serve_xverify.sh node           (default: misha00)
#   PORT          port the server is bound to                     (default: 8765)
#   N             number of samples to bench                      (default: 300)
#   THREADS       Mode B concurrency                              (default: 8)
#   SEED          workload sampling seed                          (default: 0)
#   PARQUET       training parquet to draw golds from
#                 (default: data/processed/drsci_physics_clean.parquet)
#
# A health check is run first; the bench aborts if the server is unreachable
# so you don't get a flood of "Connection refused" lines.

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
unset SIF OVERLAY
source "$ROOT/env.sh"

XVERIFY_HOST="${XVERIFY_HOST:-misha00}"
PORT="${PORT:-8765}"
N="${N:-300}"
THREADS="${THREADS:-8}"
SEED="${SEED:-0}"
PARQUET="${PARQUET:-$ROOT/data/processed/drsci_physics_clean.parquet}"

XVERIFY_URL="http://${XVERIFY_HOST}:${PORT}/judge"
HEALTH_URL="http://${XVERIFY_HOST}:${PORT}/health"

echo "=== bench_xverify.sh ==="
echo "  server  : $XVERIFY_URL"
echo "  parquet : $PARQUET"
echo "  n       : $N    threads: $THREADS    seed: $SEED"

# Pre-flight: server reachable?
if ! curl -fsS --max-time 5 "$HEALTH_URL" >/dev/null 2>&1; then
  echo "ERROR: $HEALTH_URL not reachable. Start the server with"
  echo "       bash scripts/serve_xverify.sh   on $XVERIFY_HOST"
  exit 1
fi
echo "  health  : ok"

PYTHONNOUSERSITE=1 apptainer exec \
  --overlay "$OVERLAY:ro" --no-home \
  --bind /etc/pki:/etc/pki \
  --env "PYTHONNOUSERSITE=1" \
  --env "PYTHONUNBUFFERED=1" \
  --env "PYTHONPATH=$ROOT/src:/opt/phys-extras/" \
  --env "XVERIFY_URL=$XVERIFY_URL" \
  "$SIF" \
  python3 "$ROOT/scripts/bench_xverify.py" \
    --parquet "$PARQUET" \
    --n "$N" \
    --threads "$THREADS" \
    --seed "$SEED"
