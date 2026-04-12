#!/bin/bash
#SBATCH --job-name=score_probe
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=4:00:00
#SBATCH --output=logs/%x_%j.log
#SBATCH --error=logs/%x_%j.err
#
# Scores probe rollout parquets using the training-equivalent reward function.
# Requires a running xVerify server (sbatch scripts/serve_xverify.sbatch).
#
# The scorer runs on CPU — rule verification is pure Python, xVerify calls go
# to the companion server over HTTP.
#
# Usage:
#   # First start xVerify server, then:
#   sbatch scripts/score_probe.sbatch
#
# Override inputs/output:
#   sbatch --export=ALL,PROBE_INPUTS="outputs/probe_calib_A/rollouts.parquet outputs/probe_calib_B/rollouts.parquet",PROBE_OUTPUT=outputs/probe_scored/ \
#     scripts/score_probe.sbatch

set -euo pipefail

ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "$ROOT"
unset SIF OVERLAY
source "$ROOT/env.sh"

PROBE_INPUTS="${PROBE_INPUTS:-$ROOT/outputs/probe_calib_A/rollouts.parquet $ROOT/outputs/probe_calib_B/rollouts.parquet}"
PROBE_OUTPUT="${PROBE_OUTPUT:-$ROOT/outputs/probe_scored}"

mkdir -p "$PROBE_OUTPUT" logs

# Wait for xVerify URL to appear (server may still be loading)
URL_FILE="$ROOT/outputs/xverify_endpoints/current.url"
echo "Waiting for xVerify rendezvous file: $URL_FILE"
WAITED=0
while [ ! -f "$URL_FILE" ] && [ $WAITED -lt 300 ]; do
    sleep 5
    WAITED=$((WAITED + 5))
done
if [ ! -f "$URL_FILE" ]; then
    echo "ERROR: xVerify rendezvous file not found after 300s. Start the server first."
    exit 1
fi
echo "Found: $(cat "$URL_FILE" | head -1)"

export PYTHONUNBUFFERED=1
export PYTHONNOUSERSITE=1

echo "Job:     $SLURM_JOB_ID"
echo "Node:    $SLURMD_NODENAME"
echo "Inputs:  $PROBE_INPUTS"
echo "Output:  $PROBE_OUTPUT"

# Convert space-separated list to --inputs args
INPUT_ARGS=""
for f in $PROBE_INPUTS; do
    [[ "$f" != /* ]] && f="$ROOT/$f"
    INPUT_ARGS="$INPUT_ARGS $f"
done

PYTHONNOUSERSITE=1 apptainer exec \
  --overlay "$OVERLAY:ro" \
  --no-home \
  --bind /etc/pki:/etc/pki \
  --env "PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin" \
  --env "PYTHONNOUSERSITE=1" \
  --env "PYTHONPATH=/opt/phys-extras/" \
  --env "PYTHONUNBUFFERED=1" \
  --env "HF_HOME=$HF_HOME" \
  --env "HF_DATASETS_OFFLINE=1" \
  "$SIF" \
  python3 "$ROOT/scripts/score_probe_rollouts.py" \
    --inputs $INPUT_ARGS \
    --output "$PROBE_OUTPUT"

echo "Exit: $?"
