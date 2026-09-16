#!/bin/bash
# One-shot: rewrite POLARIS parquet inside the container to add data_source.
# Uses debug qos (CPU-only ~ a few minutes, but debug is GPU partition — fine).

#SBATCH --job-name=polaris_parquet_fix
#SBATCH --output=logs/%x_%j.log
#SBATCH --error=logs/%x_%j.err
#SBATCH -A m2651
#SBATCH -C cpu
#SBATCH -q debug
#SBATCH -t 00:05:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4

set -euo pipefail
ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/../.." && pwd)}"
cd "$ROOT"
unset SIF OVERLAY
source "$ROOT/env.sh"

PYTHONNOUSERSITE=1 apptainer exec \
    --overlay "$OVERLAY:ro" \
    --no-home \
    --bind /etc/pki:/etc/pki \
    --env "PYTHONNOUSERSITE=1" \
    --env "PYTHONPATH=/opt/phys-extras/" \
    "$SIF" \
    python3 "$ROOT/scripts/polaris_parquet_fix.py"
