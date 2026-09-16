#!/bin/bash
# One-shot: inspect parquet schemas via container pyarrow.
#SBATCH --job-name=inspect_parquets
#SBATCH --output=logs/%x_%j.log
#SBATCH --error=logs/%x_%j.err
#SBATCH -A m2651
#SBATCH -C cpu
#SBATCH -q debug
#SBATCH -t 00:03:00
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
    python3 "$ROOT/scripts/inspect_parquets.py"
