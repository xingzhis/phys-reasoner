#!/bin/bash
# polaris_anchor.sbatch — K anchor run on Perlmutter.
# Tries interactive qos first (4h/1node); if that's rejected use debug (30min).
#
# Submit:
#   sbatch scripts/perlmutter/polaris_anchor.sbatch
#   # To force debug qos:
#   sbatch -q debug -t 30:00 scripts/perlmutter/polaris_anchor.sbatch

#SBATCH --job-name=polaris_anchor
#SBATCH --output=logs/%x_%j.log
#SBATCH --error=logs/%x_%j.err
#SBATCH -A m2651
#SBATCH -C gpu&hbm40g
#SBATCH -q debug
#SBATCH -t 00:30:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-node=4

set -euo pipefail
ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/../.." && pwd)}"
cd "$ROOT"
mkdir -p logs

unset SIF OVERLAY
source "$ROOT/env.sh"
[[ -f "$SIF" && -f "$OVERLAY" ]] || { echo "ERROR: SIF/OVERLAY missing" >&2; exit 1; }

echo "=== polaris_anchor.sbatch ==="
echo "  Job     : $SLURM_JOB_ID"
echo "  Nodes   : $SLURM_JOB_NODELIST"
echo "  GPUs    : $SLURM_GPUS_PER_NODE per node"
echo "  QoS     : $SLURM_JOB_QOS"
echo "  SIF     : $SIF"
echo "  OVERLAY : $OVERLAY"

# NCCL tuning (same as prod sbatch)
export NCCL_IB_TIMEOUT=32
export NCCL_IBEXT_DISABLE=1
export TORCH_NCCL_ENABLE_MONITORING=0

exec bash "$ROOT/scripts/polaris_anchor.sh"
