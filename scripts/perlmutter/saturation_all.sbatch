#!/bin/bash
# saturation_all.sbatch — interactive-QoS 1-node×4-GPU run of the 6-job
# saturation orchestrator (step_50, step_100, step_150 × TIR, CoT).
# Interactive QoS: 4h max, 4 nodes max, 2 submits/user.

#SBATCH --job-name=sat_all
#SBATCH --output=logs/%x_%j.log
#SBATCH --error=logs/%x_%j.err
#SBATCH -A m2651
#SBATCH -C gpu
#SBATCH -q interactive
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --gpus-per-node=4
#SBATCH --time=04:00:00

set -u
ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/../.." && pwd)}"
cd "$ROOT"

echo "=== saturation_all.sbatch ==="
echo "  Job  : $SLURM_JOB_ID"
echo "  Node : $(hostname)"
echo "  GPUs : $(nvidia-smi -L 2>/dev/null | wc -l) visible"
echo "  Start: $(date -Iseconds)"

bash "$ROOT/scripts/run_saturation_all.sh"
