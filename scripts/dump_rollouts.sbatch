#!/bin/bash
#SBATCH --job-name=dump_rollouts_stub
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=8:00:00
#SBATCH --output=logs/%x_%j.log
#SBATCH --error=logs/%x_%j.err
# Moved: scripts/dump_rollouts.sbatch -> eval/inference/rollout.sbatch.
# Update --export and submit path directly — this stub cannot forward SBATCH
# directives reliably. Kept only so stale submit commands don't silently run
# nothing.
echo "ERROR: scripts/dump_rollouts.sbatch moved to eval/inference/rollout.sbatch" >&2
echo "Re-submit with: sbatch eval/inference/rollout.sbatch" >&2
exit 2
