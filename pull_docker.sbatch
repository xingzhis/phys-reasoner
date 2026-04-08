#!/bin/bash
#SBATCH --job-name=pull_verl017
#SBATCH --partition=day
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=4:00:00
#SBATCH --output=logs/pull_docker_%j.log
#SBATCH --error=logs/pull_docker_%j.err

# SLURM copies the batch script to its spool dir before running, so $0 resolves
# there — not to the project. Use SLURM_SUBMIT_DIR (the dir sbatch was called from).
ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")" && pwd)}"
mkdir -p "$ROOT/logs"

# Load machine-local overrides if present
[ -f "$ROOT/.env" ] && source "$ROOT/.env"
SIF=$ROOT/verl_vllm017.latest.sif
OVERLAY=$ROOT/phys-reasoner-overlay-017.img

# Apptainer writes large temp files during OCI→SIF conversion; point it at scratch
# so it doesn't try to write to the unwritable SLURM spool dir.
export APPTAINER_TMPDIR="$ROOT/tmp_apptainer"
export APPTAINER_CACHEDIR="$ROOT/tmp_apptainer_cache"
mkdir -p "$APPTAINER_TMPDIR" "$APPTAINER_CACHEDIR"

echo "Job: $SLURM_JOB_ID on $SLURMD_NODENAME"
echo "ROOT: $ROOT"
echo "SIF:  $SIF"

# Pull verl SIF. vllm017.latest is the stable choice:
#   - vllm 0.17.0 (≥ 0.7.0 required by verl_repo)
#   - transformers 4.57.6 base (qwen3_5 support added via setup_overlay.sh)
# Note: 016.dev.qwen3_5 ships vllm 0.1.dev1 which verl_repo rejects; 018.latest
# pins the same transformers as 017. Use 017 + PYTHONPATH overlay approach instead.
apptainer pull "$SIF" docker://verlai/verl:vllm017.latest

# Create fresh overlay (6 GB)
apptainer overlay create --size 6144 "$OVERLAY"

echo "Done. SIF: $SIF"
echo "      Overlay: $OVERLAY"
echo ""
echo "Next: run  bash scripts/setup_overlay.sh  to install packages into the overlay."
