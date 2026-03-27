#!/bin/bash
#SBATCH --job-name=pull_verl017
#SBATCH --partition=test
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=4:00:00
#SBATCH --output=logs/pull_docker_%j.log
#SBATCH --error=logs/pull_docker_%j.err

ROOT="$(cd "$(dirname "$0")" && pwd)"
# Load machine-local overrides if present
[ -f "$ROOT/.env" ] && source "$ROOT/.env"
SIF=$ROOT/verl_vllm017.latest.sif
OVERLAY=$ROOT/phys-reasoner-overlay-017.img

# Pull latest verl SIF (vLLM 0.17, supports Qwen3.5)
apptainer pull docker://verlai/verl:vllm017.latest

# Create fresh overlay (4 GB — packages only; checkpoints/data live on GPFS scratch via bind mounts)
apptainer overlay create --size 4096 "$OVERLAY"

# Install project dependencies into new overlay
apptainer exec \
  --overlay "$OVERLAY" \
  --bind /etc/pki:/etc/pki \
  "$SIF" \
  pip install --no-cache-dir "math-verify[antlr4_13_2]" pint

apptainer exec \
  --overlay "$OVERLAY" \
  --bind /etc/pki:/etc/pki \
  "$SIF" \
  pip install --no-cache-dir -e "$ROOT"

echo "Done. New SIF: verl_vllm017.latest.sif, overlay: phys-reasoner-overlay-017.img"
