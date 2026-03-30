#!/bin/bash
#SBATCH --job-name=rescore-passk-xv
#SBATCH --partition=gpu
#SBATCH --qos=qos_nmi
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --gres=gpu:h200:1
#SBATCH --time=2:00:00
#SBATCH --output=logs/%x_%j.log
#SBATCH --error=logs/%x_%j.err

# Re-score a pass@k parquet (all_texts) with xVerify.
#
# Override via sbatch --export:
#   sbatch --export=ALL,XV_INPUT=data/results/zero_shot_corpus_passk.parquet,XV_SCHEMA=corpus,XV_OUTPUT=data/results/zero_shot_corpus_passk_xv7b.parquet scripts/rescore_passk_xverify.sbatch
#   sbatch --export=ALL,XV_INPUT=data/results/zero_shot_drsci_sample.parquet,XV_SCHEMA=drsci,XV_OUTPUT=data/results/zero_shot_drsci_sample_xv7b.parquet scripts/rescore_passk_xverify.sbatch

ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
source "$ROOT/env.sh"

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=8

XV_MODEL="${XV_MODEL:-IAAR-Shanghai/xVerify-7B-I}"
XV_INPUT="${XV_INPUT:-$ROOT/data/results/zero_shot_corpus_passk.parquet}"
XV_SCHEMA="${XV_SCHEMA:-corpus}"
XV_OUTPUT="${XV_OUTPUT:-$ROOT/data/results/zero_shot_corpus_passk_xv7b.parquet}"
XV_ORIG="${XV_ORIG:-$ROOT/data/processed/candidates_deduped.parquet}"

echo "Job:    $SLURM_JOB_ID  ($SLURM_JOB_NAME)"
echo "Node:   $SLURMD_NODENAME"
echo "GPU:    $CUDA_VISIBLE_DEVICES"
echo "Model:  $XV_MODEL"
echo "Input:  $XV_INPUT  (schema=$XV_SCHEMA)"
echo "Output: $XV_OUTPUT"

if [ ! -f "$XV_INPUT" ]; then
    echo "ERROR: $XV_INPUT not found."
    exit 1
fi

apptainer exec --nv \
  --overlay "$OVERLAY:ro" \
  --no-home \
  --bind /etc/pki:/etc/pki \
  --env "PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin" \
  --env "HF_HOME=$HF_HOME" \
  --env "PYTHONNOUSERSITE=1" \
  "$SIF" \
  python -u "$ROOT/scripts/rescore_passk_xverify.py" \
    --input   "$XV_INPUT" \
    --orig    "$XV_ORIG" \
    --schema  "$XV_SCHEMA" \
    --model   "$XV_MODEL" \
    --output  "$XV_OUTPUT"

echo "Done: $?"
