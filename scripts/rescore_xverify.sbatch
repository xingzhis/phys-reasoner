#!/bin/bash
#SBATCH --job-name=rescore_xv
#SBATCH --partition=gpu
#SBATCH --qos=qos_nmi
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --gres=gpu:a100:1
#SBATCH --time=4:00:00
#SBATCH --output=logs/%x_%j.log
#SBATCH --error=logs/%x_%j.err

# Required: XV_MODEL (HF model ID), XV_OUTPUT (output parquet path)
# e.g.  sbatch --export=ALL,XV_MODEL=IAAR-Shanghai/xVerify-3B-Ib,XV_OUTPUT=data/results/rescore_3b.parquet rescore_xverify.sbatch

ROOT=/gpfs/radev/scratch/krishnaswamy_smita/xs272/phys-reasoner
SIF=$ROOT/verl_vllm017.latest.sif
OVERLAY=$ROOT/phys-reasoner-overlay-017.img

if [ -z "$XV_MODEL" ] || [ -z "$XV_OUTPUT" ]; then
    echo "Error: XV_MODEL and XV_OUTPUT must be set via sbatch --export"
    exit 1
fi

export PYTHONUNBUFFERED=1
export HF_HOME=$ROOT/hf_cache
export OMP_NUM_THREADS=8

echo "Job:    $SLURM_JOB_ID  ($SLURM_JOB_NAME)"
echo "Node:   $SLURMD_NODENAME"
echo "GPU:    $CUDA_VISIBLE_DEVICES"
echo "Model:  $XV_MODEL"
echo "Output: $XV_OUTPUT"

CHUNKS="$ROOT/data/results/zero_shot_chunk0.parquet \
        $ROOT/data/results/zero_shot_chunk1.parquet \
        $ROOT/data/results/zero_shot_chunk2.parquet \
        $ROOT/data/results/zero_shot_chunk3.parquet"

apptainer exec --nv \
  --overlay "$OVERLAY:ro" \
  --bind /etc/pki:/etc/pki \
  --env "PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin" \
  --env "HF_HOME=$HF_HOME" \
  --env "PYTHONNOUSERSITE=1" \
  "$SIF" \
  python -u "$ROOT/scripts/rescore_xverify.py" \
    --chunks $CHUNKS \
    --orig   "$ROOT/data/processed/candidates_deduped.parquet" \
    --model  "$XV_MODEL" \
    --output "$XV_OUTPUT"

echo "Done: $?"
