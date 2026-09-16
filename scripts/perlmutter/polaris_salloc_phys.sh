#!/bin/bash
# polaris_salloc_phys.sh — launch polaris_anchor.sh on OUR physics data inside
# an interactive salloc allocation (1 node × 4 GPU × 4h).
#
# This is the "physics-data-sync" atom: same POLARIS hparams and sync colocate
# pipeline as the K anchor, but swap POLARIS data → our filtered physics v2_35
# parquet + our xverify-backed reward function.
#
# Usage:
#   bash scripts/perlmutter/polaris_salloc_phys.sh          # foreground (blocks ~3h)
# Claude Code runs this in the background via run_in_background=True.

set -euo pipefail
ROOT="/pscratch/sd/b/bwhou/17-phy-reasoner/phys-reasoner"
cd "$ROOT"
mkdir -p logs

TS=$(date +%Y%m%d.%H%M%S)
JOB="polaris_salloc_phys_${TS}"
LOG="$ROOT/logs/${JOB}.log"

# Read shared xverify URL (already running, slot 51910497).
URL_FILE="$ROOT/outputs/xverify_endpoints/current.url"
if [[ ! -f "$URL_FILE" ]]; then
    echo "ERROR: xverify URL file missing at $URL_FILE" | tee -a "$LOG"
    exit 1
fi
XVERIFY_URL=$(head -n1 "$URL_FILE")
export XVERIFY_URL
export PHYS_REQUIRE_XVERIFY=1

V235="$ROOT/data/processed_tir_filtered_v2_35/data"

# salloc-inline pattern: allocate, then srun the training inside the allocation.
# --gpus-per-node=4 ensures we get the full node.
salloc --nodes=1 --qos=interactive --time=03:00:00 \
       --constraint=gpu --account=m2651 \
       --gpus-per-node=4 --cpus-per-task=32 \
       --job-name="$JOB" \
    srun --ntasks=1 --gpus-per-node=4 --cpus-per-task=32 bash -c "
      set -eu
      export TRAIN_FILES='$V235/train.parquet'
      export VAL_FILES='$V235/validation.parquet'
      # Our physics reward function (xverify-backed compute_score)
      export CUSTOM_REWARD_FN_PATH='$ROOT/src/phys_reasoner/training/reward.py'
      export CUSTOM_REWARD_FN_NAME=compute_score
      export XVERIFY_URL='$XVERIFY_URL'
      export PHYS_REQUIRE_XVERIFY=1
      export EXPERIMENT=polaris_salloc_phys_v235_${TS}
      export WANDB_PROJECT=physcode_polaris_anchor
      # Scale for 1 node × 4 A100-40G, 3h budget
      export NNODES=1
      export N_GPUS=4
      export TRAIN_BATCH=32
      export ROLLOUT_N=8
      export MAX_RESPONSE_LEN=4096
      export PPO_MAX_TOKEN_LEN_PER_GPU=8192
      export TOTAL_STEPS=30
      cd '$ROOT'
      bash scripts/polaris_anchor.sh
    " 2>&1 | tee "$LOG"
