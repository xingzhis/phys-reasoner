#!/bin/bash
# sany_salloc.sh — Qwen3-0.6B + DeepScaleR sanity via salloc interactive.
# Minimal scale: 1 node × 2 GPU sync colocate. Fast iteration test.

set -euo pipefail
ROOT="/pscratch/sd/b/bwhou/17-phy-reasoner/phys-reasoner"
cd "$ROOT"
mkdir -p logs

TS=$(date +%Y%m%d.%H%M%S)
JOB="sany_qwen3_0.6b_deepscaler_${TS}"
LOG="$ROOT/logs/${JOB}.log"

DSR="$ROOT/data/deepscaler/train.parquet"
REWARD="$ROOT/src/phys_reasoner/training/reward_polaris.py"

salloc --nodes=1 --qos=interactive --time=02:00:00 \
       --constraint=gpu --account=m2651 \
       --gpus-per-node=2 --cpus-per-task=16 \
       --job-name="$JOB" \
    srun --ntasks=1 --gpus-per-node=2 --cpus-per-task=16 bash -c "
      set -eu
      export MODEL=Qwen/Qwen3-0.6B
      export TRAIN_FILES='$DSR'
      export VAL_FILES='$DSR'
      export CUSTOM_REWARD_FN_PATH='$REWARD'
      export CUSTOM_REWARD_FN_NAME=compute_score
      export EXPERIMENT=sany_qwen3_06b_deepscaler_${TS}
      export WANDB_PROJECT=physcode_polaris_anchor
      export NNODES=1
      export N_GPUS=2
      export TRAIN_BATCH=32
      export ROLLOUT_N=8
      export MAX_RESPONSE_LEN=8192
      export PPO_MAX_TOKEN_LEN_PER_GPU=10240
      export TOTAL_STEPS=60
      export TEST_FREQ=20
      export SAVE_FREQ=-1
      cd '$ROOT'
      bash scripts/polaris_anchor.sh
    " 2>&1 | tee "$LOG"
