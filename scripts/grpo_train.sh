#!/bin/bash
# PhysCode GRPO training — single-node launch script.
#
# Usage:
#   # Smoke test (10 steps, batch=4, 1 GPU):
#   bash scripts/grpo_train.sh --smoke
#
#   # Production run (full epoch, 1 GPU):
#   bash scripts/grpo_train.sh
#
# Override any param via env before calling, e.g.:
#   N_GPUS=4 TRAIN_BATCH=128 bash scripts/grpo_train.sh
#
# Algorithm note: this script runs GRPO (adv_estimator=grpo).
# To switch to DAPO/GSPO/CiSPO, change algorithm.adv_estimator and add the
# relevant algorithm.* overrides. Defer that decision until the first real run.
#
# xVerify note: reward.py uses rule-only verification by default (no xVerify GPU).
# FN rate on expression types is ~68% without xVerify. For production, run xVerify
# as a co-located reward server. Infrastructure TBD.

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
source "$ROOT/env.sh"

# ---------------------------------------------------------------------------
# Model: use HF model ID — HF auto-downloads to HF_HOME if not cached.
# On compute nodes without internet, ensure model is pre-downloaded:
#   huggingface-cli download Qwen/Qwen3.5-4B
# ---------------------------------------------------------------------------
MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3.5-4B}"
echo "Model: $MODEL_PATH  (HF_HOME=$HF_HOME)"

# ---------------------------------------------------------------------------
# Configurable params (override via env vars)
# ---------------------------------------------------------------------------
N_GPUS="${N_GPUS:-1}"
TRAIN_BATCH="${TRAIN_BATCH:-4}"          # 4 = smoke test; 128 = production
MAX_PROMPT_LEN="${MAX_PROMPT_LEN:-1024}"
# 4096 covers think+code+injection+think+answer for physics problems.
# Raise to 6144 if p1_truncated rate > 5% in Stage 0 probe.
MAX_RESPONSE_LEN="${MAX_RESPONSE_LEN:-4096}"
TOTAL_EPOCHS="${TOTAL_EPOCHS:-1}"
SAVE_FREQ="${SAVE_FREQ:-500}"
# LR: conservative for instruct RLVR fine-tuning. Try 3e-6 in ablation.
LR="${LR:-1e-6}"
ROLLOUT_TEMP="${ROLLOUT_TEMP:-0.7}"
ROLLOUT_TOP_P="${ROLLOUT_TOP_P:-0.8}"
ROLLOUT_TOP_K="${ROLLOUT_TOP_K:-20}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.85}"
TRAIN_FILES="${TRAIN_FILES:-$ROOT/data/processed/drsci_physics_clean.parquet}"

# Smoke test flag
SMOKE="${1:-}"
if [ "$SMOKE" = "--smoke" ]; then
    TRAIN_BATCH=4
    EXTRA_ARGS="trainer.total_training_steps=10"
    EXPERIMENT_SUFFIX="_smoke"
    echo "=== SMOKE TEST: 10 steps, batch=4 ==="
else
    EXTRA_ARGS=""
    EXPERIMENT_SUFFIX=""
fi

# ---------------------------------------------------------------------------
# Logging dirs
# ---------------------------------------------------------------------------
TIMESTAMP=$(date +%Y%m%d.%H%M%S)
PROJECT="physcode_tir"
EXPERIMENT="grpo_qwen35_4b${EXPERIMENT_SUFFIX}_${TIMESTAMP}"
TRAIN_DIR="$ROOT/outputs/$PROJECT/$EXPERIMENT"
mkdir -p "$TRAIN_DIR" "$ROOT/logs"

export TENSORBOARD_DIR="$TRAIN_DIR/tensorboard"
export VERL_FILE_LOGGER_PATH="$TRAIN_DIR/metrics.jsonl"

echo "Experiment: $EXPERIMENT"
echo "Train dir:  $TRAIN_DIR"
echo "N GPUs:     $N_GPUS"
echo "Batch:      $TRAIN_BATCH"
echo "Response:   $MAX_RESPONSE_LEN tokens"

# ---------------------------------------------------------------------------
# Launch via standard VeRL entrypoint.
# physcode_agent_loops.yaml tells VeRL to load PhysCodeTIRAgentLoop via
# hydra.utils.instantiate(_target_=...) — no custom wrapper needed.
#
# CRITICAL — THINKING MODE: both enable_thinking flags below MUST stay False.
# Qwen3.5 defaults to thinking=ON (<think>...</think> native CoT). If either
# flag is removed or set to True, the model breaks the TIR format and emits
# stray </think> tokens. Two flags are needed:
#   data.apply_chat_template_kwargs.enable_thinking=False  (chat template / agent loop)
#   actor_rollout_ref.model.enable_thinking=False          (model-level inference)
# See src/phys_reasoner/tir/prompts.py for full explanation.
# ---------------------------------------------------------------------------
apptainer exec --nv \
  --overlay "$OVERLAY:ro" \
  --no-home \
  --bind /etc/pki:/etc/pki \
  --env "PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin" \
  --env "PYTHONNOUSERSITE=1" \
  --env "HF_HOME=$HF_HOME" \
  --env "HF_DATASETS_OFFLINE=1" \
  --env "TENSORBOARD_DIR=$TENSORBOARD_DIR" \
  --env "VERL_FILE_LOGGER_PATH=$VERL_FILE_LOGGER_PATH" \
  "$SIF" \
  python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files="$TRAIN_FILES" \
    data.train_batch_size=$TRAIN_BATCH \
    data.max_prompt_length=$MAX_PROMPT_LEN \
    data.max_response_length=$MAX_RESPONSE_LEN \
    data.filter_overlong_prompts=True \
    data.truncation=error \
    data.apply_chat_template_kwargs.enable_thinking=False \
    actor_rollout_ref.model.path="$MODEL_PATH" \
    actor_rollout_ref.model.enable_thinking=False \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.optim.lr=$LR \
    actor_rollout_ref.actor.ppo_mini_batch_size=$TRAIN_BATCH \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.strategy=fsdp2 \
    actor_rollout_ref.actor.fsdp_config.model_dtype=bf16 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.temperature=$ROLLOUT_TEMP \
    actor_rollout_ref.rollout.top_p=$ROLLOUT_TOP_P \
    actor_rollout_ref.rollout.top_k=$ROLLOUT_TOP_K \
    actor_rollout_ref.rollout.gpu_memory_utilization=$GPU_MEM_UTIL \
    actor_rollout_ref.rollout.enable_prefix_caching=True \
    actor_rollout_ref.rollout.agent.default_agent_loop=physcode_tir \
    actor_rollout_ref.rollout.agent.agent_loop_config_path="$ROOT/scripts/physcode_agent_loops.yaml" \
    reward.custom_reward_function.path="$ROOT/src/phys_reasoner/training/reward.py" \
    reward.custom_reward_function.name=compute_score \
    trainer.project_name=$PROJECT \
    trainer.experiment_name=$EXPERIMENT \
    trainer.default_local_dir="$TRAIN_DIR" \
    trainer.n_gpus_per_node=$N_GPUS \
    trainer.nnodes=1 \
    trainer.total_epochs=$TOTAL_EPOCHS \
    trainer.save_freq=$SAVE_FREQ \
    trainer.logger='["console","tensorboard","file"]' \
    $EXTRA_ARGS \
    2>&1 | tee "$TRAIN_DIR/train.log"

echo "Exit: $?"
