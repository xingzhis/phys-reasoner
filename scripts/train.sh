#!/usr/bin/env bash
# train.sh — PhysCode GRPO production training (Qwen3.5-4B, TIR, single node).
#
# Derived from scripts/smoke_tir_qwen35.sh (validated 2026-04-02).
# All VeRL/FSDP/rollout settings are identical to that smoke script.
# Production additions over smoke: configurable epochs, checkpoint saving,
# full data paths, and wandb + tensorboard + file logging.
#
# Usage:
#   bash scripts/train.sh                          # full training run
#   TOTAL_STEPS=10 bash scripts/train.sh           # short probe run
#   TRAIN_FILES=data/processed/corpus_train.parquet bash scripts/train.sh
#
# Wandb: set WANDB_API_KEY in your environment (or `wandb login` interactively
# once on the login node — credentials are cached in ~/.netrc and work inside
# the container because we don't use --no-home on login nodes; for sbatch jobs
# export WANDB_API_KEY explicitly).

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
unset SIF OVERLAY
source "$ROOT/env.sh"
echo "  SIF:     $SIF"
echo "  OVERLAY: $OVERLAY"

# ---------------------------------------------------------------------------
# Configurable params — override via env vars
# ---------------------------------------------------------------------------
MODEL="${MODEL:-Qwen/Qwen3.5-4B}"
TRAIN_FILES="${TRAIN_FILES:-$ROOT/data/processed/drsci_physics_clean.parquet}"
VAL_FILES="${VAL_FILES:-$ROOT/data/processed/corpus_train.parquet}"

N_GPUS="${N_GPUS:-1}"
TRAIN_BATCH="${TRAIN_BATCH:-128}"
ROLLOUT_N="${ROLLOUT_N:-8}"             # GRPO needs n≥2; n=1 → zero advantages
MAX_PROMPT_LEN="${MAX_PROMPT_LEN:-1024}"
MAX_TOOL_TURNS="${MAX_TOOL_TURNS:-1}"

# Think-interrupt budget (see smoke_tir_qwen35.sh for full explanation).
THINKING_BUDGET="${THINKING_BUDGET:-}"
TOOL_CALL_BUDGET="${TOOL_CALL_BUDGET:-}"
INTERRUPT_LEN=15
MAX_TOOL_RESPONSE_LEN=512
ANSWER_BUDGET="${ANSWER_BUDGET:-}"

if [[ -n "$THINKING_BUDGET" && -n "$TOOL_CALL_BUDGET" && -n "$ANSWER_BUDGET" ]]; then
    MAX_RESPONSE_LEN=$((THINKING_BUDGET + INTERRUPT_LEN + TOOL_CALL_BUDGET + MAX_TOOL_RESPONSE_LEN + ANSWER_BUDGET))
else
    MAX_RESPONSE_LEN="${MAX_RESPONSE_LEN:-4096}"
fi

# Training schedule — use TOTAL_STEPS for a step-capped run, otherwise full epochs.
TOTAL_EPOCHS="${TOTAL_EPOCHS:-1}"
TOTAL_STEPS="${TOTAL_STEPS:-}"          # if set, overrides TOTAL_EPOCHS
SAVE_FREQ="${SAVE_FREQ:-500}"
TEST_FREQ="${TEST_FREQ:--1}"

LR="${LR:-1e-6}"
VLLM_GPU_MEM_UTIL="${VLLM_GPU_MEM_UTIL:-0.5}"

# Wandb
WANDB_PROJECT="${WANDB_PROJECT:-physcode_tir}"

# ---------------------------------------------------------------------------
# Derived settings (match smoke_tir_qwen35.sh exactly)
# ---------------------------------------------------------------------------
ROLLOUT_MAX_NUM_SEQS=$((TRAIN_BATCH * ROLLOUT_N))
PPO_MINI_BATCH=$TRAIN_BATCH
# PPO_MICRO_BS: gradient-accumulation micro-batch size. Defaults to PPO_MINI_BATCH (no
# accumulation). Set to 1 on memory-constrained GPUs (e.g. A40) to halve activation
# memory at the cost of extra forward/backward passes.
PPO_MICRO_BS="${PPO_MICRO_BS:-$PPO_MINI_BATCH}"
REF_MICRO_BATCH="${REF_MICRO_BS:-$PPO_MICRO_BS}"

TIMESTAMP=$(date +%Y%m%d.%H%M%S)
EXPERIMENT="grpo_qwen35_4b_${TIMESTAMP}"
TRAIN_DIR="$ROOT/outputs/$WANDB_PROJECT/$EXPERIMENT"
mkdir -p "$TRAIN_DIR" "$ROOT/logs"

echo "=== train.sh: PhysCode GRPO TIR training ==="
echo "  model     : $MODEL"
echo "  train     : $TRAIN_FILES"
echo "  val       : $VAL_FILES"
echo "  batch     : $TRAIN_BATCH x rollout_n=$ROLLOUT_N = $ROLLOUT_MAX_NUM_SEQS gens/step"
echo "  seq lens  : prompt=$MAX_PROMPT_LEN  response=$MAX_RESPONSE_LEN"
echo "  vLLM mem  : $VLLM_GPU_MEM_UTIL"
echo "  output    : $TRAIN_DIR"
echo "  wandb     : $WANDB_PROJECT / $EXPERIMENT"

# Guard: expandable_segments conflicts with vLLM CuMemAllocator.
if [[ "${PYTORCH_CUDA_ALLOC_CONF:-}" == *"expandable_segments:True"* ]]; then
    echo "WARNING: unsetting PYTORCH_CUDA_ALLOC_CONF (expandable_segments:True conflicts with vLLM)"
    unset PYTORCH_CUDA_ALLOC_CONF
fi

# Guard: ROCR_VISIBLE_DEVICES conflicts with CUDA_VISIBLE_DEVICES in VeRL.
if [[ -n "${ROCR_VISIBLE_DEVICES:-}" ]]; then
    echo "WARNING: unsetting ROCR_VISIBLE_DEVICES (conflicts with CUDA_VISIBLE_DEVICES in VeRL)"
    unset ROCR_VISIBLE_DEVICES
fi

# Build optional step-cap override.
STEPS_ARG=""
if [[ -n "$TOTAL_STEPS" ]]; then
    STEPS_ARG="trainer.total_training_steps=$TOTAL_STEPS"
    echo "  steps     : $TOTAL_STEPS (overrides epochs)"
else
    echo "  epochs    : $TOTAL_EPOCHS"
fi

PYTHONNOUSERSITE=1 apptainer exec --nv \
  --overlay "$OVERLAY:ro" \
  --no-home \
  --bind /etc/pki:/etc/pki \
  --env "PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin" \
  --env "PYTHONNOUSERSITE=1" \
  --env "PYTHONUNBUFFERED=1" \
  --env "PYTHONPATH=/opt/phys-extras/" \
  --env "HF_HOME=$HF_HOME" \
  --env "HF_DATASETS_OFFLINE=1" \
  --env "WANDB_API_KEY=${WANDB_API_KEY:-}" \
  --env "WANDB_PROJECT=$WANDB_PROJECT" \
  --env "WANDB_RUN_ID=$EXPERIMENT" \
  --env "VERL_DUMP_DIR=${VERL_DUMP_DIR:-}" \
  "$SIF" \
  python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    algorithm.use_kl_in_reward=False \
    data.train_files="$TRAIN_FILES" \
    data.val_files="$VAL_FILES" \
    data.train_batch_size=$TRAIN_BATCH \
    data.max_prompt_length=$MAX_PROMPT_LEN \
    data.max_response_length=$MAX_RESPONSE_LEN \
    data.filter_overlong_prompts=True \
    data.truncation=error \
    data.dataloader_num_workers=8 \
    +data.apply_chat_template_kwargs.enable_thinking=true \
    actor_rollout_ref.model.path="$MODEL" \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    '+actor_rollout_ref.model.override_config={attn_implementation:sdpa}' \
    actor_rollout_ref.actor.optim.lr=$LR \
    actor_rollout_ref.actor.ppo_mini_batch_size=$PPO_MINI_BATCH \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=$PPO_MICRO_BS \
    actor_rollout_ref.actor.ppo_epochs=1 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.actor.fsdp_config.model_dtype=bfloat16 \
    '+actor_rollout_ref.actor.fsdp_config.wrap_policy.transformer_layer_cls_to_wrap=[Qwen3_5DecoderLayer]' \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.ref.fsdp_config.model_dtype=bfloat16 \
    '+actor_rollout_ref.ref.fsdp_config.wrap_policy.transformer_layer_cls_to_wrap=[Qwen3_5DecoderLayer]' \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.n=$ROLLOUT_N \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.top_p=0.9 \
    actor_rollout_ref.rollout.gpu_memory_utilization=$VLLM_GPU_MEM_UTIL \
    actor_rollout_ref.rollout.max_model_len=$((MAX_PROMPT_LEN + MAX_RESPONSE_LEN)) \
    actor_rollout_ref.rollout.max_num_seqs=$ROLLOUT_MAX_NUM_SEQS \
    actor_rollout_ref.rollout.load_format=safetensors \
    actor_rollout_ref.rollout.enable_prefix_caching=True \
    actor_rollout_ref.rollout.enable_chunked_prefill=True \
    actor_rollout_ref.rollout.tensor_model_parallel_size=$N_GPUS \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=$REF_MICRO_BATCH \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=$REF_MICRO_BATCH \
    actor_rollout_ref.rollout.agent.default_agent_loop=tool_agent \
    actor_rollout_ref.rollout.multi_turn.enable=true \
    actor_rollout_ref.rollout.multi_turn.format=qwen3_coder \
    actor_rollout_ref.rollout.multi_turn.tool_config_path="$ROOT/scripts/physcode_tools.yaml" \
    actor_rollout_ref.rollout.multi_turn.max_assistant_turns=$((MAX_TOOL_TURNS * 2)) \
    actor_rollout_ref.rollout.multi_turn.max_user_turns=$MAX_TOOL_TURNS \
    actor_rollout_ref.rollout.multi_turn.max_parallel_calls=1 \
    actor_rollout_ref.rollout.multi_turn.max_tool_response_length=$MAX_TOOL_RESPONSE_LEN \
    actor_rollout_ref.rollout.multi_turn.tool_response_truncate_side=right \
    ${THINKING_BUDGET:++actor_rollout_ref.rollout.multi_turn.thinking_budget=$THINKING_BUDGET} \
    ${TOOL_CALL_BUDGET:++actor_rollout_ref.rollout.multi_turn.tool_call_budget=$TOOL_CALL_BUDGET} \
    reward.custom_reward_function.path="$ROOT/src/phys_reasoner/training/reward.py" \
    reward.custom_reward_function.name=compute_score \
    trainer.critic_warmup=0 \
    trainer.n_gpus_per_node=$N_GPUS \
    trainer.nnodes=1 \
    trainer.total_epochs=$TOTAL_EPOCHS \
    trainer.save_freq=$SAVE_FREQ \
    trainer.test_freq=$TEST_FREQ \
    trainer.val_before_train=false \
    trainer.project_name="$WANDB_PROJECT" \
    trainer.experiment_name="$EXPERIMENT" \
    trainer.default_local_dir="$TRAIN_DIR" \
    trainer.default_hdfs_dir=null \
    'trainer.logger=["console","wandb"]' \
    $STEPS_ARG \
    2>&1 | tee "$TRAIN_DIR/train.log"

echo "Exit: $?"
