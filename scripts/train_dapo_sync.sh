#!/bin/bash
# train_dapo_sync.sh — sync DAPO trainer (recipe.dapo.main_dapo) with TIR support.
#
# Companion to train_async.sh, but uses the SYNC DAPO trainer which has
# filter_groups (saturated-group filtering) implemented. This is the dominant
# missing piece in async (per POLARIS/DAPO/ScaleRL — see research report).
#
# Topology: hybrid_engine=True, rollout co-located with trainer on the same
# nodes. No separate rollout het group needed — simpler than the async path.
# vLLM uses sleep_mode to free GPU between rollout and update phases.
#
# Required env:
#   NNODES, N_GPUS_TRAIN, MODEL, TRAIN_FILES, VAL_FILES,
#   WANDB_PROJECT, EXPERIMENT, XVERIFY_URL,
#   TRAIN_BATCH (default 128), ROLLOUT_N (default 8),
#   GEN_BATCH (default = TRAIN_BATCH * 3 to amortize filter_groups regen),
#   LR (default 1e-6), KL_COEF (default 0.001),
#   FILTER_GROUPS_ENABLE (default true), FILTER_GROUPS_METRIC (default acc),
#   FILTER_GROUPS_MAX_GENS (default 10),
#   THINKING_BUDGET, TOOL_CALL_BUDGET, ANSWER_BUDGET,
#   TRAIN_SP (default 2), TOTAL_STEPS, TOTAL_EPOCHS

set -euo pipefail
ROOT="${ROOT:-$(cd "$(dirname "$0")/.." && pwd -P)}"
source "$ROOT/env.sh"

# --- defaults ---
NNODES="${NNODES:-4}"
N_GPUS_TRAIN="${N_GPUS_TRAIN:-4}"
MODEL="${MODEL:-Qwen/Qwen3-4B-Thinking-2507}"
TRAIN_BATCH="${TRAIN_BATCH:-128}"
ROLLOUT_N="${ROLLOUT_N:-8}"
GEN_BATCH="${GEN_BATCH:-$((TRAIN_BATCH * 3))}"   # 3× oversample for filter_groups regen headroom
PPO_MINI_BATCH="${PPO_MINI_BATCH:-32}"           # DAPO canonical mini-batch
LR="${LR:-1e-6}"
LR_WARMUP_STEPS="${LR_WARMUP_STEPS:-10}"          # DAPO canonical: 10 steps
LR_SCHEDULER_TYPE="${LR_SCHEDULER_TYPE:-constant}"
KL_COEF="${KL_COEF:-0.001}"
FILTER_GROUPS_ENABLE="${FILTER_GROUPS_ENABLE:-true}"
FILTER_GROUPS_METRIC="${FILTER_GROUPS_METRIC:-acc}"
FILTER_GROUPS_MAX_GENS="${FILTER_GROUPS_MAX_GENS:-10}"
TOTAL_EPOCHS="${TOTAL_EPOCHS:-3}"
TOTAL_STEPS="${TOTAL_STEPS:-200}"

THINKING_BUDGET="${THINKING_BUDGET:-12288}"
TOOL_CALL_BUDGET="${TOOL_CALL_BUDGET:-2048}"
ANSWER_BUDGET="${ANSWER_BUDGET:-4096}"
INTERRUPT_LEN="${INTERRUPT_LEN:-16}"
MAX_TOOL_RESPONSE_LEN="${MAX_TOOL_RESPONSE_LEN:-1024}"
MAX_TOOL_TURNS="${MAX_TOOL_TURNS:-1}"
MAX_PROMPT_LEN="${MAX_PROMPT_LEN:-1024}"
MAX_RESPONSE_LEN=$((THINKING_BUDGET + INTERRUPT_LEN + TOOL_CALL_BUDGET + MAX_TOOL_RESPONSE_LEN + ANSWER_BUDGET))

TRAIN_SP="${TRAIN_SP:-2}"
PPO_MAX_TOKEN_LEN_PER_GPU="${PPO_MAX_TOKEN_LEN_PER_GPU:-20480}"
ACTOR_OPT_OFFLOAD="${ACTOR_OPT_OFFLOAD:-True}"
ACTOR_PARAM_OFFLOAD="${ACTOR_PARAM_OFFLOAD:-False}"
VLLM_GPU_MEM_UTIL="${VLLM_GPU_MEM_UTIL:-0.72}"
DYN_BSZ_FLAG="${DYN_BSZ_FLAG:-True}"
SAVE_FREQ="${SAVE_FREQ:-50}"
TEST_FREQ="${TEST_FREQ:-50}"

WANDB_PROJECT="${WANDB_PROJECT:-physcode_tir}"
EXPERIMENT="${EXPERIMENT:-grpo_tir_dapo_sync_$(date +%Y%m%d.%H%M%S)}"
TRAIN_DIR="$ROOT/outputs/$WANDB_PROJECT/$EXPERIMENT"
mkdir -p "$TRAIN_DIR"
LOGGERS="${LOGGERS:-console,wandb}"

# Multi-turn TIR config
AGENT_LOOP="${AGENT_LOOP:-tool_agent}"
MULTI_TURN_ENABLE="${MULTI_TURN_ENABLE:-true}"

echo "=== train_dapo_sync.sh ==="
echo "  HEAD          : $(hostname)"
echo "  RAY_ADDRESS   : ${RAY_ADDRESS:-unset}"
echo "  EXPERIMENT    : $EXPERIMENT"
echo "  TRAIN_DIR     : $TRAIN_DIR"
echo "  MODEL         : $MODEL"
echo "  data          : train=$TRAIN_FILES  val=$VAL_FILES"
echo "  topology      : nnodes=$NNODES n_gpus=$N_GPUS_TRAIN sp=$TRAIN_SP (hybrid_engine=True, no separate rollout)"
echo "  batch         : train_prompt=$TRAIN_BATCH gen_prompt=$GEN_BATCH n=$ROLLOUT_N mini=$PPO_MINI_BATCH"
echo "  algo          : LR=$LR warmup=$LR_WARMUP_STEPS KL=$KL_COEF filter_groups=$FILTER_GROUPS_ENABLE/$FILTER_GROUPS_METRIC/$FILTER_GROUPS_MAX_GENS"
echo "  budgets       : think=$THINKING_BUDGET tool_call=$TOOL_CALL_BUDGET answer=$ANSWER_BUDGET (max_resp=$MAX_RESPONSE_LEN)"
echo "  xverify       : $XVERIFY_URL"

if [[ -n "${ROCR_VISIBLE_DEVICES:-}" ]]; then unset ROCR_VISIBLE_DEVICES; fi

PYTHONNOUSERSITE=1 apptainer exec --nv \
  --overlay "$OVERLAY:ro" \
  --no-home \
  --pwd "$ROOT/verl" \
  --bind /etc/pki:/etc/pki \
  --env "PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin" \
  --env "PYTHONNOUSERSITE=1" \
  --env "PYTHONUNBUFFERED=1" \
  --env "PYTHONPATH=$ROOT/.async-extras:/opt/phys-extras/" \
  --env "XDG_CACHE_HOME=/tmp/.cache" \
  --env "FLASHINFER_WORKSPACE_BASE=/tmp" \
  --env "TRITON_CACHE_DIR=/tmp/.cache/triton" \
  --env "MPLCONFIGDIR=/tmp/.cache/matplotlib" \
  --env "HF_HOME=$HF_HOME" \
  --env "HF_DATASETS_OFFLINE=1" \
  --env "HF_HUB_OFFLINE=1" \
  --env "TRANSFORMERS_OFFLINE=1" \
  --env "VLLM_USE_V1=1" \
  --env "INTERRUPT_LEN=$INTERRUPT_LEN" \
  --env "WANDB_API_KEY=${WANDB_API_KEY:-}" \
  --env "WANDB_PROJECT=$WANDB_PROJECT" \
  --env "WANDB_RUN_ID=$EXPERIMENT" \
  --env "XVERIFY_URL=${XVERIFY_URL:-}" \
  --env "PHYS_REQUIRE_XVERIFY=${PHYS_REQUIRE_XVERIFY:-1}" \
  --env "RAY_ADDRESS=${RAY_ADDRESS:-}" \
  "$SIF" \
  python3 -m recipe.dapo.main_dapo \
    algorithm.adv_estimator=grpo \
    algorithm.use_kl_in_reward=False \
    algorithm.kl_ctrl.kl_coef=$KL_COEF \
    algorithm.norm_adv_by_std_in_grpo=False \
    algorithm.filter_groups.enable=$FILTER_GROUPS_ENABLE \
    algorithm.filter_groups.metric=$FILTER_GROUPS_METRIC \
    algorithm.filter_groups.max_num_gen_batches=$FILTER_GROUPS_MAX_GENS \
    actor_rollout_ref.actor.loss_agg_mode=token-mean \
    actor_rollout_ref.actor.clip_ratio_low=0.2 \
    actor_rollout_ref.actor.clip_ratio_high=0.28 \
    actor_rollout_ref.actor.clip_ratio_c=10.0 \
    data.train_files="$TRAIN_FILES" \
    data.val_files="$VAL_FILES" \
    data.train_batch_size=$TRAIN_BATCH \
    data.gen_batch_size=$GEN_BATCH \
    data.return_raw_chat=True \
    data.max_prompt_length=$MAX_PROMPT_LEN \
    data.max_response_length=$MAX_RESPONSE_LEN \
    data.filter_overlong_prompts=True \
    data.truncation=error \
    ++data.apply_chat_template_kwargs.enable_thinking=true \
    actor_rollout_ref.model.path="$MODEL" \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.optim.lr=$LR \
    actor_rollout_ref.actor.optim.lr_scheduler_type=$LR_SCHEDULER_TYPE \
    actor_rollout_ref.actor.optim.lr_warmup_steps=$LR_WARMUP_STEPS \
    actor_rollout_ref.actor.optim.weight_decay=0.1 \
    actor_rollout_ref.actor.ppo_mini_batch_size=$PPO_MINI_BATCH \
    actor_rollout_ref.actor.use_dynamic_bsz=$DYN_BSZ_FLAG \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=$PPO_MAX_TOKEN_LEN_PER_GPU \
    actor_rollout_ref.actor.ppo_epochs=1 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.kl_loss_coef=0.0 \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.grad_clip=1.0 \
    actor_rollout_ref.actor.fsdp_config.strategy=fsdp \
    actor_rollout_ref.actor.fsdp_config.param_offload=$ACTOR_PARAM_OFFLOAD \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=$ACTOR_OPT_OFFLOAD \
    actor_rollout_ref.actor.fsdp_config.model_dtype=bfloat16 \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=$TRAIN_SP \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.ref.fsdp_config.model_dtype=bfloat16 \
    actor_rollout_ref.ref.ulysses_sequence_parallel_size=$TRAIN_SP \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=$DYN_BSZ_FLAG \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=$((PPO_MAX_TOKEN_LEN_PER_GPU * 4)) \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=$DYN_BSZ_FLAG \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=$((PPO_MAX_TOKEN_LEN_PER_GPU * 4)) \
    actor_rollout_ref.rollout.max_num_batched_tokens=$((MAX_PROMPT_LEN + MAX_RESPONSE_LEN)) \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.mode=async \
    actor_rollout_ref.rollout.n=$ROLLOUT_N \
    actor_rollout_ref.rollout.temperature=${TRAIN_TEMPERATURE:-1.4} \
    actor_rollout_ref.rollout.val_kwargs.temperature=${VAL_TEMPERATURE:-1.0} \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.rollout.val_kwargs.n=1 \
    actor_rollout_ref.rollout.top_p=1.0 \
    actor_rollout_ref.rollout.gpu_memory_utilization=$VLLM_GPU_MEM_UTIL \
    actor_rollout_ref.rollout.max_model_len=$((MAX_PROMPT_LEN + MAX_RESPONSE_LEN)) \
    actor_rollout_ref.rollout.load_format=safetensors \
    actor_rollout_ref.rollout.enable_prefix_caching=True \
    actor_rollout_ref.rollout.enable_chunked_prefill=True \
    actor_rollout_ref.rollout.enforce_eager=True \
    ++actor_rollout_ref.rollout.engine_kwargs.vllm.disable_custom_all_reduce=True \
    actor_rollout_ref.rollout.tensor_model_parallel_size=$N_GPUS_TRAIN \
    actor_rollout_ref.rollout.calculate_log_probs=True \
    actor_rollout_ref.rollout.agent.default_agent_loop=$AGENT_LOOP \
    actor_rollout_ref.rollout.multi_turn.enable=$MULTI_TURN_ENABLE \
    actor_rollout_ref.rollout.multi_turn.format=hermes \
    actor_rollout_ref.rollout.multi_turn.tool_config_path="$ROOT/scripts/physcode_tools.yaml" \
    actor_rollout_ref.rollout.multi_turn.max_assistant_turns=$((MAX_TOOL_TURNS * 2)) \
    actor_rollout_ref.rollout.multi_turn.max_user_turns=$MAX_TOOL_TURNS \
    actor_rollout_ref.rollout.multi_turn.max_parallel_calls=1 \
    actor_rollout_ref.rollout.multi_turn.max_tool_response_length=$MAX_TOOL_RESPONSE_LEN \
    actor_rollout_ref.rollout.multi_turn.tool_response_truncate_side=right \
    ++actor_rollout_ref.rollout.multi_turn.thinking_budget=${THINKING_BUDGET:-null} \
    ++actor_rollout_ref.rollout.multi_turn.tool_call_budget=${TOOL_CALL_BUDGET:-null} \
    reward.custom_reward_function.path="$ROOT/src/phys_reasoner/training/reward.py" \
    reward.custom_reward_function.name=compute_score \
    ++reward.reward_kwargs.overlong_buffer_cfg.enable=False \
    ++reward.reward_kwargs.overlong_buffer_cfg.len=0 \
    ++reward.reward_kwargs.overlong_buffer_cfg.penalty_factor=0.0 \
    ++reward.reward_kwargs.overlong_buffer_cfg.log=False \
    ++reward.reward_kwargs.max_resp_len=$MAX_RESPONSE_LEN \
    trainer.critic_warmup=0 \
    trainer.nnodes=$NNODES \
    trainer.n_gpus_per_node=$N_GPUS_TRAIN \
    trainer.total_epochs=$TOTAL_EPOCHS \
    trainer.total_training_steps=$TOTAL_STEPS \
    trainer.save_freq=$SAVE_FREQ \
    trainer.test_freq=$TEST_FREQ \
    trainer.val_before_train=${VAL_BEFORE_TRAIN:-false} \
    trainer.project_name="$WANDB_PROJECT" \
    trainer.experiment_name="$EXPERIMENT" \
    trainer.default_local_dir="$TRAIN_DIR" \
    trainer.default_hdfs_dir=null \
    trainer.log_val_generations=${LOG_VAL_GENERATIONS:-8} \
    "trainer.logger=[$(echo "$LOGGERS" | sed 's/,/","/g; s/^/"/; s/$/"/')]" \
    2>&1 | tee "$TRAIN_DIR/train.log"

echo "Exit: $?"
