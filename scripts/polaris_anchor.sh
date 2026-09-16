#!/usr/bin/env bash
# polaris_anchor.sh — K anchor: POLARIS stage1 recipe replicated on OUR verl fork
# for Qwen3-4B + POLARIS-Dataset-53K. Sync colocate (hybrid engine), single node.
#
# Goal: prove the pipeline can learn a known-working Qwen3-4B recipe when we
# strip away our stack's async/TP/SP/TIR complications. If this does NOT climb,
# the problem is in verl itself or our environment. If it DOES climb, we have
# a baseline to swap atoms against.
#
# Deltas from POLARIS stage1.sh (unavoidable due to our env):
#   - response_length: POLARIS uses 39936. We use 8192 (A100-40G + debug/interactive budget).
#   - batch: POLARIS uses 128 on 8 H800-80G. We use 64 on 4 A100-40G.
#   - dyn_sampling_polaris → filter_groups.enable=True (our verl's DAPO-style
#     replacement; does the same thing: drop saturated groups and regenerate).
#
# Everything else (LR, temp, rollout.n, clip ratios, kl coefs) matches POLARIS.

set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
unset SIF OVERLAY
source "$ROOT/env.sh"
echo "  SIF:     $SIF"
echo "  OVERLAY: $OVERLAY"

MODEL="${MODEL:-Qwen/Qwen3-4B}"
# POLARIS-Dataset-53K stage1 parquet — local copy inside phys-reasoner so apptainer
# binds it cleanly. (Original is in references/POLARIS/parquet/stage1/ outside ROOT,
# which apptainer on Perlmutter doesn't auto-bind.)
TRAIN_FILES="${TRAIN_FILES:-$ROOT/data/polaris/qwen3-4b-s1.parquet}"
VAL_FILES="${VAL_FILES:-$TRAIN_FILES}"

# 1 node × 4 A100-40G on Perlmutter. POLARIS uses 8 H800-80G; we halve the scale.
N_GPUS="${N_GPUS:-4}"
NNODES="${NNODES:-1}"

# Scale-halved from POLARIS (128→64, 40K→8K) to fit 4×A100-40G within the interactive
# walltime. Shrinks batch gradient bag by 2x, so slightly noisier but preserves
# the climbing signal.
TRAIN_BATCH="${TRAIN_BATCH:-64}"
ROLLOUT_N="${ROLLOUT_N:-8}"
MAX_PROMPT_LEN="${MAX_PROMPT_LEN:-1024}"
MAX_RESPONSE_LEN="${MAX_RESPONSE_LEN:-8192}"

# Steps: 20 is enough to see a monotone climb for Qwen3-4B on POLARIS stage1
# (reward climbs within first 10 steps in POLARIS's published logs).
TOTAL_STEPS="${TOTAL_STEPS:-20}"
SAVE_FREQ="${SAVE_FREQ:--1}"
TEST_FREQ="${TEST_FREQ:--1}"

# POLARIS stage1 hparams (verbatim where compatible)
LR="${LR:-1e-6}"
TEMP="${TEMP:-1.4}"
CLIP_LOW="${CLIP_LOW:-0.2}"
CLIP_HIGH="${CLIP_HIGH:-0.28}"
KL_COEF_CTRL="${KL_COEF_CTRL:-0.001}"

# POLARIS uses TP=1 (one vLLM engine per GPU), SP=1 (no Ulysses).
# Four GPUs on one node gives us 4 parallel engines.
ROLLOUT_TP="${ROLLOUT_TP:-1}"
TRAIN_SP="${TRAIN_SP:-1}"

# Memory on A100-40G with sync colocate (hybrid engine) is tight:
VLLM_GPU_MEM_UTIL="${VLLM_GPU_MEM_UTIL:-0.5}"
USE_DYNAMIC_BSZ="${USE_DYNAMIC_BSZ:-1}"
if [[ "$USE_DYNAMIC_BSZ" == "1" ]]; then DYN_BSZ_FLAG=True; else DYN_BSZ_FLAG=False; fi
PPO_MAX_TOKEN_LEN_PER_GPU="${PPO_MAX_TOKEN_LEN_PER_GPU:-10240}"

# POLARIS offloads both actor param and optimizer to CPU. We offload optimizer
# but keep param on GPU (4B bf16 FSDP-sharded is only ~2GB/GPU).
ACTOR_PARAM_OFFLOAD="${ACTOR_PARAM_OFFLOAD:-False}"
ACTOR_OPT_OFFLOAD="${ACTOR_OPT_OFFLOAD:-True}"

# Reward function: default = verl built-in math scorer (what POLARIS uses on their
# parquet). Set CUSTOM_REWARD_FN_PATH + CUSTOM_REWARD_FN_NAME to use our physics
# reward (xverify-backed compute_score). Both must be set together.
CUSTOM_REWARD_FN_PATH="${CUSTOM_REWARD_FN_PATH:-}"
CUSTOM_REWARD_FN_NAME="${CUSTOM_REWARD_FN_NAME:-}"

WANDB_PROJECT="${WANDB_PROJECT:-physcode_polaris_anchor}"
LOGGERS="${LOGGERS:-console,wandb}"
TIMESTAMP=$(date +%Y%m%d.%H%M%S)
EXPERIMENT="${EXPERIMENT:-polaris_anchor_qwen3b_${TIMESTAMP}}"
TRAIN_DIR="$ROOT/outputs/$WANDB_PROJECT/$EXPERIMENT"
mkdir -p "$TRAIN_DIR" "$ROOT/logs"

ROLLOUT_MAX_NUM_SEQS=$((TRAIN_BATCH * ROLLOUT_N))

echo "=== polaris_anchor.sh: K anchor (POLARIS stage1 recipe, sync colocate) ==="
echo "  model       : $MODEL"
echo "  train       : $TRAIN_FILES"
echo "  val         : $VAL_FILES"
echo "  nodes/GPUs  : ${NNODES}n × ${N_GPUS}g"
echo "  batch       : $TRAIN_BATCH × n=$ROLLOUT_N = $ROLLOUT_MAX_NUM_SEQS gens/step"
echo "  seq lens    : prompt=$MAX_PROMPT_LEN  response=$MAX_RESPONSE_LEN"
echo "  TP/SP       : ROLLOUT_TP=$ROLLOUT_TP  TRAIN_SP=$TRAIN_SP"
echo "  LR / temp   : $LR / $TEMP"
echo "  steps       : $TOTAL_STEPS"
echo "  vLLM mem    : $VLLM_GPU_MEM_UTIL"
echo "  output      : $TRAIN_DIR"

if [[ "${PYTORCH_CUDA_ALLOC_CONF:-}" == *"expandable_segments:True"* ]]; then
    unset PYTORCH_CUDA_ALLOC_CONF
fi
if [[ -n "${ROCR_VISIBLE_DEVICES:-}" ]]; then
    unset ROCR_VISIBLE_DEVICES
fi

PYTHONNOUSERSITE=1 apptainer exec --nv \
  --overlay "$OVERLAY:ro" \
  --no-home \
  --pwd "$ROOT/verl" \
  --bind /etc/pki:/etc/pki \
  --env "PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin" \
  --env "PYTHONNOUSERSITE=1" \
  --env "PYTHONUNBUFFERED=1" \
  --env "PYTHONPATH=/opt/phys-extras/" \
  --env "XDG_CACHE_HOME=/tmp/.cache" \
  --env "FLASHINFER_WORKSPACE_BASE=/tmp" \
  --env "HOME=/tmp" \
  --env "TRITON_CACHE_DIR=/tmp/.cache/triton" \
  --env "MPLCONFIGDIR=/tmp/.cache/matplotlib" \
  --env "HF_HOME=$HF_HOME" \
  --env "VLLM_USE_V1=1" \
  --env "WANDB_API_KEY=${WANDB_API_KEY:-}" \
  --env "WANDB_PROJECT=$WANDB_PROJECT" \
  --env "WANDB_RUN_ID=$EXPERIMENT" \
  --env "XVERIFY_URL=${XVERIFY_URL:-}" \
  --env "PHYS_REQUIRE_XVERIFY=${PHYS_REQUIRE_XVERIFY:-0}" \
  "$SIF" \
  python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    algorithm.use_kl_in_reward=False \
    algorithm.kl_ctrl.kl_coef=$KL_COEF_CTRL \
    algorithm.norm_adv_by_std_in_grpo=True \
    ++algorithm.filter_groups.enable=True \
    ++algorithm.filter_groups.metric=acc \
    ++algorithm.filter_groups.max_num_gen_batches=10 \
    actor_rollout_ref.actor.loss_agg_mode=token-mean \
    actor_rollout_ref.actor.clip_ratio_low=$CLIP_LOW \
    actor_rollout_ref.actor.clip_ratio_high=$CLIP_HIGH \
    data.train_files="$TRAIN_FILES" \
    data.val_files="$VAL_FILES" \
    data.train_batch_size=$TRAIN_BATCH \
    data.max_prompt_length=$MAX_PROMPT_LEN \
    data.max_response_length=$MAX_RESPONSE_LEN \
    data.filter_overlong_prompts=True \
    data.truncation=error \
    data.dataloader_num_workers=2 \
    +data.apply_chat_template_kwargs.enable_thinking=true \
    actor_rollout_ref.model.path="$MODEL" \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.optim.lr=$LR \
    actor_rollout_ref.actor.optim.lr_scheduler_type=constant \
    actor_rollout_ref.actor.optim.lr_warmup_steps=10 \
    actor_rollout_ref.actor.ppo_mini_batch_size=$TRAIN_BATCH \
    actor_rollout_ref.actor.use_dynamic_bsz=$DYN_BSZ_FLAG \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=$PPO_MAX_TOKEN_LEN_PER_GPU \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.ppo_epochs=1 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.kl_loss_coef=0.0 \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.fsdp_config.strategy=fsdp \
    actor_rollout_ref.actor.fsdp_config.param_offload=$ACTOR_PARAM_OFFLOAD \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=$ACTOR_OPT_OFFLOAD \
    actor_rollout_ref.actor.fsdp_config.model_dtype=bfloat16 \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=$TRAIN_SP \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.ref.fsdp_config.model_dtype=bfloat16 \
    actor_rollout_ref.ref.ulysses_sequence_parallel_size=$TRAIN_SP \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=$DYN_BSZ_FLAG \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=$((PPO_MAX_TOKEN_LEN_PER_GPU * 2)) \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=$DYN_BSZ_FLAG \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.n=$ROLLOUT_N \
    actor_rollout_ref.rollout.temperature=$TEMP \
    actor_rollout_ref.rollout.top_p=1.0 \
    actor_rollout_ref.rollout.top_k=-1 \
    actor_rollout_ref.rollout.gpu_memory_utilization=$VLLM_GPU_MEM_UTIL \
    actor_rollout_ref.rollout.max_model_len=$((MAX_PROMPT_LEN + MAX_RESPONSE_LEN)) \
    actor_rollout_ref.rollout.max_num_seqs=$ROLLOUT_MAX_NUM_SEQS \
    actor_rollout_ref.rollout.max_num_batched_tokens=$((MAX_PROMPT_LEN + MAX_RESPONSE_LEN)) \
    actor_rollout_ref.rollout.load_format=safetensors \
    actor_rollout_ref.rollout.enable_prefix_caching=True \
    actor_rollout_ref.rollout.enable_chunked_prefill=True \
    actor_rollout_ref.rollout.enforce_eager=False \
    +actor_rollout_ref.rollout.engine_kwargs.vllm.disable_custom_all_reduce=True \
    actor_rollout_ref.rollout.tensor_model_parallel_size=$ROLLOUT_TP \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.agent.default_agent_loop=single_turn_agent \
    actor_rollout_ref.rollout.multi_turn.enable=false \
    +actor_rollout_ref.rollout.multi_turn.thinking_budget=null \
    +actor_rollout_ref.rollout.multi_turn.tool_call_budget=null \
    ${CUSTOM_REWARD_FN_PATH:+reward.custom_reward_function.path="$CUSTOM_REWARD_FN_PATH"} \
    ${CUSTOM_REWARD_FN_NAME:+reward.custom_reward_function.name=$CUSTOM_REWARD_FN_NAME} \
    trainer.critic_warmup=0 \
    trainer.nnodes=$NNODES \
    trainer.n_gpus_per_node=$N_GPUS \
    trainer.total_training_steps=$TOTAL_STEPS \
    trainer.total_epochs=10 \
    trainer.save_freq=$SAVE_FREQ \
    trainer.test_freq=$TEST_FREQ \
    trainer.val_before_train=false \
    trainer.project_name="$WANDB_PROJECT" \
    trainer.experiment_name="$EXPERIMENT" \
    trainer.default_local_dir="$TRAIN_DIR" \
    trainer.default_hdfs_dir=null \
    "trainer.logger=[$(echo "$LOGGERS" | sed 's/,/","/g; s/^/"/; s/$/"/')]" \
    2>&1 | tee "$TRAIN_DIR/train.log"

echo "Exit: $?"
