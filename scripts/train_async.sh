#!/usr/bin/env bash
# train_async.sh — PhysCode GRPO training with VeRL fully_async_policy (disaggregated
# rollout + trainer on separate GPU pools).
#
# This is the async twin of scripts/train.sh. Use it when rollout and trainer
# should NOT share GPU memory (the colocate path OOMs on small GPUs because the
# vLLM KV cache, FSDP shards, ref copy, and AdamW state all land on the same GPU).
#
# Resource model
#   - `trainer.nnodes` / `trainer.n_gpus_per_node`   → trainer pool (owns FSDP actor + ref)
#   - `rollout.nnodes`  / `rollout.n_gpus_per_node`  → rollouter pool (owns vLLM engine)
#   Pools must be disjoint; fully_async_policy allocates them as separate Ray placement groups.
#
# Schedule (streaming, on-policy by default)
#   total_rollout_steps = TOTAL_STEPS * TRAIN_BATCH       # total prompts rolled out
#   trigger_parameter_sync_step = 1                        # sync every train step
#   require_batches              = 1                        # fetch 1 mini-batch per step
#   staleness_threshold          = 0                        # on-policy (no stale samples)
#   partial_rollout              = False                    # simplest path for debugging
#
# This gives "resource-separated but synchronous" behaviour: identical to
# colocate-sync in terms of what gets trained, but rollout and train run on
# disjoint GPUs so the trainer is no longer starved of memory by the vLLM KV cache.
# Streaming/staleness/partial_rollout can be enabled later for throughput.
#
# Usage
#   bash scripts/train_async.sh                                # uses defaults
#   TOTAL_STEPS=2 TRAIN_BATCH=4 ROLLOUT_N=4 bash scripts/train_async.sh
#   N_GPUS_ROLLOUT=4 N_GPUS_TRAIN=4 bash scripts/train_async.sh  # multi-GPU single node
#
# For smoke testing the entire pipeline end-to-end, prefer scripts/train_smoke_async.sh
# which sets small budgets, small batch, and re-stamps the smoke parquet.

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
# Defaults point at the merged parquets fetched from the HF dataset
# `xingzhi0/phys-tir` via scripts/fetch_dataset.py (docs/setup.md step 5).
# Each split is a union of the curated + Dr. SCI pools; rows carry a `pool`
# column for provenance. Smoke / rehearsal sbatch scripts override
# TRAIN_FILES to data/processed/probe_subset.parquet for speed.
TRAIN_FILES="${TRAIN_FILES:-$ROOT/data/processed_hf/data/train.parquet}"
VAL_FILES="${VAL_FILES:-$ROOT/data/processed_hf/data/validation.parquet}"

# Resource split: rollout and trainer live on disjoint GPUs on this node.
N_GPUS_ROLLOUT="${N_GPUS_ROLLOUT:-1}"
N_GPUS_TRAIN="${N_GPUS_TRAIN:-1}"
NNODES_ROLLOUT="${NNODES_ROLLOUT:-1}"
NNODES_TRAIN="${NNODES_TRAIN:-1}"

TRAIN_BATCH="${TRAIN_BATCH:-4}"        # == ppo_mini_batch_size (global across trainer GPUs)
ROLLOUT_N="${ROLLOUT_N:-4}"            # GRPO group size
MAX_PROMPT_LEN="${MAX_PROMPT_LEN:-1024}"
MAX_TOOL_TURNS="${MAX_TOOL_TURNS:-1}"

# Think-interrupt budget. Same formula and defaults semantics as train.sh.
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

# Schedule. TOTAL_STEPS is the number of local trainer updates (each consumes
# `require_batches * ppo_mini_batch_size` prompts). total_rollout_steps is the
# total number of rollouts produced by the Rollouter across the whole run.
TOTAL_STEPS="${TOTAL_STEPS:-2}"
TOTAL_ROLLOUT_STEPS=$((TOTAL_STEPS * TRAIN_BATCH))
SAVE_FREQ="${SAVE_FREQ:--1}"            # -1 = never save during smoke runs
TEST_FREQ="${TEST_FREQ:--1}"            # -1 = never validate during smoke runs

LR="${LR:-1e-6}"
# Rollout GPU is dedicated in async mode → higher vLLM mem util is safe.
VLLM_GPU_MEM_UTIL="${VLLM_GPU_MEM_UTIL:-0.8}"

# Rollout vLLM enforce_eager — disables CUDA graph capture in the rollout engine.
# Default False (CUDA graphs on) — A100 40G confirmed clean with Qwen3.5
# 0.8B end-to-end async smoke (2026-04-14). Recovers ~5-15% decode throughput
# vs enforce_eager=True. Override to True only if you see output degeneration
# (!!!) or hangs — note that the Roberts H200 sample_tokens hang we debugged
# on 2026-04-14 was NOT fixed by enforce_eager=True, so the dump_rollouts.py:300
# comment about GDN + CUDA graphs is not causal for that hang (likely a
# Hopper-specific vLLM v1 multiproc-executor issue).
ROLLOUT_ENFORCE_EAGER="${ROLLOUT_ENFORCE_EAGER:-False}"

# Wandb
WANDB_PROJECT="${WANDB_PROJECT:-physcode_tir}"
# Comma-separated list of loggers. Default to console-only to avoid needing
# WANDB_API_KEY or ~/.netrc access (which --no-home blocks). Set
# LOGGERS=console,wandb and export WANDB_API_KEY for real runs.
LOGGERS="${LOGGERS:-console}"

# Rollout inspection (for spot-checking generation quality, especially the
# Qwen3.5-4B batch-size degeneration bug — see docs/training-decisions.md):
#   LOG_VAL_GENERATIONS=N — N samples logged to wandb as a Table per validation step
#                          (tied to trainer.test_freq, so already infrequent; default 8)
#   DUMP_VAL_ROLLOUTS=1   — dump FULL validation batch to JSONL per test_freq step
#                          (low overhead, only runs at validation)
#   DUMP_TRAIN_ROLLOUTS=1 — dump FULL training batch to JSONL EVERY training step
#                          (~50-100 MB/step for TRAIN_BATCH=256 × n=8; expensive,
#                          only enable for debugging sessions)
DUMP_VAL_ROLLOUTS="${DUMP_VAL_ROLLOUTS:-1}"
DUMP_TRAIN_ROLLOUTS="${DUMP_TRAIN_ROLLOUTS:-}"

# ---------------------------------------------------------------------------
# Derived settings
# ---------------------------------------------------------------------------
ROLLOUT_MAX_NUM_SEQS=$((TRAIN_BATCH * ROLLOUT_N))

# Qwen3.5-4B + vLLM corrupts outputs on B200 (Blackwell SM_100) due to the
# auto-detected TRTLLM prefill attention kernel. See docs/training-decisions.md.
# Fix: set VLLM_USE_TRTLLM_ATTENTION=0 or avoid B200. A100/H100/H200 are unaffected.
if [[ "${SLURM_JOB_PARTITION:-}" == gpu_b200* ]] || [[ "${SBATCH_PARTITION:-}" == gpu_b200* ]]; then
    if [ -z "${VLLM_USE_TRTLLM_ATTENTION:-}" ]; then
        echo "WARNING: running on gpu_b200 without VLLM_USE_TRTLLM_ATTENTION=0." >&2
        echo "  Qwen3.5-4B rollouts will degenerate into !!! repetition on B200." >&2
        echo "  Set VLLM_USE_TRTLLM_ATTENTION=0 or use gpu_h200/gpu partition." >&2
        echo "  See docs/training-decisions.md." >&2
    fi
fi
PPO_MINI_BATCH=$TRAIN_BATCH
# Per-GPU micro batch on the trainer side. Default to 1 on the async path
# because the trainer GPU is typically the memory-constrained one in smoke runs.
PPO_MICRO_BS_PER_GPU="${PPO_MICRO_BS_PER_GPU:-1}"
LOG_PROB_MICRO_BS_PER_GPU="${LOG_PROB_MICRO_BS_PER_GPU:-$PPO_MICRO_BS_PER_GPU}"

TIMESTAMP=$(date +%Y%m%d.%H%M%S)
# EXPERIMENT can be pinned via env so resume runs land in the same default_local_dir.
# verl's trainer.resume_mode defaults to 'auto' — if a checkpoint already exists in
# default_local_dir, training resumes from it automatically.
EXPERIMENT="${EXPERIMENT:-grpo_async_${TIMESTAMP}}"
TRAIN_DIR="$ROOT/outputs/$WANDB_PROJECT/$EXPERIMENT"
mkdir -p "$TRAIN_DIR" "$ROOT/logs"

echo "=== train_async.sh: PhysCode GRPO TIR training (fully_async_policy) ==="
echo "  model       : $MODEL"
echo "  train       : $TRAIN_FILES"
echo "  val         : $VAL_FILES"
echo "  resource    : rollout ${NNODES_ROLLOUT}n x ${N_GPUS_ROLLOUT}g | train ${NNODES_TRAIN}n x ${N_GPUS_TRAIN}g"
echo "  batch       : ppo_mini=$PPO_MINI_BATCH, rollout_n=$ROLLOUT_N, total_rollout_steps=$TOTAL_ROLLOUT_STEPS"
echo "  seq lens    : prompt=$MAX_PROMPT_LEN  response=$MAX_RESPONSE_LEN"
echo "  vLLM mem    : $VLLM_GPU_MEM_UTIL"
echo "  output      : $TRAIN_DIR"
echo "  wandb       : $WANDB_PROJECT / $EXPERIMENT"

# Guards (same as train.sh).
if [[ "${PYTORCH_CUDA_ALLOC_CONF:-}" == *"expandable_segments:True"* ]]; then
    echo "WARNING: unsetting PYTORCH_CUDA_ALLOC_CONF (expandable_segments:True conflicts with vLLM)"
    unset PYTORCH_CUDA_ALLOC_CONF
fi
if [[ -n "${ROCR_VISIBLE_DEVICES:-}" ]]; then
    echo "WARNING: unsetting ROCR_VISIBLE_DEVICES (conflicts with CUDA_VISIBLE_DEVICES in VeRL)"
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
  --env "PYTHONPATH=$ROOT/.async-extras:/opt/phys-extras/" \
  --env "XDG_CACHE_HOME=/tmp/.cache" \
  --env "FLASHINFER_WORKSPACE_BASE=/tmp" \
  --env "TRITON_CACHE_DIR=/tmp/.cache/triton" \
  --env "MPLCONFIGDIR=/tmp/.cache/matplotlib" \
  --env "HF_HOME=$HF_HOME" \
  --env "HF_DATASETS_OFFLINE=0" \
  --env "VLLM_USE_V1=1" \
  --env "WANDB_API_KEY=${WANDB_API_KEY:-}" \
  --env "WANDB_PROJECT=$WANDB_PROJECT" \
  --env "WANDB_RUN_ID=$EXPERIMENT" \
  --env "VERL_DUMP_DIR=${VERL_DUMP_DIR:-}" \
  --env "XVERIFY_URL=${XVERIFY_URL:-}" \
  --env "PHYS_REQUIRE_XVERIFY=${PHYS_REQUIRE_XVERIFY:-0}" \
  --env "RAY_ADDRESS=${RAY_ADDRESS:-}" \
  --env "VERIFIER_DUMP_PATH=${VERIFIER_DUMP_PATH:-}" \
  ${VLLM_USE_TRTLLM_ATTENTION:+--env "VLLM_USE_TRTLLM_ATTENTION=$VLLM_USE_TRTLLM_ATTENTION"} \
  "$SIF" \
  python3 -m verl.experimental.fully_async_policy.fully_async_main \
    algorithm.adv_estimator=grpo \
    algorithm.use_kl_in_reward=False \
    algorithm.kl_ctrl.kl_coef=0.0 \
    algorithm.norm_adv_by_std_in_grpo=False \
    actor_rollout_ref.actor.loss_agg_mode=token-mean \
    actor_rollout_ref.actor.clip_ratio_low=0.2 \
    actor_rollout_ref.actor.clip_ratio_high=0.28 \
    actor_rollout_ref.actor.clip_ratio_c=10.0 \
    data.train_files="$TRAIN_FILES" \
    data.val_files="$VAL_FILES" \
    data.train_batch_size=0 \
    data.gen_batch_size=1 \
    data.return_raw_chat=True \
    data.max_prompt_length=$MAX_PROMPT_LEN \
    data.max_response_length=$MAX_RESPONSE_LEN \
    data.filter_overlong_prompts=True \
    data.truncation=error \
    data.dataloader_num_workers=2 \
    +data.apply_chat_template_kwargs.enable_thinking=true \
    actor_rollout_ref.hybrid_engine=False \
    actor_rollout_ref.model.path="$MODEL" \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    '+actor_rollout_ref.model.override_config={attn_implementation:sdpa}' \
    actor_rollout_ref.actor.optim.lr=$LR \
    actor_rollout_ref.actor.ppo_mini_batch_size=$PPO_MINI_BATCH \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=$PPO_MICRO_BS_PER_GPU \
    actor_rollout_ref.actor.ppo_epochs=1 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.kl_loss_coef=0.0 \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.use_rollout_log_probs=True \
    actor_rollout_ref.actor.fsdp_config.strategy=fsdp \
    actor_rollout_ref.actor.fsdp_config.param_offload=${ACTOR_PARAM_OFFLOAD:-False} \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=${ACTOR_OPT_OFFLOAD:-False} \
    actor_rollout_ref.actor.fsdp_config.model_dtype=bfloat16 \
    '+actor_rollout_ref.actor.fsdp_config.wrap_policy.transformer_layer_cls_to_wrap=[Qwen3_5DecoderLayer]' \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.ref.fsdp_config.model_dtype=bfloat16 \
    '+actor_rollout_ref.ref.fsdp_config.wrap_policy.transformer_layer_cls_to_wrap=[Qwen3_5DecoderLayer]' \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.mode=async \
    actor_rollout_ref.rollout.n=$ROLLOUT_N \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.top_p=0.9 \
    actor_rollout_ref.rollout.gpu_memory_utilization=$VLLM_GPU_MEM_UTIL \
    actor_rollout_ref.rollout.max_model_len=$((MAX_PROMPT_LEN + MAX_RESPONSE_LEN)) \
    actor_rollout_ref.rollout.max_num_seqs=$ROLLOUT_MAX_NUM_SEQS \
    actor_rollout_ref.rollout.load_format=safetensors \
    actor_rollout_ref.rollout.enable_prefix_caching=True \
    actor_rollout_ref.rollout.enable_chunked_prefill=True \
    actor_rollout_ref.rollout.enforce_eager=$ROLLOUT_ENFORCE_EAGER \
    actor_rollout_ref.rollout.tensor_model_parallel_size=$N_GPUS_ROLLOUT \
    actor_rollout_ref.rollout.calculate_log_probs=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=$LOG_PROB_MICRO_BS_PER_GPU \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=$LOG_PROB_MICRO_BS_PER_GPU \
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
    algorithm.rollout_correction.bypass_mode=True \
    async_training.trigger_parameter_sync_step=1 \
    async_training.require_batches=1 \
    async_training.staleness_threshold=0 \
    async_training.partial_rollout=False \
    async_training.use_trainer_do_validate=False \
    reward.custom_reward_function.path="$ROOT/src/phys_reasoner/training/reward.py" \
    reward.custom_reward_function.name=compute_score \
    trainer.critic_warmup=0 \
    trainer.nnodes=$NNODES_TRAIN \
    trainer.n_gpus_per_node=$N_GPUS_TRAIN \
    rollout.nnodes=$NNODES_ROLLOUT \
    rollout.n_gpus_per_node=$N_GPUS_ROLLOUT \
    rollout.total_rollout_steps=$TOTAL_ROLLOUT_STEPS \
    trainer.total_epochs=1 \
    trainer.save_freq=$SAVE_FREQ \
    trainer.test_freq=$TEST_FREQ \
    trainer.val_before_train=false \
    trainer.project_name="$WANDB_PROJECT" \
    trainer.experiment_name="$EXPERIMENT" \
    trainer.default_local_dir="$TRAIN_DIR" \
    trainer.default_hdfs_dir=null \
    trainer.log_val_generations=${LOG_VAL_GENERATIONS:-8} \
    ${DUMP_TRAIN_ROLLOUTS:++trainer.rollout_data_dir=$TRAIN_DIR/rollout_dumps} \
    ${DUMP_VAL_ROLLOUTS:++trainer.validation_data_dir=$TRAIN_DIR/validation_dumps} \
    "trainer.logger=[$(echo "$LOGGERS" | sed 's/,/","/g; s/^/"/; s/$/"/')]" \
    2>&1 | tee "$TRAIN_DIR/train.log"

echo "Exit: $?"
