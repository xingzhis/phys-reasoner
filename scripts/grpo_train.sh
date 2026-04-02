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
VAL_FILES="${VAL_FILES:-$TRAIN_FILES}" # [TODO] this should not be the same file in production; use a separate val set or split from train.
ROLLOUT_DTYPE="${ROLLOUT_DTYPE:-bfloat16}" # [TODO] we will revisit dtype as recent literature recommended higher precision in some stage.
FSDP_DTYPE="${FSDP_DTYPE:-bf16}"
USE_EXPLICIT_DTYPES="${USE_EXPLICIT_DTYPES:-0}"
USE_FSDP2="${USE_FSDP2:-0}"
USE_PREFIX_CACHING="${USE_PREFIX_CACHING:-1}"
SMOKE_FAKE_DATA="${SMOKE_FAKE_DATA:-0}"
VAL_BEFORE_TRAIN="${VAL_BEFORE_TRAIN:-true}"
TEST_FREQ="${TEST_FREQ:--1}"
ROLLOUT_MAX_NUM_SEQS="${ROLLOUT_MAX_NUM_SEQS:-1024}"
ROLLOUT_TP_SIZE="${ROLLOUT_TP_SIZE:-1}"
ROLLOUT_MAX_MODEL_LEN="${ROLLOUT_MAX_MODEL_LEN:-}"
ROLLOUT_MAX_BATCHED_TOKENS="${ROLLOUT_MAX_BATCHED_TOKENS:-}"
ROLLOUT_ENFORCE_EAGER="${ROLLOUT_ENFORCE_EAGER:-false}"
ROLLOUT_ENABLE_CHUNKED_PREFILL="${ROLLOUT_ENABLE_CHUNKED_PREFILL:-true}"
ROLLOUT_LAYERED_SUMMON="${ROLLOUT_LAYERED_SUMMON:-false}"
ROLLOUT_LOAD_FORMAT="${ROLLOUT_LOAD_FORMAT:-safetensors}"
DATA_NUM_WORKERS="${DATA_NUM_WORKERS:-8}"
ACTOR_PARAM_OFFLOAD="${ACTOR_PARAM_OFFLOAD:-false}"
ACTOR_OPTIMIZER_OFFLOAD="${ACTOR_OPTIMIZER_OFFLOAD:-false}"
REF_PARAM_OFFLOAD="${REF_PARAM_OFFLOAD:-false}"
TINY_STEPS="${TINY_STEPS:-2}"
ENABLE_TIR="${ENABLE_TIR:-1}"
SMOKE_BASELINE="${SMOKE_BASELINE:-0}"

# Smoke test flag
# [TODO] should we remove the smoke content to keep it clean?
SMOKE="${1:-}"
if [ "$SMOKE" = "--smoke" ]; then
    TRAIN_BATCH=1
    MAX_PROMPT_LEN=256
    MAX_RESPONSE_LEN=96
    GPU_MEM_UTIL=0.10
    SMOKE_FAKE_DATA=1
    VAL_BEFORE_TRAIN=false
    TEST_FREQ=-1
    DATA_NUM_WORKERS=1
    ROLLOUT_MAX_NUM_SEQS=1
    ROLLOUT_MAX_MODEL_LEN=$((MAX_PROMPT_LEN + MAX_RESPONSE_LEN + 64))
    ROLLOUT_MAX_BATCHED_TOKENS=$ROLLOUT_MAX_MODEL_LEN
    ROLLOUT_ENFORCE_EAGER=true
    ROLLOUT_ENABLE_CHUNKED_PREFILL=false
    ROLLOUT_LAYERED_SUMMON=false
    ROLLOUT_LOAD_FORMAT=safetensors
    ACTOR_PARAM_OFFLOAD=true
    ACTOR_OPTIMIZER_OFFLOAD=true
    REF_PARAM_OFFLOAD=true
    EXTRA_ARGS="trainer.total_training_steps=$TINY_STEPS"
    EXPERIMENT_SUFFIX="_smoke"
    echo "=== SMOKE TEST: tiny settings for end-to-end pipeline bring-up ==="

    if [ "$SMOKE_BASELINE" = "1" ]; then
        ENABLE_TIR=0
        ROLLOUT_TEMP=1.0
        ROLLOUT_TOP_P=0.9
        ROLLOUT_TOP_K=-1
        GPU_MEM_UTIL=0.38
        ROLLOUT_MAX_NUM_SEQS=8
        ROLLOUT_MAX_MODEL_LEN=""
        ROLLOUT_MAX_BATCHED_TOKENS=""
        ROLLOUT_ENFORCE_EAGER=false
        ROLLOUT_ENABLE_CHUNKED_PREFILL=true
        ROLLOUT_LOAD_FORMAT=safetensors  # use actual weights like crystal's 'auto'; dummy can crash silently during FSDP→vLLM weight sync
        ACTOR_PARAM_OFFLOAD=false
        ACTOR_OPTIMIZER_OFFLOAD=false
        REF_PARAM_OFFLOAD=true
        # rollout.n=8 matches crystal; n=1 gives all-zero GRPO advantages (no learning signal)
        # filter_overlong_prompts=False: skip tokenizer-based length check (prompt is tiny; filter
        #   can silently drop all rows if tokenizer fails at data-load time before GPU is init'd)
        # truncation=right: don't error on overlong — just truncate (smoke prompt is short anyway)
        EXTRA_ARGS="trainer.total_training_steps=$TINY_STEPS actor_rollout_ref.rollout.n=8 data.filter_overlong_prompts=False data.truncation=right"
        echo "=== BASELINE MODE: crystal-like GRPO without TIR/tool agent ==="
    fi
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
echo "Prompt:     $MAX_PROMPT_LEN tokens"
echo "Response:   $MAX_RESPONSE_LEN tokens"
echo "Val file:   $VAL_FILES"
echo "Rollout:    dtype=$ROLLOUT_DTYPE mem=$GPU_MEM_UTIL seqs=$ROLLOUT_MAX_NUM_SEQS"
echo "FSDP:       dtype=$FSDP_DTYPE actor_offload=$ACTOR_PARAM_OFFLOAD ref_offload=$REF_PARAM_OFFLOAD"

# [TODO] should we remove the smoke content to keep it clean?
if [ "$SMOKE_FAKE_DATA" = "1" ]; then
    TINY_DATA_PATH="$TRAIN_DIR/tiny_smoke.parquet"
    echo "Writing tiny synthetic smoke dataset: $TINY_DATA_PATH"
    apptainer exec --nv \
      --overlay "$OVERLAY:ro" \
      --no-home \
      --bind /etc/pki:/etc/pki \
      --env "PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin" \
      --env "PYTHONNOUSERSITE=1" \
      "$SIF" \
      python3 - <<'PY' "$TINY_DATA_PATH" "$ENABLE_TIR"
import sys

import pandas as pd

path = sys.argv[1]
enable_tir = True
if len(sys.argv) > 2:
    enable_tir = sys.argv[2] == "1"

system_text = "Use the python tool once if needed. End with a boxed final answer."
user_text = "What is 2 + 2? Use the python tool once, then answer with \\\\boxed{}."
if not enable_tir:
    system_text = "Solve the problem and end with a boxed final answer."
    user_text = "What is 2 + 2? Answer with \\\\boxed{}."

row = {
    "prompt": [
        {
            "role": "system",
            "content": system_text,
        },
        {
            "role": "user",
            "content": user_text,
        },
    ],
    "data_source": "tiny_smoke",
    # reward_model must be a struct (dict column) — VeRL's naive reward manager
    # accesses data["reward_model"]["ground_truth"].
    "reward_model": {"ground_truth": "4", "style": "rule"},
    # answer_type, unit, tolerance go into extra_info; compute_score reads from there.
    "extra_info": {
        "index": 0,
        "answer_type": "numerical",
        "unit": "",
        "tolerance": 0.0,
        "problem": "What is 2 + 2?",
    },
}
pd.DataFrame([row]).to_parquet(path, index=False)
PY
    TRAIN_FILES="$TINY_DATA_PATH"
    VAL_FILES="$TINY_DATA_PATH"
fi

TIR_ARGS=""
if [ "$ENABLE_TIR" = "1" ]; then
    TIR_ARGS="
    actor_rollout_ref.rollout.agent.default_agent_loop=tool_agent \
    actor_rollout_ref.rollout.multi_turn.enable=true \
    actor_rollout_ref.rollout.multi_turn.format=qwen3_coder \
    actor_rollout_ref.rollout.multi_turn.tool_config_path=$ROOT/scripts/physcode_tools.yaml \
    actor_rollout_ref.rollout.multi_turn.max_assistant_turns=2 \
    actor_rollout_ref.rollout.multi_turn.max_user_turns=1 \
    actor_rollout_ref.rollout.multi_turn.max_parallel_calls=1 \
    actor_rollout_ref.rollout.multi_turn.max_tool_response_length=512 \
    actor_rollout_ref.rollout.multi_turn.tool_response_truncate_side=right \
    "
fi

FSDP_STRATEGY_ARGS=""
if [ "$USE_FSDP2" = "1" ]; then
    FSDP_STRATEGY_ARGS="
    actor_rollout_ref.actor.strategy=fsdp2 \
    actor_rollout_ref.ref.strategy=fsdp2 \
    "
fi

DTYPE_ARGS=""
if [ "$USE_EXPLICIT_DTYPES" = "1" ]; then
    DTYPE_ARGS="
    actor_rollout_ref.actor.fsdp_config.model_dtype=$FSDP_DTYPE \
    actor_rollout_ref.ref.fsdp_config.model_dtype=$FSDP_DTYPE \
    actor_rollout_ref.rollout.dtype=$ROLLOUT_DTYPE \
    "
fi

# ---------------------------------------------------------------------------
# Launch via standard VeRL entrypoint.
# Uses VeRL's built-in ToolAgentLoop ("tool_agent") with our PythonSandboxTool.
# Tool config: scripts/physcode_tools.yaml
#
# multi_turn parameter notes:
#   format=qwen3_coder   — Qwen3.5 tokenizer emits XML/coder tool-call format.
#                          DO NOT use format=hermes here; that is for Qwen3 (0.6B/1.7B/4B)
#                          which emits JSON format. Wrong format → silent parse failure,
#                          tool never runs, all rollouts score zero reward.
#                          Confirmed 2026-04-01. See prompts.py FORMAT docstring.
#   max_assistant_turns=2 — LLM generates exactly twice per trajectory:
#                            turn 1: reasoning + <tool_call>...</tool_call>
#                            turn 2: final answer with \boxed{} after tool response is injected.
#                            (The check fires AFTER each generation, so 1 would terminate
#                             before the tool ever runs — 2 is the correct value for single-call TIR.)
#   max_parallel_calls=1  — We expect exactly 1 tool call per turn; this caps parallel execution.
#                            No effect in practice since the model emits one call, but prevents
#                            runaway multi-call trajectories from consuming extra sandbox processes.
#   max_tool_response_length=512 — Truncate sandbox stdout to 512 tokens if it overflows.
#                                   Physics outputs are short (a few numbers); 512 is generous.
#   tool_response_truncate_side=right — Keep the start of stdout (the answer) if truncation needed.
#
# THINKING MODE: enable_thinking=True for phase 1 (tool-call turn).
# With thinking suppressed the tool-call format has no plain-content slot, so the
# model moves all reasoning into Python code comments. With thinking ON, it reasons
# in <think>...</think> then calls the tool with clean code. </think> closes before
# <tool_call> cleanly — no leakage observed (confirmed 2026-04-02, Qwen3.5-4B).
#
# TOKEN LENGTH NOTE: thinking traces on hard physics problems can be long. Planned
# mitigations: token-budget checkpoint in system prompt, stop-thinking injection at
# threshold, RL length penalty (λ>0) in late GRPO ablations. See prompts.py.
#
# VeRL knob: data.apply_chat_template_kwargs.enable_thinking=True (below).
# DO NOT add actor_rollout_ref.model.enable_thinking — not a valid field in this
# VeRL build, crashes worker init.
# ---------------------------------------------------------------------------
# Guard: vLLM CuMemAllocator is incompatible with expandable_segments:True.
# Unset the variable on the host before passing env to apptainer so that
# vLLM does not hang or crash during memory pool initialization.
if [[ "${PYTORCH_CUDA_ALLOC_CONF:-}" == *"expandable_segments:True"* ]]; then
    echo "WARNING: Unsetting PYTORCH_CUDA_ALLOC_CONF (expandable_segments:True conflicts with vLLM CuMemAllocator)"
    unset PYTORCH_CUDA_ALLOC_CONF
fi

# Guard: SLURM on some clusters sets ROCR_VISIBLE_DEVICES (AMD ROCm variable).
# VeRL's worker init raises ValueError if both ROCR_VISIBLE_DEVICES and
# CUDA_VISIBLE_DEVICES are set simultaneously on an NVIDIA node.
if [[ -n "${ROCR_VISIBLE_DEVICES:-}" ]]; then
    echo "WARNING: unsetting ROCR_VISIBLE_DEVICES (conflicts with CUDA_VISIBLE_DEVICES in VeRL)"
    unset ROCR_VISIBLE_DEVICES
fi

apptainer exec --nv \
  --overlay "$OVERLAY:ro" \
  --no-home \
  --bind /etc/pki:/etc/pki \
  --env "PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin" \
  --env "PYTHONNOUSERSITE=1" \
  --env "PYTHONUNBUFFERED=1" \
  --env "HF_HOME=$HF_HOME" \
  --env "HF_DATASETS_OFFLINE=1" \
  --env "TENSORBOARD_DIR=$TENSORBOARD_DIR" \
  --env "VERL_FILE_LOGGER_PATH=$VERL_FILE_LOGGER_PATH" \
  "$SIF" \
  python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    data.train_files="$TRAIN_FILES" \
    data.val_files="$VAL_FILES" \
    data.train_batch_size=$TRAIN_BATCH \
    data.val_batch_size=$TRAIN_BATCH \
    data.max_prompt_length=$MAX_PROMPT_LEN \
    data.max_response_length=$MAX_RESPONSE_LEN \
    data.dataloader_num_workers=$DATA_NUM_WORKERS \
    data.filter_overlong_prompts=True \
    data.truncation=error \
    +data.apply_chat_template_kwargs.enable_thinking=True \
    data.trust_remote_code=True \
    actor_rollout_ref.model.path="$MODEL_PATH" \
    actor_rollout_ref.model.trust_remote_code=True \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    '+actor_rollout_ref.model.override_config={attn_implementation:sdpa}' \
    actor_rollout_ref.actor.optim.lr=$LR \
    actor_rollout_ref.actor.ppo_mini_batch_size=$TRAIN_BATCH \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=$TRAIN_BATCH \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=$TRAIN_BATCH \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=$TRAIN_BATCH \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.ppo_epochs=1 \
    actor_rollout_ref.actor.fsdp_config.param_offload=$ACTOR_PARAM_OFFLOAD \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=$ACTOR_OPTIMIZER_OFFLOAD \
    '+actor_rollout_ref.actor.fsdp_config.wrap_policy.transformer_layer_cls_to_wrap=[Qwen3_5DecoderLayer]' \
    actor_rollout_ref.ref.fsdp_config.param_offload=$REF_PARAM_OFFLOAD \
    '+actor_rollout_ref.ref.fsdp_config.wrap_policy.transformer_layer_cls_to_wrap=[Qwen3_5DecoderLayer]' \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.tensor_model_parallel_size=$ROLLOUT_TP_SIZE \
    actor_rollout_ref.rollout.temperature=$ROLLOUT_TEMP \
    actor_rollout_ref.rollout.top_p=$ROLLOUT_TOP_P \
    actor_rollout_ref.rollout.top_k=$ROLLOUT_TOP_K \
    actor_rollout_ref.rollout.gpu_memory_utilization=$GPU_MEM_UTIL \
    actor_rollout_ref.rollout.enforce_eager=$ROLLOUT_ENFORCE_EAGER \
    actor_rollout_ref.rollout.max_num_seqs=$ROLLOUT_MAX_NUM_SEQS \
    actor_rollout_ref.rollout.enable_chunked_prefill=$ROLLOUT_ENABLE_CHUNKED_PREFILL \
    actor_rollout_ref.rollout.layered_summon=$ROLLOUT_LAYERED_SUMMON \
    actor_rollout_ref.rollout.load_format=$ROLLOUT_LOAD_FORMAT \
    actor_rollout_ref.rollout.enable_prefix_caching=$USE_PREFIX_CACHING \
    reward.custom_reward_function.path="$ROOT/src/phys_reasoner/training/reward.py" \
    reward.custom_reward_function.name=compute_score \
    trainer.project_name=$PROJECT \
    trainer.experiment_name=$EXPERIMENT \
    trainer.default_local_dir="$TRAIN_DIR" \
    trainer.n_gpus_per_node=$N_GPUS \
    trainer.nnodes=1 \
    trainer.val_before_train=$VAL_BEFORE_TRAIN \
    trainer.test_freq=$TEST_FREQ \
    trainer.total_epochs=$TOTAL_EPOCHS \
    trainer.save_freq=$SAVE_FREQ \
    trainer.logger='["console","tensorboard","file"]' \
    $FSDP_STRATEGY_ARGS \
    $DTYPE_ARGS \
    ${ROLLOUT_MAX_MODEL_LEN:+actor_rollout_ref.rollout.max_model_len=$ROLLOUT_MAX_MODEL_LEN} \
    ${ROLLOUT_MAX_BATCHED_TOKENS:+actor_rollout_ref.rollout.max_num_batched_tokens=$ROLLOUT_MAX_BATCHED_TOKENS} \
    $TIR_ARGS \
    $EXTRA_ARGS \
    2>&1 | tee "$TRAIN_DIR/train.log"

echo "Exit: $?"
