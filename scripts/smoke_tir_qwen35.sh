#!/usr/bin/env bash
# smoke_tir_qwen35.sh — TIR smoke test for Qwen3.5-4B on H200.
#
# *** VALIDATED configuration for Qwen3.5-4B with format=qwen3_coder ***
# For Qwen3-0.6B use smoke_tir.sh (format=hermes) instead.
#
# TOOL-CALL FORMAT NOTE — why this file exists separately from smoke_tir.sh:
#   Qwen3   (0.6B, 1.7B, 4B, ...) → tokenizer emits hermes/JSON format:
#       <tool_call>
#       {"name": "python", "arguments": {"code": "..."}}
#       </tool_call>
#       VeRL setting: multi_turn.format=hermes
#
#   Qwen3.5 (0.8B, 4B, ...)        → tokenizer emits Qwen XML/coder format:
#       <tool_call>
#       <function=python>
#       <parameter=code>...</parameter>
#       </function>
#       </tool_call>
#       VeRL setting: multi_turn.format=qwen3_coder
#
#   This was confirmed 2026-04-01 by inspecting the rendered phase 1 prompts
#   via dump_rollouts.py. The Qwen3.5-4B tokenizer's apply_chat_template injects
#   XML-format tool-call instructions into the system message; the hermes parser
#   in VeRL cannot parse this format and silently drops all tool calls.
#   VeRL's qwen3_coder parser (verl/experimental/agent_loop/tool_parser.py:173)
#   explicitly handles this format.
#
# Architecture:
#   - VeRL ToolAgentLoop (tool_agent) handles the 2-phase TIR cycle:
#       Turn 1: model generates reasoning + <tool_call>...</tool_call>
#       [ToolAgentLoop extracts code, calls PythonSandboxTool, injects <tool_response>]
#       Turn 2: model sees tool output, generates final answer with \boxed{}
#   - MAX_TOOL_TURNS=1 (default): single-shot TIR, one tool call per trajectory
#     MAX_TOOL_TURNS=2: iterative TIR, model can recover from errors with a second call
#   - Tool schema is injected by apply_chat_template(tools=...) inside the agent loop;
#     the parquet prompts don't need to pre-inject it.
#
# GPU: H200 (80 GB). VLLM_GPU_MEM_UTIL=0.5 is safe for 4B + training overhead.
# For A100/A40, lower VLLM_GPU_MEM_UTIL and increase param_offload as needed.
#
# Usage:
#   bash scripts/smoke_tir_qwen35.sh
#   REAL_PARQUET=data/processed/drsci_train.parquet bash scripts/smoke_tir_qwen35.sh
#   SMOKE_N=4 bash scripts/smoke_tir_qwen35.sh

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
# Unset SIF/OVERLAY from SLURM-inherited env so env.sh/.env always wins
unset SIF OVERLAY
source "$ROOT/env.sh"
echo "  SIF:     $SIF"
echo "  OVERLAY: $OVERLAY"

MODEL="${MODEL:-Qwen/Qwen3.5-4B}"
# Default to corpus_train (smaller, cleaner). Override via env.
REAL_PARQUET="${REAL_PARQUET:-data/processed/corpus_train.parquet}"
SMOKE_N="${SMOKE_N:-2}"
# MAX_TOOL_TURNS: number of tool call rounds allowed per trajectory.
# 1 = single-shot TIR (original design, cleaner reward signal).
# 2+ = iterative TIR (model can recover from errors; higher compute per rollout).
MAX_TOOL_TURNS="${MAX_TOOL_TURNS:-1}"

# Resolve relative path against ROOT
[[ "$REAL_PARQUET" != /* ]] && REAL_PARQUET="$ROOT/$REAL_PARQUET"

# ---------- H200 (80 GB) settings ----------
N_GPUS=1
TRAIN_BATCH=2
ROLLOUT_N=8
PPO_MINI_BATCH=2
REF_MICRO_BATCH=$PPO_MINI_BATCH
TOTAL_GENS=$((TRAIN_BATCH * ROLLOUT_N))
ROLLOUT_MAX_NUM_SEQS=$TOTAL_GENS
# 4B on H200: 0.5 leaves ample room for FSDP + Ray overhead alongside vLLM.
# Override via env for smaller GPUs (e.g. VLLM_GPU_MEM_UTIL=0.3 for A40).
VLLM_GPU_MEM_UTIL="${VLLM_GPU_MEM_UTIL:-0.5}"

# TIR response budget — all five components must be set; MAX_RESPONSE_LEN is derived.
#
#   response_length = THINKING_BUDGET + INTERRUPT_LEN
#                   + TOOL_CALL_BUDGET + MAX_TOOL_RESPONSE_LEN
#                   + ANSWER_BUDGET
#
# INTERRUPT_LEN: exact token count of THINK_INTERRUPT_PHRASE for Qwen3.5-4B.
# To recompute for a different model:
#   python3 -c "from transformers import AutoTokenizer; t=AutoTokenizer.from_pretrained('<model>', local_files_only=True); \
#     print(len(t.encode('\nOkay, time is up. Let me stop thinking and formulate a final answer\n</think>\n', add_special_tokens=False)))"
MAX_PROMPT_LEN="${MAX_PROMPT_LEN:-1024}"
THINKING_BUDGET="${THINKING_BUDGET:-}"       # required when think-interrupt is enabled
TOOL_CALL_BUDGET="${TOOL_CALL_BUDGET:-}"     # required when think-interrupt is enabled
INTERRUPT_LEN=15                             # exact for Qwen3.5-4B
MAX_TOOL_RESPONSE_LEN=512                    # must match multi_turn.max_tool_response_length below
ANSWER_BUDGET="${ANSWER_BUDGET:-}"           # required when think-interrupt is enabled

# Derive MAX_RESPONSE_LEN from budgets when think-interrupt is enabled; otherwise use override.
if [[ -n "$THINKING_BUDGET" && -n "$TOOL_CALL_BUDGET" && -n "$ANSWER_BUDGET" ]]; then
    MAX_RESPONSE_LEN=$((THINKING_BUDGET + INTERRUPT_LEN + TOOL_CALL_BUDGET + MAX_TOOL_RESPONSE_LEN + ANSWER_BUDGET))
else
    MAX_RESPONSE_LEN="${MAX_RESPONSE_LEN:-4096}"
fi

TIMESTAMP=$(date +%Y%m%d.%H%M%S)
TRAIN_DIR="$ROOT/outputs/smoke_tir_qwen35_$TIMESTAMP"
mkdir -p "$TRAIN_DIR"

echo "=== smoke_tir_qwen35: TIR with ToolAgentLoop + qwen3_coder format ==="
echo "  model     : $MODEL"
echo "  data      : $REAL_PARQUET (n=$SMOKE_N)"
echo "  batch     : $TRAIN_BATCH prompts x $ROLLOUT_N rollouts = $TOTAL_GENS gens/step"
echo "  seq lens  : prompt=$MAX_PROMPT_LEN  response=$MAX_RESPONSE_LEN"
echo "  vLLM mem  : $VLLM_GPU_MEM_UTIL  (H200)"
echo "  output    : $TRAIN_DIR"

# Guard: expandable_segments is incompatible with vLLM CuMemAllocator.
if [[ "${PYTORCH_CUDA_ALLOC_CONF:-}" == *"expandable_segments:True"* ]]; then
    echo "WARNING: unsetting PYTORCH_CUDA_ALLOC_CONF (expandable_segments:True conflicts with vLLM)"
    unset PYTORCH_CUDA_ALLOC_CONF
fi

# Guard: SLURM on some clusters sets ROCR_VISIBLE_DEVICES (AMD ROCm variable).
# VeRL's worker init raises ValueError if both ROCR_VISIBLE_DEVICES and
# CUDA_VISIBLE_DEVICES are set simultaneously on an NVIDIA node.
if [[ -n "${ROCR_VISIBLE_DEVICES:-}" ]]; then
    echo "WARNING: unsetting ROCR_VISIBLE_DEVICES (conflicts with CUDA_VISIBLE_DEVICES in VeRL)"
    unset ROCR_VISIBLE_DEVICES
fi

# ---------- sample smoke parquet (inside container for correct pandas/pyarrow) ----------
# Stable path so train_smoke.sbatch can always find it without a timestamp.
SMOKE_DATA="$ROOT/data/processed/smoke.parquet"
echo ""
echo "Sampling $SMOKE_N rows from: $REAL_PARQUET → $SMOKE_DATA"
PYTHONNOUSERSITE=1 apptainer exec \
  --overlay "$OVERLAY:ro" --no-home \
  --bind /etc/pki:/etc/pki \
  --env "PYTHONNOUSERSITE=1" \
  --env "PYTHONPATH=/opt/phys-extras/" \
  "$SIF" \
  python3 - <<'PY' "$REAL_PARQUET" "$SMOKE_DATA" "$SMOKE_N"
import sys
import pandas as pd

src, dst, n = sys.argv[1], sys.argv[2], int(sys.argv[3])
df = pd.read_parquet(src)
sample = df.sample(n=min(n, len(df)), random_state=42).reset_index(drop=True)
sample.to_parquet(dst, index=False)
print(f"Sampled {len(sample)} rows from {src} → {dst}")
print(f"  columns: {list(sample.columns)}")
print(f"  answer_types: {[r.get('answer_type') for r in sample['extra_info']]}")
# Show problem snippets for sanity
for i, row in sample.iterrows():
    user_content = row["prompt"][1]["content"][:120]
    gold = row["reward_model"]["ground_truth"]
    print(f"  [{i}] gold={gold!r}  problem={user_content!r}...")
PY

echo ""
echo "Launching VeRL TIR training..."
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
  --env "VERL_DUMP_DIR=${VERL_DUMP_DIR:-}" \
  "$SIF" \
  python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    algorithm.use_kl_in_reward=False \
    data.train_files="$SMOKE_DATA" \
    data.val_files="$SMOKE_DATA" \
    data.train_batch_size=$TRAIN_BATCH \
    data.max_prompt_length=$MAX_PROMPT_LEN \
    data.max_response_length=$MAX_RESPONSE_LEN \
    data.filter_overlong_prompts=False \
    data.truncation=right \
    data.dataloader_num_workers=1 \
    +data.apply_chat_template_kwargs.enable_thinking=true \
    actor_rollout_ref.model.path="$MODEL" \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    '+actor_rollout_ref.model.override_config={attn_implementation:sdpa}' \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=$PPO_MINI_BATCH \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=$PPO_MINI_BATCH \
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
    trainer.total_training_steps=2 \
    trainer.save_freq=50 \
    trainer.test_freq=-1 \
    trainer.val_before_train=false \
    trainer.project_name=physcode_smoke \
    trainer.experiment_name="smoke_tir_qwen35_$TIMESTAMP" \
    trainer.default_local_dir="$TRAIN_DIR" \
    trainer.default_hdfs_dir=null \
    'trainer.logger=["console"]' \
    2>&1 | tee "$TRAIN_DIR/run.log"

echo "Exit code: $?"
