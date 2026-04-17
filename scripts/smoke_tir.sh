#!/usr/bin/env bash
# smoke_tir.sh — Step 4: TIR smoke test with VeRL's native ToolAgentLoop.
#
# *** VALIDATED for Qwen3-0.6B with format=hermes ***
# For Qwen3.5-4B use smoke_tir_qwen35.sh (format=qwen3_coder) instead.
#
# TOOL-CALL FORMAT NOTE (easy to get wrong):
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
#   Setting the wrong format causes the ToolAgentLoop to fail to parse tool
#   calls silently — the model generates correct output but no tool ever runs.
#   Confirmed 2026-04-01 by inspecting rendered prompts via dump_rollouts.py.
#
# Validates the full TIR pipeline:
#   real physics data → tool_agent loop → hermes tool-call format
#   → PythonSandboxTool execution → phys_reasoner reward → 2 training steps
#
# Architecture:
#   - VeRL ToolAgentLoop (tool_agent) handles the 2-phase TIR cycle:
#       Turn 1: model generates reasoning + <tool_call>...</tool_call>
#       [ToolAgentLoop extracts code, calls PythonSandboxTool, injects <tool_response>]
#       Turn 2: model sees tool output, generates final answer with \boxed{}
#   - max_assistant_turns=2 enforces exactly one tool call per trajectory
#   - Tool schema is injected by apply_chat_template(tools=...) inside the agent loop;
#     the parquet prompts don't need to pre-inject it.
#
# Usage:
#   bash scripts/smoke_tir.sh
#   REAL_PARQUET=data/processed/drsci_train.parquet bash scripts/smoke_tir.sh
#   SMOKE_N=4 bash scripts/smoke_tir.sh
#   MODEL=Qwen/Qwen3-1.7B bash scripts/smoke_tir.sh

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
source "$ROOT/env.sh"

MODEL="${MODEL:-Qwen/Qwen3-0.6B}"
# Default to corpus_train (smaller, cleaner). Override via env.
REAL_PARQUET="${REAL_PARQUET:-data/processed/corpus_train.parquet}"
SMOKE_N="${SMOKE_N:-2}"

# Resolve relative path against ROOT
[[ "$REAL_PARQUET" != /* ]] && REAL_PARQUET="$ROOT/$REAL_PARQUET"

# ---------- crystal-matching settings for A40 (48 GB) ----------
N_GPUS=1
TRAIN_BATCH=2
ROLLOUT_N=8
PPO_MINI_BATCH=2
REF_MICRO_BATCH=$PPO_MINI_BATCH
TOTAL_GENS=$((TRAIN_BATCH * ROLLOUT_N))
ROLLOUT_MAX_NUM_SEQS=$TOTAL_GENS
VLLM_GPU_MEM_UTIL=0.38

# TIR responses need two phases: reasoning+tool_call → injection → final answer.
# 1024 prompt + 1536 response covers most physics problems with room to spare.
MAX_PROMPT_LEN="${MAX_PROMPT_LEN:-1024}"
MAX_RESPONSE_LEN="${MAX_RESPONSE_LEN:-1536}"

TIMESTAMP=$(date +%Y%m%d.%H%M%S)
TRAIN_DIR="$ROOT/outputs/smoke_tir_$TIMESTAMP"
mkdir -p "$TRAIN_DIR"

echo "=== smoke_tir Step 4: TIR with ToolAgentLoop + qwen3_coder format ==="
echo "  model     : $MODEL"
echo "  data      : $REAL_PARQUET (n=$SMOKE_N)"
echo "  batch     : $TRAIN_BATCH prompts x $ROLLOUT_N rollouts = $TOTAL_GENS gens/step"
echo "  seq lens  : prompt=$MAX_PROMPT_LEN  response=$MAX_RESPONSE_LEN"
echo "  vLLM mem  : $VLLM_GPU_MEM_UTIL"
echo "  output    : $TRAIN_DIR"

# Guard: expandable_segments is incompatible with vLLM CuMemAllocator.
if [[ "${PYTORCH_CUDA_ALLOC_CONF:-}" == *"expandable_segments:True"* ]]; then
    echo "WARNING: unsetting PYTORCH_CUDA_ALLOC_CONF (expandable_segments:True conflicts with vLLM)"
    unset PYTORCH_CUDA_ALLOC_CONF
fi

# ---------- sample smoke parquet (inside container for correct pandas/pyarrow) ----------
SMOKE_DATA="$TRAIN_DIR/smoke.parquet"
echo ""
echo "Sampling $SMOKE_N rows from: $REAL_PARQUET → $SMOKE_DATA"
PYTHONNOUSERSITE=1 apptainer exec \
  --overlay "$OVERLAY:ro" --no-home \
  --bind /etc/pki:/etc/pki \
  --env "PYTHONNOUSERSITE=1" \
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
  --env "HF_HOME=$HF_HOME" \
  --env "HF_DATASETS_OFFLINE=1" \
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
    +data.apply_chat_template_kwargs.enable_thinking=false \
    actor_rollout_ref.model.path="$MODEL" \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=$PPO_MINI_BATCH \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=$PPO_MINI_BATCH \
    actor_rollout_ref.actor.ppo_epochs=1 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.actor.fsdp_config.model_dtype=bfloat16 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.ref.fsdp_config.model_dtype=bfloat16 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.n=$ROLLOUT_N \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.top_p=0.9 \
    actor_rollout_ref.rollout.gpu_memory_utilization=$VLLM_GPU_MEM_UTIL \
    actor_rollout_ref.rollout.max_num_seqs=$ROLLOUT_MAX_NUM_SEQS \
    actor_rollout_ref.rollout.load_format=safetensors \
    actor_rollout_ref.rollout.tensor_model_parallel_size=$N_GPUS \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=$REF_MICRO_BATCH \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=$REF_MICRO_BATCH \
    actor_rollout_ref.rollout.agent.default_agent_loop=tool_agent \
    actor_rollout_ref.rollout.multi_turn.enable=true \
    actor_rollout_ref.rollout.multi_turn.format=hermes \
    actor_rollout_ref.rollout.multi_turn.tool_config_path="$ROOT/scripts/physcode_tools.yaml" \
    actor_rollout_ref.rollout.multi_turn.max_assistant_turns=2 \
    actor_rollout_ref.rollout.multi_turn.max_user_turns=1 \
    actor_rollout_ref.rollout.multi_turn.max_parallel_calls=1 \
    actor_rollout_ref.rollout.multi_turn.max_tool_response_length=1024 \
    actor_rollout_ref.rollout.multi_turn.tool_response_truncate_side=right \
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
    trainer.experiment_name="smoke_tir_$TIMESTAMP" \
    trainer.default_local_dir="$TRAIN_DIR" \
    trainer.default_hdfs_dir=null \
    'trainer.logger=["console"]' \
    2>&1 | tee "$TRAIN_DIR/run.log"

echo "Exit code: $?"
