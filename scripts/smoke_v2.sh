#!/usr/bin/env bash
# smoke_v2.sh — Crystal-style GRPO smoke in phys-reasoner's apptainer env.
#
# Goal: verify the full VeRL + apptainer training pipeline completes 2 steps.
#
# Progression plan:
#   Step 1 (default):  synthetic 2-row data + standalone reward (no phys_reasoner)
#   Step 2:            synthetic 2-row data + real phys_reasoner reward
#   Step 3:            2 rows sampled from real physics parquet + phys_reasoner reward
#   Step 4:            re-enable TIR (separate script)
#
# Usage:
#   bash scripts/smoke_v2.sh
#   MODEL=Qwen/Qwen3-1.7B bash scripts/smoke_v2.sh
#   REWARD_STEP=2 bash scripts/smoke_v2.sh
#   REWARD_STEP=2 REAL_PARQUET=data/processed/corpus_train.parquet bash scripts/smoke_v2.sh
#   REWARD_STEP=2 REAL_PARQUET=data/processed/corpus_train.parquet SMOKE_N=4 bash scripts/smoke_v2.sh

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
source "$ROOT/env.sh"

MODEL="${MODEL:-Qwen/Qwen3-0.6B}"
REWARD_STEP="${REWARD_STEP:-1}"   # 1=simple standalone, 2=phys_reasoner reward
# Optional: path to a real parquet (relative to ROOT or absolute). When set,
# samples SMOKE_N rows from it instead of generating synthetic data.
REAL_PARQUET="${REAL_PARQUET:-}"
SMOKE_N="${SMOKE_N:-2}"          # how many rows to sample from REAL_PARQUET

# ---------- crystal-matching settings for A40 (48 GB) ----------
N_GPUS=1
TRAIN_BATCH=2         # matches 2-row smoke dataset
ROLLOUT_N=8           # 8 completions/prompt → 16 total gens/step (like crystal)
PPO_MINI_BATCH=2      # 16 total gens divisible by 2 ✓
REF_MICRO_BATCH=$PPO_MINI_BATCH
TOTAL_GENS=$((TRAIN_BATCH * ROLLOUT_N))
ROLLOUT_MAX_NUM_SEQS=$TOTAL_GENS  # cap at total gens (crystal pattern)
VLLM_GPU_MEM_UTIL=0.38            # crystal's ≤50 GB setting

# Prompt/response lengths: real physics prompts are much longer than synthetic ones.
# Synthetic: system (~60 tok) + "What is 2+2?" (~10 tok) ≈ 70 tokens → 256 is fine.
# Real:      TIR system prompt (~180 tok) + physics problem (~200+ tok) ≈ 400+ tokens.
if [[ -n "$REAL_PARQUET" ]]; then
    MAX_PROMPT_LEN="${MAX_PROMPT_LEN:-1024}"
    MAX_RESPONSE_LEN="${MAX_RESPONSE_LEN:-512}"
else
    MAX_PROMPT_LEN="${MAX_PROMPT_LEN:-256}"
    MAX_RESPONSE_LEN="${MAX_RESPONSE_LEN:-256}"
fi

TIMESTAMP=$(date +%Y%m%d.%H%M%S)
TRAIN_DIR="$ROOT/outputs/smoke_v2_$TIMESTAMP"
mkdir -p "$TRAIN_DIR"

if [[ "$REWARD_STEP" == "1" ]]; then
    REWARD_PATH="$ROOT/scripts/smoke_reward_simple.py"
    echo "=== smoke_v2 Step 1: standalone reward (no phys_reasoner imports) ==="
else
    REWARD_PATH="$ROOT/src/phys_reasoner/training/reward.py"
    echo "=== smoke_v2 Step 2: phys_reasoner reward ==="
fi

echo "  model    : $MODEL"
echo "  batch    : $TRAIN_BATCH prompts x $ROLLOUT_N rollouts = $TOTAL_GENS gens/step"
echo "  ppo mini : $PPO_MINI_BATCH"
echo "  vLLM mem : $VLLM_GPU_MEM_UTIL"
echo "  reward   : $REWARD_PATH"
echo "  output   : $TRAIN_DIR"

# Guard: expandable_segments is incompatible with vLLM CuMemAllocator.
if [[ "${PYTORCH_CUDA_ALLOC_CONF:-}" == *"expandable_segments:True"* ]]; then
    echo "WARNING: unsetting PYTORCH_CUDA_ALLOC_CONF (expandable_segments:True conflicts with vLLM)"
    unset PYTORCH_CUDA_ALLOC_CONF
fi

# ---------- generate smoke parquet (inside container for correct pandas/pyarrow) ----------
SMOKE_DATA="$TRAIN_DIR/smoke.parquet"
echo ""

if [[ -n "$REAL_PARQUET" ]]; then
    # Resolve relative paths against ROOT
    [[ "$REAL_PARQUET" != /* ]] && REAL_PARQUET="$ROOT/$REAL_PARQUET"
    echo "Sampling $SMOKE_N rows from real parquet: $REAL_PARQUET → $SMOKE_DATA"
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
PY
else
    echo "Writing $SMOKE_N synthetic smoke rows: $SMOKE_DATA"
    PYTHONNOUSERSITE=1 apptainer exec \
      --overlay "$OVERLAY:ro" --no-home \
      --bind /etc/pki:/etc/pki \
      --env "PYTHONNOUSERSITE=1" \
      "$SIF" \
      python3 - <<'PY' "$SMOKE_DATA"
import sys
import pandas as pd

path = sys.argv[1]
sys_msg = "Solve the problem. End your answer with \\boxed{answer}."
rows = [
    {
        "prompt": [
            {"role": "system", "content": sys_msg},
            {"role": "user",   "content": "What is 2 + 2?"},
        ],
        "data_source": "smoke",
        "reward_model": {"ground_truth": "4", "style": "rule"},
        "extra_info": {
            "answer_type": "numerical", "unit": "", "tolerance": 0.0,
            "problem": "What is 2 + 2?",
        },
    },
    {
        "prompt": [
            {"role": "system", "content": sys_msg},
            {"role": "user",   "content": "What is 3 + 3?"},
        ],
        "data_source": "smoke",
        "reward_model": {"ground_truth": "6", "style": "rule"},
        "extra_info": {
            "answer_type": "numerical", "unit": "", "tolerance": 0.0,
            "problem": "What is 3 + 3?",
        },
    },
]
pd.DataFrame(rows).to_parquet(path, index=False)
print(f"Wrote {len(rows)} rows → {path}")
PY
fi

echo ""
echo "Launching VeRL training..."
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
    reward.custom_reward_function.path="$REWARD_PATH" \
    reward.custom_reward_function.name=compute_score \
    trainer.critic_warmup=0 \
    trainer.n_gpus_per_node=$N_GPUS \
    trainer.nnodes=1 \
    trainer.total_training_steps=2 \
    trainer.save_freq=50 \
    trainer.test_freq=-1 \
    trainer.val_before_train=false \
    trainer.project_name=physcode_smoke \
    trainer.experiment_name="smoke_v2_$TIMESTAMP" \
    trainer.default_local_dir="$TRAIN_DIR" \
    trainer.default_hdfs_dir=null \
    'trainer.logger=["console"]' \
    2>&1 | tee "$TRAIN_DIR/run.log"

echo "Exit code: $?"
