#!/bin/bash
# Z2e-sync — Z2 config in SYNC main_ppo mode + fp16 autocast (mixed precision).
# Why sync: our fully_async path uses verl/workers/engine/fsdp/transformer_impl.py
# which has hardcoded bf16 autocast and no ShardedGradScaler. The sail-sg fp16
# patch (already merged in our verl via 65f9badf) lives in dp_actor.py which
# the SYNC main_ppo path uses — so only sync gives us the ShardedGradScaler.
#
# Correct fp16 knob in our verl: actor_rollout_ref.actor.fsdp_config.dtype=float16
# (triggers ShardedGradScaler in dp_actor.py). NOT model_dtype (master weights)
# and NOT a top-level actor.dtype (doesn't exist).
#
# Single-knob swap from Z2 defaults (fp32 master + bf16 autocast = the recipe
# that climbed to 0.21 at gstep 199): change actor autocast bf16 → fp16 and
# rollout dtype bf16 → fp16.
# If climbs like Z2 -> fp16 recipe works; next step is porting patch to engine/
# If still plateau -> precision isn't the learning blocker
# If NaN -> investigate ShardedGradScaler wiring

#SBATCH --job-name=z2e_sync_fp16
#SBATCH --output=logs/%x_%j.log
#SBATCH --error=logs/%x_%j.err

#SBATCH -A m2651
#SBATCH -C gpu&hbm40g
#SBATCH -q premium
#SBATCH -t 01:00:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-node=4

set -euo pipefail
ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/../.." && pwd)}"
cd "$ROOT"
mkdir -p logs

unset SIF OVERLAY
source "$ROOT/env.sh"
[[ -f "$SIF" && -f "$OVERLAY" ]] || { echo "ERROR: SIF/OVERLAY missing" >&2; exit 1; }

echo "=== z2e_sync_fp16.sbatch ==="
echo "  Job   : $SLURM_JOB_ID"
echo "  Nodes : $SLURM_JOB_NODELIST"

APT_CMD=(
    apptainer exec --nv
    --overlay "$OVERLAY:ro"
    --no-home
    --pwd "$ROOT/verl"
    --bind /etc/pki:/etc/pki
    --env "PYTHONNOUSERSITE=1"
    --env "PYTHONUNBUFFERED=1"
    --env "PYTHONPATH=$ROOT/.async-extras:/opt/phys-extras/"
    --env "HF_HOME=$HF_HOME"
    --env "XDG_CACHE_HOME=/tmp/.cache"
    --env "FLASHINFER_WORKSPACE_BASE=/tmp"
    --env "HOME=/tmp"
    --env "TRITON_CACHE_DIR=/tmp/.cache/triton"
    --env "MPLCONFIGDIR=/tmp/.cache/matplotlib"
    --env "PHYS_REQUIRE_XVERIFY=0"
    --env "PYTORCH_ALLOC_CONF=expandable_segments:True"
    --env "NCCL_IB_TIMEOUT=32"
    --env "NCCL_IBEXT_DISABLE=1"
    --env "TORCH_NCCL_ENABLE_MONITORING=0"
    "$SIF"
)
export APT_CMD

# Single-node sync + hybrid_engine — let verl.trainer.main_ppo init ray itself.

EXP_TS=$(date +%Y%m%d.%H%M%S)
TRAIN_DIR="$ROOT/outputs/physcode_ablation/z2e_sync_fp16_${EXP_TS}"
mkdir -p "$TRAIN_DIR"

MODEL_PATH=Qwen/Qwen2.5-0.5B-Instruct
PHYSICS_TRAIN=$ROOT/data/processed_cot_z2_numerical_easy/data/train.parquet
PHYSICS_VAL=$ROOT/data/processed_cot_z2_numerical_easy/data/validation.parquet
REWARD_PATH=$ROOT/src/phys_reasoner/training/reward_polaris.py

echo "[driver] launching Z2e-sync (Qwen2.5-0.5B + fp16 e2e, sync main_ppo)"
"${APT_CMD[@]}" \
    python3 -m verl.trainer.main_ppo \
        algorithm.adv_estimator=grpo \
        algorithm.use_kl_in_reward=False \
        algorithm.kl_ctrl.kl_coef=0.0 \
        algorithm.norm_adv_by_std_in_grpo=True \
        ++algorithm.filter_groups.enable=False \
        ++algorithm.filter_groups.metric=acc \
        ++algorithm.filter_groups.max_num_gen_batches=10 \
        data.train_files="$PHYSICS_TRAIN" \
        data.val_files="$PHYSICS_VAL" \
        data.prompt_key=prompt \
        data.truncation=left \
        data.max_prompt_length=1024 \
        data.max_response_length=2048 \
        data.train_batch_size=64 \
        data.return_raw_chat=True \
        data.dataloader_num_workers=2 \
        actor_rollout_ref.hybrid_engine=True \
        actor_rollout_ref.model.path=$MODEL_PATH \
        actor_rollout_ref.model.use_remove_padding=True \
        actor_rollout_ref.model.enable_gradient_checkpointing=True \
        actor_rollout_ref.actor.fsdp_config.dtype=float16 \
        actor_rollout_ref.actor.optim.lr=1e-6 \
        actor_rollout_ref.actor.ppo_mini_batch_size=16 \
        actor_rollout_ref.actor.use_dynamic_bsz=True \
        actor_rollout_ref.actor.ppo_max_token_len_per_gpu=12288 \
        actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
        actor_rollout_ref.actor.ppo_epochs=1 \
        actor_rollout_ref.actor.grad_clip=1.0 \
        actor_rollout_ref.actor.use_kl_loss=False \
        actor_rollout_ref.actor.kl_loss_coef=0.0 \
        actor_rollout_ref.actor.entropy_coeff=0 \
        actor_rollout_ref.actor.clip_ratio_low=0.2 \
        actor_rollout_ref.actor.clip_ratio_high=0.28 \
        actor_rollout_ref.actor.loss_agg_mode=token-mean \
        actor_rollout_ref.actor.fsdp_config.strategy=fsdp2 \
        actor_rollout_ref.actor.fsdp_config.param_offload=False \
        actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
        actor_rollout_ref.actor.fsdp_config.fsdp_size=1 \
        actor_rollout_ref.actor.ulysses_sequence_parallel_size=1 \
        actor_rollout_ref.ref.fsdp_config.param_offload=True \
        actor_rollout_ref.ref.fsdp_config.strategy=fsdp2 \
        actor_rollout_ref.ref.ulysses_sequence_parallel_size=1 \
        actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True \
        actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=49152 \
        actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
        actor_rollout_ref.rollout.dtype=float16 \
        actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True \
        actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
        actor_rollout_ref.rollout.name=vllm \
        actor_rollout_ref.rollout.mode=sync \
        actor_rollout_ref.rollout.n=16 \
        actor_rollout_ref.rollout.temperature=1.0 \
        actor_rollout_ref.rollout.val_kwargs.temperature=1.0 \
        actor_rollout_ref.rollout.top_p=1.0 \
        actor_rollout_ref.rollout.top_k=-1 \
        actor_rollout_ref.rollout.gpu_memory_utilization=0.7 \
        actor_rollout_ref.rollout.max_model_len=3072 \
        actor_rollout_ref.rollout.max_num_seqs=256 \
        actor_rollout_ref.rollout.max_num_batched_tokens=3072 \
        actor_rollout_ref.rollout.load_format=safetensors \
        actor_rollout_ref.rollout.enable_prefix_caching=True \
        actor_rollout_ref.rollout.enable_chunked_prefill=True \
        actor_rollout_ref.rollout.enforce_eager=False \
        +actor_rollout_ref.rollout.engine_kwargs.vllm.disable_custom_all_reduce=True \
        actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
        actor_rollout_ref.rollout.calculate_log_probs=True \
        actor_rollout_ref.rollout.agent.default_agent_loop=single_turn_agent \
        actor_rollout_ref.rollout.multi_turn.enable=false \
        +actor_rollout_ref.rollout.multi_turn.thinking_budget=null \
        +actor_rollout_ref.rollout.multi_turn.tool_call_budget=null \
        reward_model.custom_reward_function.path="$REWARD_PATH" \
        reward_model.custom_reward_function.name=compute_score \
        trainer.critic_warmup=0 \
        trainer.nnodes=1 \
        trainer.n_gpus_per_node=4 \
        trainer.total_epochs=30 \
        trainer.save_freq=-1 \
        trainer.test_freq=-1 \
        trainer.val_before_train=false \
        trainer.project_name=physcode_ablation \
        trainer.experiment_name=z2e_sync_fp16_${EXP_TS} \
        trainer.default_local_dir=$TRAIN_DIR \
        trainer.default_hdfs_dir=null \
        'trainer.logger=["console","wandb"]' \
    2>&1 | tee "$TRAIN_DIR/train.log"
DRIVER_EXIT=$?
echo "[driver] exited with $DRIVER_EXIT"
exit $DRIVER_EXIT
