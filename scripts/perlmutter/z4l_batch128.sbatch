#!/bin/bash
# Z4l — Z4h + ppo_mini_batch_size 16 → 128 (single-knob).
# Tests optimizer stability under 8x larger batch on Z4h-Goldilocks-245.
# Side-effect: drop rollout.n 16 → 8 to keep rollout/step manageable
# (128 × 8 = 1024 rollouts/step on 2 rollout nodes; vs Z4h 16 × 16 = 256/step).
# Caveat: at batch=128 on 245 problems, each step samples ~half dataset →
# curriculum dynamics ≠ prod's 17k>>128 regime. This atom answers stability,
# not curriculum-realism.

#SBATCH --job-name=z4l_batch128
#SBATCH --output=logs/%x_%j.log
#SBATCH --error=logs/%x_%j.err

# HET-GROUP 0: TRAINER 4×4 on 40G
#SBATCH -A m2651
#SBATCH -C gpu&hbm40g
#SBATCH -q premium
#SBATCH -t 06:00:00
#SBATCH -N 4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-node=4

#SBATCH hetjob
# HET-GROUP 1: ROLLOUT 2×4 on 40G
#SBATCH -A m2651
#SBATCH -C gpu&hbm40g
#SBATCH -q premium
#SBATCH -t 06:00:00
#SBATCH -N 2
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

echo "=== z4l_batch128.sbatch ==="
echo "  Job       : $SLURM_JOB_ID"
echo "  Trainer   : $SLURM_JOB_NODELIST_HET_GROUP_0"
echo "  Rollout   : $SLURM_JOB_NODELIST_HET_GROUP_1"

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
    --env "WANDB_API_KEY=${WANDB_API_KEY:-}"
    "$SIF"
)
export APT_CMD

TRAIN_NODES=( $(scontrol show hostnames "$SLURM_JOB_NODELIST_HET_GROUP_0") )
HEAD_NODE="${TRAIN_NODES[0]}"
TRAIN_WORKER_NODES=( "${TRAIN_NODES[@]:1}" )
WORKER_NODES=( $(scontrol show hostnames "$SLURM_JOB_NODELIST_HET_GROUP_1") )
export HEAD_NODE TRAIN_WORKER_NODES WORKER_NODES

export N_GPUS_TRAIN=4
export N_GPUS_ROLLOUT=4
export NNODES_TRAIN=4
export NNODES_ROLLOUT=2
export HET_SRUN_HEAD="--het-group=0"
export HET_SRUN_WORKER="--het-group=1"
source "$ROOT/scripts/perlmutter/_ray_bringup.sh"

cleanup() {
    local rc=$?
    type ray_teardown &>/dev/null && ray_teardown || true
    return $rc
}
trap cleanup EXIT

EXP_TS=$(date +%Y%m%d.%H%M%S)
TRAIN_DIR="$ROOT/outputs/physcode_ablation/z4l_batch128_${EXP_TS}"
mkdir -p "$TRAIN_DIR"

MODEL_PATH=Qwen/Qwen3-4B
PHYSICS_TRAIN=$ROOT/data/processed_cot_z4h_goldilocks/data/train.parquet
PHYSICS_VAL=$ROOT/data/processed_cot_z4h_goldilocks/data/validation.parquet
REWARD_PATH=$ROOT/src/phys_reasoner/training/reward_polaris.py

echo "[driver] launching Z4l on $HEAD_NODE  (ppo_mini=128, rollout.n=8)"
srun --overlap --het-group=0 --nodes=1 --ntasks=1 -w "$HEAD_NODE" \
    "${APT_CMD[@]}" \
    python3 -m verl.experimental.fully_async_policy.fully_async_main \
        algorithm.adv_estimator=grpo \
        algorithm.use_kl_in_reward=False \
        algorithm.kl_ctrl.kl_coef=0.0 \
        algorithm.norm_adv_by_std_in_grpo=True \
        ++algorithm.filter_groups.enable=False \
        ++algorithm.filter_groups.metric=acc \
        ++algorithm.filter_groups.max_num_gen_batches=10 \
        actor_rollout_ref.actor.loss_agg_mode=token-mean \
        actor_rollout_ref.actor.clip_ratio_low=0.2 \
        actor_rollout_ref.actor.clip_ratio_high=0.28 \
        data.train_files="$PHYSICS_TRAIN" \
        data.val_files="$PHYSICS_VAL" \
        data.prompt_key=prompt \
        data.truncation=left \
        data.max_prompt_length=1024 \
        data.max_response_length=8192 \
        data.train_batch_size=0 \
        data.gen_batch_size=1 \
        data.return_raw_chat=True \
        data.dataloader_num_workers=2 \
        actor_rollout_ref.hybrid_engine=False \
        actor_rollout_ref.model.path=$MODEL_PATH \
        actor_rollout_ref.model.use_remove_padding=True \
        actor_rollout_ref.model.enable_gradient_checkpointing=True \
        actor_rollout_ref.actor.optim.lr=1e-6 \
        actor_rollout_ref.actor.ppo_mini_batch_size=128 \
        actor_rollout_ref.actor.use_dynamic_bsz=True \
        actor_rollout_ref.actor.ppo_max_token_len_per_gpu=12288 \
        actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
        actor_rollout_ref.actor.ppo_epochs=1 \
        actor_rollout_ref.actor.grad_clip=1.0 \
        actor_rollout_ref.actor.use_kl_loss=False \
        actor_rollout_ref.actor.kl_loss_coef=0.0 \
        actor_rollout_ref.actor.entropy_coeff=0 \
        actor_rollout_ref.actor.fsdp_config.strategy=fsdp2 \
        actor_rollout_ref.actor.fsdp_config.param_offload=False \
        actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
        actor_rollout_ref.actor.fsdp_config.fsdp_size=-1 \
        actor_rollout_ref.actor.ulysses_sequence_parallel_size=1 \
        actor_rollout_ref.ref.fsdp_config.param_offload=True \
        actor_rollout_ref.ref.fsdp_config.strategy=fsdp2 \
        actor_rollout_ref.ref.ulysses_sequence_parallel_size=1 \
        actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True \
        actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=49152 \
        actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True \
        actor_rollout_ref.rollout.name=vllm \
        actor_rollout_ref.rollout.mode=async \
        actor_rollout_ref.rollout.n=8 \
        actor_rollout_ref.rollout.temperature=1.0 \
        actor_rollout_ref.rollout.val_kwargs.temperature=1.0 \
        actor_rollout_ref.rollout.top_p=1.0 \
        actor_rollout_ref.rollout.top_k=-1 \
        actor_rollout_ref.rollout.gpu_memory_utilization=0.7 \
        actor_rollout_ref.rollout.max_model_len=9216 \
        actor_rollout_ref.rollout.max_num_seqs=1024 \
        actor_rollout_ref.rollout.max_num_batched_tokens=9216 \
        actor_rollout_ref.rollout.load_format=safetensors \
        actor_rollout_ref.rollout.enable_prefix_caching=True \
        actor_rollout_ref.rollout.enable_chunked_prefill=True \
        actor_rollout_ref.rollout.enforce_eager=False \
        +actor_rollout_ref.rollout.engine_kwargs.vllm.disable_custom_all_reduce=True \
        actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
        actor_rollout_ref.rollout.calculate_log_probs=True \
        actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
        actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
        actor_rollout_ref.rollout.agent.default_agent_loop=single_turn_agent \
        actor_rollout_ref.rollout.multi_turn.enable=false \
        +actor_rollout_ref.rollout.multi_turn.thinking_budget=null \
        +actor_rollout_ref.rollout.multi_turn.tool_call_budget=null \
        async_training.trigger_parameter_sync_step=4 \
        async_training.require_batches=1 \
        async_training.staleness_threshold=0.5 \
        async_training.partial_rollout=True \
        async_training.use_trainer_do_validate=False \
        reward.custom_reward_function.path="$REWARD_PATH" \
        reward.custom_reward_function.name=compute_score \
        trainer.critic_warmup=0 \
        trainer.nnodes=4 \
        trainer.n_gpus_per_node=4 \
        rollout.nnodes=2 \
        rollout.n_gpus_per_node=4 \
        rollout.total_rollout_steps=6400 \
        trainer.total_epochs=30 \
        trainer.save_freq=-1 \
        trainer.test_freq=-1 \
        trainer.val_before_train=false \
        trainer.project_name=physcode_ablation \
        trainer.experiment_name=z4l_batch128_${EXP_TS} \
        trainer.default_local_dir=$TRAIN_DIR \
        trainer.default_hdfs_dir=null \
        'trainer.logger=["console","wandb"]' \
    2>&1 | tee "$TRAIN_DIR/train.log"
DRIVER_EXIT=$?
echo "[driver] exited with $DRIVER_EXIT"
exit $DRIVER_EXIT
