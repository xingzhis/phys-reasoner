#!/bin/bash
# Z4n — diagnostic: Z4m recipe on equation-only data.
# Z4m on mixed-type Goldilocks-from-pilot oscillated 0.66-0.77 over 47 steps —
# never climbed like Z4h/Z4l/Z4k did. Single-knob change vs Z4m: data is
# numerical-only (Z4h's 245 + pilot's 52 numerical = 297 problems).
# Outcomes:
#   - Z4n climbs cleanly → multi-type / verifier-heterogeneity is the culprit;
#     test multi-task ideas next (balanced sampling, per-type reward norm,
#     DAPO filter_groups, decompose+distill).
#   - Z4n doesn't climb → recipe (think-interrupt + sp=2 + 19488 budget +
#     reward.py) itself is the issue; narrow further.

#SBATCH --job-name=z4q_eq
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
# HET-GROUP 1: ROLLOUT 4×4 on 40G (doubled from Z4m's 2 nodes — Z4m's 79% trainer-idle
# showed it was severely rollout-bound at 4T+2R for 19488-token sequences).
#SBATCH -A m2651
#SBATCH -C gpu&hbm40g
#SBATCH -q premium
#SBATCH -t 06:00:00
#SBATCH -N 4
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

echo "=== z4q_eq_only.sbatch ==="
echo "  Job       : $SLURM_JOB_ID"
echo "  Trainer   : $SLURM_JOB_NODELIST_HET_GROUP_0"
echo "  Rollout   : $SLURM_JOB_NODELIST_HET_GROUP_1"

# --- xverify URL (must be set before APT_CMD references it) ---
URL_FILE="${SHARED_URL_FILE:-$ROOT/outputs/xverify_endpoints/current.url}"
[[ -f "$URL_FILE" ]] || { echo "ERROR: xverify URL file missing at $URL_FILE" >&2; exit 1; }
XVERIFY_URL=$(head -n1 "$URL_FILE")
echo "[xverify] URL = $XVERIFY_URL"
export XVERIFY_URL

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
    --env "PHYS_REQUIRE_XVERIFY=1"
    --env "XVERIFY_URL=$XVERIFY_URL"
    --env "INTERRUPT_LEN=16"
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
export NNODES_ROLLOUT=4
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
TRAIN_DIR="$ROOT/outputs/physcode_ablation/z4q_eq_${EXP_TS}"
mkdir -p "$TRAIN_DIR"

MODEL_PATH=Qwen/Qwen3-4B
PHYSICS_TRAIN=$ROOT/data/processed_cot_z4q_eq/data/train.parquet
PHYSICS_VAL=$ROOT/data/processed_cot_z4q_eq/data/validation.parquet
REWARD_PATH=$ROOT/src/phys_reasoner/training/reward.py

# Budget: 1024 prompt + 12288 thinking + 16 interrupt + 7184 post-interrupt = 20496 model_len.
# data.max_response_length is the response side only = 12288 + 16 + 7184 = 19488.
# (Slack of 16 vs 19472 for safety; verl truncates to response_length anyway.)
RESPONSE_LEN=19488
MODEL_LEN=$((1024 + RESPONSE_LEN))

echo "[driver] launching Z4n on $HEAD_NODE  (response_len=$RESPONSE_LEN, model_len=$MODEL_LEN, data=equation-only-21, xverify=$XVERIFY_URL)"
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
        data.max_response_length=$RESPONSE_LEN \
        data.train_batch_size=0 \
        data.gen_batch_size=1 \
        data.return_raw_chat=True \
        data.dataloader_num_workers=2 \
        actor_rollout_ref.hybrid_engine=False \
        actor_rollout_ref.model.path=$MODEL_PATH \
        actor_rollout_ref.model.use_remove_padding=True \
        actor_rollout_ref.model.enable_gradient_checkpointing=True \
        actor_rollout_ref.actor.optim.lr=1e-6 \
        actor_rollout_ref.actor.ppo_mini_batch_size=16 \
        actor_rollout_ref.actor.use_dynamic_bsz=True \
        actor_rollout_ref.actor.ppo_max_token_len_per_gpu=20480 \
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
        actor_rollout_ref.actor.ulysses_sequence_parallel_size=2 \
        actor_rollout_ref.ref.fsdp_config.param_offload=True \
        actor_rollout_ref.ref.fsdp_config.strategy=fsdp2 \
        actor_rollout_ref.ref.ulysses_sequence_parallel_size=2 \
        actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True \
        actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=49152 \
        actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True \
        actor_rollout_ref.rollout.name=vllm \
        actor_rollout_ref.rollout.mode=async \
        actor_rollout_ref.rollout.n=16 \
        actor_rollout_ref.rollout.temperature=1.0 \
        actor_rollout_ref.rollout.val_kwargs.temperature=1.0 \
        actor_rollout_ref.rollout.top_p=1.0 \
        actor_rollout_ref.rollout.top_k=-1 \
        actor_rollout_ref.rollout.gpu_memory_utilization=0.7 \
        actor_rollout_ref.rollout.max_model_len=$MODEL_LEN \
        actor_rollout_ref.rollout.max_num_seqs=256 \
        actor_rollout_ref.rollout.max_num_batched_tokens=$MODEL_LEN \
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
        +actor_rollout_ref.rollout.multi_turn.thinking_budget=12288 \
        +actor_rollout_ref.rollout.multi_turn.tool_call_budget=2048 \
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
        rollout.nnodes=4 \
        rollout.n_gpus_per_node=4 \
        rollout.total_rollout_steps=3200 \
        trainer.total_epochs=30 \
        trainer.save_freq=-1 \
        trainer.test_freq=-1 \
        trainer.val_before_train=false \
        trainer.project_name=physcode_ablation \
        trainer.experiment_name=z4q_eq_${EXP_TS} \
        trainer.default_local_dir=$TRAIN_DIR \
        trainer.default_hdfs_dir=null \
        'trainer.logger=["console","wandb"]' \
    2>&1 | tee "$TRAIN_DIR/train.log"
DRIVER_EXIT=$?
echo "[driver] exited with $DRIVER_EXIT"
exit $DRIVER_EXIT
