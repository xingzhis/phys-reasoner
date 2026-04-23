#!/bin/bash
# probe_qwen3_4b_sentinel.sbatch — 1-node sentinel to:
#   1. Measure gpu_regular hbm40g queue wait for 1-node 1h jobs
#   2. Validate 4-GPU parallel probe throughput on the real data
#
# Runs: 200 prompts from data/processed_tir/data/train.parquet × n=4 rollouts
# Split across 4 GPUs (50 prompts per GPU = 200 rollouts per GPU). At measured
# ~15 rollouts/min/GPU, each GPU finishes in ~13-15 min. Total wall ~20 min
# after init. If this fits in 1h walltime with comfortable margin, the full
# fleet (20 × 1-node jobs, each 850 rollouts/GPU) is viable.
#
# Output: outputs/probe_qwen3_4b/sentinel/chunk_<gpu>.parquet
# Also emits [sentinel-metric] lines to log for throughput calculation.

#SBATCH --job-name=probe_qwen3_4b_sentinel
#SBATCH --output=logs/%x_%j.log
#SBATCH --error=logs/%x_%j.err
#SBATCH -A m2651
#SBATCH -C gpu&hbm40g
#SBATCH -q regular
#SBATCH -t 01:00:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-node=4

set -euo pipefail
ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/../.." && pwd)}"
cd "$ROOT"
mkdir -p logs outputs/probe_qwen3_4b/sentinel

unset SIF OVERLAY
source "$ROOT/env.sh"
[[ -f "$SIF" && -f "$OVERLAY" ]] || { echo "ERROR: SIF/OVERLAY missing" >&2; exit 1; }

echo "=== probe_qwen3_4b_sentinel ==="
echo "  Job     : $SLURM_JOB_ID"
echo "  Node    : $(hostname)"
echo "  Submit  : ${SLURM_SUBMIT_TIME:-unknown}"
echo "  Start   : $(date -Iseconds)"
nvidia-smi -L

# ---- config ----
TRAIN_PARQUET="${TRAIN_PARQUET:-$ROOT/data/processed_tir/data/train.parquet}"
MODEL="${MODEL:-Qwen/Qwen3-4B}"
N_PROMPTS_TOTAL="${N_PROMPTS_TOTAL:-200}"    # 50/GPU × 4 GPUs
N_ROLLOUTS="${N_ROLLOUTS:-4}"
TEMPERATURE="${TEMPERATURE:-1.4}"
SEED="${SEED:-42}"
OUT_BASE="${OUT_BASE:-$ROOT/outputs/probe_qwen3_4b/sentinel}"

# Budgets — match training
THINKING_BUDGET=12288
TOOL_CALL_BUDGET=2048
ANSWER_BUDGET=4096
MAX_PROMPT_LEN=1024
# TIR mode; probe uses same tool-call agent loop but we skip phase-2 (no
# xverify scoring in probe proper — scoring is a separate post-step)
MODE="tir"

APT_CMD=(
  apptainer exec --nv
  --overlay "$OVERLAY:ro" --no-home --bind /etc/pki:/etc/pki
  --env "PYTHONNOUSERSITE=1" --env "PYTHONUNBUFFERED=1"
  --env "PYTHONPATH=$ROOT/src:$ROOT/.async-extras:/opt/phys-extras/"
  --env "HF_HOME=$HF_HOME" --env "XDG_CACHE_HOME=/tmp/.cache" --env "HOME=/tmp"
  --env "FLASHINFER_WORKSPACE_BASE=/tmp"
  --env "TRITON_CACHE_DIR=/tmp/.cache/triton"
  --env "MPLCONFIGDIR=/tmp/.cache/matplotlib"
  "$SIF"
)

# Split N_PROMPTS_TOTAL across 4 GPUs
PROMPTS_PER_GPU=$(( (N_PROMPTS_TOTAL + 3) / 4 ))
echo "[sentinel] $N_PROMPTS_TOTAL prompts × $N_ROLLOUTS rollouts across 4 GPUs"
echo "[sentinel] Prompts per GPU: $PROMPTS_PER_GPU"

T_LAUNCH=$(date +%s)
pids=()
for gpu in 0 1 2 3; do
  s=$(( gpu * PROMPTS_PER_GPU ))
  e=$(( (gpu + 1) * PROMPTS_PER_GPU ))
  (( e > N_PROMPTS_TOTAL )) && e=$N_PROMPTS_TOTAL
  (( s >= N_PROMPTS_TOTAL )) && continue

  OUT_DIR="$OUT_BASE/gpu_$gpu"
  mkdir -p "$OUT_DIR"
  echo "[sentinel] GPU $gpu: rows [$s, $e) → $OUT_DIR"

  CUDA_VISIBLE_DEVICES=$gpu "${APT_CMD[@]}" \
    python3 -u eval/inference/rollout.py \
      --mode "$MODE" \
      --model "$MODEL" \
      --parquet "$TRAIN_PARQUET" \
      --n -1 \
      --n_rollouts $N_ROLLOUTS \
      --start_idx $s --end_idx $e \
      --seed $SEED \
      --temperature $TEMPERATURE \
      --top_p 1.0 \
      --max_prompt_len $MAX_PROMPT_LEN \
      --thinking_budget $THINKING_BUDGET \
      --tool_call_budget $TOOL_CALL_BUDGET \
      --answer_budget $ANSWER_BUDGET \
      --gpu_mem 0.75 \
      --out_dir "$OUT_DIR" \
      > "logs/${SLURM_JOB_NAME}_${SLURM_JOB_ID}_gpu${gpu}.log" 2>&1 &
  pids+=($!)
done

echo "[sentinel] Launched ${#pids[@]} worker processes. Waiting..."
RC=0
for pid in "${pids[@]}"; do
  wait "$pid" || { echo "[sentinel] worker $pid FAILED"; RC=1; }
done
T_END=$(date +%s)
ELAPSED=$(( T_END - T_LAUNCH ))

TOTAL_ROLLOUTS=$(( N_PROMPTS_TOTAL * N_ROLLOUTS ))
ROLLOUTS_PER_GPU_MIN=$(awk -v r=$TOTAL_ROLLOUTS -v s=$ELAPSED 'BEGIN { printf "%.2f", r / 4.0 / (s / 60.0) }')

echo "[sentinel-metric] total_rollouts=$TOTAL_ROLLOUTS"
echo "[sentinel-metric] elapsed_seconds=$ELAPSED"
echo "[sentinel-metric] rollouts_per_gpu_min=$ROLLOUTS_PER_GPU_MIN"
echo "[sentinel-metric] projected_fleet_size_for_17827_n4_in_1h=$(awk -v rate=$ROLLOUTS_PER_GPU_MIN 'BEGIN { printf "%.0f", 17827 * 4 / (rate * 60) }') GPUs"
echo "[sentinel] Exit: $RC"
exit $RC
