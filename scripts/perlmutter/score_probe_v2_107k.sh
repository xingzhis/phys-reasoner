#!/bin/bash
# score_probe_v2_107k.sh — inner script to run the xVerify vLLM scorer on the
# merged 107k rollouts parquet. Designed to be driven by a trailing-command
# salloc so we can use interactive QoS without an interactive shell:
#
#   cd /pscratch/sd/b/bwhou/17-phy-reasoner/phys-reasoner
#   mkdir -p logs
#   nohup salloc -N 1 -q interactive -t 04:00:00 -C gpu -A m2651 \
#     bash scripts/perlmutter/score_probe_v2_107k.sh \
#     > logs/score_probe_v2_107k_$(date +%Y%m%d_%H%M%S).log 2>&1 &
#   disown
#
# Outputs → outputs/probe_qwen3_4b_v2_107k/rollouts_scored.parquet
# (distinct from probe1's outputs/probe_qwen3_4b/rollouts_scored.parquet).

set -euo pipefail
ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/../.." && pwd)}"
cd "$ROOT"

unset SIF OVERLAY
source "$ROOT/env.sh"
[[ -f "$SIF" && -f "$OVERLAY" ]] || { echo "ERROR: SIF/OVERLAY missing" >&2; exit 1; }

mkdir -p outputs/probe_qwen3_4b_v2_107k

MERGED="${MERGED:-$ROOT/outputs/probe_qwen3_4b_v2_107k/rollouts_merged_107k.parquet}"
OUT_DIR="${OUT_DIR:-$ROOT/outputs/probe_qwen3_4b_v2_107k}"
XV_MODEL="${XV_MODEL:-IAAR-Shanghai/xVerify-7B-I}"
GPU_MEM="${GPU_MEM:-0.85}"
# xVerify-7B-I is Qwen2-7B-based (native 32k RoPE). 16384 covers long physics
# reasoning traces that blew past 8192. A few rows still exceed even this;
# the Python scorer truncates them from the LEFT (keeping the boxed answer,
# which is at the end of the reasoning trace).
MAX_MODEL_LEN="${MAX_MODEL_LEN:-16384}"

[[ -f "$MERGED" ]] || { echo "ERROR: merged parquet missing: $MERGED" >&2; exit 1; }

echo "=== score_probe_v2_107k ==="
echo "  Job    : ${SLURM_JOB_ID:-N/A}"
echo "  Node   : $(hostname)"
echo "  GPUs   : $(nvidia-smi -L 2>/dev/null | wc -l)"
echo "  Merged : $MERGED"
echo "  Out    : $OUT_DIR"
echo "  xV mod : $XV_MODEL"
echo "  Start  : $(date -Iseconds)"

# IMPORTANT: salloc's trailing command runs on the LOGIN node, not the allocated
# compute node. We must `srun` to step onto the compute node (and get GPU
# visibility) before invoking apptainer --nv. Without this, the rule pass runs
# on the login CPU (slow / risks hanging a login host) and the vLLM xVerify
# phase would fail because no CUDA device is visible.
T0=$(date +%s)
srun --ntasks=1 --nodes=1 --gpus-per-node=1 --cpus-per-task=32 --overlap \
  bash -c "PYTHONNOUSERSITE=1 apptainer exec --nv \
  --overlay '$OVERLAY:ro' --no-home --bind /etc/pki:/etc/pki \
  --env 'PYTHONNOUSERSITE=1' \
  --env 'PYTHONUNBUFFERED=1' \
  --env 'PYTHONPATH=$ROOT/src:/opt/phys-extras/' \
  --env 'HF_HOME=$HF_HOME' \
  --env 'HF_HUB_OFFLINE=1' \
  --env 'TRANSFORMERS_OFFLINE=1' \
  --env 'XDG_CACHE_HOME=/tmp/.cache' --env 'HOME=/tmp' \
  --env 'FLASHINFER_WORKSPACE_BASE=/tmp' \
  --env 'TRITON_CACHE_DIR=/tmp/.cache/triton' \
  --env 'MPLCONFIGDIR=/tmp/.cache/matplotlib' \
  --env 'VLLM_USE_V1=1' \
  '$SIF' \
  python3 -u '$ROOT/scripts/score_and_filter_vllm.py' \
    --merged '$MERGED' \
    --out_dir '$OUT_DIR' \
    --xverify_model '$XV_MODEL' \
    --gpu_mem '$GPU_MEM' \
    --max_model_len '$MAX_MODEL_LEN' \
    --rule_timeout 5 \
    --no_filter"

RC=$?
T_END=$(date +%s)
echo "[score] exit=$RC elapsed=$((T_END-T0))s"
exit $RC
