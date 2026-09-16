#!/bin/bash
# score_probe_v2_107k.sbatch — interactive-QoS xVerify scoring for the
# 107,977-row probe_v2 merged rollouts.
#
# Runs scripts/score_and_filter_vllm.py in vLLM-batched mode (single GPU node,
# no HTTP server needed). Per the script docstring, scoring ~131k rollouts took
# ~10-15 min on one H200; our 863,816 rollouts should take ~70-90 min — well
# within the 4 h interactive cap.
#
# QoS: Perlmutter's `interactive` QoS is salloc-only (sbatch is rejected with
# "Cannot submit batch jobs to gpu_interactive_ss11"). We use `shared` instead
# which:
#   - allows sbatch with partial-node GPU requests (1 of 4 GPUs here)
#   - is NOT premium, so does not contend with the user's training tests there
#   - typically has short queue for 1-GPU jobs
# If the shared queue happens to be slow, alternative: user starts
#   salloc --nodes 1 --qos interactive --time 04:00:00 -C gpu --account=m2651
# and then runs the apptainer command below by hand; same command, same output.
#
# Inputs:
#   outputs/probe_qwen3_4b_v2_107k/rollouts_merged_107k.parquet  (from merge_probe_v2_107k.py)
#
# Outputs (distinct from probe1's outputs/probe_qwen3_4b/ tree):
#   outputs/probe_qwen3_4b_v2_107k/rollouts_scored.parquet
#
# NB: --no_filter is passed so this ONLY scores; it does not rebuild the
# filtered training parquets. Filter decisions happen downstream once the user
# has reviewed score distributions against the 107k source.

#SBATCH --job-name=score_probe_v2_107k
#SBATCH --output=logs/%x_%j.log
#SBATCH --error=logs/%x_%j.err
#SBATCH -A m2651
#SBATCH -C gpu&hbm40g
#SBATCH -q shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus=1
#SBATCH --time=04:00:00

set -euo pipefail
ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/../.." && pwd)}"
cd "$ROOT"

unset SIF OVERLAY
source "$ROOT/env.sh"
[[ -f "$SIF" && -f "$OVERLAY" ]] || { echo "ERROR: SIF/OVERLAY missing" >&2; exit 1; }

mkdir -p logs outputs/probe_qwen3_4b_v2_107k

MERGED="${MERGED:-$ROOT/outputs/probe_qwen3_4b_v2_107k/rollouts_merged_107k.parquet}"
OUT_DIR="${OUT_DIR:-$ROOT/outputs/probe_qwen3_4b_v2_107k}"
XV_MODEL="${XV_MODEL:-IAAR-Shanghai/xVerify-7B-I}"
GPU_MEM="${GPU_MEM:-0.85}"

[[ -f "$MERGED" ]] || { echo "ERROR: merged parquet missing: $MERGED" >&2; exit 1; }

echo "=== score_probe_v2_107k ==="
echo "  Job    : $SLURM_JOB_ID"
echo "  Node   : $(hostname)"
echo "  GPUs   : $(nvidia-smi -L 2>/dev/null | wc -l)"
echo "  Merged : $MERGED"
echo "  Out    : $OUT_DIR"
echo "  xV mod : $XV_MODEL"
echo "  Start  : $(date -Iseconds)"

T0=$(date +%s)
PYTHONNOUSERSITE=1 apptainer exec --nv \
  --overlay "$OVERLAY:ro" --no-home --bind /etc/pki:/etc/pki \
  --env "PYTHONNOUSERSITE=1" \
  --env "PYTHONUNBUFFERED=1" \
  --env "PYTHONPATH=$ROOT/src:/opt/phys-extras/" \
  --env "HF_HOME=$HF_HOME" \
  --env "HF_HUB_OFFLINE=1" \
  --env "TRANSFORMERS_OFFLINE=1" \
  --env "XDG_CACHE_HOME=/tmp/.cache" --env "HOME=/tmp" \
  --env "VLLM_USE_V1=1" \
  "$SIF" \
  python3 "$ROOT/scripts/score_and_filter_vllm.py" \
    --merged "$MERGED" \
    --out_dir "$OUT_DIR" \
    --xverify_model "$XV_MODEL" \
    --gpu_mem "$GPU_MEM" \
    --no_filter

RC=$?
T_END=$(date +%s)
echo "[score] exit=$RC elapsed=$((T_END-T0))s"
exit $RC
