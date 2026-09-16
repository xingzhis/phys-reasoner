#!/bin/bash
# Train-matched scoring for the 17k-latest-HF subset.
# Runs the two-pass collector + vLLM batch + replay scorer that mirrors VeRL
# training's verify_answer control flow (boxed pred, unit-appended gold,
# multi-part permutation matching).
#
# Driven by salloc trailing-command so interactive QoS works without an
# interactive shell:
#
#   cd /pscratch/sd/b/bwhou/17-phy-reasoner/phys-reasoner
#   mkdir -p logs
#   nohup salloc -N 1 -q interactive -t 04:00:00 -C gpu -A m2651 \
#     bash scripts/perlmutter/score_train_matched_17k.sh \
#     > logs/score_train_matched_17k_$(date +%Y%m%d_%H%M%S).log 2>&1 &
#   disown
#
# Output (does NOT overwrite earlier scoring files):
#   outputs/probe_qwen3_4b_v2_107k/rollouts_scored_trainmatched_17k.parquet

set -euo pipefail
ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/../.." && pwd)}"
cd "$ROOT"

unset SIF OVERLAY
source "$ROOT/env.sh"
[[ -f "$SIF" && -f "$OVERLAY" ]] || { echo "ERROR: SIF/OVERLAY missing" >&2; exit 1; }

mkdir -p outputs/probe_qwen3_4b_v2_107k

MERGED="${MERGED:-$ROOT/outputs/probe_qwen3_4b_v2_107k/rollouts_merged_17k_latest.parquet}"
OUT_DIR="${OUT_DIR:-$ROOT/outputs/probe_qwen3_4b_v2_107k}"
OUT_NAME="${OUT_NAME:-rollouts_scored_trainmatched_17k.parquet}"
RULE_CACHE="${RULE_CACHE:-$ROOT/outputs/probe_qwen3_4b_v2_107k/_rule_pass_cache.pkl}"
# NB: the rule cache was computed over the 107k merged parquet (863,816 rows);
# the 17k subset parquet has 142,616 rows in a different ORDER. The script
# checks cache length vs df length and ignores the cache on mismatch. That is
# the correct behavior here — skip the cache for this subset run.
unset RULE_CACHE_FOR_17K  # placeholder in case we later build a per-subset cache

XV_MODEL="${XV_MODEL:-IAAR-Shanghai/xVerify-7B-I}"
GPU_MEM="${GPU_MEM:-0.85}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-16384}"
ROW_TIMEOUT="${ROW_TIMEOUT:-5}"

[[ -f "$MERGED" ]] || { echo "ERROR: merged parquet missing: $MERGED" >&2; exit 1; }

echo "=== score_train_matched_17k ==="
echo "  Job    : ${SLURM_JOB_ID:-N/A}"
echo "  Node   : $(hostname)"
echo "  Merged : $MERGED"
echo "  Out    : $OUT_DIR/$OUT_NAME"
echo "  Start  : $(date -Iseconds)"

T0=$(date +%s)
srun --ntasks=1 --nodes=1 --gpus-per-node=1 --cpus-per-task=32 --overlap \
  bash -c "PYTHONNOUSERSITE=1 apptainer exec --nv \
  --overlay '$OVERLAY:ro' --no-home --bind /etc/pki:/etc/pki \
  --env 'PYTHONNOUSERSITE=1' --env 'PYTHONUNBUFFERED=1' \
  --env 'PYTHONPATH=$ROOT/src:/opt/phys-extras/' \
  --env 'HF_HOME=$HF_HOME' --env 'HF_HUB_OFFLINE=1' --env 'TRANSFORMERS_OFFLINE=1' \
  --env 'XDG_CACHE_HOME=/tmp/.cache' --env 'HOME=/tmp' \
  --env 'FLASHINFER_WORKSPACE_BASE=/tmp' \
  --env 'TRITON_CACHE_DIR=/tmp/.cache/triton' \
  --env 'MPLCONFIGDIR=/tmp/.cache/matplotlib' \
  --env 'VLLM_USE_V1=1' \
  '$SIF' \
  python3 -u '$ROOT/scripts/score_train_matched.py' \
    --merged '$MERGED' \
    --out_dir '$OUT_DIR' \
    --out_name '$OUT_NAME' \
    --xverify_model '$XV_MODEL' \
    --gpu_mem '$GPU_MEM' \
    --max_model_len '$MAX_MODEL_LEN' \
    --row_timeout '$ROW_TIMEOUT'"

RC=$?
echo "[score_train_matched_17k] exit=$RC elapsed=$(($(date +%s)-T0))s"
exit $RC
