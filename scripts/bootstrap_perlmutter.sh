#!/usr/bin/env bash
# bootstrap_perlmutter.sh — first-run setup for a fresh cluster (Perlmutter or any).
#
# Idempotent. Safe to re-run. Designed to be the first thing the collaborator runs
# on Perlmutter on bring-up day. Steps 1–4 are CPU-only and can run on a login or
# devel node; step 5 (smoke) auto-skips if no GPU is visible.
#
# Prerequisites:
#   1. Repo cloned to $ROOT
#   2. SIF + overlay + .async-extras built — i.e. these have been run already:
#         sbatch pull_docker.sbatch
#         bash scripts/setup_overlay.sh
#   3. .env present with SBATCH_PARTITION (and SBATCH_QOS / SBATCH_ACCOUNT if cluster needs them)
#   4. Outbound HTTPS available so HF can download Qwen3.5-4B + xVerify-7B
#
# This script does NOT install anything — installation is setup_overlay.sh's job.
# It only verifies the environment, pre-fetches model weights, and runs a smoke.
#
# What it does:
#   1. Sources env.sh, verifies SIF / OVERLAY / HF_HOME / .async-extras
#   2. Container import sanity (vllm, transformers, verl, ray, qwen3_5, numpy 2.x)
#   3. Pre-fetches Qwen/Qwen3.5-4B + IAAR-Shanghai/xVerify-7B-I into HF_HOME
#   4. Verifies required parquets exist
#   5. Optional 1-GPU async smoke (1 trainer + 1 rollout, TOTAL_STEPS=1)
#
# Usage:
#   bash scripts/bootstrap_perlmutter.sh                # CPU node: stops after step 5
#   srun --gres=gpu:1 --pty bash scripts/bootstrap_perlmutter.sh   # full pipeline incl. smoke

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd -P)"
cd "$ROOT"

step() { echo; echo "=== $* ==="; }
fail() { echo "FAIL: $*" >&2; exit 1; }

# ---------------------------------------------------------------------------
# 0. Environment
# ---------------------------------------------------------------------------
step "0  env.sh"
unset SIF OVERLAY
source "$ROOT/env.sh"
echo "  ROOT     = $ROOT"
echo "  SIF      = $SIF"
echo "  OVERLAY  = $OVERLAY"
echo "  HF_HOME  = $HF_HOME"
[[ -f "$SIF" ]]     || fail "SIF not found — run 'sbatch pull_docker.sbatch' first"
[[ -f "$OVERLAY" ]] || fail "Overlay not found — run pull_docker.sbatch then 'bash scripts/setup_overlay.sh'"
[[ -d "$ROOT/.async-extras/numpy" ]] || fail ".async-extras missing — run 'bash scripts/setup_overlay.sh' (it now installs them as step 5)"
mkdir -p "$HF_HOME"

# Helper: run a command inside the container with the same env every script uses.
APT() {
    local nvflag="${APT_NV:-}"
    PYTHONNOUSERSITE=1 apptainer exec $nvflag \
        --overlay "$OVERLAY:ro" --no-home --bind /etc/pki:/etc/pki \
        --env "PYTHONNOUSERSITE=1" \
        --env "PYTHONPATH=$ROOT/.async-extras:/opt/phys-extras/" \
        --env "HF_HOME=$HF_HOME" \
        --env "XDG_CACHE_HOME=/tmp/.cache" \
        "$SIF" "$@"
}

# ---------------------------------------------------------------------------
# 1. Container import sanity
# ---------------------------------------------------------------------------
step "1  container imports"
APT python3 - <<'PY'
import vllm, transformers, verl, ray, numpy
from transformers.models.auto.configuration_auto import CONFIG_MAPPING_NAMES
print('  vllm        :', vllm.__version__)
print('  transformers:', transformers.__version__)
print('  verl        :', verl.__version__)
print('  ray         :', ray.__version__)
print('  numpy       :', numpy.__version__)
assert 'qwen3_5' in CONFIG_MAPPING_NAMES, 'qwen3_5 model type missing — check setup_overlay.sh'
assert numpy.__version__.startswith('2.'), f'numpy 2.x expected, got {numpy.__version__} — check .async-extras on PYTHONPATH'
print('  qwen3_5 model type: OK')
print('  numpy 2.x         : OK')
PY

# ---------------------------------------------------------------------------
# 2. Pre-fetch model weights into HF_HOME so the first training run does not
#    race the network. snapshot_download is idempotent — already-cached files
#    are skipped.
# ---------------------------------------------------------------------------
step "2  pre-fetch model weights"
# Optional but recommended: avoids multi-rank lockfile contention on first
# distributed run. Set SKIP_PREFETCH=1 to skip if you trust the network.
if [[ "${SKIP_PREFETCH:-0}" == "1" ]]; then
    echo "  SKIP_PREFETCH=1 — skipping (HF will download on first train run)"
else
    APT python3 - <<'PY'
from huggingface_hub import snapshot_download
for repo in ['Qwen/Qwen3.5-4B', 'IAAR-Shanghai/xVerify-7B-I']:
    print(f'  fetching {repo} ...')
    p = snapshot_download(
        repo_id=repo,
        allow_patterns=['*.json', '*.safetensors', '*.txt', 'tokenizer*', '*.model'],
    )
    print(f'    -> {p}')
PY
fi

# ---------------------------------------------------------------------------
# 3. Data parquets
# ---------------------------------------------------------------------------
step "3  data parquets"
need=(
    "data/processed/drsci_train.parquet"
    "data/processed/drsci_dev.parquet"
    "data/processed/corpus_train.parquet"
    "data/processed/corpus_dev.parquet"
)
missing=()
for f in "${need[@]}"; do
    [[ -f "$ROOT/$f" ]] || missing+=("$f")
done
if (( ${#missing[@]} )); then
    echo "  MISSING:"
    for f in "${missing[@]}"; do echo "    - $f"; done
    echo
    echo "  Build pipeline (see .claude/plans/data-pipeline.md):"
    echo "    bash scripts/prepare_drsci.sh"
    echo "    APT python3 scripts/build_training_parquets.py"
    echo "    APT python3 scripts/split_train_dev_test.py"
    echo "    APT python3 scripts/subsample_probe.py"
    fail "data parquets missing — build them then re-run this script"
fi
echo "  all required parquets present"

# ---------------------------------------------------------------------------
# 4. 1-GPU smoke. Requires a GPU allocation. Skipped (with warning) otherwise.
# ---------------------------------------------------------------------------
step "4  single-GPU async smoke"
if ! command -v nvidia-smi >/dev/null 2>&1 || ! nvidia-smi -L >/dev/null 2>&1; then
    echo "  no GPU visible — skipping smoke"
    echo "  re-run inside a GPU allocation:"
    echo "    srun --gres=gpu:1 --time=30:00 --pty bash scripts/bootstrap_perlmutter.sh"
    echo
    echo "=== bootstrap (steps 0-3) PASSED ==="
    exit 0
fi

nvidia-smi -L | sed 's/^/  /'

TOTAL_STEPS=1 \
TRAIN_BATCH=2 \
ROLLOUT_N=2 \
N_GPUS_ROLLOUT=1 \
N_GPUS_TRAIN=1 \
NNODES_ROLLOUT=1 \
NNODES_TRAIN=1 \
SAVE_FREQ=-1 \
EXPERIMENT="bootstrap_smoke" \
LOGGERS=console \
THINKING_BUDGET= \
TOOL_CALL_BUDGET= \
ANSWER_BUDGET= \
bash "$ROOT/scripts/train_async.sh"

echo
echo "=== bootstrap PASSED — ready for 2-node rehearsal ==="
echo "  next: sbatch scripts/train_async_2node.sbatch"
