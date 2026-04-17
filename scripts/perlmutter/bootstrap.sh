#!/usr/bin/env bash
# bootstrap.sh — first-run environment verification on Perlmutter.
#
# Runs on a login/devel node (CPU-only is fine). Idempotent, safe to re-run.
# Does NOT build anything — installation is setup_overlay.sh's job (run once
# per cluster before this).
#
# What it does:
#   1. Sources env.sh, confirms SIF / OVERLAY / HF_HOME / .async-extras exist
#   2. Apptainer import sanity (vllm, transformers, verl, ray, qwen3_5, numpy 2.x)
#   3. Pre-fetches Qwen/Qwen3.5-4B + IAAR-Shanghai/xVerify-7B-I into HF_HOME
#      (avoids first-run race when many ranks hit the HF API simultaneously)
#   4. Verifies training parquets are present
#
# The GPU smoke is intentionally NOT part of bootstrap — use
# scripts/perlmutter/smoke_tir_het.sbatch (or smoke_tir_companion.sbatch) instead.

set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd -P)"
cd "$ROOT"

step() { echo; echo "=== $* ==="; }
fail() { echo "FAIL: $*" >&2; exit 1; }

# --- 0. env ---
step "0  env.sh"
unset SIF OVERLAY
source "$ROOT/env.sh"
echo "  ROOT     = $ROOT"
echo "  SIF      = $SIF"
echo "  OVERLAY  = $OVERLAY"
echo "  HF_HOME  = $HF_HOME"
[[ -f "$SIF" ]]                         || fail "SIF missing — run pull_docker.sbatch"
[[ -f "$OVERLAY" ]]                     || fail "Overlay missing — run setup_overlay.sh"
[[ -d "$ROOT/.async-extras/numpy" ]]    || fail ".async-extras missing — run setup_overlay.sh (installs numpy 2.x host-side)"
mkdir -p "$HF_HOME"

APT() {
    PYTHONNOUSERSITE=1 apptainer exec \
        --overlay "$OVERLAY:ro" --no-home --bind /etc/pki:/etc/pki \
        --env "PYTHONNOUSERSITE=1" \
        --env "PYTHONPATH=$ROOT/.async-extras:/opt/phys-extras/" \
        --env "HF_HOME=$HF_HOME" \
        --env "XDG_CACHE_HOME=/tmp/.cache" \
        "$SIF" "$@"
}

# --- 1. import sanity ---
step "1  container imports"
APT python3 - <<'PY'
import vllm, transformers, verl, ray, numpy
from transformers.models.auto.configuration_auto import CONFIG_MAPPING_NAMES
print(f'  vllm={vllm.__version__}  transformers={transformers.__version__}')
print(f'  verl={verl.__version__}  ray={ray.__version__}  numpy={numpy.__version__}')
assert 'qwen3_5' in CONFIG_MAPPING_NAMES, 'qwen3_5 model type missing — rebuild overlay'
assert numpy.__version__.startswith('2.'), f'need numpy 2.x (got {numpy.__version__}); .async-extras not on PYTHONPATH'
print('  OK')
PY

# --- 2. prefetch weights ---
step "2  pre-fetch model weights"
if [[ "${SKIP_PREFETCH:-0}" == "1" ]]; then
    echo "  SKIP_PREFETCH=1 — skipping"
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

# --- 3. data parquets ---
step "3  data parquets"
# FETCH_OUT_DIR must match the --out-dir passed to scripts/fetch_dataset.py.
# Default tracks fetch_dataset.py's own default (data/processed_tir).
FETCH_OUT_DIR="${FETCH_OUT_DIR:-data/processed_tir}"
# Second path only needed if the CoT baseline run (prod_cot_het.sbatch / prod_cot_companion.sbatch) will be submitted.
# Set FETCH_COT_OUT_DIR="" or pass SKIP_COT_CHECK=1 to bypass if you only need TIR.
FETCH_COT_OUT_DIR="${FETCH_COT_OUT_DIR:-data/processed_cot}"
need=(
    "$FETCH_OUT_DIR/data/train.parquet"
    "$FETCH_OUT_DIR/data/validation.parquet"
    "$FETCH_OUT_DIR/data/test.parquet"
)
if [[ "${SKIP_COT_CHECK:-0}" != "1" && -n "$FETCH_COT_OUT_DIR" ]]; then
    need+=(
        "$FETCH_COT_OUT_DIR/data/train.parquet"
        "$FETCH_COT_OUT_DIR/data/validation.parquet"
        "$FETCH_COT_OUT_DIR/data/test.parquet"
    )
fi
missing=()
for f in "${need[@]}"; do
    [[ -f "$ROOT/$f" ]] || missing+=("$f")
done
if (( ${#missing[@]} )); then
    echo "  MISSING (looked under $FETCH_OUT_DIR):"
    for f in "${missing[@]}"; do echo "    - $f"; done
    echo
    echo "  Fetch them inside apptainer (see docs/setup.md):"
    echo "    HF_TOKEN=hf_... APT python3 scripts/fetch_dataset.py --repo-id <user>/phys-tir \\"
    echo "                                                         --out-dir $FETCH_OUT_DIR"
    echo "  Or override the check path: FETCH_OUT_DIR=<your-dir> bash scripts/perlmutter/bootstrap.sh"
    fail "parquets missing"
fi
echo "  all required parquets present under $FETCH_OUT_DIR"

echo
echo "=== bootstrap PASSED ==="
echo "  Next:"
echo "    1. (optional) python3 scripts/perlmutter/probe_xverify.py --url <server>"
echo "    2. sbatch scripts/perlmutter/smoke_tir_het.sbatch   # or smoke_tir_companion.sbatch"
