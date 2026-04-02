#!/usr/bin/env bash
# setup_overlay.sh — install all packages into the active SIF+overlay.
#
# Reads SIF and OVERLAY from env.sh (or .env override), so just update env.sh
# to point at a new SIF and re-run this script — no edits needed here.
#
# Prerequisites: SIF exists and overlay has been created (pull_docker.sbatch).
#
# Usage:
#   bash scripts/setup_overlay.sh

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
source "$ROOT/env.sh"

echo "=== setup_overlay.sh ==="
echo "  SIF:     $SIF"
echo "  OVERLAY: $OVERLAY"
echo ""

if [[ ! -f "$SIF" ]]; then
    echo "ERROR: SIF not found: $SIF"
    echo "  Run: sbatch pull_docker.sbatch"
    exit 1
fi
if [[ ! -f "$OVERLAY" ]]; then
    echo "ERROR: Overlay not found: $OVERLAY"
    echo "  Run: sbatch pull_docker.sbatch"
    exit 1
fi

APT() {
    PYTHONNOUSERSITE=1 apptainer exec \
        --overlay "$OVERLAY" \
        --no-home \
        --bind /etc/pki:/etc/pki \
        --env "PYTHONNOUSERSITE=1" \
        --env "PYTHONPATH=/opt/phys-extras/" \
        --env "HF_HOME=$HF_HOME" \
        "$SIF" "$@"
}

# --- 1. Verify SIF base is reachable and log starting versions ---
echo "=== 1/5  Verify SIF base ==="
APT python3 -c "
import vllm, huggingface_hub, transformers
print('  vllm:         ', vllm.__version__)
print('  hub:          ', huggingface_hub.__version__)
print('  transformers: ', transformers.__version__, '(will be upgraded to 5.3.0)')
# vllm must be >= 0.7.0 for verl_repo compatibility
from packaging.version import Version
v = Version(vllm.__version__)
if v < Version('0.7.0'):
    raise SystemExit(f'ERROR: vllm {vllm.__version__} < 0.7.0 — wrong SIF (need vllm017.latest or newer)')
print('  vllm >= 0.7.0: OK')
"

# --- 2. Project (phys-reasoner) with dev extras ---
# Installs all declared dependencies (math-verify, pint, scipy, sympy, etc.)
# plus dev extras (pytest, ruff). No separate pip install needed for those.
echo ""
echo "=== 2/5  Install phys-reasoner[dev] ==="
APT pip install --no-cache-dir -e "$ROOT[dev]"

# --- 3. verl from local repo, --no-deps (SIF already has all verl deps) ---
echo ""
echo "=== 3/5  Install verl --no-deps ==="
VERL_REPO="$ROOT/verl_repo"
if [[ -d "$VERL_REPO" ]]; then
    echo "  Using local verl_repo: $VERL_REPO"
    APT pip install --no-deps -e "$VERL_REPO"
else
    echo "  verl_repo not found — installing latest from PyPI"
    APT pip install --no-deps verl
fi

# --- 4. Install transformers 5.3.0 + hub 1.8.0 + flash-linear-attention into /opt/phys-extras/ ---
# Strategy: install to a SEPARATE directory (/opt/phys-extras/) rather than overwriting
# the SIF's dist-packages. Set PYTHONPATH=/opt/phys-extras/ to prioritize these newer
# versions at runtime. This avoids OverlayFS whiteout issues on RHEL 8 compute nodes where
# pip-uninstall/reinstall of SIF packages fails silently (compute node sees old 0.36.2 hub).
#
# Use --no-deps since the SIF already has compatible tokenizers (0.22.2), torch, etc.
# Only install what's newer than the SIF: transformers 5.3.0, hub 1.8.0, hf-xet 1.4.3,
# and flash-linear-attention with its fla-core dependency.
# (verl PR #5381: https://github.com/verl-project/verl/pull/5381)
echo ""
echo "=== 4/5  Install transformers==5.3.0 + hub==1.8.0 + flash-linear-attention ==="
APT pip install --no-cache-dir --no-deps --target /opt/phys-extras/ \
    "transformers==5.3.0" \
    "huggingface_hub==1.8.0" \
    "hf-xet==1.4.3" \
    "flash-linear-attention==0.4.2" \
    "fla-core==0.4.2"

# --- 5. Final smoke check ---
echo ""
echo "=== 5/5  Smoke-check ==="
APT python3 -c "
import verl, transformers, vllm
from transformers import AutoConfig
from transformers.models.auto.configuration_auto import CONFIG_MAPPING_NAMES
import os

print('  verl:         ', verl.__version__)
print('  transformers: ', transformers.__version__)
print('  vllm:         ', vllm.__version__)
supported = 'qwen3_5' in CONFIG_MAPPING_NAMES
print('  qwen3_5:      ', 'SUPPORTED' if supported else 'MISSING — transformers upgrade failed?')
if not supported:
    raise SystemExit(1)

hf_home = '$HF_HOME'
model_path = os.path.join(hf_home, 'hub', 'models--Qwen--Qwen3.5-4B', 'snapshots')
snaps = os.listdir(model_path) if os.path.exists(model_path) else []
if snaps:
    cfg = AutoConfig.from_pretrained(os.path.join(model_path, snaps[0]), local_files_only=True)
    print(f'  Qwen3.5-4B:   model_type={cfg.model_type} ✓')
else:
    print('  Qwen3.5-4B:   not in cache — skipping')
"

echo ""
echo "=== Finalize  e2fsck ==="
# Run e2fsck to mark the overlay filesystem clean so compute nodes can mount it :ro.
# Without this, an unclean ext2 superblock can prevent proper mounting on some nodes.
e2fsck -fp "$OVERLAY" || true

echo ""
echo "=== DONE — overlay is ready ==="
echo "  sbatch scripts/smoke_tir_qwen35.sbatch"
