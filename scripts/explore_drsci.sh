#!/bin/bash
# Explore MiniByte-666/Dr.SCI: filter physics + rule_verifiable,
# print row counts and answer-type distribution.
#
# Usage:
#   bash scripts/explore_drsci.sh
#   bash scripts/explore_drsci.sh --save data/processed/drsci_physics.parquet

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
source "$ROOT/env.sh"

export PYTHONUNBUFFERED=1

# Dataset exploration requires the legacy overlay: it has datasets==4.7.0 +
# huggingface_hub==1.7.2 with proper dist-info. The 017 overlay lacks hub
# dist-info so `import datasets` fails. No model loading here, so hub 1.7.2
# is fine (the 1.7.2 breakage only affects transformers model loading).
DATASETS_OVERLAY="${ROOT}/phys-reasoner-overlay.img"

apptainer exec \
  --overlay "$DATASETS_OVERLAY" \
  --bind /etc/pki:/etc/pki \
  "$SIF" \
  python3 "$ROOT/scripts/explore_drsci.py" \
    --cache_dir "$ROOT/data/hf_cache" \
    "$@"
