#!/bin/bash
# prepare_drsci.sh — full Dr. SCI data pipeline from scratch.
# Run once after cloning (or after any dedup/clean logic changes).
#
# Produces data/processed/drsci_physics_clean.parquet (~108k rows) —
# the ready-to-use Dr. SCI training corpus.
#
# Usage (from project root):
#   bash scripts/prepare_drsci.sh
#
# To skip the download step (if drsci_physics.parquet already exists):
#   SKIP_DOWNLOAD=1 bash scripts/prepare_drsci.sh
#
# To run the optional GPU validation step (requires a node with GPU):
#   RUN_XVERIFY=1 bash scripts/prepare_drsci.sh

set -e

export PYTHONNOUSERSITE=1   # must be exported before any apptainer invocation

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SIF=$ROOT/verl_vllm017.latest.sif
OVERLAY=$ROOT/phys-reasoner-overlay-017.img
[ -f "$ROOT/.env" ] && source "$ROOT/.env"

# Legacy overlay for the dedup step (needs working HF hub importlib.metadata)
OVERLAY_LEGACY=$ROOT/phys-reasoner-overlay.img

RUN="apptainer exec --overlay $OVERLAY:ro --bind /etc/pki:/etc/pki $SIF"
RUN_LEGACY="apptainer exec --overlay $OVERLAY_LEGACY --bind /etc/pki:/etc/pki $SIF"
RUN_GPU="apptainer exec --nv --overlay $OVERLAY_LEGACY --bind /etc/pki:/etc/pki $SIF"

RAW="$ROOT/data/processed/drsci_physics.parquet"
DEDUPED="$ROOT/data/processed/drsci_physics_deduped.parquet"
CLEAN="$ROOT/data/processed/drsci_physics_clean.parquet"

# ---------------------------------------------------------------------------

if [ "${SKIP_DOWNLOAD:-0}" = "1" ] && [ -f "$RAW" ]; then
    echo "=== [1/3] Download skipped (SKIP_DOWNLOAD=1, file exists) ==="
else
    echo "=== [1/3] Download Dr. SCI from HuggingFace ==="
    $RUN python "$ROOT/scripts/download_drsci.py" \
        --cache_dir "$ROOT/data/hf_cache" \
        --output    "$RAW"
fi

echo ""
echo "=== [2/3] Dedup (exact + fuzzy + eval-contamination check) ==="
# Uses legacy overlay: contamination check loads eval sets via the datasets library,
# which requires working HF hub importlib.metadata (broken in -017.img).
$RUN_LEGACY python "$ROOT/scripts/drsci_dedup.py" \
    --input     "$RAW" \
    --output    "$DEDUPED" \
    --corpus    "$ROOT/data/processed/candidates_deduped.parquet" \
    --cache_dir "$ROOT/data/hf_cache"

echo ""
echo "=== [3/3] Clean ground_truth strings ==="
$RUN python "$ROOT/scripts/drsci_clean.py" \
    --input  "$DEDUPED" \
    --output "$CLEAN"

echo ""
echo "=== Done ==="
echo "Dr. SCI training corpus: $CLEAN"

# ---------------------------------------------------------------------------
# Optional: audit report (informational, no GPU needed)
# ---------------------------------------------------------------------------
if [ "${RUN_AUDIT:-0}" = "1" ]; then
    echo ""
    echo "=== [optional] Audit report ==="
    $RUN python "$ROOT/scripts/drsci_audit.py" \
        --input "$CLEAN"
fi

# ---------------------------------------------------------------------------
# Optional: xVerify round-trip validation (requires GPU node)
# ---------------------------------------------------------------------------
if [ "${RUN_XVERIFY:-0}" = "1" ]; then
    echo ""
    echo "=== [optional] xVerify round-trip validation (GPU) ==="
    $RUN_GPU python "$ROOT/scripts/drsci_xverify_test.py" \
        --input     "$CLEAN" \
        --n_samples 200 \
        --model     "IAAR-Shanghai/xVerify-3B-Ib"
fi
