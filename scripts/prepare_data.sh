#!/bin/bash
# prepare_data.sh — full data pipeline from scratch.
# Run this once after cloning (or after any loader/dedup logic changes).
#
# Usage (from project root):
#   . .env                        # load machine-local overrides if any
#   bash scripts/prepare_data.sh

set -e

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SIF=$ROOT/verl_vllm017.latest.sif
OVERLAY=$ROOT/phys-reasoner-overlay-017.img
[ -f "$ROOT/.env" ] && source "$ROOT/.env"

RUN="PYTHONNOUSERSITE=1 apptainer exec --overlay $OVERLAY:ro --bind /etc/pki:/etc/pki $SIF"

echo "=== [1/4] Download all datasets ==="
$RUN python "$ROOT/scripts/download_datasets.py" \
    --cache_dir "$ROOT/data/hf_cache" \
    --raw_dir   "$ROOT/data/raw"

echo ""
echo "=== [2/4] Validate loaders ==="
$RUN python "$ROOT/scripts/validate_loaders.py"

echo ""
echo "=== [3/4] Build raw candidates parquet ==="
$RUN python "$ROOT/scripts/explore_quality.py" \
    --save "$ROOT/data/processed/candidates_raw.parquet"

echo ""
echo "=== [4/4] Dedup (exact + fuzzy + contamination check) ==="
$RUN python "$ROOT/scripts/run_dedup.py" \
    --input     "$ROOT/data/processed/candidates_raw.parquet" \
    --output    "$ROOT/data/processed/candidates_deduped.parquet" \
    --cache_dir "$ROOT/data/hf_cache"

echo ""
echo "=== Done ==="
echo "Training corpus: $ROOT/data/processed/candidates_deduped.parquet"
