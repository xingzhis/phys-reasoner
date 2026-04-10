#!/bin/bash
# prepare_data.sh — full data pipeline from scratch.
# Run this once after cloning (or after any loader/dedup logic changes).
#
# Usage (from project root):
#   . .env                        # load machine-local overrides if any
#   bash scripts/prepare_data.sh

set -e

export PYTHONNOUSERSITE=1   # must be exported before any apptainer invocation

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SIF=$ROOT/verl_vllm017.latest.sif
OVERLAY=$ROOT/phys-reasoner-overlay-017.img
[ -f "$ROOT/.env" ] && source "$ROOT/.env"

RUN="apptainer exec --overlay $OVERLAY:ro --bind /etc/pki:/etc/pki $SIF"

echo "=== [1/6] Download all datasets ==="
$RUN python "$ROOT/scripts/download_datasets.py" \
    --cache_dir "$ROOT/data/hf_cache" \
    --raw_dir   "$ROOT/data/raw"

echo ""
echo "=== [2/6] Validate loaders ==="
$RUN python "$ROOT/scripts/validate_loaders.py"

echo ""
echo "=== [3/6] Build raw candidates parquet ==="
$RUN python "$ROOT/scripts/explore_quality.py" \
    --save "$ROOT/data/processed/candidates_raw.parquet"

echo ""
echo "=== [4/6] Dedup (exact + fuzzy + contamination check) ==="
$RUN python "$ROOT/scripts/run_dedup.py" \
    --input     "$ROOT/data/processed/candidates_raw.parquet" \
    --output    "$ROOT/data/processed/candidates_deduped.parquet" \
    --cache_dir "$ROOT/data/hf_cache"

echo ""
echo "=== [5/6] Data quality filter (MCQ/TF norm, explanation removal, etc.) ==="
$RUN python "$ROOT/scripts/filter_data_quality.py" \
    --input  "$ROOT/data/processed/candidates_deduped.parquet" \
    --output "$ROOT/data/processed/candidates_filtered.parquet"

echo ""
echo "=== [6/6] Build VeRL-ready training parquet (metadata enrichment + prompt rebuild) ==="
$RUN python "$ROOT/scripts/build_training_parquets.py" \
    --corpus-only \
    --corpus-input  "$ROOT/data/processed/candidates_filtered.parquet" \
    --corpus-output "$ROOT/data/processed/corpus_train.parquet"

echo ""
echo "=== Done ==="
echo "Intermediate deduped:   $ROOT/data/processed/candidates_deduped.parquet"
echo "Intermediate filtered:  $ROOT/data/processed/candidates_filtered.parquet"
echo "VeRL training corpus:   $ROOT/data/processed/corpus_train.parquet"
