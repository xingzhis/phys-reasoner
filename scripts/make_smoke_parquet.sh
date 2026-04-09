#!/usr/bin/env bash
# make_smoke_parquet.sh — sample a stratified smoke-test parquet from both training corpora.
#
# Run this on the login node (CPU-only, no GPU) whenever:
#   - TIR_SYSTEM_PROMPT or MCQ/TF hints change in prompts.py
#   - build_training_parquets.py is re-run (prompt or data changes)
#   - You want a fresh smoke sample with different problems
#
# Sampling strategy:
#   Corpus  (N_CORPUS rows): stratified across 4 scalar types
#                            (numerical / expression / equation / mcq),
#                            skipping list-answer rows.
#   Dr. SCI (N_DRSCI rows):  proportional to type distribution in that corpus
#                            (equation / numerical / expression / mcq only;
#                            "unknown" excluded).
#
# Output: outputs/smoke_tir_qwen35_<timestamp>/smoke<total>.parquet
# Update BASE_PARQUET in train_smoke_async.sh to point at the new file.
#
# Usage:
#   bash scripts/make_smoke_parquet.sh                    # default: 8 corpus + 8 drsci = 16
#   N_CORPUS=4 N_DRSCI=4 bash scripts/make_smoke_parquet.sh
#   SEED=123 bash scripts/make_smoke_parquet.sh

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
unset SIF OVERLAY
source "$ROOT/env.sh"

CORPUS_SRC="${CORPUS_SRC:-$ROOT/data/processed/corpus_train.parquet}"
DRSCI_SRC="${DRSCI_SRC:-$ROOT/data/processed/drsci_train.parquet}"
N_CORPUS="${N_CORPUS:-8}"   # rows from corpus (scalar types only, stratified)
N_DRSCI="${N_DRSCI:-8}"     # rows from Dr. SCI (proportional across 4 main types)
SEED="${SEED:-42}"

TOTAL=$((N_CORPUS + N_DRSCI))
TIMESTAMP=$(date +%Y%m%d.%H%M%S)
OUT_DIR="$ROOT/outputs/smoke_tir_qwen35_${TIMESTAMP}"
OUT_FILE="$OUT_DIR/smoke${TOTAL}.parquet"
mkdir -p "$OUT_DIR"

echo "=== make_smoke_parquet.sh ==="
echo "  corpus src : $CORPUS_SRC"
echo "  drsci src  : $DRSCI_SRC"
echo "  n_corpus=$N_CORPUS  n_drsci=$N_DRSCI  seed=$SEED  total=$TOTAL"
echo "  output     : $OUT_FILE"

PYTHONNOUSERSITE=1 apptainer exec \
  --overlay "$OVERLAY:ro" --no-home \
  --bind /etc/pki:/etc/pki \
  --env "PYTHONNOUSERSITE=1" \
  --env "PYTHONPATH=/opt/phys-extras/" \
  "$SIF" \
  python3 - <<PY "$CORPUS_SRC" "$DRSCI_SRC" "$OUT_FILE" "$N_CORPUS" "$N_DRSCI" "$SEED"
import sys
sys.path.insert(0, "$ROOT/src")
import pandas as pd
import math

corpus_src, drsci_src, dst, n_corpus, n_drsci, seed = \
    sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), int(sys.argv[5]), int(sys.argv[6])

SCALAR_TYPES = ["numerical", "expression", "equation", "mcq"]
parts = []

# --- corpus: stratified across 4 scalar types, skip list-answer rows ---
if n_corpus > 0:
    corpus = pd.read_parquet(corpus_src)
    corpus_type = corpus["extra_info"].apply(lambda x: x.get("answer_type"))
    corpus_scalar = corpus[corpus_type.isin(SCALAR_TYPES)].copy()
    corpus_scalar["_type"] = corpus_scalar["extra_info"].apply(lambda x: x.get("answer_type"))

    per_type = max(1, n_corpus // len(SCALAR_TYPES))
    remainder = n_corpus - per_type * len(SCALAR_TYPES)
    corpus_parts = []
    for t in SCALAR_TYPES:
        pool = corpus_scalar[corpus_scalar["_type"] == t]
        n = min(per_type, len(pool))
        if n > 0:
            corpus_parts.append(pool.sample(n=n, random_state=seed))
    # fill remainder from any type if needed
    if remainder > 0 and corpus_parts:
        leftover_pool = corpus_scalar[~corpus_scalar.index.isin(
            pd.concat(corpus_parts).index)]
        extra = min(remainder, len(leftover_pool))
        if extra > 0:
            corpus_parts.append(leftover_pool.sample(n=extra, random_state=seed + 1))

    corpus_sample = pd.concat(corpus_parts, ignore_index=True).drop(columns=["_type"])
    parts.append(corpus_sample)
    type_counts = corpus_sample["extra_info"].apply(lambda x: x.get("answer_type")).value_counts().to_dict()
    print(f"Corpus: sampled {len(corpus_sample)} rows — {type_counts}")

# --- Dr. SCI: proportional to equation/numerical/expression/mcq counts ---
if n_drsci > 0:
    drsci = pd.read_parquet(drsci_src)
    drsci_type = drsci["extra_info"].apply(lambda x: x.get("answer_type"))
    drsci_main = drsci[drsci_type.isin(SCALAR_TYPES)].copy()
    drsci_main["_type"] = drsci_main["extra_info"].apply(lambda x: x.get("answer_type"))

    type_totals = drsci_main["_type"].value_counts()
    grand_total = type_totals.sum()
    drsci_parts = []
    allocated = 0
    for t in SCALAR_TYPES:
        n = min(round(n_drsci * type_totals.get(t, 0) / grand_total), len(drsci_main[drsci_main["_type"] == t]))
        allocated += n
        if n > 0:
            drsci_parts.append(drsci_main[drsci_main["_type"] == t].sample(n=n, random_state=seed))
    # top up to n_drsci if rounding lost rows
    still_need = n_drsci - allocated
    if still_need > 0 and drsci_parts:
        leftover_pool = drsci_main[~drsci_main.index.isin(
            pd.concat(drsci_parts).index)]
        extra = min(still_need, len(leftover_pool))
        if extra > 0:
            drsci_parts.append(leftover_pool.sample(n=extra, random_state=seed + 2))

    drsci_sample = pd.concat(drsci_parts, ignore_index=True).drop(columns=["_type"])
    parts.append(drsci_sample)
    type_counts = drsci_sample["extra_info"].apply(lambda x: x.get("answer_type")).value_counts().to_dict()
    print(f"Dr. SCI: sampled {len(drsci_sample)} rows — {type_counts}")

sample = pd.concat(parts, ignore_index=True).sample(frac=1, random_state=seed).reset_index(drop=True)
sample.to_parquet(dst, index=False)

print(f"\nWrote {len(sample)} rows → {dst}")
print("\nRow summary:")
for i, row in sample.iterrows():
    ei = row["extra_info"]
    src = ei.get("data_source", ei.get("source", "?"))
    user_tail = row["prompt"][1]["content"][-80:]
    print(f"  [{i}] source={src!r}  answer_type={ei.get('answer_type')!r}")
    print(f"       user tail: {user_tail!r}")
PY

echo ""
echo "Done. Update BASE_PARQUET in scripts/train_smoke_async.sh:"
echo "  BASE_PARQUET=\"$OUT_FILE\""
