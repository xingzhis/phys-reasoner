#!/usr/bin/env bash
# make_smoke_parquet.sh — sample a small smoke-test parquet from the training data.
#
# Run this on the login node (CPU-only, no GPU) whenever:
#   - TIR_SYSTEM_PROMPT or MCQ/TF hints change in prompts.py
#   - build_training_parquets.py is re-run (prompt or data changes)
#   - You want a fresh smoke sample with different problems
#
# Output: outputs/smoke_tir_qwen35_<timestamp>/smoke.parquet
# Update train_smoke.sbatch BASE_PARQUET to point at the new file.
#
# Usage:
#   bash scripts/make_smoke_parquet.sh                    # default: 2 corpus + 1 MCQ
#   N_CORPUS=4 N_MCQ=2 bash scripts/make_smoke_parquet.sh
#   SEED=123 bash scripts/make_smoke_parquet.sh

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
unset SIF OVERLAY
source "$ROOT/env.sh"

CORPUS_SRC="${CORPUS_SRC:-$ROOT/data/processed/corpus_train.parquet}"
DRSCI_SRC="${DRSCI_SRC:-$ROOT/data/processed/drsci_train.parquet}"
N_CORPUS="${N_CORPUS:-2}"   # rows from corpus
N_MCQ="${N_MCQ:-1}"         # MCQ rows from drsci (for hint verification)
SEED="${SEED:-42}"

TIMESTAMP=$(date +%Y%m%d.%H%M%S)
OUT_DIR="$ROOT/outputs/smoke_tir_qwen35_${TIMESTAMP}"
OUT_FILE="$OUT_DIR/smoke.parquet"
mkdir -p "$OUT_DIR"

echo "=== make_smoke_parquet.sh ==="
echo "  corpus src : $CORPUS_SRC"
echo "  drsci src  : $DRSCI_SRC"
echo "  n_corpus=$N_CORPUS  n_mcq=$N_MCQ  seed=$SEED"
echo "  output     : $OUT_FILE"

PYTHONNOUSERSITE=1 apptainer exec \
  --overlay "$OVERLAY:ro" --no-home \
  --bind /etc/pki:/etc/pki \
  --env "PYTHONNOUSERSITE=1" \
  --env "PYTHONPATH=/opt/phys-extras/" \
  "$SIF" \
  python3 - <<PY "$CORPUS_SRC" "$DRSCI_SRC" "$OUT_FILE" "$N_CORPUS" "$N_MCQ" "$SEED"
import sys
sys.path.insert(0, "$ROOT/src")
import pandas as pd

corpus_src, drsci_src, dst, n_corpus, n_mcq, seed = \
    sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), int(sys.argv[5]), int(sys.argv[6])

parts = []

if n_corpus > 0:
    corpus = pd.read_parquet(corpus_src)
    parts.append(corpus.sample(n=min(n_corpus, len(corpus)), random_state=seed))
    print(f"Sampled {n_corpus} rows from corpus ({len(corpus)} total)")

if n_mcq > 0:
    drsci = pd.read_parquet(drsci_src)
    mcq_pool = drsci[drsci["extra_info"].apply(lambda x: x.get("answer_type") == "mcq")]
    parts.append(mcq_pool.sample(n=min(n_mcq, len(mcq_pool)), random_state=seed))
    print(f"Sampled {n_mcq} MCQ rows from drsci ({len(mcq_pool)} MCQ total)")

sample = pd.concat(parts, ignore_index=True)
sample.to_parquet(dst, index=False)

print(f"\nWrote {len(sample)} rows → {dst}")
print("\nRow summary:")
for i, row in sample.iterrows():
    ei = row["extra_info"]
    user_tail = row["prompt"][1]["content"][-100:]
    sys_tail  = row["prompt"][0]["content"][-60:]
    print(f"  [{i}] answer_type={ei.get('answer_type')!r}")
    print(f"       user tail: {user_tail!r}")
    print(f"       sys tail:  {sys_tail!r}")
PY

echo ""
echo "Done. Update train_smoke.sbatch:"
echo "  BASE_PARQUET=\"$OUT_FILE\""
