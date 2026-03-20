# sketches/

Speculative implementations written before the actual datasets were inspected.
Kept for reference — field names, answer formats, and config names were inferred
from papers/READMEs, not from real data.

Do not import from here. The real implementations live in `src/phys_reasoner/data/`
once written against actual HuggingFace dataset schemas.

Files:
- `data/loaders.py` — guessed loader implementations for all 5 datasets
- `data/prepare.py` — dedup/split/parquet pipeline (depends on loaders)
- `tests/test_loaders.py` — tests using hand-coded fake rows
- `scripts/catalog_answers.py` — answer format catalog script
