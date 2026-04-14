"""Push merged train/validation/test parquet splits to a private HF dataset.

Pre-requisite: run scripts/merge_splits.py to produce the three files in
data/processed/merged/ (union of curated + Dr. SCI with `pool` column).

Usage:
  HF_TOKEN=hf_xxx python scripts/push_dataset.py \
    --repo-id <user>/phys-tir \
    --data-dir data/processed/merged

Repo layout:
  data/train.parquet    data/validation.parquet    data/test.parquet
  README.md   (declares a single config with the three standard splits)
"""
import argparse
from pathlib import Path

from huggingface_hub import HfApi, create_repo

SPLITS = ["train", "validation", "test"]

README_TEMPLATE = """\
---
license: other
configs:
- config_name: default
  data_files:
  - split: train
    path: data/train.parquet
  - split: validation
    path: data/validation.parquet
  - split: test
    path: data/test.parquet
---

# phys-tir

Physics problem dataset for tool-integrated reasoning (TIR) RLVR training.
Pre-split train / validation / test, each a union of two pools:

| Pool | Train | Validation | Test |
|---|---|---|---|
| `curated` (5 public physics benchmarks) | 6,817 | 200 | 200 |
| `drsci` (Dr. SCI physics subset)        | 102,563 | 2,000 | 2,000 |
| **total**                                | **109,380** | **2,200** | **2,200** |

Each row carries a top-level `pool` column (`"curated"` / `"drsci"`) so the
two sources can be separated for ablations.

## Schema

```
data_source   : large_string       # original per-row source tag
prompt        : list<struct<content: string, role: string>>
reward_model  : struct<ground_truth: string, style: string>
extra_info    : struct<...>        # union schema (see below); absent fields are null
pool          : string             # "curated" | "drsci"
```

`extra_info` is a union of the fields from both pools; fields that originate
from only one pool are null on rows from the other pool. No rows were dropped
during merge and no fields were discarded. The only type coercion is
`extra_info.difficulty`, which was promoted to string (Dr. SCI's float64
difficulty is stringified, preserving the numeric value).

| Field | Type | Populated on |
|---|---|---|
| answer_type | string | both |
| problem | string | both |
| tolerance | float64 | both |
| unit | string | both |
| difficulty | string | both (drsci stringified from float) |
| domain, domain_coarse, primary_answer_type, source | string | curated only |
| from, subject | string | drsci only |

## Pools

### `curated` — five public physics benchmarks
Filtered to rule-verifiable answer types (numerical, expression, equation, MCQ,
true/false) and deduped across sources:

| Source | Rows (approx) | Notes |
|---|---|---|
| UGPhysics (EN) | ~5,520 | NV / EX / EQ / MC / TF, dirty answer-type labels cleaned |
| desimfj/PHYSICS | ~1,500 | non-Open-end only, Chinese + multi-alternative rows dropped |
| OlympiadBench OE_TO_physics_en_COMP | 236 | text-only EN competition |
| SciBench-RL (physics subset) | 427 | fund / thermo / quan / calculus; chem excluded |
| PHYBench (filtered) | ~700 | simple algebraic subset; prose answers dropped |

### `drsci` — Dr. SCI physics
Sourced from the unofficial release at
[MiniByte-666/Dr.SCI](https://huggingface.co/datasets/MiniByte-666/Dr.SCI),
re-implementing the dataset described in the Dr. SCI paper
([arXiv:2602.08321](https://arxiv.org/abs/2602.08321)). Filtered to physics
problems with rule-verifiable answers (numerical / expression / equation /
MCQ / true-false); non-verifiable or degenerate entries removed.


## Loading for VeRL

```python
from huggingface_hub import snapshot_download
path = snapshot_download("<user>/phys-tir", repo_type="dataset",
                         allow_patterns=["data/*.parquet"])
# data.train_files=path/data/train.parquet
# data.val_files=path/data/validation.parquet
```

See `scripts/fetch_dataset.py` for the full wiring.
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-id", required=True, help="e.g. xingzhisun/phys-tir")
    ap.add_argument("--data-dir", default="data/processed/merged")
    ap.add_argument("--public", action="store_true", help="make repo public (default: private)")
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    files = [data_dir / f"{split}.parquet" for split in SPLITS]
    missing = [str(p) for p in files if not p.exists()]
    if missing:
        raise SystemExit(f"Missing parquet files: {missing} — run scripts/merge_splits.py first")

    api = HfApi()
    create_repo(args.repo_id, repo_type="dataset", private=not args.public, exist_ok=True)

    for path in files:
        api.upload_file(
            path_or_fileobj=str(path),
            path_in_repo=f"data/{path.name}",
            repo_id=args.repo_id,
            repo_type="dataset",
        )
        print(f"uploaded {path.name}")

    readme_path = Path("/tmp/phys_tir_README.md")
    readme_path.write_text(README_TEMPLATE)
    api.upload_file(
        path_or_fileobj=str(readme_path),
        path_in_repo="README.md",
        repo_id=args.repo_id,
        repo_type="dataset",
    )
    print(f"done: https://huggingface.co/datasets/{args.repo_id}")


if __name__ == "__main__":
    main()
