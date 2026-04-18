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

README_FRONTMATTER = """\
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

"""

README_HEADER_TIR = """\
# phys-tir

Physics problem dataset for tool-integrated reasoning (TIR) RLVR training.
Pre-split train / validation / test, each a union of two pools (`drsci` and
`curated`).

Every row's `prompt` uses the TIR system prompt (Python/SymPy sandbox tool
available). See `phys-cot` for the matched CoT-only baseline variant.

Composition rule (one filter per source, then include):
- **Dr.SCI physics, difficulty ≥ 0.5** — retains the tractable band as labeled
  by Qwen3-32B 8-rollout sampling (Dr.SCI's own difficulty proxy). This is the
  floor of Dr.SCI's dynamic-curriculum starting band [0.51, 0.99]; Spearman
  ρ=0.42 vs observed 4B TIR pass-rate on a 2k probe validates the filter.
- **UGPhysics (EN)** — no per-problem difficulty labels, included as-is.
- **PHYSICS (desimfj)** — no per-problem difficulty labels, stratified 80/20
  self-split (no public leaderboard to dilute).
- **SciBench-RL** — physics subset (excluding atkins/chemmc chemistry). Uses
  the official native train/test split: native train is in training, native
  test (153 problems from disjoint textbooks class/diff/matter) is held out.

External benchmarks held out for contamination-free evaluation:
OlympiadBench (236), PHYBench (1000), ABench (800), CritPt (70).

"""

README_HEADER_COT = """\
# phys-cot

Physics problem dataset for chain-of-thought (CoT) RLVR training. Matched
CoT-only baseline variant of `phys-tir`: identical rows, identical splits,
identical user messages and ground truths. The only difference is the system
prompt — CoT mentions no tools/Python/code. Used for the TIR-vs-CoT ablation.

Pre-split train / validation / test, each a union of two pools (`drsci` and
`curated`). See `phys-tir` for the composition rule and source attribution.

External benchmarks held out for contamination-free evaluation:
OlympiadBench (236), PHYBench (1000), ABench (800), CritPt (70).

"""

README_POOLS = """
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

### `curated` — UGPhysics + PHYSICS + SciBench-RL (physics)
Hand-curated physics sources, all rule-verifiable:

- **UGPhysics (EN)** — Undergraduate physics from
  [UGPhysics/ugphysics](https://huggingface.co/datasets/UGPhysics/ugphysics),
  English subset only. Filtered to rule-verifiable answer types (NV / EX / EQ
  / MC / TF), dirty answer-type labels cleaned, deduped.
- **PHYSICS (desimfj)** — Undergraduate textbook physics from
  [desimfj/PHYSICS](https://huggingface.co/datasets/desimfj/PHYSICS). Chinese
  rows dropped, multi-alternative answers excluded, open-ended items removed.
  Stratified 80/20 train/eval self-split (no public leaderboard for this set).
- **SciBench-RL** — Textbook STEM problems from
  [Sihangli/scibench-rl](https://huggingface.co/datasets/Sihangli/scibench-rl),
  physics sources only (excluding atkins/chemmc chemistry). Uses the official
  native train/test divide: native train in training pool, native test (153
  problems from disjoint textbooks class/diff/matter) held out.

Other public physics benchmarks (OlympiadBench, PHYBench, ABench, CritPt) are
excluded from training so they remain clean external eval targets.

### `drsci` — Dr.SCI physics (difficulty ≥ 0.5)
Sourced from the unofficial release at
[MiniByte-666/Dr.SCI](https://huggingface.co/datasets/MiniByte-666/Dr.SCI),
re-implementing the dataset described in the Dr.SCI paper
([arXiv:2602.08321](https://arxiv.org/abs/2602.08321)). Filtered to physics
problems with rule-verifiable answers (numerical / expression / equation /
MCQ / true-false).

**Difficulty filter:** retains only problems with Dr.SCI's per-problem
`difficulty ≥ 0.5` (= Qwen3-32B 8-rollout pass-rate ≥ 4/8). This matches the
floor of Dr.SCI's dynamic-curriculum starting band [0.51, 0.99]. Validated
against a 2,000-row 4B TIR probe: Spearman ρ = 0.42, p < 1e-80.


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


def _detect_variant(data_dir: Path) -> str:
    """Return 'tir' or 'cot' by inspecting the system prompt in a sample row."""
    import pandas as pd
    df = pd.read_parquet(data_dir / "train.parquet")
    sys_msg = df["prompt"].iloc[0][0]["content"].lower()
    if "tool" in sys_msg or "python" in sys_msg or "code" in sys_msg:
        return "tir"
    return "cot"


def _build_readme(data_dir: Path) -> str:
    """Build README dynamically from actual parquet row counts + detected variant."""
    import pandas as pd

    def _counts(split: str) -> tuple[int, int, int]:
        p = data_dir / f"{split}.parquet"
        df = pd.read_parquet(p)
        if "pool" in df.columns:
            cur = int((df["pool"] == "curated").sum())
            drs = int((df["pool"] == "drsci").sum())
        else:
            cur = drs = 0
        return cur, drs, len(df)

    tr_c, tr_d, tr_t = _counts("train")
    va_c, va_d, va_t = _counts("validation")
    te_c, te_d, te_t = _counts("test")

    # Break down the curated pool by source tag for a more informative table.
    def _by_source(split: str) -> dict[str, int]:
        p = data_dir / f"{split}.parquet"
        df = pd.read_parquet(p)
        if "pool" not in df.columns:
            return {}
        cur = df[df["pool"] == "curated"]
        if len(cur) == 0:
            return {}
        return cur["data_source"].astype(str).value_counts().to_dict()

    tr_cur_srcs = _by_source("train")
    va_cur_srcs = _by_source("validation")
    te_cur_srcs = _by_source("test")
    all_cur_keys = set(tr_cur_srcs) | set(va_cur_srcs) | set(te_cur_srcs)

    src_rows = ""
    for k in ("UGPhysics", "PHYSICS", "SciBench_RL"):
        if k in all_cur_keys:
            src_rows += (
                f"| &nbsp;&nbsp;↳ `{k}` | "
                f"{tr_cur_srcs.get(k, 0):,} | {va_cur_srcs.get(k, 0):,} | {te_cur_srcs.get(k, 0):,} |\n"
            )

    table = (
        "| Pool | Train | Validation | Test |\n"
        "|---|---|---|---|\n"
        f"| `curated` (UGPhysics + PHYSICS + SciBench-RL) | {tr_c:,} | {va_c:,} | {te_c:,} |\n"
        + src_rows
        + f"| `drsci` (Dr.SCI physics, difficulty ≥ 0.5)   | {tr_d:,} | {va_d:,} | {te_d:,} |\n"
        + f"| **total** | **{tr_t:,}** | **{va_t:,}** | **{te_t:,}** |\n"
    )

    variant = _detect_variant(data_dir)
    header = README_HEADER_TIR if variant == "tir" else README_HEADER_COT
    return README_FRONTMATTER + header + table + README_POOLS


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
    readme_path.write_text(_build_readme(data_dir))
    api.upload_file(
        path_or_fileobj=str(readme_path),
        path_in_repo="README.md",
        repo_id=args.repo_id,
        repo_type="dataset",
    )
    print(f"done: https://huggingface.co/datasets/{args.repo_id}")


if __name__ == "__main__":
    main()
