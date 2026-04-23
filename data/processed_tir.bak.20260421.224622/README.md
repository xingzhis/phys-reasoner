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

| Pool | Train | Validation | Test |
|---|---|---|---|
| `curated` (UGPhysics + PHYSICS + SciBench-RL) | 5,762 | 317 | 561 |
| &nbsp;&nbsp;↳ `UGPhysics` | 4,980 | 217 | 217 |
| &nbsp;&nbsp;↳ `PHYSICS` | 502 | 100 | 191 |
| &nbsp;&nbsp;↳ `SciBench_RL` | 280 | 0 | 153 |
| `drsci` (Dr.SCI physics, difficulty ≥ 0.5)   | 17,827 | 503 | 503 |
| **total** | **23,589** | **820** | **1,064** |

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
