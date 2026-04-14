"""Reconcile curated (corpus_*) and Dr. SCI (drsci_*) parquets into unified
train / validation / test parquets with a lossless union schema.

Reconciliation rules:
  - `extra_info` is promoted to a union struct containing every field from either
    source. Fields missing on a given row become null. No row is dropped.
  - `extra_info.difficulty` is cast to string on both sides (curated is already
    string; Dr. SCI stores float64, stringified to preserve the numeric value).
  - A top-level `pool` column is added: "curated" or "drsci".
  - `extra_info.source` (existing in curated as per-source tag, e.g. "UGPhysics")
    is kept as-is and is null for drsci rows.
  - `prompt`, `reward_model`, `data_source` are untouched.

Output: data/processed/merged/{train,validation,test}.parquet

Running (inside the overlay):
  PYTHONNOUSERSITE=1 apptainer exec \
    --overlay phys-reasoner-overlay-017.img --bind /etc/pki:/etc/pki \
    --env PYTHONPATH=/opt/phys-extras/ verl_vllm017.latest.sif \
    python3 scripts/merge_splits.py
"""
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

DATA_DIR = Path("data/processed")
OUT_DIR = DATA_DIR / "merged"

SPLIT_MAP = {"train": "train", "validation": "dev", "test": "test"}

UNION_EXTRA_INFO_TYPE = pa.struct([
    ("answer_type", pa.string()),
    ("difficulty", pa.string()),          # promoted from float64 on drsci side
    ("domain", pa.string()),              # curated only
    ("domain_coarse", pa.string()),       # curated only
    ("from", pa.string()),                # drsci only
    ("primary_answer_type", pa.string()), # curated only
    ("problem", pa.string()),
    ("source", pa.string()),              # curated only (per-source tag)
    ("subject", pa.string()),             # drsci only
    ("tolerance", pa.float64()),
    ("unit", pa.string()),
])


def reconcile(path: Path, pool_tag: str) -> pa.Table:
    t = pq.read_table(str(path))
    n = len(t)

    ei = t["extra_info"].to_pylist()
    for d in ei:
        if d is None:
            continue
        if d.get("difficulty") is not None:
            d["difficulty"] = str(d["difficulty"])

    new_ei = pa.array(ei, type=UNION_EXTRA_INFO_TYPE)
    idx = t.schema.get_field_index("extra_info")
    t = t.set_column(idx, pa.field("extra_info", UNION_EXTRA_INFO_TYPE), new_ei)

    pool_col = pa.array([pool_tag] * n, type=pa.string())
    t = t.append_column("pool", pool_col)
    return t


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    summary = []
    for out_split, in_split in SPLIT_MAP.items():
        cur = reconcile(DATA_DIR / f"corpus_{in_split}.parquet", "curated")
        drs = reconcile(DATA_DIR / f"drsci_{in_split}.parquet", "drsci")
        merged = pa.concat_tables([cur, drs])
        out = OUT_DIR / f"{out_split}.parquet"
        pq.write_table(merged, str(out))
        summary.append((out_split, len(cur), len(drs), len(merged), str(out)))
        print(f"wrote {out}: curated={len(cur)} + drsci={len(drs)} = {len(merged)}")

    print("\nDone.")
    for row in summary:
        print("  ", row)


if __name__ == "__main__":
    main()
