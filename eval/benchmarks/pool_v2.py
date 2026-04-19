"""In-distribution loader: slice pool v2 test.parquet by source.

The pool_v2 test split (1,064 rows) carries four sources mixed in one parquet:

    data_source == 'Dr. SCI'      → 503 (drsci stratified test)
    data_source == 'UGPhysics'    → 217 (ugphysics stratified test)
    data_source == 'PHYSICS'      → 191 (physics self-split test)
    data_source == 'SciBench_RL'  → 153 (native test split)

This loader filters by data_source and writes a per-source parquet that already
matches rollout.py's input schema (columns: data_source, prompt, reward_model,
extra_info, pool). Idempotent: returns out_path unchanged if it already exists.

The input parquet path defaults to data/processed_tir/data/test.parquet (the
post-`scripts/fetch_dataset.py` layout from HF mirror xingzhi0/phys-tir). Pass
--src to override (e.g. data/processed/pool_v2/tir/test.parquet for a local
build_pool_v2.py output).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

# data_source values written by scripts/build_pool_v2.py
SOURCE_ALIASES = {
    "drsci":      "Dr. SCI",
    "ugphysics":  "UGPhysics",
    "physics":    "PHYSICS",
    "scibench":   "SciBench_RL",
}


def load(
    out_path: str,
    source: str,
    src_parquet: str = "data/processed_tir/data/test.parquet",
    force: bool = False,
) -> str:
    out = Path(out_path)
    if out.exists() and not force:
        n = len(pd.read_parquet(out))
        print(f"[pool_v2:{source}] reuse {out} ({n} rows)")
        return str(out)

    canonical = SOURCE_ALIASES.get(source, source)
    df = pd.read_parquet(src_parquet)
    if "data_source" not in df.columns:
        raise ValueError(f"{src_parquet} has no `data_source` column (cols={list(df.columns)})")

    sub = df[df["data_source"] == canonical].reset_index(drop=True)
    if len(sub) == 0:
        avail = sorted(df["data_source"].unique().tolist())
        raise ValueError(f"No rows for data_source={canonical!r} in {src_parquet}; available: {avail}")

    out.parent.mkdir(parents=True, exist_ok=True)
    sub.to_parquet(out, index=False)
    print(f"[pool_v2:{source}] wrote {len(sub)} rows → {out}  (source={canonical})")
    return str(out)


def main() -> None:
    p = argparse.ArgumentParser(description="Slice pool_v2 test.parquet by data_source")
    p.add_argument("--source", required=True, choices=sorted(SOURCE_ALIASES.keys()),
                   help="Short alias for the source filter")
    p.add_argument("--src", default="data/processed_tir/data/test.parquet",
                   help="Path to the pool_v2 test parquet")
    p.add_argument("--out", required=True, help="Output parquet path")
    p.add_argument("--force", action="store_true", help="Overwrite if --out exists")
    args = p.parse_args()
    load(out_path=args.out, source=args.source, src_parquet=args.src, force=args.force)


if __name__ == "__main__":
    main()
