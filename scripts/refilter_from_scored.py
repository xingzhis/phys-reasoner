"""Re-filter the existing scored rollouts to a new (lo,hi) range.

Reads outputs/probe_qwen3_4b/rollouts_scored.parquet, computes per-problem
n_correct over n=8, keeps problems with lo <= n_correct <= hi, then writes
filtered TIR (and optionally CoT) train.parquet into a NEW directory keyed
on the filter range. Validation + test parquets are copied unchanged.

Refuses to overwrite existing dst directories — you must remove them first.

Usage:
  python3 scripts/refilter_from_scored.py --lo 2 --hi 6
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import pandas as pd


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scored", default="outputs/probe_qwen3_4b/rollouts_scored.parquet")
    ap.add_argument("--lo", type=int, required=True, help="min n_correct (inclusive)")
    ap.add_argument("--hi", type=int, required=True, help="max n_correct (inclusive)")
    ap.add_argument("--n_total", type=int, default=8)
    ap.add_argument("--modes", nargs="+", default=["tir", "cot"])
    ap.add_argument("--root", default="/pscratch/sd/b/bwhou/17-phy-reasoner/phys-reasoner")
    args = ap.parse_args()

    root = Path(args.root)
    suffix = f"{args.lo}{args.hi}"

    print(f"[refilter] reading {args.scored}")
    df = pd.read_parquet(root / args.scored)

    agg = df.groupby("problem_idx")["correct"].agg(["sum", "count"]).reset_index()
    agg = agg[agg["count"] == args.n_total]
    print(f"[refilter] {len(agg)} problems with full {args.n_total} rollouts")

    keep = agg[(agg["sum"] >= args.lo) & (agg["sum"] <= args.hi)]
    print(f"[refilter] kept {len(keep)}/{len(agg)} = {100*len(keep)/len(agg):.1f}%  "
          f"(range [{args.lo},{args.hi}])")

    keep_idx = sorted(int(i) for i in keep["problem_idx"].tolist())

    for mode in args.modes:
        src_dir = root / f"data/processed_{mode}/data"
        dst_dir = root / f"data/processed_{mode}_filtered_{suffix}/data"
        if dst_dir.exists():
            raise FileExistsError(f"{dst_dir} already exists — refusing to overwrite")
        dst_dir.mkdir(parents=True, exist_ok=False)

        # train: filter
        train_src = src_dir / "train.parquet"
        train_full = pd.read_parquet(train_src)
        kept_train = train_full.iloc[keep_idx].reset_index(drop=True)
        train_dst = dst_dir / "train.parquet"
        kept_train.to_parquet(train_dst, index=False)
        print(f"[write] {mode} train: {len(train_full)} → {len(kept_train)}  ({train_dst})")

        # validation + test: copy unchanged
        for split in ("validation", "test"):
            src = src_dir / f"{split}.parquet"
            if src.exists():
                shutil.copy2(src, dst_dir / f"{split}.parquet")
                print(f"[copy ] {mode} {split} unchanged → {dst_dir / (split + '.parquet')}")

    print(f"[done ] suffix=_{suffix}; submit prod_tir with TRAIN_FILES=data/processed_tir_filtered_{suffix}/data/train.parquet")


if __name__ == "__main__":
    main()
