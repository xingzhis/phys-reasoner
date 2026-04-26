"""Merge cot_pilot_thinkinterrupt chunks and print per-(type,band) Goldilocks
summary. Used to evaluate P-pilot health before committing to P-full.
"""
import argparse
import glob
import os

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_glob", default="outputs/cot_pilot_thinkinterrupt/chunk_*.parquet")
    ap.add_argument("--out", default="outputs/cot_pilot_thinkinterrupt/merged.parquet")
    args = ap.parse_args()

    files = sorted(glob.glob(args.in_glob))
    print(f"[merge] {len(files)} chunks: {[os.path.basename(f) for f in files]}")
    if not files:
        print("[merge] no chunks found"); return
    dfs = [pq.read_table(f).to_pandas() for f in files]
    df = pd.concat(dfs, ignore_index=True)
    print(f"[merge] total rollouts: {len(df)}")

    pq.write_table(pa.Table.from_pandas(df, preserve_index=False), args.out)
    print(f"[merge] wrote {args.out}")

    pp = df.groupby("problem_hash").agg(
        answer_type=("answer_type", "first"),
        pass_rate=("correct", "mean"),
        n=("correct", "size"),
    ).reset_index()
    print(f"[merge] {len(pp)} problems × n=8 pilot")
    print(f"[merge] interrupt fired: {df['interrupt_fired'].mean():.3f}")
    print(f"[merge] phase1_tokens p50={np.percentile(df['phase1_tokens'], 50):.0f}, "
          f"p90={np.percentile(df['phase1_tokens'], 90):.0f}")
    if df["interrupt_fired"].any():
        ph2 = df[df["interrupt_fired"]]["phase2_tokens"]
        print(f"[merge] phase2_tokens (when fired): p50={ph2.median():.0f} p90={ph2.quantile(0.9):.0f}")

    bins = [-0.001, 0.001, 0.124, 0.375, 0.624, 0.875, 1.001]
    labels = ["0/8","1/8","2-3/8","4-5/8","6-7/8","8/8"]
    pp["band"] = pd.cut(pp["pass_rate"], bins=bins, labels=labels)
    print()
    print("[merge] (answer_type, pass-band) distribution from CoT prod-budget pilot:")
    print(pd.crosstab(pp["answer_type"], pp["band"], margins=True))

    gold = pp[(pp["pass_rate"] > 0) & (pp["pass_rate"] < 1)]
    print()
    print(f"[merge] Goldilocks band (0,1) count: {len(gold)}/{len(pp)} ({100.0*len(gold)/len(pp):.1f}%)")
    print(f"[merge] all-zero (TOO HARD)         : {(pp['pass_rate']==0).sum()}/{len(pp)} ({100.0*(pp['pass_rate']==0).mean():.1f}%)")
    print(f"[merge] all-one  (TOO EASY)         : {(pp['pass_rate']==1).sum()}/{len(pp)} ({100.0*(pp['pass_rate']==1).mean():.1f}%)")
    print()
    print("[merge] Goldilocks counts per type:")
    g_per_type = gold.groupby("answer_type").size().sort_values(ascending=False)
    print(g_per_type.to_string())


if __name__ == "__main__":
    main()
