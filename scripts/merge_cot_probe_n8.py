"""Merge n=8 CoT probe chunks into a single file with per-problem pass@8.

Run after all chunk parquets land in outputs/cot_probe_4b_z2_n8/.
"""
import os
import glob
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq
import pandas as pd

CHUNK_DIR = "/pscratch/sd/b/bwhou/17-phy-reasoner/phys-reasoner/outputs/cot_probe_4b_z2_n8"
OUT_ALL = f"{CHUNK_DIR}/rollouts_pass8_all.parquet"
OUT_PER_PROB = f"{CHUNK_DIR}/per_problem_pass8.parquet"


def main() -> None:
    chunk_files = sorted(glob.glob(f"{CHUNK_DIR}/chunk_*.parquet"))
    print(f"[merge] found {len(chunk_files)} chunks: {[os.path.basename(p) for p in chunk_files]}")
    if not chunk_files:
        print("[merge] no chunks yet")
        return
    dfs = [ds.dataset(p).to_table().to_pandas() for p in chunk_files]
    big = pd.concat(dfs, ignore_index=True)
    print(f"[merge] total rollouts: {len(big)}")

    pq.write_table(pa.Table.from_pandas(big, preserve_index=False), OUT_ALL)

    pp = big.groupby("problem_hash").agg(
        n_rollouts=("correct", "size"),
        pass_rate=("correct", "mean"),
        len_mean=("response_length_tokens", "mean"),
        len_max=("response_length_tokens", "max"),
    ).reset_index()
    print(f"[merge] unique problems: {len(pp)}")
    print(f"[merge] pass_rate distribution:")
    print(f"  mean        : {pp.pass_rate.mean():.3f}")
    print(f"  frac=0.0    : {(pp.pass_rate == 0).mean():.3f}")
    print(f"  frac=1.0    : {(pp.pass_rate == 1).mean():.3f}")
    print(f"  goldilocks  : {((pp.pass_rate > 0) & (pp.pass_rate < 1)).mean():.3f}")
    print(f"  in [0.125, 0.875]: {((pp.pass_rate >= 0.125) & (pp.pass_rate <= 0.875)).mean():.3f}")
    print(f"  in [0.25, 0.75] : {((pp.pass_rate >= 0.25) & (pp.pass_rate <= 0.75)).mean():.3f}")

    pq.write_table(pa.Table.from_pandas(pp, preserve_index=False), OUT_PER_PROB)
    print(f"[merge] wrote {OUT_ALL} and {OUT_PER_PROB}")


if __name__ == "__main__":
    main()
