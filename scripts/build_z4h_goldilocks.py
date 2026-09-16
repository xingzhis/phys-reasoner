"""Build Z4h: Z2's numerical_easy subset filtered to Goldilocks pass@8 band.

Reads the merged n=8 CoT probe at outputs/cot_probe_4b_z2_n8/per_problem_pass8.parquet
and selects problems with pass@8 strictly between LOW_THRESH and HIGH_THRESH —
i.e. groups that will sometimes succeed and sometimes fail at training rollout
time, providing maximal GRPO signal.

Default band [0.125, 0.875] = at least 1/8 right and at least 1/8 wrong in the
probe. Tunable via env vars.

Caveat: probe was at max_tokens=4096; training uses 8192. So pass rates here
slightly underestimate training pass rates (truncation bias); see notes in
the chat session. Acceptable as a v1 filter.
"""
import os
import hashlib
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq

SRC_COT = "/pscratch/sd/b/bwhou/17-phy-reasoner/phys-reasoner/data/processed_cot_z2_numerical_easy/data/train.parquet"
PROBE_PER_PROB = "/pscratch/sd/b/bwhou/17-phy-reasoner/phys-reasoner/outputs/cot_probe_4b_z2_n8/per_problem_pass8.parquet"
DST_DIR = "/pscratch/sd/b/bwhou/17-phy-reasoner/phys-reasoner/data/processed_cot_z4h_goldilocks"

LOW = float(os.environ.get("LOW_THRESH", "0.125"))
HIGH = float(os.environ.get("HIGH_THRESH", "0.875"))


def main() -> None:
    print(f"[z4h] loading probe per-problem pass@8: {PROBE_PER_PROB}")
    pp = ds.dataset(PROBE_PER_PROB).to_table().to_pandas()
    print(f"[z4h] probe rows: {len(pp)}")
    print(f"[z4h] applying Goldilocks band: pass_rate ∈ ({LOW}, {HIGH})")
    keep = pp[(pp.pass_rate > 0) & (pp.pass_rate < 1)]  # any mixed group
    print(f"[z4h] passes filter: {len(keep)} ({100*len(keep)/len(pp):.1f}%)")
    keep_hashes = set(keep.problem_hash.tolist())

    cot = ds.dataset(SRC_COT).to_table().to_pandas()
    def row_hash(row) -> str:
        prob = row["extra_info"].get("problem", "") if isinstance(row["extra_info"], dict) else ""
        gold = row["reward_model"].get("ground_truth", "") if isinstance(row["reward_model"], dict) else ""
        return hashlib.md5(f"{prob}|||{gold}".encode("utf-8")).hexdigest()[:16]
    cot["_h"] = cot.apply(row_hash, axis=1)
    matched = cot[cot._h.isin(keep_hashes)].drop(columns=["_h"])
    print(f"[z4h] Z2 ∩ Goldilocks: {len(matched)} rows ({100*len(matched)/len(cot):.1f}% of Z2)")

    if len(matched) == 0:
        print("[z4h] ERROR: empty subset; loosen thresholds")
        return

    os.makedirs(f"{DST_DIR}/data", exist_ok=True)
    out = pa.Table.from_pandas(matched, preserve_index=False)
    pq.write_table(out, f"{DST_DIR}/data/train.parquet")
    print(f"[z4h] wrote train.parquet ({len(matched)} rows)")
    split = int(len(matched) * 0.9)
    val = matched.iloc[split:]
    pq.write_table(pa.Table.from_pandas(val, preserve_index=False), f"{DST_DIR}/data/validation.parquet")
    print(f"[z4h] wrote validation.parquet ({len(val)} rows)")


if __name__ == "__main__":
    main()
