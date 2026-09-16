"""Build Z4g: Z2's numerical_easy subset filtered to problems where base-4B
got the correct answer at least once in CoT mode (pass@1=1 from CoT probe).

Rationale: Z4f's all_wrong_frac=65% means 2/3 of GRPO groups have no
informative gradient (16/16 rollouts wrong). The TIR-based filter we used for
Z2 doesn't transfer to CoT mode (where the model can't delegate math to a
tool). Use the actual CoT probe pass@1 to keep only problems the model can
sometimes solve.

With pass@1=1 single-rollout-probe filter, we expect:
  - ~330 problems retained (pass@1 mean was 0.191 across 1724)
  - all_wrong_frac drops sharply (problem provably solvable at least once)
  - Plenty of GRPO signal per step
"""
import os
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq

SRC_COT = "/pscratch/sd/b/bwhou/17-phy-reasoner/phys-reasoner/data/processed_cot_z2_numerical_easy/data/train.parquet"
COT_PROBE = "/pscratch/sd/b/bwhou/17-phy-reasoner/phys-reasoner/outputs/cot_probe_4b_z2/rollouts_pass1.parquet"
DST_DIR = "/pscratch/sd/b/bwhou/17-phy-reasoner/phys-reasoner/data/processed_cot_z4g_cot_solvable"


def main() -> None:
    print("[z4g] loading CoT probe", COT_PROBE)
    probe = ds.dataset(COT_PROBE).to_table().to_pandas()
    print(f"[z4g] probe rows: {len(probe)}, cols: {probe.columns.tolist()}")
    print(f"[z4g] probe pass@1 mean: {probe.correct.mean():.3f}")

    solvable_hashes = set(probe[probe.correct == 1].problem_hash.tolist())
    print(f"[z4g] solvable (pass@1=1) problems: {len(solvable_hashes)}")

    print("[z4g] loading Z2 source CoT parquet")
    cot = ds.dataset(SRC_COT).to_table().to_pandas()
    import hashlib
    def row_hash(row) -> str:
        prob = row["extra_info"].get("problem", "") if isinstance(row["extra_info"], dict) else ""
        gold = row["reward_model"].get("ground_truth", "") if isinstance(row["reward_model"], dict) else ""
        return hashlib.md5(f"{prob}|||{gold}".encode("utf-8")).hexdigest()[:16]
    cot["_h"] = cot.apply(row_hash, axis=1)
    matched = cot[cot._h.isin(solvable_hashes)].drop(columns=["_h"])
    print(f"[z4g] Z2 ∩ CoT-solvable: {len(matched)} rows ({100*len(matched)/len(cot):.1f}% of Z2)")

    if len(matched) == 0:
        print("[z4g] ERROR: no overlap")
        return

    os.makedirs(f"{DST_DIR}/data", exist_ok=True)
    out = pa.Table.from_pandas(matched, preserve_index=False)
    pq.write_table(out, f"{DST_DIR}/data/train.parquet")
    print(f"[z4g] wrote train.parquet ({len(matched)} rows)")
    split = int(len(matched) * 0.9)
    val = matched.iloc[split:]
    pq.write_table(pa.Table.from_pandas(val, preserve_index=False), f"{DST_DIR}/data/validation.parquet")
    print(f"[z4g] wrote validation.parquet ({len(val)} rows)")


if __name__ == "__main__":
    main()
