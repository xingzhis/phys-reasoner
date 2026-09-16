"""What does Z2's numerical_easy subset look like in TIR probe lengths?"""
import hashlib
import pyarrow.dataset as ds
import numpy as np

SRC_COT = "/pscratch/sd/b/bwhou/17-phy-reasoner/phys-reasoner/data/processed_cot_z2_numerical_easy/data/train.parquet"
SCORED = "/pscratch/sd/b/bwhou/17-phy-reasoner/phys-reasoner/outputs/probe_qwen3_4b_v2_107k/rollouts_scored_trainmatched_107k.parquet"


def main() -> None:
    cot = ds.dataset(SRC_COT).to_table().to_pandas()
    def row_hash(row) -> str:
        prob = row["extra_info"].get("problem", "") if isinstance(row["extra_info"], dict) else ""
        gold = row["reward_model"].get("ground_truth", "") if isinstance(row["reward_model"], dict) else ""
        return hashlib.md5(f"{prob}|||{gold}".encode("utf-8")).hexdigest()[:16]
    cot["_h"] = cot.apply(row_hash, axis=1)
    z2_hashes = set(cot._h.tolist())
    print(f"[z2-len] Z2 problems: {len(z2_hashes)}")

    scored = ds.dataset(SCORED).to_table().to_pandas()
    scored = scored[scored.problem_hash.isin(z2_hashes)]
    scored["_chars"] = (
        scored["phase1_text"].fillna("").str.len()
        + scored["phase1b_text"].fillna("").str.len()
        + scored["phase2_text"].fillna("").str.len()
    )
    scored["_tok"] = scored["_chars"] / 3.5

    per_prob = scored.groupby("problem_hash").agg(
        len_p95=("_tok", lambda x: np.percentile(x, 95)),
        len_max=("_tok", "max"),
    ).reset_index()
    print(f"[z2-len] probed problems in Z2: {len(per_prob)}")
    print(f"[z2-len] len_p95 distribution (tokens):")
    s = per_prob.len_p95
    print(f"  p10={np.percentile(s,10):.0f} p50={np.percentile(s,50):.0f} p90={np.percentile(s,90):.0f} p95={np.percentile(s,95):.0f} max={s.max():.0f}")
    print(f"[z2-len] len_max distribution (tokens):")
    s = per_prob.len_max
    print(f"  p10={np.percentile(s,10):.0f} p50={np.percentile(s,50):.0f} p90={np.percentile(s,90):.0f} p95={np.percentile(s,95):.0f} max={s.max():.0f}")

    print("[z2-len] Subset sizes for various p95-token thresholds (of Z2's subset):")
    for thr in [200, 300, 400, 500, 600, 800, 1000, 1200, 1500]:
        kept = per_prob[per_prob.len_p95 < thr]
        pct = 100 * len(kept) / len(per_prob) if len(per_prob) else 0
        print(f"  p95<{thr}: {len(kept)} problems ({pct:.1f}%)")


if __name__ == "__main__":
    main()
