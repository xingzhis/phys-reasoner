"""Build Z4d: Z2 numerical_easy subset filtered to problems where base-4B
TIR-probe p95 response length is short.

Rationale: Z4 (4B CoT) observed clip_ratio ~80% at max_response_length=2048.
A single-knob data subset of problems the base 4B already answers concisely
should drop clip_ratio toward 0% without changing any training config.

Length source: probe rollouts parquet (TIR format). TIR length is a lower
bound on CoT length for the same problem, so a tight TIR p95 threshold picks
problems whose reasoning is short in both modes.

Filter: TIR p95 (tokens, estimated from chars/3.5) < THRESHOLD.
"""
import os
import numpy as np
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq

SRC_COT = "/pscratch/sd/b/bwhou/17-phy-reasoner/phys-reasoner/data/processed_cot_z2_numerical_easy/data/train.parquet"
SCORED = "/pscratch/sd/b/bwhou/17-phy-reasoner/phys-reasoner/outputs/probe_qwen3_4b_v2_107k/rollouts_scored_trainmatched_107k.parquet"
DST_DIR = "/pscratch/sd/b/bwhou/17-phy-reasoner/phys-reasoner/data/processed_cot_z4d_numerical_short"
# Keep only problems whose base-4B TIR p95 (tokens) is below this. Aggressive
# threshold since CoT is 2-3x longer than TIR; 1200-token TIR p95 ≈ 2500-3500
# CoT, which still truncates at 2048 but for a smaller fraction.
P95_TOKEN_MAX = 400  # aim: CoT p95 under 2048 (CoT is ~5x longer than TIR)


def main() -> None:
    print("[z4d] loading TIR probe scored parquet")
    scored = ds.dataset(SCORED).to_table().to_pandas()
    print(f"[z4d]   {len(scored)} rollouts, {scored['problem_hash'].nunique()} problems")
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

    short_hashes = set(per_prob[per_prob.len_p95 < P95_TOKEN_MAX].problem_hash.tolist())
    print(f"[z4d] short-TIR problems (p95<{P95_TOKEN_MAX} tok): {len(short_hashes)}")

    print("[z4d] loading Z2 numerical_easy CoT training parquet")
    cot = ds.dataset(SRC_COT).to_table().to_pandas()
    print(f"[z4d]   Z2 size: {len(cot)} rows")

    # Z2 parquet was built with problem_hash in 'extra_info' — match there.
    # (See scripts/build_z3_expression_filter.py for the md5-from-(problem,gold) pattern.)
    import hashlib
    def row_hash(row) -> str:
        prob = row["extra_info"].get("problem", "") if isinstance(row["extra_info"], dict) else ""
        gold = row["reward_model"].get("ground_truth", "") if isinstance(row["reward_model"], dict) else ""
        return hashlib.md5(f"{prob}|||{gold}".encode("utf-8")).hexdigest()[:16]
    cot["_problem_hash"] = cot.apply(row_hash, axis=1)
    matched = cot[cot._problem_hash.isin(short_hashes)].drop(columns=["_problem_hash"])
    print(f"[z4d] Z2 ∩ short-TIR: {len(matched)} rows ({100*len(matched)/len(cot):.1f}% of Z2)")

    if len(matched) == 0:
        print("[z4d] ERROR: no overlap")
        return

    os.makedirs(f"{DST_DIR}/data", exist_ok=True)
    out = pa.Table.from_pandas(matched, preserve_index=False)
    pq.write_table(out, f"{DST_DIR}/data/train.parquet")
    print(f"[z4d] wrote train.parquet ({len(matched)} rows)")
    split = int(len(matched) * 0.9)
    val = matched.iloc[split:]
    pq.write_table(pa.Table.from_pandas(val, preserve_index=False), f"{DST_DIR}/data/validation.parquet")
    print(f"[z4d] wrote validation.parquet ({len(val)} rows)")


if __name__ == "__main__":
    main()
