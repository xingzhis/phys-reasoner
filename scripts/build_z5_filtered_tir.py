"""Build Z5 TIR-filtered subset — same filter as Z2 but on TIR-107k source.

Same filter criteria as Z2: numerical + 4B pass_rate >= 0.75. Only the source
parquet differs (TIR version has tool-use system prompt).

Output: verl-ready TIR parquet with filtered problems only.
"""
import hashlib
import os
import pyarrow.dataset as ds
import pyarrow as pa
import pyarrow.parquet as pq

SRC_TIR = "/pscratch/sd/b/bwhou/17-phy-reasoner/phys-reasoner/data/phys_tir_107k/train.parquet"
SCORED = "/pscratch/sd/b/bwhou/17-phy-reasoner/phys-reasoner/outputs/probe_qwen3_4b_v2_107k/rollouts_scored_trainmatched_107k.parquet"
DST_DIR = "/pscratch/sd/b/bwhou/17-phy-reasoner/phys-reasoner/data/processed_tir_z5_numerical_easy"
PASS_RATE_MIN = 0.75


def compute_hash(problem: str, gold: str) -> str:
    return hashlib.md5(f"{problem}|||{gold}".encode("utf-8")).hexdigest()[:16]


def main() -> None:
    print("[z5] loading scored parquet...")
    scored = ds.dataset(SCORED).to_table().to_pandas()
    per_prob = (
        scored.groupby("problem_idx")
        .agg(
            n_rollouts=("correct", "size"),
            n_correct=("correct", "sum"),
            answer_type=("answer_type", "first"),
            problem_hash=("problem_hash", "first"),
        )
        .reset_index()
    )
    per_prob["pass_rate"] = per_prob.n_correct / per_prob.n_rollouts
    keep = per_prob[(per_prob.answer_type == "numerical") & (per_prob.pass_rate >= PASS_RATE_MIN)]
    keep_hashes = set(keep.problem_hash.tolist())
    print(f"[z5] filter matches: {len(keep)} problems (numerical + 4B pass>={PASS_RATE_MIN})")

    print("[z5] loading TIR 107k source...")
    tir = ds.dataset(SRC_TIR).to_table().to_pandas()
    print(f"[z5] TIR source: {len(tir)} rows")

    def row_hash(row) -> str:
        prob = row["extra_info"].get("problem", "") if isinstance(row["extra_info"], dict) else ""
        gold = row["reward_model"].get("ground_truth", "") if isinstance(row["reward_model"], dict) else ""
        return compute_hash(str(prob), str(gold))

    tir["_problem_hash"] = tir.apply(row_hash, axis=1)
    matched = tir[tir._problem_hash.isin(keep_hashes)].drop(columns=["_problem_hash"])
    print(f"[z5] joined: {len(matched)} of {len(tir)} TIR rows match filter")

    if len(matched) == 0:
        print("[z5] ERROR: zero matches. Sample hashes:")
        print("  scored:", list(keep.problem_hash.head(5)))
        print("  tir:", list(tir._problem_hash.head(5)))
        return

    os.makedirs(f"{DST_DIR}/data", exist_ok=True)
    out = pa.Table.from_pandas(matched, preserve_index=False)
    pq.write_table(out, f"{DST_DIR}/data/train.parquet")
    print(f"[z5] wrote {len(matched)} rows -> {DST_DIR}/data/train.parquet")
    split = int(len(matched) * 0.9)
    val = matched.iloc[split:]
    pq.write_table(pa.Table.from_pandas(val, preserve_index=False), f"{DST_DIR}/data/validation.parquet")
    print(f"[z5] wrote {len(val)} val rows")


if __name__ == "__main__":
    main()
