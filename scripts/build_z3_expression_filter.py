"""Build Z3 expression-type filtered subset.

Purpose: stress xverify on its primary use case (expression answers — where
rule-based can't decide and xverify is the decider). This is the REAL xverify
atom test (numerical filter mostly shortcuts to rule-based).

Filter: answer_type == 'expression' AND 4B pass_rate >= 0.5 (moderate threshold
to get ~800-1600 problems; xverify-scored probe data, so high pass_rates are
"xverify's strong cases").
"""
import hashlib
import os
import pyarrow.dataset as ds
import pyarrow as pa
import pyarrow.parquet as pq

SRC_COT = "/pscratch/sd/b/bwhou/17-phy-reasoner/phys-reasoner/data/phys_cot_107k/train.parquet"
SCORED = "/pscratch/sd/b/bwhou/17-phy-reasoner/phys-reasoner/outputs/probe_qwen3_4b_v2_107k/rollouts_scored_trainmatched_107k.parquet"
DST_DIR = "/pscratch/sd/b/bwhou/17-phy-reasoner/phys-reasoner/data/processed_cot_z3_expression_easy"
PASS_RATE_MIN = 0.5  # relaxed from 0.75 since expression probes use xverify


def compute_hash(problem: str, gold: str) -> str:
    return hashlib.md5(f"{problem}|||{gold}".encode("utf-8")).hexdigest()[:16]


def main() -> None:
    print("[z3] loading scored parquet...")
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
    keep = per_prob[(per_prob.answer_type == "expression") & (per_prob.pass_rate >= PASS_RATE_MIN)]
    keep_hashes = set(keep.problem_hash.tolist())
    print(f"[z3] filter matches: {len(keep)} expression problems (pass>={PASS_RATE_MIN})")

    print("[z3] loading source CoT 107k...")
    cot = ds.dataset(SRC_COT).to_table().to_pandas()
    print(f"[z3] source: {len(cot)} rows")

    def row_hash(row) -> str:
        prob = row["extra_info"].get("problem", "") if isinstance(row["extra_info"], dict) else ""
        gold = row["reward_model"].get("ground_truth", "") if isinstance(row["reward_model"], dict) else ""
        return compute_hash(str(prob), str(gold))

    cot["_problem_hash"] = cot.apply(row_hash, axis=1)
    matched = cot[cot._problem_hash.isin(keep_hashes)].drop(columns=["_problem_hash"])
    print(f"[z3] joined: {len(matched)} rows")

    if len(matched) == 0:
        return

    os.makedirs(f"{DST_DIR}/data", exist_ok=True)
    out = pa.Table.from_pandas(matched, preserve_index=False)
    pq.write_table(out, f"{DST_DIR}/data/train.parquet")
    print(f"[z3] wrote train.parquet ({len(matched)} rows)")
    split = int(len(matched) * 0.9)
    val = matched.iloc[split:]
    pq.write_table(pa.Table.from_pandas(val, preserve_index=False), f"{DST_DIR}/data/validation.parquet")
    print(f"[z3] wrote validation.parquet ({len(val)} rows)")


if __name__ == "__main__":
    main()
