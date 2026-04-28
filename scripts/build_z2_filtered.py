"""Build Z2 diagnostic subset: processed_cot filtered to 4B-easy numerical problems.

Filter:
  - answer_type == 'numerical' (rule-based reward can match cleanly)
  - Qwen3-4B pass rate >= 0.75 (moderately easy for 4B → 0.5B has some headroom)

Output: a CoT parquet keeping the same schema as the source (verl-ready).
"""
import hashlib
import pyarrow.dataset as ds
import pyarrow as pa
import pyarrow.parquet as pq

SRC_COT = "/pscratch/sd/b/bwhou/17-phy-reasoner/phys-reasoner/data/phys_cot_107k/train.parquet"
SCORED = "/pscratch/sd/b/bwhou/17-phy-reasoner/phys-reasoner/outputs/probe_qwen3_4b_v2_107k/rollouts_scored_trainmatched_107k.parquet"
DST_DIR = "/pscratch/sd/b/bwhou/17-phy-reasoner/phys-reasoner/data/processed_cot_z2_numerical_easy"

PASS_RATE_MIN = 0.75


def compute_hash(problem: str, gold: str) -> str:
    s = f"{problem}|||{gold}".encode("utf-8")
    return hashlib.md5(s).hexdigest()[:16]


def main() -> None:
    import os
    import pandas as pd

    print("[z2] loading scored parquet...")
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

    keep = per_prob[
        (per_prob.answer_type == "numerical") & (per_prob.pass_rate >= PASS_RATE_MIN)
    ]
    keep_hashes = set(keep.problem_hash.tolist())
    print(f"[z2] scored: {len(per_prob)} problems total; {len(keep)} pass filter (numerical+pass>={PASS_RATE_MIN})")

    print("[z2] loading source processed_cot...")
    cot = ds.dataset(SRC_COT).to_table().to_pandas()
    print(f"[z2] processed_cot: {len(cot)} rows")

    # Compute hash from problem text + ground_truth
    def row_hash(row) -> str:
        prob = row["extra_info"].get("problem", "") if isinstance(row["extra_info"], dict) else ""
        gold = row["reward_model"].get("ground_truth", "") if isinstance(row["reward_model"], dict) else ""
        return compute_hash(str(prob), str(gold))

    cot["_problem_hash"] = cot.apply(row_hash, axis=1)
    matched = cot[cot._problem_hash.isin(keep_hashes)].drop(columns=["_problem_hash"])
    print(f"[z2] joined: {len(matched)} of {len(cot)} CoT rows match filter")

    if len(matched) == 0:
        import sys
        print("[z2] ERROR: no matches — hash format mismatch. Investigating...", file=sys.stderr)
        # Diagnostic: show sample hashes from each
        print("[z2 diag] sample scored hashes:", list(keep.problem_hash.head(5)))
        print("[z2 diag] sample cot hashes:", list(cot._problem_hash.head(5)))
        print("[z2 diag] sample cot raw: problem=", cot.iloc[0]["extra_info"].get("problem", "")[:80],
              " gt=", cot.iloc[0]["reward_model"].get("ground_truth", ""))
        return

    os.makedirs(f"{DST_DIR}/data", exist_ok=True)
    out_tbl = pa.Table.from_pandas(matched, preserve_index=False)
    out_path = f"{DST_DIR}/data/train.parquet"
    pq.write_table(out_tbl, out_path)
    print(f"[z2] wrote {len(matched)} rows to {out_path}")

    # Also create a small validation split (last 10% of matched)
    split = int(len(matched) * 0.9)
    val = matched.iloc[split:]
    val_tbl = pa.Table.from_pandas(val, preserve_index=False)
    pq.write_table(val_tbl, f"{DST_DIR}/data/validation.parquet")
    print(f"[z2] wrote {len(val)} rows to validation.parquet")


if __name__ == "__main__":
    main()
