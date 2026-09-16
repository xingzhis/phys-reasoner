"""Build Z4k mixed dataset: Z4h numerical (245) + small expression slice (~80) from
the 107k TIR probe filtered to expression-type with TIR-pass ∈ (0.125, 0.875).

Z4k is an integration atom for xverify-on-4B. Single-knob change from Z4h:
  reward = polaris (rule-only) → reward.py (rule + xverify fallback)
The expression slice exercises the xverify path; numerical slice keeps a stable
baseline so we can see if xverify integration breaks the climb.

System prompt is rewritten to the CoT format (matches Z4h numerical and the
single_turn_agent rollout we'll use).
"""
import argparse
import os
import hashlib

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


COT_SYSTEM_PROMPT = (
    "You are an expert physics problem solver.\n"
    "Solve the problem step by step. Show your reasoning concisely.\n\n"
    "End with your final answer as \\boxed{<value>}.\n"
)


def row_hash(row) -> str:
    prob = row["extra_info"].get("problem", "") if isinstance(row["extra_info"], dict) else ""
    gold = row["reward_model"].get("ground_truth", "") if isinstance(row["reward_model"], dict) else ""
    return hashlib.md5(f"{prob}|||{gold}".encode("utf-8")).hexdigest()[:16]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--z4h_train", default="data/processed_cot_z4h_goldilocks/data/train.parquet")
    ap.add_argument("--z4h_val", default="data/processed_cot_z4h_goldilocks/data/validation.parquet")
    ap.add_argument("--scored_probe", default="outputs/probe_qwen3_4b_v2_107k/rollouts_scored_trainmatched_107k.parquet")
    ap.add_argument("--source_train", default="data/phys_tir_107k/train.parquet")
    ap.add_argument("--n_expression", type=int, default=80,
                    help="Expression problems to add. Stratified across TIR-pass bands.")
    ap.add_argument("--out_train", default="data/processed_cot_z4k_xverify_mix/data/train.parquet")
    ap.add_argument("--out_val", default="data/processed_cot_z4k_xverify_mix/data/validation.parquet")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    print(f"[build] loading Z4h numerical: {args.z4h_train}")
    z4h_train = pq.read_table(args.z4h_train).to_pandas()
    z4h_val = pq.read_table(args.z4h_val).to_pandas()
    print(f"[build]   z4h_train={len(z4h_train)} z4h_val={len(z4h_val)}")

    print(f"[build] loading scored probe: {args.scored_probe}")
    probe = pq.read_table(args.scored_probe,
        columns=["problem_idx", "answer_type", "correct", "problem_hash"]).to_pandas()
    agg = (probe.groupby("problem_idx")
        .agg(answer_type=("answer_type","first"),
             pass_rate=("correct","mean"),
             problem_hash=("problem_hash","first"))
        .reset_index())
    expr = agg[(agg["answer_type"] == "expression") & (agg["pass_rate"] > 0) & (agg["pass_rate"] < 1)].copy()
    print(f"[build]   expression-Goldilocks problems in probe: {len(expr)}")

    bins = [-0.001, 0.124, 0.375, 0.624, 0.875, 1.001]
    labels = ["1/8","2-3/8","4-5/8","6-7/8","≥7/8"]
    expr["band"] = pd.cut(expr["pass_rate"], bins=bins, labels=labels)
    per_band = max(args.n_expression // 4, 5)
    rng = pd.Series(range(len(expr))).sample(frac=1.0, random_state=args.seed).index
    sampled = (expr.iloc[rng]
        .groupby("band", observed=True)
        .head(per_band)
        .reset_index(drop=True))
    sampled = sampled.head(args.n_expression)
    print(f"[build]   sampled {len(sampled)} expression problems:")
    print(sampled["band"].value_counts())

    print(f"[build] loading source training parquet: {args.source_train}")
    src = pq.read_table(args.source_train).to_pandas()
    src["problem_hash"] = src.apply(row_hash, axis=1)
    src_by_hash = {h: i for i, h in enumerate(src["problem_hash"].tolist())}
    matched_idx = [src_by_hash[h] for h in sampled["problem_hash"].tolist() if h in src_by_hash]
    expr_rows = src.iloc[matched_idx].drop(columns=["problem_hash"]).reset_index(drop=True)

    def rewrite_to_cot(prompt):
        lst = list(prompt)
        if lst and isinstance(lst[0], dict) and lst[0].get("role") == "system":
            lst[0] = {"role": "system", "content": COT_SYSTEM_PROMPT}
        return lst
    expr_rows["prompt"] = expr_rows["prompt"].apply(rewrite_to_cot)
    print(f"[build]   matched {len(expr_rows)}/{len(sampled)} sampled expression rows")

    # Sanity: required columns match
    common_cols = ["data_source", "prompt", "reward_model", "extra_info", "pool"]
    for c in common_cols:
        assert c in z4h_train.columns and c in expr_rows.columns, f"missing {c}"

    train_combined = pd.concat([z4h_train[common_cols], expr_rows[common_cols]], ignore_index=True)
    val_combined = z4h_val[common_cols]  # keep Z4h val unchanged
    print(f"[build] combined train: {len(train_combined)} ({len(z4h_train)} num + {len(expr_rows)} expr)")
    print(f"[build] val (unchanged): {len(val_combined)}")

    for path, df in [(args.out_train, train_combined), (args.out_val, val_combined)]:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        pq.write_table(pa.Table.from_pandas(df, preserve_index=False), path)
        print(f"[build] wrote {path}")


if __name__ == "__main__":
    main()
