"""Build a CoT-Goldilocks training parquet from the P-pilot probe.

The pilot probe (cot_pilot_thinkinterrupt) ran the prod CoT rollout config
(single_turn_agent + think-interrupt at 12288/7184) on 1200 stratified problems
from the 107k phys-tir corpus. Per-problem pass@8 in (0,1) defines the Goldilocks
band — the 237 problems where GRPO has maximal informative gradient signal under
the prod rollout config.

This is a smaller-but-prod-aligned alternative to Z4h-Goldilocks-245 (which was
filtered at max_tokens=4096, no think-interrupt). Use as the next-stage atom data
for testing prod recipe knobs (bf16-off, scale up, etc.) before committing to a
full P-full re-probe.
"""
import argparse
import hashlib
import os

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
    ap.add_argument("--pilot_merged", default="outputs/cot_pilot_thinkinterrupt/merged.parquet")
    ap.add_argument("--source_train", default="data/phys_tir_107k/train.parquet")
    ap.add_argument("--out_train", default="data/processed_cot_goldilocks_from_pilot/data/train.parquet")
    ap.add_argument("--out_val", default="data/processed_cot_goldilocks_from_pilot/data/validation.parquet")
    ap.add_argument("--val_frac", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    print(f"[build] loading pilot merged: {args.pilot_merged}")
    pilot = pq.read_table(args.pilot_merged).to_pandas()
    pp = pilot.groupby("problem_hash").agg(
        answer_type=("answer_type", "first"),
        pass_rate=("correct", "mean"),
        gt=("gt", "first"),
    ).reset_index()
    gold = pp[(pp["pass_rate"] > 0) & (pp["pass_rate"] < 1)].copy()
    print(f"[build]   {len(gold)} Goldilocks problems in pilot")
    print(f"[build]   per type: {gold.groupby('answer_type').size().to_dict()}")

    print(f"[build] loading source training parquet: {args.source_train}")
    src = pq.read_table(args.source_train).to_pandas()
    src["problem_hash"] = src.apply(row_hash, axis=1)
    src_by_hash = {h: i for i, h in enumerate(src["problem_hash"].tolist())}
    matched_idx = [src_by_hash[h] for h in gold["problem_hash"].tolist() if h in src_by_hash]
    matched = src.iloc[matched_idx].drop(columns=["problem_hash"]).reset_index(drop=True)
    print(f"[build]   matched {len(matched)}/{len(gold)} pilot Goldilocks rows in source parquet")

    def rewrite_to_cot(prompt):
        lst = list(prompt)
        if lst and isinstance(lst[0], dict) and lst[0].get("role") == "system":
            lst[0] = {"role": "system", "content": COT_SYSTEM_PROMPT}
        return lst
    matched["prompt"] = matched["prompt"].apply(rewrite_to_cot)

    rng = pd.Series(range(len(matched))).sample(frac=1.0, random_state=args.seed).index
    matched = matched.iloc[rng].reset_index(drop=True)

    n_val = max(int(len(matched) * args.val_frac), 8)
    val = matched.iloc[:n_val].reset_index(drop=True)
    train = matched.iloc[n_val:].reset_index(drop=True)
    print(f"[build]   train={len(train)}, val={n_val}")

    common_cols = ["data_source", "prompt", "reward_model", "extra_info", "pool"]
    for path, df in [(args.out_train, train[common_cols]),
                     (args.out_val, val[common_cols])]:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        pq.write_table(pa.Table.from_pandas(df, preserve_index=False), path)
        print(f"[build] wrote {path}")


if __name__ == "__main__":
    main()
