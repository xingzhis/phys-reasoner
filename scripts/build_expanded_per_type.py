"""Expand per-type isolation datasets to ~200 each for fair comparison.

Pilot CoT-Goldilocks gives us small numbers per type (37 expr, 23 eq). For
controlled isolation atoms we want ~150-200 problems per type (Z4h's 245 was
the proven baseline). Augment the expression and equation slices by sampling
from the existing 107k TIR-probe Goldilocks-band (TIR-pass ∈ (0,1)).

Caveat: TIR-pass != CoT-pass, so some augmented problems may not actually be
in the CoT-Goldilocks band. But for the diagnostic question "does this type
climb in isolation?", noise in band-membership is acceptable — Z4h-245 had
the same caveat (filtered at TIR + max_tokens=4096) and climbed cleanly.

Output:
  - data/processed_cot_z4p_expr_v2/data/{train,validation}.parquet
  - data/processed_cot_z4q_eq_v2/data/{train,validation}.parquet
  Each containing: pilot's {expression, equation} + sampled additions from 107k.
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
    ap.add_argument("--scored_probe", default="outputs/probe_qwen3_4b_v2_107k/rollouts_scored_trainmatched_107k.parquet")
    ap.add_argument("--source_train", default="data/phys_tir_107k/train.parquet")
    ap.add_argument("--pilot_train", default="data/processed_cot_goldilocks_from_pilot/data/train.parquet")
    ap.add_argument("--target_per_type", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    print(f"[build] loading scored probe: {args.scored_probe}")
    probe = pq.read_table(args.scored_probe,
        columns=["problem_idx", "answer_type", "correct", "problem_hash"]).to_pandas()
    agg = (probe.groupby("problem_idx")
        .agg(answer_type=("answer_type","first"),
             pass_rate=("correct","mean"),
             problem_hash=("problem_hash","first"))
        .reset_index())

    print(f"[build] loading source training parquet: {args.source_train}")
    src = pq.read_table(args.source_train).to_pandas()
    src["problem_hash"] = src.apply(row_hash, axis=1)
    src_by_hash = {h: i for i, h in enumerate(src["problem_hash"].tolist())}

    print(f"[build] loading pilot Goldilocks: {args.pilot_train}")
    pilot = pq.read_table(args.pilot_train).to_pandas()
    pilot["answer_type"] = pilot["extra_info"].apply(
        lambda x: x.get("answer_type","") if isinstance(x, dict) else "")
    pilot["problem_hash"] = pilot.apply(row_hash, axis=1)

    def rewrite_to_cot(prompt):
        lst = list(prompt)
        if lst and isinstance(lst[0], dict) and lst[0].get("role") == "system":
            lst[0] = {"role": "system", "content": COT_SYSTEM_PROMPT}
        return lst

    common_cols = ["data_source", "prompt", "reward_model", "extra_info", "pool"]

    for type_name, dirname in [("expression","z4p_expr_v2"), ("equation","z4q_eq_v2")]:
        pilot_t = pilot[pilot["answer_type"]==type_name]
        existing_hashes = set(pilot_t["problem_hash"].tolist())
        n_to_add = max(args.target_per_type - len(pilot_t), 0)

        print(f"\n[build] === {type_name} ===")
        print(f"[build]   pilot Goldilocks: {len(pilot_t)} (target {args.target_per_type}, need +{n_to_add})")

        # Sample from 107k TIR-Goldilocks-band, excluding pilot hashes
        candidates = agg[(agg["answer_type"]==type_name)
                         & (agg["pass_rate"]>0) & (agg["pass_rate"]<1)
                         & (~agg["problem_hash"].isin(existing_hashes))]
        print(f"[build]   107k TIR-Goldilocks-band candidates: {len(candidates)}")

        if n_to_add > 0:
            rng = candidates.sample(n=min(n_to_add, len(candidates)), random_state=args.seed)
            sampled_hashes = rng["problem_hash"].tolist()
            matched_idx = [src_by_hash[h] for h in sampled_hashes if h in src_by_hash]
            added = src.iloc[matched_idx].copy()
            added["problem_hash"] = [src["problem_hash"].iloc[i] for i in matched_idx]
            added["prompt"] = added["prompt"].apply(rewrite_to_cot)
            print(f"[build]   added {len(added)} rows from 107k")
            combined = pd.concat([pilot_t, added], ignore_index=True)
        else:
            combined = pilot_t.copy()

        # Drop helper cols & shuffle
        combined = combined.drop(columns=["answer_type","problem_hash"]).reset_index(drop=True)
        combined = combined.sample(frac=1.0, random_state=args.seed).reset_index(drop=True)
        n_val = max(min(int(len(combined) * 0.05), 10), 4)
        val_t = combined.iloc[:n_val][common_cols].reset_index(drop=True)
        train_t = combined.iloc[n_val:][common_cols].reset_index(drop=True)
        print(f"[build]   final: train={len(train_t)}, val={len(val_t)}")

        out_dir = f"data/processed_cot_{dirname}/data"
        os.makedirs(out_dir, exist_ok=True)
        pq.write_table(pa.Table.from_pandas(train_t, preserve_index=False), f"{out_dir}/train.parquet")
        pq.write_table(pa.Table.from_pandas(val_t, preserve_index=False), f"{out_dir}/validation.parquet")
        print(f"[build]   wrote {out_dir}/")


if __name__ == "__main__":
    main()
