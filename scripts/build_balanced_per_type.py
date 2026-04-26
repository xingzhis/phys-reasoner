"""Build a per-type balanced multi-type Goldilocks dataset.

The Goldilocks-from-pilot data is mcq-heavy (mcq:123, num:54, expr:37, eq:23).
Random sampling under async path yields per-batch type fractions that mirror
this skew, leaving equation under-represented in the gradient. Async strict
batch-level stratification requires verl-side sampler hooks; here we use the
no-code-change alternative: rebuild the dataset to be uniform per type so that
random sampling yields balanced per-batch composition in expectation.

Sources:
  - pilot Goldilocks (CoT-prod-budget-validated): mcq=123, num=54, expr=37, eq=23
  - 107k TIR-Goldilocks-band: ~7-11k per type (TIR-pass != CoT-pass; diagnostic-grade)

Strategy: for each of the four main types, take all of the pilot Goldilocks
problems (highest fidelity), then augment from 107k TIR-Goldilocks to reach a
target count. Mirrors build_expanded_per_type.py but does both expansion and
truncation/balancing.

Output:
  - data/processed_cot_balanced_per_type/data/{train,validation}.parquet
  - data/processed_tir_balanced_per_type/data/{train,validation}.parquet
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
    ap.add_argument("--target_per_type", type=int, default=100)
    ap.add_argument("--out_cot_dir", default="data/processed_cot_balanced_per_type")
    ap.add_argument("--out_tir_dir", default="data/processed_tir_balanced_per_type")
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

    main_types = ["numerical", "mcq", "expression", "equation"]
    selected_hashes_by_type = {}

    for t in main_types:
        pilot_t = pilot[pilot["answer_type"]==t]
        existing = set(pilot_t["problem_hash"].tolist())
        n_pilot = len(pilot_t)

        if n_pilot >= args.target_per_type:
            kept = pilot_t.sample(n=args.target_per_type, random_state=args.seed)
            selected_hashes_by_type[t] = kept["problem_hash"].tolist()
            print(f"[build]   {t}: pilot {n_pilot} -> kept {args.target_per_type} (random subsample)")
        else:
            # Augment from 107k TIR-Goldilocks-band
            n_to_add = args.target_per_type - n_pilot
            candidates = agg[(agg["answer_type"]==t)
                             & (agg["pass_rate"]>0) & (agg["pass_rate"]<1)
                             & (~agg["problem_hash"].isin(existing))]
            sampled = candidates.sample(n=min(n_to_add, len(candidates)), random_state=args.seed)
            combined_hashes = pilot_t["problem_hash"].tolist() + sampled["problem_hash"].tolist()
            selected_hashes_by_type[t] = combined_hashes
            print(f"[build]   {t}: pilot {n_pilot} + {len(sampled)} from 107k = {len(combined_hashes)}")

    # Build train + val for both CoT and TIR
    rng = pd.Series(range(args.target_per_type * 4)).sample(frac=1.0, random_state=args.seed).tolist()
    val_n_per_type = 3   # tiny val per type, just for verl format

    def rewrite_to_cot(prompt):
        lst = list(prompt)
        if lst and isinstance(lst[0], dict) and lst[0].get("role") == "system":
            lst[0] = {"role": "system", "content": COT_SYSTEM_PROMPT}
        return lst

    common_cols = ["data_source", "prompt", "reward_model", "extra_info", "pool"]

    for mode_name, out_dir, rewrite_fn in [
        ("cot", args.out_cot_dir, rewrite_to_cot),
        ("tir", args.out_tir_dir, lambda p: list(p)),  # keep TIR system prompt
    ]:
        all_train_rows = []
        all_val_rows = []
        for t in main_types:
            hashes = selected_hashes_by_type[t]
            idx = [src_by_hash[h] for h in hashes if h in src_by_hash]
            rows = src.iloc[idx][common_cols].copy().reset_index(drop=True)
            rows["prompt"] = rows["prompt"].apply(rewrite_fn)
            # Per-type val/train split
            val_rows = rows.iloc[:val_n_per_type].reset_index(drop=True)
            train_rows = rows.iloc[val_n_per_type:].reset_index(drop=True)
            all_train_rows.append(train_rows)
            all_val_rows.append(val_rows)

        # Interleave types in the train rows so dataset order is balanced
        # (each consecutive 4 rows = one of each type). Async sampler handles
        # remaining randomization; this ensures type-balance baked into ordering.
        train_dfs = all_train_rows
        max_len = max(len(d) for d in train_dfs)
        interleaved = []
        for i in range(max_len):
            for d in train_dfs:
                if i < len(d):
                    interleaved.append(d.iloc[[i]])
        train_combined = pd.concat(interleaved, ignore_index=True)
        val_combined = pd.concat(all_val_rows, ignore_index=True)

        os.makedirs(f"{out_dir}/data", exist_ok=True)
        pq.write_table(pa.Table.from_pandas(train_combined, preserve_index=False),
                       f"{out_dir}/data/train.parquet")
        pq.write_table(pa.Table.from_pandas(val_combined, preserve_index=False),
                       f"{out_dir}/data/validation.parquet")
        print(f"[build] {mode_name}: train={len(train_combined)}, val={len(val_combined)} "
              f"-> {out_dir}/data/")
        print(f"[build]   train type counts: {train_combined['extra_info'].apply(lambda x: x.get('answer_type','') if isinstance(x, dict) else '').value_counts().to_dict()}")


if __name__ == "__main__":
    main()
