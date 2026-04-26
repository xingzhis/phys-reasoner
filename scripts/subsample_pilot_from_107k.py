"""Stratified pilot subsample for the prod-budget CoT re-probe.

Picks problems from the existing TIR-scored 107k probe (option 2 per discussion):
  - filter to four main answer types {numerical, expression, equation, mcq}
  - bin by TIR pass-rate band (0/8, 1/8, 2-3/8, 4-5/8, 6-7/8, 8/8)
  - sample N_PER_CELL per (type, band) cell → ~1-2k pilot

Then matches each sampled problem_hash to its row in the canonical training parquet
(by recomputing problem_hash via the same recipe used in cot_probe_base_4b.py)
so the output is a probe-input parquet with the right `prompt`/`reward_model`/
`extra_info` schema.
"""
import argparse
import hashlib
import os
import sys

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


def row_hash(row) -> str:
    prob = row["extra_info"].get("problem", "") if isinstance(row["extra_info"], dict) else ""
    gold = row["reward_model"].get("ground_truth", "") if isinstance(row["reward_model"], dict) else ""
    return hashlib.md5(f"{prob}|||{gold}".encode("utf-8")).hexdigest()[:16]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--scored_probe",
        default="outputs/probe_qwen3_4b_v2_107k/rollouts_scored_trainmatched_107k.parquet",
    )
    ap.add_argument(
        "--source_train",
        default="data/phys_tir_107k/train.parquet",
        help="Canonical training parquet to pull prompts from. The 107k version has "
        "the TIR system prompt; we rewrite it to the CoT system prompt below to match "
        "single_turn_agent rollouts.",
    )
    ap.add_argument(
        "--cot_system_prompt",
        default=(
            "You are an expert physics problem solver.\n"
            "Solve the problem step by step. Show your reasoning concisely.\n\n"
            "End with your final answer as \\boxed{<value>}.\n"
        ),
        help="CoT system prompt that replaces the TIR system prompt (matches "
        "data/processed_cot/data/train.parquet).",
    )
    ap.add_argument("--output", default="data/processed_cot_pilot_1k/data/train.parquet")
    ap.add_argument("--n_per_cell", type=int, default=50,
                    help="Per (answer_type, pass-band) cell. 4 types x 6 bands x 50 = 1200.")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    print(f"[subsample] loading scored probe: {args.scored_probe}")
    probe = pq.read_table(
        args.scored_probe, columns=["problem_idx", "answer_type", "correct", "problem_hash"]
    ).to_pandas()
    print(f"[subsample]   {len(probe)} rollout rows, {probe['problem_idx'].nunique()} problems")

    agg = (
        probe.groupby("problem_idx")
        .agg(answer_type=("answer_type", "first"),
             pass_rate=("correct", "mean"),
             n=("correct", "size"),
             problem_hash=("problem_hash", "first"))
        .reset_index()
    )
    main_types = ["numerical", "expression", "equation", "mcq"]
    agg = agg[agg["answer_type"].isin(main_types)].copy()
    print(f"[subsample]   {len(agg)} problems in main types {main_types}")

    bins = [-0.001, 0.001, 0.124, 0.375, 0.624, 0.875, 1.001]
    labels = ["0/8", "1/8", "2-3/8", "4-5/8", "6-7/8", "8/8"]
    agg["band"] = pd.cut(agg["pass_rate"], bins=bins, labels=labels)

    rng = pd.Series(range(len(agg))).sample(frac=1.0, random_state=args.seed).index
    sampled = (
        agg.iloc[rng]
        .groupby(["answer_type", "band"], observed=True)
        .head(args.n_per_cell)
        .reset_index(drop=True)
    )
    print(f"[subsample]   sampled {len(sampled)} problems via stratified (type x band):")
    print(pd.crosstab(sampled["answer_type"], sampled["band"], margins=True))

    print(f"[subsample] loading source training parquet: {args.source_train}")
    src = pq.read_table(args.source_train).to_pandas()
    print(f"[subsample]   {len(src)} rows")
    src["problem_hash"] = src.apply(row_hash, axis=1)
    src_by_hash = {h: i for i, h in enumerate(src["problem_hash"].tolist())}

    sampled_hashes = set(sampled["problem_hash"].tolist())
    matched_idx = [src_by_hash[h] for h in sampled_hashes if h in src_by_hash]
    matched = src.iloc[matched_idx].drop(columns=["problem_hash"]).reset_index(drop=True)
    print(f"[subsample] matched {len(matched)}/{len(sampled)} sampled hashes in source parquet")

    if len(matched) < 0.5 * len(sampled):
        print("[subsample] WARNING: < 50% match rate — source parquet may not cover the probe's full corpus.")
        print("[subsample] Falling back to all matched rows; pilot may under-represent some bands.")

    # Rewrite the TIR system prompt to the CoT system prompt so the probe
    # exercises CoT mode (single_turn_agent), not TIR.
    def rewrite_to_cot(prompt):
        # prompt is a list/array of {role, content} dicts
        lst = list(prompt)
        if lst and isinstance(lst[0], dict) and lst[0].get("role") == "system":
            lst[0] = {"role": "system", "content": args.cot_system_prompt}
        return lst

    matched["prompt"] = matched["prompt"].apply(rewrite_to_cot)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    pq.write_table(pa.Table.from_pandas(matched, preserve_index=False), args.output)
    print(f"[subsample] wrote {args.output} ({len(matched)} rows, system prompt rewritten to CoT)")


if __name__ == "__main__":
    main()
