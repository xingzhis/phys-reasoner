"""Filter Goldilocks-from-pilot to numerical-only — diagnostic dataset for Z4n.

Z4m was prod recipe (think-interrupt + sp=2 + reward.py + 19488 budget) on the
mixed-type Goldilocks-from-pilot data. It oscillated rather than climbed.
Z4n isolates: same recipe, single-type (numerical) data only. If Z4n climbs,
multi-type interference / verifier heterogeneity is the culprit; if it doesn't,
the recipe itself is the issue (and Z4h's recipe would be the differentiating axis).
"""
import argparse
import hashlib
import os

import pyarrow as pa
import pyarrow.parquet as pq


COT_SYSTEM_PROMPT = (
    "You are an expert physics problem solver.\n"
    "Solve the problem step by step. Show your reasoning concisely.\n\n"
    "End with your final answer as \\boxed{<value>}.\n"
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src_train", default="data/processed_cot_goldilocks_from_pilot/data/train.parquet")
    ap.add_argument("--src_val", default="data/processed_cot_goldilocks_from_pilot/data/validation.parquet")
    ap.add_argument("--out_train", default="data/processed_cot_goldilocks_pilot_numerical/data/train.parquet")
    ap.add_argument("--out_val", default="data/processed_cot_goldilocks_pilot_numerical/data/validation.parquet")
    ap.add_argument("--keep_type", default="numerical")
    args = ap.parse_args()

    for src, out in [(args.src_train, args.out_train), (args.src_val, args.out_val)]:
        print(f"[filter] {src}")
        df = pq.read_table(src).to_pandas()
        df["answer_type"] = df["extra_info"].apply(
            lambda x: x.get("answer_type", "") if isinstance(x, dict) else "")
        kept = df[df["answer_type"] == args.keep_type].drop(columns=["answer_type"]).reset_index(drop=True)
        print(f"[filter]   {len(df)} → {len(kept)} ({args.keep_type} only)")

        os.makedirs(os.path.dirname(out), exist_ok=True)
        pq.write_table(pa.Table.from_pandas(kept, preserve_index=False), out)
        print(f"[filter]   wrote {out}")


if __name__ == "__main__":
    main()
