"""Build a tiny CoT smoke parquet for exercising the think-interrupt path.

Samples N rows from data/processed_cot/data/train.parquet (long problems that
WILL hit the interrupt budget) and appends one synthetic short row ("what is
2+2") that is expected to finish naturally inside the thinking budget. The
two populations together let us spot-check:

  - interrupt fires on long rows (phrase appears, mask=0 segment present)
  - interrupt does NOT fire on the 2+2 row (pure [1]-mask, \\boxed{4})

Usage:
  python3 scripts/build_cot_smoke_parquet.py \
      --src data/processed_cot/data/train.parquet \
      --dst outputs/smoke_cot_async/smoke.parquet \
      --n 7
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def build(src: Path, dst: Path, n: int, seed: int) -> None:
    df = pd.read_parquet(src)
    long_sample = df.sample(n=n, random_state=seed).reset_index(drop=True)

    # Build the 2+2 row by cloning the schema of row 0 and overriding the fields
    # that are actually read by the trainer/reward path.
    template = df.iloc[0].to_dict()
    system_prompt = long_sample.iloc[0]["prompt"][0]["content"]
    simple_row = dict(template)
    simple_row["prompt"] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": "What is 2+2? Answer in one sentence."},
    ]
    simple_row["reward_model"] = {"ground_truth": "\\boxed{4}", "style": "rule"}
    simple_row["extra_info"] = {
        **dict(template["extra_info"]),
        "answer_type": "numerical",
        "primary_answer_type": "numerical",
        "problem": "What is 2+2?",
        "unit": "",
        "tolerance": 0.0,
        "domain": "arithmetic",
        "domain_coarse": "arithmetic",
        "source": "smoke",
    }
    simple_row["data_source"] = "smoke"
    simple_row["pool"] = "smoke"

    combined = pd.concat(
        [long_sample, pd.DataFrame([simple_row])], ignore_index=True
    )
    dst.parent.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(dst, index=False)
    print(f"Wrote {len(combined)} rows to {dst}")
    for i, r in combined.iterrows():
        q = r["extra_info"].get("problem", "")
        q_short = q[:80].replace("\n", " ")
        print(f"  [{i}] {r['data_source']:10s} {r['pool']:8s}  {q_short}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, type=Path)
    ap.add_argument("--dst", required=True, type=Path)
    ap.add_argument("--n", type=int, default=7, help="number of long rows to sample")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    build(args.src, args.dst, args.n, args.seed)


if __name__ == "__main__":
    main()
