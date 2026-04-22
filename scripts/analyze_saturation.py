"""Compute GRPO group saturation stats from a scored rollouts parquet.

Reads a parquet produced by eval/scoring/our_verifier.py (columns:
problem_idx, rollout_idx, verdict, correct, unverifiable, answer_type)
and groups by problem_idx to compute:

  - per-group reward mean and std
  - saturated group = all correct (mean=1) OR all wrong (mean=0)
    (unverifiable rows count as 0.0 for this purpose — they yield zero
     advantage under GRPO just like confirmed wrong answers)
  - saturation fraction and its 95% CI
  - histogram of group reward means
  - breakdown by answer_type

This is the diagnostic for the "GRPO group saturation" hypothesis: if most
groups are all-right or all-wrong, the advantage signal within each group
is zero, and the policy has little to learn.

Usage:
    python3 scripts/analyze_saturation.py \\
        --scored outputs/saturation_analysis/tir_step100/scored.parquet \\
        --n_rollouts 8 \\
        --out outputs/saturation_analysis/tir_step100/saturation.json
"""
from __future__ import annotations

import argparse
import json
import math
import os
from collections import Counter


def analyze(scored_path: str, n_rollouts: int, out_path: str) -> dict:
    import pandas as pd  # noqa: PLC0415

    df = pd.read_parquet(scored_path)
    n = len(df)
    if "problem_idx" not in df.columns:
        raise KeyError("scored parquet missing 'problem_idx' column")

    # Binary correctness: unverifiable counts as 0.0 (matches training reward
    # where verify_answer returns -1 but the reward mapping clamps <1 to 0 for
    # the scalar in the advantage computation).
    df["reward"] = df["correct"].astype(float)

    groups = df.groupby("problem_idx").agg(
        reward_mean=("reward", "mean"),
        reward_std=("reward", "std"),
        n_rollouts=("reward", "size"),
        answer_type=("answer_type", "first"),
    ).reset_index()

    # Replace NaN std (singleton groups) with 0 for cleanness.
    groups["reward_std"] = groups["reward_std"].fillna(0.0)

    n_groups = len(groups)

    # A group is saturated under GRPO when its within-group advantage is zero,
    # i.e. all rollouts have identical reward. For binary rewards, that's
    # equivalent to std == 0, equivalent to mean in {0, 1}.
    saturated = groups["reward_std"] == 0.0
    all_right = groups["reward_mean"] == 1.0
    all_wrong = groups["reward_mean"] == 0.0

    sat_frac = float(saturated.mean()) if n_groups else float("nan")
    allright_frac = float(all_right.mean()) if n_groups else float("nan")
    allwrong_frac = float(all_wrong.mean()) if n_groups else float("nan")

    # Wilson 95% CI for the saturated fraction
    if n_groups:
        p = sat_frac
        z = 1.96
        denom = 1 + z * z / n_groups
        center = (p + z * z / (2 * n_groups)) / denom
        half = z * math.sqrt((p * (1 - p) + z * z / (4 * n_groups)) / n_groups) / denom
        ci_lo, ci_hi = max(0.0, center - half), min(1.0, center + half)
    else:
        ci_lo = ci_hi = float("nan")

    # Per-type breakdown
    per_type = {}
    for atype, sub in groups.groupby("answer_type"):
        per_type[str(atype)] = {
            "n_groups": int(len(sub)),
            "mean_reward": float(sub["reward_mean"].mean()),
            "saturated_frac": float((sub["reward_std"] == 0.0).mean()),
            "all_right_frac": float((sub["reward_mean"] == 1.0).mean()),
            "all_wrong_frac": float((sub["reward_mean"] == 0.0).mean()),
        }

    # Histogram of group reward means in bins of 1/n_rollouts
    bins = [i / n_rollouts for i in range(n_rollouts + 1)]
    hist = Counter()
    for m in groups["reward_mean"]:
        k = round(m * n_rollouts) / n_rollouts
        hist[k] += 1
    hist_sorted = [(round(b, 4), hist.get(round(i / n_rollouts, 4), 0)) for i, b in enumerate(bins)]

    result = {
        "scored_path": scored_path,
        "n_rollouts_total": int(n),
        "n_groups": int(n_groups),
        "expected_rollouts_per_group": int(n_rollouts),
        "mean_reward": float(df["reward"].mean()) if n else float("nan"),
        "saturated_frac": sat_frac,
        "saturated_ci95": [ci_lo, ci_hi],
        "all_right_frac": allright_frac,
        "all_wrong_frac": allwrong_frac,
        "informative_frac": float(1.0 - sat_frac) if n_groups else float("nan"),
        "per_type": per_type,
        "group_mean_histogram": hist_sorted,
    }

    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)

    # Human-readable summary
    print("=" * 72)
    print(f"Saturation analysis: {scored_path}")
    print("=" * 72)
    print(f"  total rollouts        : {n}")
    print(f"  n_groups              : {n_groups}")
    print(f"  overall mean reward   : {result['mean_reward']:.4f}")
    print(f"  saturated groups      : {sat_frac:.3f}  "
          f"(95% CI [{ci_lo:.3f}, {ci_hi:.3f}])")
    print(f"    all-right (top)     : {allright_frac:.3f}")
    print(f"    all-wrong (bottom)  : {allwrong_frac:.3f}")
    print(f"  informative groups    : {1 - sat_frac:.3f}  (non-zero within-group advantage)")
    print()
    print("  Group reward-mean histogram:")
    for b, c in hist_sorted:
        bar = "#" * int(50 * c / max(1, n_groups))
        print(f"    {b:.3f}  {c:5d}  {bar}")
    print()
    print("  Per answer_type:")
    for atype, stats in per_type.items():
        print(f"    {atype:14s}  n={stats['n_groups']:4d}  "
              f"mean={stats['mean_reward']:.3f}  sat={stats['saturated_frac']:.3f}  "
              f"allR={stats['all_right_frac']:.3f}  allW={stats['all_wrong_frac']:.3f}")
    print()
    print(f"Saved JSON → {out_path}")
    return result


def main() -> None:
    p = argparse.ArgumentParser(description="GRPO group saturation analysis")
    p.add_argument("--scored", required=True, help="scored.parquet from our_verifier.py")
    p.add_argument("--n_rollouts", type=int, default=8,
                   help="rollouts per prompt (for histogram bin width only)")
    p.add_argument("--out", required=True, help="Path to saturation.json output")
    args = p.parse_args()
    analyze(args.scored, args.n_rollouts, args.out)


if __name__ == "__main__":
    main()
