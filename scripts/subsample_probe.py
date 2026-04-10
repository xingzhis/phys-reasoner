"""Stratified probe subsample from train splits for rollout scoring.

Samples a manageable subset of the *train split only* for the rollout probe
(Step 3 in the data pipeline). Dev and test rows are excluded.

Target: ~2,500 rows (2,000 Dr. SCI + 500 corpus).
Aim for ~40 rows per stratum with a minimum floor of 10.

Stratification keys (same as split_train_dev_test.py):
  Dr. SCI: (answer_type × from × difficulty_bin)
  Corpus:  (primary_answer_type × source)

Outputs:
    data/processed/probe_subset.parquet

Usage:
    python scripts/subsample_probe.py
    python scripts/subsample_probe.py --seed 42
    python scripts/subsample_probe.py --report   # stats only, no save
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Difficulty binning (same as split_train_dev_test.py)
# ---------------------------------------------------------------------------

def difficulty_bin(d) -> str:
    try:
        d = float(d)
    except (ValueError, TypeError):
        return "low"
    if d <= 0.0:
        return "low"
    elif d <= 0.375:
        return "medium"
    else:
        return "high"


# ---------------------------------------------------------------------------
# Stratified sampling with floor
# ---------------------------------------------------------------------------

def stratified_sample(
    df: pd.DataFrame,
    stratum_col: str,
    n_target: int,
    min_per_stratum: int,
    seed: int,
) -> pd.DataFrame:
    """Sample n_target rows from df, stratified by stratum_col.

    Each stratum gets at least min_per_stratum rows (capped at stratum size).
    Remaining budget is allocated proportionally.
    """
    rng = np.random.RandomState(seed)
    stratum_sizes = df[stratum_col].value_counts()
    strata = stratum_sizes.index.tolist()
    total = len(df)

    # Phase 1: allocate floor
    alloc: dict[str, int] = {}
    for s in strata:
        alloc[s] = min(min_per_stratum, stratum_sizes[s])

    floor_total = sum(alloc.values())
    remaining = n_target - floor_total

    if remaining > 0:
        # Phase 2: distribute remaining proportionally
        # Only strata that have headroom above their floor get extra
        headroom = {s: stratum_sizes[s] - alloc[s] for s in strata}
        total_headroom = sum(headroom.values())

        if total_headroom > 0:
            for s in strata:
                if headroom[s] <= 0:
                    continue
                extra = round(remaining * headroom[s] / total_headroom)
                extra = min(extra, headroom[s])
                alloc[s] += extra

    # Adjust to hit target exactly
    current = sum(alloc.values())
    diff = n_target - current

    if diff != 0:
        sorted_strata = sorted(strata, key=lambda s: stratum_sizes[s], reverse=True)
        for i in range(abs(diff)):
            s = sorted_strata[i % len(sorted_strata)]
            if diff > 0 and alloc[s] < stratum_sizes[s]:
                alloc[s] += 1
            elif diff < 0 and alloc[s] > min(min_per_stratum, stratum_sizes[s]):
                alloc[s] -= 1

    # Sample from each stratum
    parts = []
    for s in strata:
        n = alloc[s]
        if n <= 0:
            continue
        group = df[df[stratum_col] == s]
        n = min(n, len(group))
        parts.append(group.sample(n=n, random_state=rng))

    return pd.concat(parts, ignore_index=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--drsci-input",
                        default="data/processed/drsci_train_split.parquet")
    parser.add_argument("--corpus-input",
                        default="data/processed/corpus_train_split.parquet")
    parser.add_argument("--output",
                        default="data/processed/probe_subset.parquet")
    parser.add_argument("--n-drsci", type=int, default=2000)
    parser.add_argument("--n-corpus", type=int, default=500)
    parser.add_argument("--min-per-stratum", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--report", action="store_true",
                        help="Print stats only, no save")
    args = parser.parse_args()

    # --- Dr. SCI ---
    print("=== Dr. SCI Probe Subsample ===")
    drsci = pd.read_parquet(args.drsci_input)
    print(f"  Train split: {len(drsci):,} rows")

    drsci["_answer_type"] = drsci["extra_info"].apply(lambda x: x["answer_type"])
    drsci["_from"] = drsci["extra_info"].apply(lambda x: x["from"])
    drsci["_difficulty_bin"] = drsci["extra_info"].apply(
        lambda x: difficulty_bin(x["difficulty"])
    )
    drsci["_stratum"] = (drsci["_answer_type"] + "|" + drsci["_from"]
                         + "|" + drsci["_difficulty_bin"])

    n_strata_d = drsci["_stratum"].nunique()
    print(f"  {n_strata_d} strata")

    drsci_sample = stratified_sample(
        drsci, "_stratum", args.n_drsci, args.min_per_stratum, args.seed
    )
    # Tag dataset origin
    drsci_sample["_dataset"] = "drsci"
    print(f"  Sampled {len(drsci_sample):,} rows")

    # --- Corpus ---
    print("\n=== Corpus Probe Subsample ===")
    corpus = pd.read_parquet(args.corpus_input)
    print(f"  Train split: {len(corpus):,} rows")

    corpus["_primary_answer_type"] = corpus["extra_info"].apply(
        lambda x: x["primary_answer_type"]
    )
    corpus["_source"] = corpus["extra_info"].apply(lambda x: x["source"])
    corpus["_stratum"] = corpus["_primary_answer_type"] + "|" + corpus["_source"]

    n_strata_c = corpus["_stratum"].nunique()
    print(f"  {n_strata_c} strata")

    corpus_sample = stratified_sample(
        corpus, "_stratum", args.n_corpus, args.min_per_stratum, args.seed
    )
    corpus_sample["_dataset"] = "corpus"
    print(f"  Sampled {len(corpus_sample):,} rows")

    # --- Merge and clean ---
    # Drop temp columns, keep _dataset as metadata
    drsci_sample.drop(
        columns=["_answer_type", "_from", "_difficulty_bin", "_stratum"],
        inplace=True,
    )
    corpus_sample.drop(
        columns=["_primary_answer_type", "_source", "_stratum"],
        inplace=True,
    )

    # Normalize difficulty type: drsci has float, corpus has string.
    # Convert drsci difficulty to string so Arrow can merge the extra_info structs.
    drsci_sample["extra_info"] = drsci_sample["extra_info"].apply(
        lambda d: {**d, "difficulty": str(d["difficulty"])}
    )

    probe = pd.concat([drsci_sample, corpus_sample], ignore_index=True)
    # Shuffle
    probe = probe.sample(frac=1, random_state=args.seed).reset_index(drop=True)

    print(f"\n=== Combined Probe Subset ===")
    print(f"  Total: {len(probe):,} rows "
          f"({len(drsci_sample):,} drsci + {len(corpus_sample):,} corpus)")

    # Print stratum stats
    print(f"\n  Dr. SCI per-stratum sizes:")
    _print_sample_stats(drsci, drsci_sample, "_stratum")
    print(f"\n  Corpus per-stratum sizes:")
    _print_sample_stats(corpus, corpus_sample, "_stratum")

    if not args.report:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        probe.to_parquet(args.output, index=False)
        print(f"\n  Saved → {args.output}")
    else:
        print("\n(--report mode: no files written)")


def _print_sample_stats(full_df, sample_df, stratum_col):
    """Print per-stratum sample size vs population size."""
    # Reconstruct stratum on sample (already dropped temp cols, but _dataset is still there)
    # Use the full_df's stratum info: match by index
    pop = full_df[stratum_col].value_counts().sort_index()
    # For sample, reconstruct stratum from extra_info
    if "_stratum" in sample_df.columns:
        samp = sample_df[stratum_col].value_counts().sort_index()
    else:
        # Already dropped — reconstruct from extra_info
        if sample_df["_dataset"].iloc[0] == "drsci":
            sample_df = sample_df.copy()
            sample_df["_s"] = sample_df["extra_info"].apply(
                lambda x: f"{x['answer_type']}|{x['from']}|{difficulty_bin(x['difficulty'])}"
            )
        else:
            sample_df = sample_df.copy()
            sample_df["_s"] = sample_df["extra_info"].apply(
                lambda x: f"{x['primary_answer_type']}|{x['source']}"
            )
        samp = sample_df["_s"].value_counts().sort_index()

    print(f"    {'stratum':<45} {'pop':>6} {'sample':>6} {'rate':>6}")
    for s in pop.index:
        p = pop[s]
        sa = samp.get(s, 0)
        rate = f"{sa/p:.1%}" if p > 0 else "—"
        print(f"    {s:<45} {p:>6} {sa:>6} {rate:>6}")


if __name__ == "__main__":
    main()
