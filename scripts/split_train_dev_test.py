"""Stratified train/dev/test split for Dr. SCI and Corpus training parquets.

Reads the enriched train parquets (output of build_training_parquets.py) and
produces stratified splits:

  Dr. SCI:  2,000 dev + 2,000 test + remainder train
    Stratification key: (answer_type × from × difficulty_bin)
    difficulty_bin: low (0.0) / medium (0.125–0.375) / high (0.5–0.75)

  Corpus:   200 dev + 200 test + remainder train
    Stratification key: (primary_answer_type × source)

Within each stratum, rows are randomly shuffled and then allocated proportionally
to dev/test/train.  If a stratum is too small for proportional allocation, the
minimum floor ensures at least 1 row goes to dev and 1 to test (if the stratum
has >= 3 rows; strata with < 3 rows go entirely to train).

Outputs:
    data/processed/drsci_train_split.parquet
    data/processed/drsci_dev.parquet
    data/processed/drsci_test.parquet
    data/processed/corpus_train_split.parquet
    data/processed/corpus_dev.parquet
    data/processed/corpus_test.parquet

Usage:
    python scripts/split_train_dev_test.py
    python scripts/split_train_dev_test.py --seed 42
    python scripts/split_train_dev_test.py --report   # stats only, no save
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Difficulty binning (Dr. SCI)
# ---------------------------------------------------------------------------

def difficulty_bin(d: float) -> str:
    """Map float difficulty to low / medium / high."""
    if d <= 0.0:
        return "low"
    elif d <= 0.375:
        return "medium"
    else:
        return "high"


# ---------------------------------------------------------------------------
# Stratified split logic
# ---------------------------------------------------------------------------

def stratified_split(
    df: pd.DataFrame,
    stratum_col: str,
    n_dev: int,
    n_test: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split df into (train, dev, test) stratified by stratum_col.

    Allocation is proportional to stratum size.  Minimum: 1 dev + 1 test per
    stratum if stratum has >= 3 rows.  Strata with < 3 rows go to train only.

    Returns (train_df, dev_df, test_df) with original index reset.
    """
    rng = np.random.RandomState(seed)

    strata = df[stratum_col].unique()
    stratum_sizes = df[stratum_col].value_counts()
    total = len(df)

    # Compute per-stratum allocation
    dev_alloc: dict[str, int] = {}
    test_alloc: dict[str, int] = {}

    for s in strata:
        sz = stratum_sizes[s]
        if sz < 3:
            dev_alloc[s] = 0
            test_alloc[s] = 0
            continue
        # Proportional allocation
        frac = sz / total
        d = max(1, round(n_dev * frac))
        t = max(1, round(n_test * frac))
        # Don't take more than sz - 1 (leave at least 1 for train)
        if d + t >= sz:
            d = max(1, sz // 3)
            t = max(1, sz // 3)
        dev_alloc[s] = d
        test_alloc[s] = t

    # Adjust totals to hit target exactly
    dev_alloc = _adjust_totals(dev_alloc, n_dev, stratum_sizes)
    test_alloc = _adjust_totals(test_alloc, n_test, stratum_sizes)

    train_parts, dev_parts, test_parts = [], [], []

    for s in strata:
        mask = df[stratum_col] == s
        group = df[mask].sample(frac=1, random_state=rng).reset_index(drop=True)
        nd = dev_alloc.get(s, 0)
        nt = test_alloc.get(s, 0)
        dev_parts.append(group.iloc[:nd])
        test_parts.append(group.iloc[nd:nd + nt])
        train_parts.append(group.iloc[nd + nt:])

    train_df = pd.concat(train_parts, ignore_index=True)
    dev_df = pd.concat(dev_parts, ignore_index=True)
    test_df = pd.concat(test_parts, ignore_index=True)

    return train_df, dev_df, test_df


def _adjust_totals(
    alloc: dict[str, int],
    target: int,
    stratum_sizes: pd.Series,
) -> dict[str, int]:
    """Nudge allocations so they sum to exactly target."""
    current = sum(alloc.values())
    diff = target - current

    if diff == 0:
        return alloc

    # Sort strata by size descending — adjust the largest strata first
    sorted_strata = sorted(
        [s for s, n in alloc.items() if n > 0],
        key=lambda s: stratum_sizes[s],
        reverse=True,
    )

    if diff > 0:
        # Need more: add 1 to largest strata until we hit target
        for i in range(abs(diff)):
            s = sorted_strata[i % len(sorted_strata)]
            # Don't exceed stratum capacity (leave room for train)
            if alloc[s] + 1 < stratum_sizes[s]:
                alloc[s] += 1
    else:
        # Need fewer: remove 1 from largest strata
        for i in range(abs(diff)):
            s = sorted_strata[i % len(sorted_strata)]
            if alloc[s] > 1:
                alloc[s] -= 1

    return alloc


# ---------------------------------------------------------------------------
# Dr. SCI split
# ---------------------------------------------------------------------------

def split_drsci(input_path: str, output_dir: str, n_dev: int, n_test: int,
                seed: int, report: bool) -> pd.DataFrame | None:
    print(f"\n=== Dr. SCI Split ===")
    df = pd.read_parquet(input_path)
    print(f"  Loaded {len(df):,} rows from {input_path}")

    # Build stratum key
    df["_answer_type"] = df["extra_info"].apply(lambda x: x["answer_type"])
    df["_from"] = df["extra_info"].apply(lambda x: x["from"])
    df["_difficulty"] = df["extra_info"].apply(lambda x: x["difficulty"])
    df["_difficulty_bin"] = df["_difficulty"].apply(difficulty_bin)
    df["_stratum"] = df["_answer_type"] + "|" + df["_from"] + "|" + df["_difficulty_bin"]

    n_strata = df["_stratum"].nunique()
    print(f"  {n_strata} strata (answer_type × from × difficulty_bin)")

    train, dev, test = stratified_split(df, "_stratum", n_dev, n_test, seed)

    print(f"  Split sizes: train={len(train):,}  dev={len(dev):,}  test={len(test):,}")
    _print_stratum_report_direct(df, train, dev, test,
                                 ["_answer_type", "_from", "_difficulty_bin"])

    # Drop temporary columns
    tmp_cols = ["_answer_type", "_from", "_difficulty", "_difficulty_bin", "_stratum"]
    for split_df in (train, dev, test):
        split_df.drop(columns=tmp_cols, inplace=True)

    if not report:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        train.to_parquet(out / "drsci_train_split.parquet", index=False)
        dev.to_parquet(out / "drsci_dev.parquet", index=False)
        test.to_parquet(out / "drsci_test.parquet", index=False)
        print(f"  Saved → {out}/drsci_{{train_split,dev,test}}.parquet")

    return df


def split_corpus(input_path: str, output_dir: str, n_dev: int, n_test: int,
                 seed: int, report: bool, stratum_key: str = "source") -> pd.DataFrame | None:
    print(f"\n=== Corpus Split ===")
    df = pd.read_parquet(input_path)
    print(f"  Loaded {len(df):,} rows from {input_path}")

    # Build stratum key
    df["_primary_answer_type"] = df["extra_info"].apply(lambda x: x["primary_answer_type"])
    if stratum_key == "source":
        df["_secondary"] = df["extra_info"].apply(lambda x: x["source"])
        stratum_label = "primary_answer_type × source"
        report_cols = ["_primary_answer_type", "_secondary"]
    elif stratum_key == "domain_coarse":
        df["_secondary"] = df["extra_info"].apply(lambda x: x.get("domain_coarse", "other"))
        stratum_label = "primary_answer_type × domain_coarse"
        report_cols = ["_primary_answer_type", "_secondary"]
    else:
        raise ValueError(f"Unknown stratum_key: {stratum_key!r} (use 'source' or 'domain_coarse')")
    df["_stratum"] = df["_primary_answer_type"] + "|" + df["_secondary"]

    n_strata = df["_stratum"].nunique()
    print(f"  {n_strata} strata ({stratum_label})")

    train, dev, test = stratified_split(df, "_stratum", n_dev, n_test, seed)

    print(f"  Split sizes: train={len(train):,}  dev={len(dev):,}  test={len(test):,}")
    _print_stratum_report_direct(df, train, dev, test, report_cols)

    # Drop temporary columns
    tmp_cols = ["_primary_answer_type", "_secondary", "_stratum"]
    for split_df in (train, dev, test):
        split_df.drop(columns=tmp_cols, inplace=True)

    if not report:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        train.to_parquet(out / "corpus_train_split.parquet", index=False)
        dev.to_parquet(out / "corpus_dev.parquet", index=False)
        test.to_parquet(out / "corpus_test.parquet", index=False)
        print(f"  Saved → {out}/corpus_{{train_split,dev,test}}.parquet")

    return df


# ---------------------------------------------------------------------------
# Report helpers
# ---------------------------------------------------------------------------

def _print_stratum_report_direct(df_orig, train, dev, test, key_cols):
    """Print per-dimension distribution across splits.

    Assumes the temporary _-prefixed columns still exist on all DataFrames.
    """
    for col in key_cols:
        label = col.lstrip("_")
        orig_counts = df_orig[col].value_counts()
        train_counts = train[col].value_counts()
        dev_counts = dev[col].value_counts()
        test_counts = test[col].value_counts()

        print(f"\n  {label} distribution:")
        print(f"    {'value':<25} {'total':>7} {'train':>7} {'dev':>5} {'test':>5}  {'dev%':>5} {'test%':>5}")
        for val in orig_counts.index:
            t = orig_counts.get(val, 0)
            tr = train_counts.get(val, 0)
            dv = dev_counts.get(val, 0)
            ts = test_counts.get(val, 0)
            dp = f"{dv/t:.1%}" if t > 0 else "—"
            tp = f"{ts/t:.1%}" if t > 0 else "—"
            print(f"    {str(val):<25} {t:>7} {tr:>7} {dv:>5} {ts:>5}  {dp:>5} {tp:>5}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--drsci-input", default="data/processed/drsci_train.parquet")
    parser.add_argument("--corpus-input", default="data/processed/corpus_train.parquet")
    parser.add_argument("--output-dir", default="data/processed")
    parser.add_argument("--drsci-dev", type=int, default=2000)
    parser.add_argument("--drsci-test", type=int, default=2000)
    parser.add_argument("--corpus-dev", type=int, default=200)
    parser.add_argument("--corpus-test", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--report", action="store_true", help="Print stats only, no save")
    parser.add_argument("--drsci-only", action="store_true")
    parser.add_argument("--corpus-only", action="store_true")
    parser.add_argument("--corpus-stratum", choices=["source", "domain_coarse"],
                        default="source",
                        help="Corpus stratification key. 'source' (default, legacy) stratifies by "
                             "(primary_answer_type × source). 'domain_coarse' stratifies by "
                             "(primary_answer_type × domain_coarse) — use when corpus is single-source.")
    args = parser.parse_args()

    if not args.corpus_only:
        split_drsci(args.drsci_input, args.output_dir,
                    args.drsci_dev, args.drsci_test, args.seed, args.report)
    if not args.drsci_only:
        split_corpus(args.corpus_input, args.output_dir,
                     args.corpus_dev, args.corpus_test, args.seed, args.report,
                     stratum_key=args.corpus_stratum)

    print("\nDone.")
    if args.report:
        print("(--report mode: no files written)")


if __name__ == "__main__":
    main()
