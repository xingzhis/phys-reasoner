"""Sanity check: metadata statistics, type verification, and cross-tabulations.

Checks the full pipeline from source parquets → enriched train → splits → probe.
Reports:
  1. Per-field distributions with types
  2. Cross-tabulations (answer_type × from, answer_type × difficulty_bin, etc.)
  3. Difficulty conversion audit (source float vs output float, fallback count)
  4. Split consistency (train + dev + test = total, no row loss)
  5. Sample rows for manual inspection

No files modified.

Usage:
    python scripts/sanity_check_pipeline.py
    python scripts/sanity_check_pipeline.py --show-samples 5
    python scripts/sanity_check_pipeline.py --no-source    # skip source comparison
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _section(title: str):
    print(f"\n{'=' * 70}")
    print(f"  {title}")
    print(f"{'=' * 70}")


def _subsection(title: str):
    print(f"\n  --- {title} ---")


def _ei(df: pd.DataFrame, key: str) -> pd.Series:
    """Extract a key from the extra_info dict column."""
    return df["extra_info"].apply(lambda x: x.get(key))


def _dist_table(series: pd.Series, label: str, max_rows: int = 30):
    """Print a value distribution table with type info."""
    vc = series.value_counts(dropna=False)
    n_total = len(series)
    n_null = series.isna().sum()
    types = series.dropna().apply(type).apply(lambda t: t.__name__).value_counts()
    type_str = ", ".join(f"{t}({n})" for t, n in types.items())

    print(f"\n  {label}  [n={n_total:,}, null={n_null}, types: {type_str}]")
    print(f"    {'value':<40} {'count':>7} {'pct':>6}")
    print(f"    {'-'*40} {'-'*7} {'-'*6}")
    for i, (v, n) in enumerate(vc.items()):
        if i >= max_rows:
            remaining = len(vc) - max_rows
            print(f"    ... and {remaining} more values")
            break
        v_str = repr(v) if not isinstance(v, (int, float)) else str(v)
        if len(v_str) > 40:
            v_str = v_str[:37] + "..."
        print(f"    {v_str:<40} {n:>7} {n/n_total:>5.1%}")


def _crosstab(df: pd.DataFrame, key_a: str, key_b: str, label_a: str, label_b: str):
    """Print a cross-tabulation of two extra_info keys."""
    a = _ei(df, key_a)
    b = _ei(df, key_b)
    ct = pd.crosstab(a, b, margins=True)
    _subsection(f"Cross-tab: {label_a} × {label_b}")
    print(ct.to_string(col_space=8))


def _show_samples(df: pd.DataFrame, n: int, label: str):
    if n <= 0:
        return
    _subsection(f"Sample rows ({label}, {n} rows)")
    for i, row in df.head(n).iterrows():
        ei = row["extra_info"]
        print(f"\n  [{i}] data_source={row['data_source']!r}")
        for k, v in sorted(ei.items()):
            if k == "problem":
                print(f"       {k}: {str(v)[:100]}...")
            else:
                print(f"       {k}: {v!r}  <{type(v).__name__}>")
        rm = row["reward_model"]
        if isinstance(rm, dict):
            print(f"       ground_truth: {str(rm.get('ground_truth', ''))[:80]}")


# ---------------------------------------------------------------------------
# Source vs output comparison (difficulty audit)
# ---------------------------------------------------------------------------

def check_drsci_difficulty_audit(source_path: str, output_path: str):
    """Compare difficulty values between source and output to detect fallbacks."""
    _subsection("Difficulty conversion audit (source → output)")

    if not Path(source_path).exists():
        print(f"  SKIP: source {source_path} not found")
        return
    if not Path(output_path).exists():
        print(f"  SKIP: output {output_path} not found")
        return

    src = pd.read_parquet(source_path)
    out = pd.read_parquet(output_path)

    # Source difficulty
    src_diff = src["extra_info.difficulty"]
    print(f"  Source ({source_path}):")
    print(f"    dtype: {src_diff.dtype}")
    print(f"    nulls: {src_diff.isna().sum()}")
    print(f"    unique: {sorted(src_diff.dropna().unique())}")
    non_float = 0
    for v in src_diff:
        try:
            float(v)
        except (ValueError, TypeError):
            non_float += 1
    print(f"    non-float-convertible: {non_float}")

    # Output difficulty
    out_diff = _ei(out, "difficulty")
    print(f"\n  Output ({output_path}):")
    out_types = out_diff.apply(type).apply(lambda t: t.__name__).value_counts()
    print(f"    types: {out_types.to_dict()}")
    print(f"    nulls: {out_diff.isna().sum()}")
    print(f"    unique: {sorted(out_diff.dropna().unique())}")

    # Check for potential fallback contamination
    genuine_zero = (src_diff == 0.0).sum()
    output_zero = (out_diff == 0.0).sum()
    # After filtering (NaN from + unknown answer_type + figure filter),
    # the number of 0.0s should only decrease or stay same
    print(f"\n  Fallback contamination check:")
    print(f"    Source rows with difficulty=0.0: {genuine_zero:,} "
          f"({genuine_zero/len(src):.1%} of source)")
    print(f"    Output rows with difficulty=0.0: {output_zero:,} "
          f"({output_zero/len(out):.1%} of output)")

    # If output has MORE zeros than expected after filtering, that means fallbacks fired
    # Estimate: source had some rows dropped (NaN from, unknown type, fig filter)
    # so output zeros should be <= source zeros
    if output_zero > genuine_zero:
        extra = output_zero - genuine_zero
        print(f"    ⚠ WARNING: {extra} extra zero-difficulty rows in output!")
        print(f"      These are likely fallback conversions from non-numeric values.")
    else:
        dropped = genuine_zero - output_zero
        print(f"    ✓ OK: {dropped} zero-difficulty rows removed by filtering, "
              f"no fallback contamination")


# ---------------------------------------------------------------------------
# Per-dataset checks
# ---------------------------------------------------------------------------

def check_drsci(data_dir: str, n_samples: int, check_source: bool):
    path = f"{data_dir}/drsci_train.parquet"
    if not Path(path).exists():
        print(f"  SKIP: {path} not found")
        return None

    _section(f"Dr. SCI Train: {path}")
    df = pd.read_parquet(path)
    print(f"  Rows: {len(df):,}")
    print(f"  Columns: {list(df.columns)}")
    ei0 = df["extra_info"].iloc[0]
    print(f"  extra_info keys: {sorted(ei0.keys())}")

    # Per-field distributions with types
    _dist_table(_ei(df, "answer_type"), "answer_type")
    _dist_table(_ei(df, "from"), "from")
    _dist_table(_ei(df, "difficulty"), "difficulty")
    _dist_table(_ei(df, "subject"), "subject")

    # Cross-tabs
    _crosstab(df, "answer_type", "from", "answer_type", "from")

    # difficulty_bin cross-tab
    diff = _ei(df, "difficulty")
    df["_difficulty_bin"] = diff.apply(_difficulty_bin)
    _dist_table(df["_difficulty_bin"], "difficulty_bin (derived)")

    ct = pd.crosstab(
        _ei(df, "answer_type"),
        df["_difficulty_bin"],
        margins=True,
    )
    _subsection("Cross-tab: answer_type × difficulty_bin")
    print(ct.to_string(col_space=8))

    ct2 = pd.crosstab(
        _ei(df, "from"),
        df["_difficulty_bin"],
        margins=True,
    )
    _subsection("Cross-tab: from × difficulty_bin")
    print(ct2.to_string(col_space=8))
    df.drop(columns=["_difficulty_bin"], inplace=True)

    # Difficulty audit
    if check_source:
        check_drsci_difficulty_audit(
            f"{data_dir}/drsci_physics_clean.parquet", path)

    _show_samples(df, n_samples, "Dr. SCI")
    return df


def check_corpus(data_dir: str, n_samples: int):
    path = f"{data_dir}/corpus_train.parquet"
    if not Path(path).exists():
        print(f"  SKIP: {path} not found")
        return None

    _section(f"Corpus Train: {path}")
    df = pd.read_parquet(path)
    print(f"  Rows: {len(df):,}")
    print(f"  Columns: {list(df.columns)}")
    ei0 = df["extra_info"].iloc[0]
    print(f"  extra_info keys: {sorted(ei0.keys())}")

    # Per-field distributions with types
    _dist_table(_ei(df, "primary_answer_type"), "primary_answer_type")
    _dist_table(_ei(df, "answer_type"), "answer_type (raw)")
    _dist_table(_ei(df, "domain_coarse"), "domain_coarse")
    _dist_table(_ei(df, "domain"), "domain (raw)")
    _dist_table(_ei(df, "source"), "source")
    _dist_table(_ei(df, "difficulty"), "difficulty")
    _dist_table(df["data_source"], "data_source (column)")

    # Cross-tabs
    _crosstab(df, "primary_answer_type", "source", "primary_answer_type", "source")
    _crosstab(df, "domain_coarse", "source", "domain_coarse", "source")
    _crosstab(df, "primary_answer_type", "domain_coarse",
              "primary_answer_type", "domain_coarse")

    _show_samples(df, n_samples, "Corpus")
    return df


def _difficulty_bin(d) -> str:
    try:
        d = float(d)
    except (ValueError, TypeError):
        return "non-numeric"
    if d <= 0.0:
        return "low"
    elif d <= 0.375:
        return "medium"
    else:
        return "high"


# ---------------------------------------------------------------------------
# Split consistency
# ---------------------------------------------------------------------------

def check_splits(data_dir: str, prefix: str, name: str, full_df: pd.DataFrame | None):
    _section(f"{name} Split Consistency")

    paths = {
        "train_split": f"{data_dir}/{prefix}_train_split.parquet",
        "dev": f"{data_dir}/{prefix}_dev.parquet",
        "test": f"{data_dir}/{prefix}_test.parquet",
    }
    dfs = {}
    for split_name, path in paths.items():
        if not Path(path).exists():
            print(f"  SKIP: {path} not found")
            return
        dfs[split_name] = pd.read_parquet(path)

    total_split = sum(len(d) for d in dfs.values())
    print(f"  {'split':<12} {'rows':>7}")
    print(f"  {'-'*12} {'-'*7}")
    for split_name, split_df in dfs.items():
        print(f"  {split_name:<12} {len(split_df):>7,}")
    print(f"  {'TOTAL':<12} {total_split:>7,}")

    if full_df is not None:
        expected = len(full_df)
        if total_split == expected:
            print(f"  ✓ Matches full train parquet ({expected:,} rows)")
        else:
            print(f"  ⚠ MISMATCH: splits sum to {total_split:,} "
                  f"but full train has {expected:,} rows")

    # Check stratification proportions
    _subsection("Dev/test proportions by answer_type")
    for split_name in ["dev", "test"]:
        split_df = dfs[split_name]
        at = _ei(split_df, "answer_type")
        # For corpus, use primary_answer_type if available
        ei0 = split_df["extra_info"].iloc[0]
        if "primary_answer_type" in ei0:
            at = _ei(split_df, "primary_answer_type")
        vc = at.value_counts()
        total_split_n = len(split_df)
        print(f"\n  {split_name}:")
        for v, n in vc.items():
            print(f"    {str(v):<25} {n:>5} ({n/total_split_n:.1%})")


# ---------------------------------------------------------------------------
# Probe check
# ---------------------------------------------------------------------------

def check_probe(data_dir: str, n_samples: int):
    path = f"{data_dir}/probe_subset.parquet"
    if not Path(path).exists():
        print(f"  SKIP: {path} not found")
        return

    _section(f"Probe Subset: {path}")
    df = pd.read_parquet(path)
    print(f"  Rows: {len(df):,}")
    print(f"  Columns: {list(df.columns)}")

    if "_dataset" in df.columns:
        _dist_table(df["_dataset"], "_dataset")

    _dist_table(df["data_source"], "data_source")

    # Per-dataset breakdowns
    for ds in (df["_dataset"].unique() if "_dataset" in df.columns else ["all"]):
        sub = df[df["_dataset"] == ds] if ds != "all" else df
        _subsection(f"Probe — {ds} subset ({len(sub):,} rows)")
        _dist_table(_ei(sub, "answer_type"), f"answer_type ({ds})", max_rows=10)

        # Check types of difficulty in probe
        diff = _ei(sub, "difficulty")
        diff_types = diff.dropna().apply(type).apply(lambda t: t.__name__).value_counts()
        print(f"\n  difficulty types ({ds}): {diff_types.to_dict()}")

    # Verify probe rows come from train split only (not dev/test)
    _subsection("Probe vs dev/test overlap check")
    for prefix, name in [("drsci", "Dr. SCI"), ("corpus", "Corpus")]:
        for split in ["dev", "test"]:
            split_path = f"{data_dir}/{prefix}_{split}.parquet"
            if not Path(split_path).exists():
                continue
            split_df = pd.read_parquet(split_path)
            # Compare by problem text (most reliable dedup key)
            probe_problems = set(
                _ei(df[df.get("_dataset", pd.Series("all")) != "IMPOSSIBLE"], "problem")
            )
            split_problems = set(_ei(split_df, "problem"))
            overlap = probe_problems & split_problems
            if overlap:
                print(f"  ⚠ {len(overlap)} probe problems found in {name} {split}!")
            else:
                print(f"  ✓ No overlap between probe and {name} {split}")

    _show_samples(df, n_samples, "Probe")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--show-samples", type=int, default=2)
    parser.add_argument("--data-dir", default="data/processed")
    parser.add_argument("--no-source", action="store_true",
                        help="Skip source parquet comparison")
    args = parser.parse_args()

    d = args.data_dir
    n = args.show_samples

    drsci_df = check_drsci(d, n, check_source=not args.no_source)
    corpus_df = check_corpus(d, n)
    check_splits(d, "drsci", "Dr. SCI", drsci_df)
    check_splits(d, "corpus", "Corpus", corpus_df)
    check_probe(d, n)

    _section("Summary")
    print("  Review the tables above. Key things to check:")
    print("  1. difficulty types are float (drsci) — no string '0.0' leaks")
    print("  2. No fallback contamination in difficulty=0.0 rows")
    print("  3. primary_answer_type has exactly 7 values (corpus)")
    print("  4. domain_coarse has exactly 6 values (corpus)")
    print("  5. Splits sum to full train parquet")
    print("  6. No probe/dev/test overlap")


if __name__ == "__main__":
    main()
