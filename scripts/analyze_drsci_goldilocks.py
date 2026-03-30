"""Goldilocks profiling analysis for Dr. SCI.

Phase 3: Reads zero_shot_drsci_sample.parquet (scores) and
drsci_physics_deduped.parquet (full post-dedup set) to:

  1. Compute pass@1 by difficulty bucket and source
  2. Identify the Goldilocks slice: difficulty buckets where pass@1 ∈ [0.15, 0.85]
  3. Estimate effective training set size
  4. Compare difficulty distribution to existing 6.8k corpus
  5. Print corpus-decision recommendation

Also supports --save_sample to produce drsci_goldilocks_sample.parquet
(the 600-row stratified input for run_zero_shot_drsci.py).

Usage
-----
  # Create the 600-row sample for zero-shot profiling:
  python scripts/analyze_drsci_goldilocks.py --save_sample

  # Analyze results after zero-shot inference:
  python scripts/analyze_drsci_goldilocks.py \\
      --scores data/results/zero_shot_drsci_sample.parquet \\
      --deduped data/processed/drsci_physics_deduped.parquet \\
      --corpus  data/processed/candidates_deduped.parquet
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


_DIFF_BINS = [-0.001, 0.1, 0.25, 0.5, 0.75, 1.01]
_DIFF_LABELS = ["0.0", "0.1-0.25", "0.25-0.5", "0.5-0.75", "0.75+"]

_GOLDILOCKS_LOW = 0.15
_GOLDILOCKS_HIGH = 0.85


def _diff_bucket_col(series: pd.Series) -> pd.Categorical:
    return pd.cut(
        pd.to_numeric(series, errors="coerce").fillna(0.0),
        bins=_DIFF_BINS,
        labels=_DIFF_LABELS,
    )


# ---------------------------------------------------------------------------
# Phase 3a: Stratified 600-row sample
# ---------------------------------------------------------------------------

def build_goldilocks_sample(
    deduped_path: str,
    n: int = 600,
    seed: int = 42,
) -> pd.DataFrame:
    """Build a stratified sample for zero-shot Goldilocks profiling."""
    df = pd.read_parquet(deduped_path)
    diff_col = "extra_info.difficulty"
    from_col = "extra_info.from"

    has_diff = diff_col in df.columns
    has_from = from_col in df.columns

    if has_diff:
        df = df.copy()
        df["_diff_bucket"] = _diff_bucket_col(df[diff_col])

    group_cols = []
    if has_diff:
        group_cols.append("_diff_bucket")
    if has_from:
        group_cols.append(from_col)

    if not group_cols:
        return df.sample(min(n, len(df)), random_state=seed).reset_index(drop=True)

    parts = []
    for _, grp in df.groupby(group_cols, observed=True):
        n_take = max(1, round(n * len(grp) / len(df)))
        parts.append(grp.sample(min(len(grp), n_take), random_state=seed))
    sampled = pd.concat(parts).reset_index(drop=True)

    # Ensure at least 50 per difficulty bucket (oversample harder ones)
    if has_diff:
        bucket_counts = sampled["_diff_bucket"].value_counts()
        for bucket in _DIFF_LABELS:
            if bucket_counts.get(bucket, 0) < 50:
                extras = df[df["_diff_bucket"] == bucket].drop(
                    sampled.index, errors="ignore"
                )
                need = 50 - bucket_counts.get(bucket, 0)
                if len(extras):
                    extra_rows = extras.sample(min(need, len(extras)), random_state=seed)
                    sampled = pd.concat([sampled, extra_rows]).reset_index(drop=True)

    # Trim to n
    if len(sampled) > n:
        sampled = sampled.sample(n, random_state=seed).reset_index(drop=True)

    # Drop internal helper column
    if "_diff_bucket" in sampled.columns:
        sampled = sampled.drop(columns=["_diff_bucket"])

    return sampled


# ---------------------------------------------------------------------------
# Phase 3c: Goldilocks analysis
# ---------------------------------------------------------------------------

def analyze_goldilocks(
    scores_df: pd.DataFrame,
    deduped_df: pd.DataFrame,
    corpus_df: pd.DataFrame | None,
) -> None:
    """Print full Goldilocks analysis."""

    verifiable = scores_df[scores_df["score"] != -1.0].copy()
    if len(verifiable) == 0:
        print("ERROR: no verifiable rows in scores — check answer_type distribution.")
        return

    print(f"Verifiable rows: {len(verifiable)} / {len(scores_df)}")

    verifiable = verifiable.copy()
    verifiable["diff_bucket"] = _diff_bucket_col(verifiable["difficulty"])

    # --- 1. Pass@1 by difficulty bucket ---
    print("\n=== Pass@1 by difficulty bucket ===")
    goldilocks_buckets = []
    bucket_pass_rates = {}
    for bucket in _DIFF_LABELS:
        grp = verifiable[verifiable["diff_bucket"] == bucket]
        if len(grp) == 0:
            print(f"  diff={bucket:<10} (no verifiable samples)")
            continue
        acc = grp["score"].mean()
        bucket_pass_rates[bucket] = acc
        in_gz = _GOLDILOCKS_LOW <= acc <= _GOLDILOCKS_HIGH
        flag = " <-- GOLDILOCKS" if in_gz else ""
        print(f"  diff={bucket:<10} pass@1={acc:.1%}  n={len(grp)}{flag}")
        if in_gz:
            goldilocks_buckets.append(bucket)

    # --- 2. Pass@1 by source ---
    from_col = "drsci_from"
    if from_col in verifiable.columns:
        print("\n=== Pass@1 by source ===")
        for src, grp in verifiable.groupby(from_col, observed=True):
            acc = grp["score"].mean()
            print(f"  {str(src):<35} pass@1={acc:.1%}  n={len(grp)}")

    # --- 3. Goldilocks slice + effective training set size ---
    print(f"\n=== Goldilocks slice (pass@1 ∈ [{_GOLDILOCKS_LOW:.0%}, {_GOLDILOCKS_HIGH:.0%}]) ===")
    if not goldilocks_buckets:
        print("  No buckets fall in the Goldilocks zone!")
        print("  Recommendation: widen zone or use full Dr. SCI (accept harder/easier problems)")
        effective_fraction = 1.0
    else:
        print(f"  Goldilocks buckets: {goldilocks_buckets}")

        # Estimate fraction of full post-dedup set in Goldilocks buckets
        if "extra_info.difficulty" in deduped_df.columns:
            deduped_df = deduped_df.copy()
            deduped_df["diff_bucket"] = _diff_bucket_col(deduped_df["extra_info.difficulty"])
            in_gz = deduped_df["diff_bucket"].isin(goldilocks_buckets)
            n_in_gz = in_gz.sum()
            effective_fraction = n_in_gz / max(len(deduped_df), 1)
            n_effective = n_in_gz
        else:
            effective_fraction = 1.0
            n_effective = len(deduped_df)

        print(f"  Fraction of post-dedup set in Goldilocks zone: {effective_fraction:.1%}")
        print(f"  Effective training set size estimate: ~{n_effective:,} rows")

        if n_effective < 10_000:
            print(f"  WARNING: effective set < 10k — mix with existing 6.8k corpus")
        elif n_effective < 20_000:
            print(f"  NOTE: {n_effective:,} usable rows — Goldilocks slice + 6.8k supplement")
        else:
            print(f"  OK: {n_effective:,} usable rows — Goldilocks slice alone is sufficient")

    # --- 4. Difficulty distribution comparison ---
    if corpus_df is not None and "difficulty" in corpus_df.columns:
        print("\n=== Difficulty comparison: Dr. SCI vs existing 6.8k ===")
        deduped_df_cp = deduped_df.copy() if "extra_info.difficulty" in deduped_df.columns else None
        if deduped_df_cp is not None:
            drsci_diffs = pd.to_numeric(
                deduped_df_cp.get("extra_info.difficulty", pd.Series(dtype=float)),
                errors="coerce",
            ).dropna()
            corpus_diffs = pd.to_numeric(corpus_df["difficulty"], errors="coerce").dropna()
            print(f"  Dr. SCI   — mean={drsci_diffs.mean():.3f}  median={drsci_diffs.median():.3f}  "
                  f"n={len(drsci_diffs):,}")
            print(f"  Corpus 6.8k — mean={corpus_diffs.mean():.3f}  median={corpus_diffs.median():.3f}  "
                  f"n={len(corpus_diffs):,}")

    # --- 5. Corpus-decision recommendation ---
    n_deduped = len(deduped_df)
    goldilocks_eff_pct = effective_fraction

    print(f"\n=== Corpus decision recommendation ===")
    print(f"  Post-dedup Dr. SCI: {n_deduped:,} rows")
    if n_deduped < 30_000:
        print(f"  SIZE: < 30k → Use Dr. SCI as SUPPLEMENT to existing 6.8k")
    elif n_deduped < 80_000:
        print(f"  SIZE: 30k–80k → Dr. SCI becomes PRIMARY; 6.8k as hard-problem supplement")
    else:
        print(f"  SIZE: > 80k → Dr. SCI is PRIMARY; 6.8k moved to additional eval tier")

    if goldilocks_eff_pct < 0.30:
        print(f"  GOLDILOCKS: < 30% in zone → filter to Goldilocks slice only; mix with 6.8k")
    else:
        print(f"  GOLDILOCKS: ≥ 30% in zone ({goldilocks_eff_pct:.1%}) → "
              "Goldilocks slice alone is sufficient (≥20k usable rows)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--scores",  default="data/results/zero_shot_drsci_sample.parquet",
        help="Output of run_zero_shot_drsci.py",
    )
    parser.add_argument(
        "--deduped", default="data/processed/drsci_physics_clean.parquet",
        help="Cleaned Dr. SCI parquet (has inferred_answer_type column)",
    )
    parser.add_argument(
        "--corpus",  default="data/processed/candidates_deduped.parquet",
        help="Existing 6.8k training corpus (for difficulty comparison)",
    )
    parser.add_argument(
        "--save_sample", default=None, nargs="?", const="data/processed/drsci_goldilocks_sample.parquet",
        metavar="PATH",
        help="Build and save 600-row Goldilocks sample (default path if no PATH given: "
             "data/processed/drsci_goldilocks_sample.parquet)",
    )
    parser.add_argument("--n_sample", type=int, default=600)
    parser.add_argument("--seed",     type=int, default=42)
    args = parser.parse_args()

    # --- Save sample mode ---
    if args.save_sample is not None:
        if not Path(args.deduped).exists():
            print(f"ERROR: {args.deduped} not found. Run drsci_dedup.py first.")
            sys.exit(1)
        print(f"Building {args.n_sample}-row Goldilocks sample from {args.deduped}...", flush=True)
        deduped_df = pd.read_parquet(args.deduped)
        sample = build_goldilocks_sample(args.deduped, n=args.n_sample, seed=args.seed)
        Path(args.save_sample).parent.mkdir(parents=True, exist_ok=True)
        sample.to_parquet(args.save_sample, index=False)
        print(f"Saved {len(sample)}-row sample → {args.save_sample}")
        print("\nDifficulty bucket distribution in sample:")
        if "extra_info.difficulty" in sample.columns:
            sample["diff_bucket"] = _diff_bucket_col(sample["extra_info.difficulty"])
            print(sample["diff_bucket"].value_counts().sort_index().to_string())
        if "extra_info.from" in sample.columns:
            print("\nSource distribution in sample:")
            print(sample["extra_info.from"].value_counts().to_string())
        return

    # --- Analysis mode ---
    if not Path(args.scores).exists():
        print(f"ERROR: {args.scores} not found. Run run_zero_shot_drsci.py first.")
        print("  Or run:  python scripts/analyze_drsci_goldilocks.py --save_sample")
        print("  Then:    sbatch scripts/zero_shot_drsci_sample.sbatch")
        sys.exit(1)

    print(f"Loading scores from {args.scores}...", flush=True)
    scores_df = pd.read_parquet(args.scores)
    print(f"  {len(scores_df)} rows")

    deduped_df = pd.read_parquet(args.deduped) if Path(args.deduped).exists() else pd.DataFrame()
    corpus_df  = pd.read_parquet(args.corpus)  if Path(args.corpus).exists() else None

    analyze_goldilocks(scores_df, deduped_df, corpus_df)


if __name__ == "__main__":
    main()
