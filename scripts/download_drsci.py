"""Download and filter Dr. SCI physics subset from HuggingFace.

Loads MiniByte-666/Dr.SCI, filters for physics subject rows where
match_rule is True, flattens nested columns to dot-notation, and saves to
data/processed/drsci_physics.parquet (~115k rows).

Usage
-----
  # From project root:
  python scripts/download_drsci.py

  # With explicit cache / output:
  python scripts/download_drsci.py --cache_dir data/hf_cache --output data/processed/drsci_physics.parquet

  # Dry-run: print stats only, no save:
  python scripts/download_drsci.py --report
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cache_dir", default="data/hf_cache")
    parser.add_argument("--output",    default="data/processed/drsci_physics.parquet")
    parser.add_argument("--report",    action="store_true", help="Print stats only — no save")
    args = parser.parse_args()

    os.environ.setdefault("HF_HOME", str(Path(args.cache_dir).resolve()))

    import datasets
    import pandas as pd

    print("Loading MiniByte-666/Dr.SCI from HuggingFace cache...", flush=True)
    ds = datasets.load_dataset(
        "MiniByte-666/Dr.SCI",
        cache_dir=args.cache_dir,
        trust_remote_code=True,
    )
    # Dataset has a single "train" split
    split = ds["train"] if "train" in ds else ds[list(ds.keys())[0]]
    print(f"  Raw rows: {len(split):,}")

    df = split.to_pandas()

    # --- Flatten nested dict columns to dot-notation ---
    # The HF dataset may deliver extra_info and reward_model as dicts; flatten them.
    nested_cols = [c for c in df.columns if df[c].dtype == object
                   and isinstance(df[c].iloc[0], dict)]
    for col in nested_cols:
        expanded = pd.json_normalize(df[col].tolist())
        expanded.columns = [f"{col}.{sub}" for sub in expanded.columns]
        df = pd.concat([df.drop(columns=[col]), expanded], axis=1)

    print(f"  Columns after flatten: {df.columns.tolist()}")

    # --- Filter: physics subject + match_rule == True ---
    subject_col = next((c for c in df.columns if c.endswith(".subject") or c == "subject"), None)
    rule_col    = next((c for c in df.columns if c.endswith(".match_rule") or c == "match_rule"), None)

    if subject_col:
        before = len(df)
        df = df[df[subject_col].astype(str).str.lower() == "physics"]
        print(f"  After subject=physics filter: {len(df):,}  (dropped {before - len(df):,})")
    else:
        print("  WARNING: no subject column found — keeping all rows")

    if rule_col:
        before = len(df)
        df = df[df[rule_col].astype(str).str.lower().isin({"true", "1"})]
        print(f"  After match_rule=True filter: {len(df):,}  (dropped {before - len(df):,})")
    else:
        print("  WARNING: no match_rule column found — keeping all rows")

    df = df.reset_index(drop=True)

    # --- Summary ---
    from_col = next((c for c in df.columns if c.endswith(".from")), None)
    if from_col:
        print("\n  Source distribution (extra_info.from):")
        for src, n in df[from_col].value_counts().items():
            print(f"    {str(src):<35} {n:>7}  ({n/len(df):.1%})")

    print(f"\n  Final rows: {len(df):,}")

    if not args.report:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(args.output, index=False)
        print(f"  Saved → {args.output}")


if __name__ == "__main__":
    main()
