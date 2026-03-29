"""Explore the MiniByte-666/Dr.SCI dataset.

Prints schema, filters physics + rule_verifiable rows, and documents
row counts and answer-type distribution.

Usage
-----
  python scripts/explore_drsci.py
  python scripts/explore_drsci.py --save data/processed/drsci_physics.parquet
"""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cache_dir", default="data/hf_cache",
                        help="HuggingFace cache dir (default: data/hf_cache)")
    parser.add_argument("--save", default=None,
                        help="If set, save filtered physics+match_rule rows to this path")
    args = parser.parse_args()

    from datasets import load_dataset
    import pandas as pd

    print("Loading MiniByte-666/Dr.SCI ...")
    ds = load_dataset("MiniByte-666/Dr.SCI", cache_dir=args.cache_dir)
    print(f"Splits: { {k: len(v) for k, v in ds.items()} }")

    split_name = list(ds.keys())[0]
    data = ds[split_name]

    print(f"\n=== Schema (split='{split_name}', {len(data)} rows) ===")
    for col, feat in data.features.items():
        print(f"  {col}: {feat}")

    df = data.to_pandas()

    # Expand nested dict columns into flat dotted columns
    for nested_col in ("extra_info", "reward_model"):
        if nested_col in df.columns and isinstance(df[nested_col].iloc[0], dict):
            expanded = pd.json_normalize(df[nested_col].tolist())
            expanded.columns = [f"{nested_col}.{c}" for c in expanded.columns]
            df = pd.concat([df.drop(columns=[nested_col]), expanded], axis=1)
            print(f"  Expanded '{nested_col}' → {list(expanded.columns)}")

    print(f"\n=== Flat columns ===")
    for col in df.columns:
        try:
            uniq = df[col].dropna().unique()
            if len(uniq) <= 30:
                print(f"  {col}: {sorted(str(v) for v in uniq)}")
            else:
                print(f"  {col}: {len(uniq)} unique values — sample: {[str(v) for v in uniq[:5]]}")
        except TypeError:
            print(f"  {col}: (unhashable — nested list/dict)")

    # ---- subject distribution ----
    subject_col = next((c for c in df.columns if c == "extra_info.subject" or c == "subject"), None)
    if subject_col:
        print(f"\n=== Subject distribution ['{subject_col}'] ===")
        for val, cnt in df[subject_col].value_counts().items():
            marker = " <-- PHYSICS" if "phys" in str(val).lower() else ""
            print(f"  {val}: {cnt}{marker}")
    else:
        print("\n[WARN] No subject column found")

    # ---- match_rule / rule_verifiable ----
    rv_col = next((c for c in df.columns if c in ("extra_info.match_rule", "match_rule")), None)
    if rv_col:
        print(f"\n=== match_rule ['{rv_col}'] ===")
        print(df[rv_col].value_counts().to_string())
    else:
        print("\n[WARN] No match_rule column found")

    # ---- reward style ----
    style_col = next((c for c in df.columns if c == "reward_model.style"), None)
    if style_col:
        print(f"\n=== reward_model.style ===")
        print(df[style_col].value_counts().to_string())

    # ---- filter physics ----
    if subject_col:
        df_phys = df[df[subject_col].str.lower().str.contains("phys", na=False)].copy()
        print(f"\n=== Physics rows: {len(df_phys)} / {len(df)} ===")
    else:
        df_phys = df.copy()
        print("\n[WARN] No subject column — using all rows")

    # ---- filter match_rule ----
    if rv_col:
        df_rv = df_phys[df_phys[rv_col].astype(bool)].copy()
        print(f"  + match_rule=True: {len(df_rv)} / {len(df_phys)}")
    else:
        df_rv = df_phys.copy()

    # ---- reward style in filtered set ----
    if style_col:
        print(f"\n=== reward_model.style (physics + match_rule) ===")
        print(df_rv[style_col].value_counts().to_string())

    # ---- difficulty distribution ----
    diff_col = "extra_info.difficulty"
    if diff_col in df_rv.columns:
        print(f"\n=== Difficulty distribution (physics + match_rule) ===")
        print(df_rv[diff_col].describe().to_string())

    # ---- sample rows ----
    print(f"\n=== Sample rows (first 3, physics + match_rule=True) ===")
    show_cols = [c for c in (
        "extra_info.question", "extra_info.reference_answer",
        "extra_info.subject", "extra_info.difficulty",
        "extra_info.from", "extra_info.match_rule",
        "reward_model.style", "reward_model.ground_truth",
    ) if c in df_rv.columns]
    for i, row in df_rv.head(3).iterrows():
        print(f"\n--- Row {i} ---")
        for col in show_cols:
            print(f"  {col}: {str(row[col])[:300]}")

    # ---- save ----
    if args.save:
        out = Path(args.save)
        out.parent.mkdir(parents=True, exist_ok=True)
        df_rv.to_parquet(out, index=False)
        print(f"\nSaved {len(df_rv)} rows → {out}")

    print("\n=== Summary ===")
    print(f"  Total rows:                     {len(df)}")
    print(f"  Physics rows:                   {len(df_phys)}")
    print(f"  Physics + match_rule=True rows: {len(df_rv)}")


if __name__ == "__main__":
    main()
