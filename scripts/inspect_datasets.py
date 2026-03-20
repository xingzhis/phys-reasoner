"""Inspect real dataset schemas before writing loaders.

Downloads each HuggingFace dataset and prints:
  - dataset.features (column names + types)
  - first N rows as raw dicts

Usage
-----
  python scripts/inspect_datasets.py --dataset all --n 3 --cache_dir data/hf_cache
  python scripts/inspect_datasets.py --dataset physics
  python scripts/inspect_datasets.py --dataset ugphysics --subject ClassicalMechanics
  python scripts/inspect_datasets.py --dataset olympiad
  python scripts/inspect_datasets.py --dataset critpt
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _print_section(title: str) -> None:
    print(f"\n{'=' * 70}")
    print(f"  {title}")
    print('=' * 70)


def _show(rows: list[dict], n: int, label: str = "") -> None:
    if label:
        print(f"\n--- {label} ---")
    for i, row in enumerate(rows[:n]):
        print(f"\n[row {i}]")
        for k, v in row.items():
            v_repr = repr(v)
            if len(v_repr) > 200:
                v_repr = v_repr[:197] + "..."
            print(f"  {k}: {v_repr}")


def inspect_physics(n: int, cache_dir: str | None) -> None:
    _print_section("PHYSICS  (desimfj/PHYSICS)")
    from datasets import load_dataset
    ds = load_dataset("desimfj/PHYSICS", cache_dir=cache_dir)
    print(f"\nSplits: {list(ds.keys())}")
    for split_name, split_ds in ds.items():
        print(f"\n  [{split_name}]  rows={len(split_ds)}")
        print(f"  features: {split_ds.features}")
        _show(list(split_ds)[:n], n, label=split_name)


def inspect_ugphysics(n: int, cache_dir: str | None, subject: str) -> None:
    _print_section(f"UGPhysics  (UGPhysics/ugphysics)  subject={subject}")
    from datasets import load_dataset
    ds = load_dataset("UGPhysics/ugphysics", subject, cache_dir=cache_dir)
    print(f"\nSplits: {list(ds.keys())}")
    for split_name, split_ds in ds.items():
        print(f"\n  [{split_name}]  rows={len(split_ds)}")
        print(f"  features: {split_ds.features}")
        _show(list(split_ds)[:n], n, label=split_name)


_ALL_UGPHYSICS_SUBJECTS = [
    "AtomicPhysics", "ClassicalElectromagnetism", "ClassicalMechanics",
    "Electrodynamics", "GeometricalOptics", "QuantumMechanics", "Relativity",
    "SemiconductorPhysics", "Solid-StatePhysics", "StatisticalMechanics",
    "TheoreticalMechanics", "Thermodynamics", "WaveOptics",
]


def inspect_ugphysics_all_subjects(n: int, cache_dir: str | None) -> None:
    """Print just the split sizes for all 13 subjects."""
    _print_section("UGPhysics — all subjects (size summary)")
    from datasets import load_dataset
    for subject in _ALL_UGPHYSICS_SUBJECTS:
        try:
            ds = load_dataset("UGPhysics/ugphysics", subject, cache_dir=cache_dir)
            sizes = {k: len(v) for k, v in ds.items()}
            print(f"  {subject:<30} {sizes}")
        except Exception as e:
            print(f"  {subject:<30} ERROR: {e}")


def inspect_olympiadbench(n: int, cache_dir: str | None) -> None:
    _print_section("OlympiadBench  (lscpku/OlympiadBench-official)")
    from datasets import load_dataset, get_dataset_config_names

    try:
        configs = get_dataset_config_names("lscpku/OlympiadBench-official")
        physics_configs = [c for c in configs if "physics" in c.lower()]
        print(f"\nAll configs ({len(configs)} total), physics subset:")
        for c in physics_configs:
            print(f"  {c}")
    except Exception as e:
        print(f"  Could not list configs: {e}")
        physics_configs = ["OE_TO_physics_en_COMP", "OE_TO_physics_zh_CEE"]

    for config in physics_configs:
        try:
            ds = load_dataset("lscpku/OlympiadBench-official", config, cache_dir=cache_dir)
            print(f"\n  [{config}]  splits={list(ds.keys())}")
            for split_name, split_ds in ds.items():
                print(f"    {split_name}: rows={len(split_ds)}")
                print(f"    features: {split_ds.features}")
                _show(list(split_ds)[:n], n, label=f"{config}/{split_name}")
        except Exception as e:
            print(f"  [{config}] ERROR: {e}")


def inspect_critpt(n: int, cache_dir: str | None) -> None:
    _print_section("CritPt  (CritPt-Benchmark/CritPt)")
    from datasets import load_dataset
    ds = load_dataset("CritPt-Benchmark/CritPt", cache_dir=cache_dir)
    print(f"\nSplits: {list(ds.keys())}")
    for split_name, split_ds in ds.items():
        print(f"\n  [{split_name}]  rows={len(split_ds)}")
        print(f"  features: {split_ds.features}")
        _show(list(split_ds)[:n], n, label=split_name)


def inspect_abench(n: int) -> None:
    _print_section("ABench Phy_B  (local CSV — no HF release)")
    csv_path = Path(__file__).parent.parent / "data" / "raw" / "Phy_B_dynamic_100.csv"
    if not csv_path.exists():
        print(f"\n  Not yet downloaded. Instructions:")
        print(f"  1. Go to: https://github.com/inclusionAI/ABench/tree/main/Physics")
        print(f"  2. Download the Phy_B CSV file")
        print(f"  3. Save to: {csv_path}")
        return

    import csv
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    print(f"\n  rows={len(rows)}")
    print(f"  columns: {list(rows[0].keys()) if rows else 'empty'}")
    _show(rows, n)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset", default="all",
        choices=["all", "physics", "ugphysics", "ugphysics_all", "olympiad", "critpt", "abench"],
        help="Which dataset to inspect",
    )
    parser.add_argument("--n", type=int, default=2, help="Rows to print per split")
    parser.add_argument("--cache_dir", default="data/hf_cache", help="HuggingFace cache dir")
    parser.add_argument(
        "--subject", default="ClassicalMechanics",
        help="UGPhysics subject (used when --dataset ugphysics)",
    )
    args = parser.parse_args()

    cache_dir = args.cache_dir or None

    if args.dataset in ("all", "physics"):
        inspect_physics(args.n, cache_dir)

    if args.dataset in ("all", "ugphysics"):
        inspect_ugphysics(args.n, cache_dir, args.subject)

    if args.dataset == "ugphysics_all":
        inspect_ugphysics_all_subjects(args.n, cache_dir)

    if args.dataset in ("all", "olympiad"):
        inspect_olympiadbench(args.n, cache_dir)

    if args.dataset in ("all", "critpt"):
        inspect_critpt(args.n, cache_dir)

    if args.dataset in ("all", "abench"):
        inspect_abench(args.n)


if __name__ == "__main__":
    main()
