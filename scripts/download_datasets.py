"""Download all HuggingFace datasets to local cache.

Run this once before any loader/training work. ABench requires a
separate manual download (see instructions printed at the end).

Usage
-----
  python scripts/download_datasets.py --cache_dir data/hf_cache
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


UGPHYSICS_SUBJECTS = [
    "AtomicPhysics",
    "ClassicalElectromagnetism",
    "ClassicalMechanics",
    "Electrodynamics",
    "GeometricalOptics",
    "QuantumMechanics",
    "Relativity",
    "SemiconductorPhysics",
    "Solid-StatePhysics",
    "StatisticalMechanics",
    "TheoreticalMechanics",
    "Thermodynamics",
    "WaveOptics",
]

# Text-only, open-ended physics configs (no images, no proof-only problems)
# Using aggregate configs that cover both competition + CEE problems
OLYMPIAD_CONFIGS = [
    "physics_en_no_proof",   # 692 rows — all English OE non-proof
    "physics_zh_no_proof",   # 1598 rows — all Chinese OE non-proof
    # Keep individual configs for reference/filtering
    "OE_TO_physics_en_COMP", # 236 rows — English, text-only, competition
    "OE_TO_physics_zh_CEE",  # 115 rows — Chinese, text-only, CEE
]


def download_physics(cache_dir: str) -> None:
    from datasets import load_dataset
    print("\n[1/4] Downloading PHYSICS (desimfj/PHYSICS)...")
    ds = load_dataset("desimfj/PHYSICS", cache_dir=cache_dir)
    total = sum(len(v) for v in ds.values())
    print(f"  Done. Splits: { {k: len(v) for k, v in ds.items()} }  total={total}")


def download_ugphysics(cache_dir: str) -> None:
    from datasets import load_dataset
    print(f"\n[2/4] Downloading UGPhysics ({len(UGPHYSICS_SUBJECTS)} subjects)...")
    totals: dict[str, int] = {}
    for subject in UGPHYSICS_SUBJECTS:
        ds = load_dataset("UGPhysics/ugphysics", subject, cache_dir=cache_dir)
        n = sum(len(v) for v in ds.values())
        totals[subject] = n
        print(f"  {subject:<30} {n} rows")
    print(f"  Total UGPhysics rows: {sum(totals.values())}")


def download_olympiad(cache_dir: str) -> None:
    from datasets import load_dataset
    print(f"\n[3/4] Downloading OlympiadBench ({len(OLYMPIAD_CONFIGS)} configs)...")
    for config in OLYMPIAD_CONFIGS:
        ds = load_dataset("lscpku/OlympiadBench-official", config, cache_dir=cache_dir)
        n = sum(len(v) for v in ds.values())
        print(f"  {config:<35} {n} rows")


def download_critpt(cache_dir: str) -> None:
    from datasets import load_dataset
    print("\n[4/4] Downloading CritPt (CritPt-Benchmark/CritPt)...")
    ds = load_dataset("CritPt-Benchmark/CritPt", cache_dir=cache_dir)
    total = sum(len(v) for v in ds.values())
    print(f"  Done. Splits: { {k: len(v) for k, v in ds.items()} }  total={total}")


def print_abench_instructions(cache_dir: str) -> None:
    target = Path(cache_dir).parent / "raw" / "abench"
    print(f"""
[ABench — manual download required]
  ABench Phy_B is not on HuggingFace. Download it from GitHub:

  1. Go to: https://github.com/inclusionAI/ABench
  2. Navigate to: Physics/
  3. Download the Phy_B CSV file (dynamic benchmark)
  4. Save to: {target}/Phy_B_dynamic_100.csv

  Then run: python scripts/inspect_datasets.py --dataset abench
""")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache_dir", default="data/hf_cache")
    parser.add_argument(
        "--dataset", default="all",
        choices=["all", "physics", "ugphysics", "olympiad", "critpt"],
    )
    args = parser.parse_args()

    cache_dir = args.cache_dir
    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    print(f"Cache directory: {Path(cache_dir).resolve()}")

    if args.dataset in ("all", "physics"):
        download_physics(cache_dir)
    if args.dataset in ("all", "ugphysics"):
        download_ugphysics(cache_dir)
    if args.dataset in ("all", "olympiad"):
        download_olympiad(cache_dir)
    if args.dataset in ("all", "critpt"):
        download_critpt(cache_dir)

    if args.dataset == "all":
        print_abench_instructions(cache_dir)

    print("\nAll HuggingFace datasets downloaded.")


if __name__ == "__main__":
    main()
