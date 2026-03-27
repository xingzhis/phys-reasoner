"""Download all datasets (HuggingFace + GitHub) to local cache.

Run this once before any loader/training work.

Usage
-----
  python scripts/download_datasets.py                  # download everything
  python scripts/download_datasets.py --dataset yale   # single source
"""

from __future__ import annotations

import argparse
import urllib.request
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

OLYMPIAD_CONFIGS = [
    "physics_en_no_proof",    # 692 rows — all English OE non-proof
    "physics_zh_no_proof",    # 1598 rows — all Chinese OE non-proof
    "OE_TO_physics_en_COMP",  # 236 rows — English, text-only, competition
    "OE_TO_physics_zh_CEE",   # 115 rows — Chinese, text-only, CEE
]

# Yale NLP Physics: only textonly tier is used (eval_tier3 + dedup contamination check)
_YALE_BASE = "https://raw.githubusercontent.com/yale-nlp/Physics/main/PHYSICS/PHYSICS-textonly"
_YALE_FILES = [
    "atomic_dataset_textonly.jsonl",
    "electro_dataset_textonly.jsonl",
    "mechanics_dataset_textonly.jsonl",
    "optics_dataset_textonly.jsonl",
    "quantum_dataset_textonly.jsonl",
    "statistics_dataset_textonly.jsonl",
]

# ABench Physics: Phy_A (static) + Phy_B (dynamic variants) — eval_tier2
_ABENCH_BASE = "https://raw.githubusercontent.com/inclusionAI/ABench/main/Physics/data"
_ABENCH_FILES = [
    "Phy_A_fixed_400.csv",
    "Phy_B_dynamic_100.csv",
]


def _download_file(url: str, dest: Path) -> None:
    if dest.exists():
        print(f"  skip  {dest.name} (already exists)")
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"  fetch {dest.name} ...", end=" ", flush=True)
    urllib.request.urlretrieve(url, dest)
    print(f"ok ({dest.stat().st_size // 1024} KB)")


def download_physics(cache_dir: str) -> None:
    from datasets import load_dataset
    print("\n[1/6] Downloading PHYSICS (desimfj/PHYSICS)...")
    ds = load_dataset("desimfj/PHYSICS", cache_dir=cache_dir)
    total = sum(len(v) for v in ds.values())
    print(f"  Done. Splits: { {k: len(v) for k, v in ds.items()} }  total={total}")


def download_ugphysics(cache_dir: str) -> None:
    from datasets import load_dataset
    print(f"\n[2/6] Downloading UGPhysics ({len(UGPHYSICS_SUBJECTS)} subjects)...")
    totals: dict[str, int] = {}
    for subject in UGPHYSICS_SUBJECTS:
        ds = load_dataset("UGPhysics/ugphysics", subject, cache_dir=cache_dir)
        n = sum(len(v) for v in ds.values())
        totals[subject] = n
        print(f"  {subject:<30} {n} rows")
    print(f"  Total UGPhysics rows: {sum(totals.values())}")


def download_olympiad(cache_dir: str) -> None:
    from datasets import load_dataset
    print(f"\n[3/6] Downloading OlympiadBench ({len(OLYMPIAD_CONFIGS)} configs)...")
    for config in OLYMPIAD_CONFIGS:
        ds = load_dataset("lscpku/OlympiadBench-official", config, cache_dir=cache_dir)
        n = sum(len(v) for v in ds.values())
        print(f"  {config:<35} {n} rows")


def download_critpt(cache_dir: str) -> None:
    from datasets import load_dataset
    print("\n[4/6] Downloading CritPt (CritPt-Benchmark/CritPt)...")
    ds = load_dataset("CritPt-Benchmark/CritPt", cache_dir=cache_dir)
    total = sum(len(v) for v in ds.values())
    print(f"  Done. Splits: { {k: len(v) for k, v in ds.items()} }  total={total}")


def download_yale(raw_dir: str) -> None:
    """Download Yale NLP PHYSICS-textonly tier (eval_tier3) from GitHub."""
    print("\n[5/6] Downloading Yale NLP Physics (PHYSICS-textonly, 6 files)...")
    dest_dir = Path(raw_dir) / "yale-physics" / "PHYSICS" / "PHYSICS-textonly"
    for fname in _YALE_FILES:
        _download_file(f"{_YALE_BASE}/{fname}", dest_dir / fname)
    print(f"  Done. Files in {dest_dir}")


def download_abench(raw_dir: str) -> None:
    """Download ABench Phy_A and Phy_B CSVs (eval_tier2) from GitHub."""
    print("\n[6/6] Downloading ABench Physics (Phy_A + Phy_B CSVs)...")
    dest_dir = Path(raw_dir) / "abench"
    for fname in _ABENCH_FILES:
        _download_file(f"{_ABENCH_BASE}/{fname}", dest_dir / fname)
    print(f"  Done. Files in {dest_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cache_dir", default="data/hf_cache",
                        help="HuggingFace dataset cache (default: data/hf_cache)")
    parser.add_argument("--raw_dir", default="data/raw",
                        help="Directory for raw file downloads (default: data/raw)")
    parser.add_argument(
        "--dataset", default="all",
        choices=["all", "physics", "ugphysics", "olympiad", "critpt", "yale", "abench"],
        help="Which dataset(s) to download (default: all)",
    )
    args = parser.parse_args()

    Path(args.cache_dir).mkdir(parents=True, exist_ok=True)
    print(f"HF cache:  {Path(args.cache_dir).resolve()}")
    print(f"Raw files: {Path(args.raw_dir).resolve()}")

    if args.dataset in ("all", "physics"):
        download_physics(args.cache_dir)
    if args.dataset in ("all", "ugphysics"):
        download_ugphysics(args.cache_dir)
    if args.dataset in ("all", "olympiad"):
        download_olympiad(args.cache_dir)
    if args.dataset in ("all", "critpt"):
        download_critpt(args.cache_dir)
    if args.dataset in ("all", "yale"):
        download_yale(args.raw_dir)
    if args.dataset in ("all", "abench"):
        download_abench(args.raw_dir)

    print("\nAll datasets downloaded.")


if __name__ == "__main__":
    main()
