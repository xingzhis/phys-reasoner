"""Quick catalog of answer formats across all datasets.

Run this after prepare_data.py to understand what the verifier must handle.

Usage
-----
  python scripts/catalog_answers.py --n_examples 5
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import Counter, defaultdict
from pathlib import Path

# Allow running from project root
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from phys_reasoner.data.loaders import (
    load_physics,
    load_ugphysics,
    load_olympiadbench,
    load_abench_phy_b,
    load_critpt,
)
from phys_reasoner.data.schema import PhysicsProblem

logger = logging.getLogger(__name__)


def catalog(problems: list[PhysicsProblem], n_examples: int = 3) -> None:
    by_type: dict[str, list[PhysicsProblem]] = defaultdict(list)
    for p in problems:
        if isinstance(p.answer_type, list):
            for t in p.answer_type:
                by_type[t].append(p)
        else:
            by_type[p.answer_type].append(p)

    print(f"\n{'='*60}")
    print(f"Source: {problems[0].source if problems else '?'}  ({len(problems)} problems)")
    print(f"{'='*60}")

    for atype, items in sorted(by_type.items(), key=lambda x: -len(x[1])):
        print(f"\n  [{atype}]  n={len(items)}")
        for p in items[:n_examples]:
            ans = p.answer if not isinstance(p.answer, list) else p.answer
            unit = f"  unit={p.unit!r}" if p.unit else ""
            tol = f"  tol={p.tolerance!r}" if p.tolerance else ""
            print(f"    answer={ans!r}{unit}{tol}")


def main() -> None:
    logging.basicConfig(level=logging.WARNING)

    parser = argparse.ArgumentParser()
    parser.add_argument("--n_examples", type=int, default=3, help="Examples per answer type")
    parser.add_argument("--language", default="en")
    parser.add_argument("--cache_dir", default=None)
    parser.add_argument("--datasets", nargs="*",
                        default=["physics", "ugphysics", "olympiad", "abench", "critpt"],
                        help="Datasets to catalog")
    args = parser.parse_args()

    lang = args.language

    if "physics" in args.datasets:
        try:
            probs = load_physics(split="test", language=lang, cache_dir=args.cache_dir)
            catalog(probs, args.n_examples)
        except Exception as e:
            print(f"PHYSICS: {e}")

    if "ugphysics" in args.datasets:
        try:
            # Load just one subject to avoid long runtime
            probs = load_ugphysics(subjects=["ClassicalMechanics"], language=lang, cache_dir=args.cache_dir)
            catalog(probs, args.n_examples)
        except Exception as e:
            print(f"UGPhysics: {e}")

    if "olympiad" in args.datasets:
        try:
            probs = load_olympiadbench(language=lang, cache_dir=args.cache_dir)
            catalog(probs, args.n_examples)
        except Exception as e:
            print(f"OlympiadBench: {e}")

    if "abench" in args.datasets:
        try:
            probs = load_abench_phy_b()
            catalog(probs, args.n_examples)
        except FileNotFoundError as e:
            print(f"ABench Phy_B (not downloaded yet): {e}")

    if "critpt" in args.datasets:
        try:
            probs = load_critpt(cache_dir=args.cache_dir)
            catalog(probs, args.n_examples)
        except Exception as e:
            print(f"CritPt: {e}")


if __name__ == "__main__":
    main()
