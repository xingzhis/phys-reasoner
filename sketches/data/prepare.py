"""Prepare datasets for RLVR training.

Steps:
  1. Load all training sources (PHYSICS + UGPhysics + OlympiadBench).
  2. Deduplicate by exact problem text.
  3. Split into train / val.
  4. Save as parquet files compatible with VeRL's data loader.

Usage
-----
  python -m phys_reasoner.data.prepare --out_dir data/processed/ --language en
"""

from __future__ import annotations

import argparse
import hashlib
import logging
from pathlib import Path

import pandas as pd

from .loaders import load_all_training, load_abench_phy_b, load_critpt
from .schema import PhysicsProblem

logger = logging.getLogger(__name__)


def dedup(problems: list[PhysicsProblem]) -> list[PhysicsProblem]:
    """Remove duplicates by SHA-256 of (source, problem_text)."""
    seen: set[str] = set()
    deduped: list[PhysicsProblem] = []
    for p in problems:
        key = hashlib.sha256(p.problem.strip().encode()).hexdigest()
        if key not in seen:
            seen.add(key)
            deduped.append(p)
    removed = len(problems) - len(deduped)
    if removed:
        logger.info("Dedup: removed %d duplicates (%d → %d)", removed, len(problems), len(deduped))
    return deduped


def train_val_split(
    problems: list[PhysicsProblem],
    val_frac: float = 0.05,
    seed: int = 42,
) -> tuple[list[PhysicsProblem], list[PhysicsProblem]]:
    """Split train problems into train/val.

    Only problems with split="train" are re-split.  Those already
    marked split="test" are kept as-is (used for eval, not training).
    """
    import random
    rng = random.Random(seed)

    train_pool = [p for p in problems if p.split == "train"]
    eval_pool = [p for p in problems if p.split != "train"]

    rng.shuffle(train_pool)
    n_val = max(1, int(len(train_pool) * val_frac))
    val = train_pool[:n_val]
    train = train_pool[n_val:]

    for p in val:
        p.split = "val"

    logger.info("Split: %d train, %d val, %d eval-only", len(train), len(val), len(eval_pool))
    return train, val


def to_verl_format(problems: list[PhysicsProblem]) -> pd.DataFrame:
    """Convert to VeRL-compatible parquet schema.

    VeRL expects columns: data_source, prompt, ability, reward_model, extra_info.
    We also keep our schema fields for verifier use.
    """
    records = []
    for p in problems:
        # Serialize multi-part answers to JSON string for parquet storage
        import json
        answer_serialized = json.dumps(p.answer) if isinstance(p.answer, list) else p.answer
        atype_serialized = json.dumps(p.answer_type) if isinstance(p.answer_type, list) else p.answer_type

        records.append({
            # VeRL required fields
            "data_source": p.source,
            "prompt": [{"role": "user", "content": p.problem}],
            "ability": "physics",
            "reward_model": {"style": "rule", "ground_truth": answer_serialized},
            # Our schema fields
            "problem_id": p.problem_id,
            "answer": answer_serialized,
            "answer_type": atype_serialized,
            "unit": p.unit,
            "tolerance": p.tolerance,
            "difficulty": p.difficulty,
            "domain": p.domain,
            "language": p.language,
            "split": p.split,
        })
    return pd.DataFrame(records)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description="Prepare physics RLVR training data")
    parser.add_argument("--out_dir", default="data/processed", help="Output directory")
    parser.add_argument("--language", default="en", choices=["en", "zh", "all"],
                        help="Language filter ('all' = no filter)")
    parser.add_argument("--val_frac", type=float, default=0.05, help="Validation fraction")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cache_dir", default=None, help="HuggingFace cache dir")
    parser.add_argument("--skip_abench", action="store_true")
    parser.add_argument("--skip_critpt", action="store_true")
    args = parser.parse_args()

    lang = None if args.language == "all" else args.language
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- Training sources ---
    logger.info("=== Loading training sources ===")
    problems = load_all_training(language=lang, cache_dir=args.cache_dir)
    problems = dedup(problems)
    train, val = train_val_split(problems, val_frac=args.val_frac, seed=args.seed)

    to_verl_format(train).to_parquet(out_dir / "train.parquet", index=False)
    to_verl_format(val).to_parquet(out_dir / "val.parquet", index=False)
    logger.info("Saved train.parquet (%d rows) and val.parquet (%d rows)", len(train), len(val))

    # --- Eval sources ---
    if not args.skip_abench:
        logger.info("=== Loading ABench Phy_B ===")
        try:
            abench = load_abench_phy_b()
            to_verl_format(abench).to_parquet(out_dir / "abench_phy_b.parquet", index=False)
            logger.info("Saved abench_phy_b.parquet (%d rows)", len(abench))
        except FileNotFoundError as e:
            logger.warning("%s", e)

    if not args.skip_critpt:
        logger.info("=== Loading CritPt ===")
        try:
            critpt = load_critpt(cache_dir=args.cache_dir)
            to_verl_format(critpt).to_parquet(out_dir / "critpt.parquet", index=False)
            logger.info("Saved critpt.parquet (%d rows)", len(critpt))
        except Exception as e:
            logger.warning("CritPt failed: %s", e)

    # --- Catalog answer formats ---
    logger.info("=== Answer format catalog ===")
    all_problems = train + val
    from collections import Counter
    type_counts: Counter = Counter()
    source_counts: Counter = Counter()
    for p in all_problems:
        if isinstance(p.answer_type, list):
            for t in p.answer_type:
                type_counts[t] += 1
        else:
            type_counts[p.answer_type] += 1
        source_counts[p.source] += 1

    print("\n=== Answer type distribution (training set) ===")
    for t, c in type_counts.most_common():
        print(f"  {t:<20} {c:>6}")
    print("\n=== Source distribution ===")
    for s, c in source_counts.most_common():
        print(f"  {s:<25} {c:>6}")


if __name__ == "__main__":
    main()
