"""Exploratory QC script for training candidate datasets.

Run → inspect output → fix issues in loaders or clean.py → re-run.
Phase B ends when this script reports zero flagged rows.

Usage
-----
  python scripts/explore_quality.py
  python scripts/explore_quality.py --sources physics ugphysics olympiad
  python scripts/explore_quality.py --save data/processed/candidates_raw.parquet
"""

from __future__ import annotations

import argparse
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

# Ensure project src is on path when run directly
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from phys_reasoner.data.schema import PhysicsProblem


# ---------------------------------------------------------------------------
# Valid answer type codes (used in invariant checks)
# ---------------------------------------------------------------------------

_VALID_AT_CODES = {
    # PHYSICS dataset
    "Numerical", "Expression", "Equation", "MCQ", "Open-end", "True/False", "Interval",
    # UGPhysics
    "NV", "EX", "EQ", "MC", "TF", "IN",
    # OlympiadBench
    "numerical", "expression", "equation", "interval",
    # Inferred
    "symbolic", "numerical", "code", "unknown",
}


# ---------------------------------------------------------------------------
# Load all training candidates
# ---------------------------------------------------------------------------

def load_all_training(cache_dir: str, sources: list[str]) -> list[PhysicsProblem]:
    from phys_reasoner.data.loaders import (
        load_physics,
        load_ugphysics,
        load_olympiadbench,
        load_phybench,
        load_scibench_rl,
    )

    rows: list[PhysicsProblem] = []

    if "physics" in sources:
        print("Loading PHYSICS...", flush=True)
        rows.extend(load_physics(cache_dir=cache_dir))

    if "ugphysics" in sources:
        print("Loading UGPhysics (all 13 subjects)...", flush=True)
        rows.extend(load_ugphysics(cache_dir=cache_dir))

    if "olympiad" in sources:
        print("Loading OlympiadBench (text-only EN)...", flush=True)
        rows.extend(load_olympiadbench("OE_TO_physics_en_COMP", cache_dir=cache_dir))

    if "phybench" in sources:
        print("Loading PHYBench...", flush=True)
        rows.extend(load_phybench(cache_dir=cache_dir))

    if "scibench" in sources:
        print("Loading SciBench-RL...", flush=True)
        rows.extend(load_scibench_rl(cache_dir=cache_dir))

    # Filter to training candidates only
    train = [r for r in rows if r.split == "train_candidate"]
    print(f"\nTotal rows loaded: {len(rows)}")
    print(f"Training candidates: {len(train)}")
    return train


# ---------------------------------------------------------------------------
# QC checks
# ---------------------------------------------------------------------------

def _at_str(at: str | list[str]) -> str:
    if isinstance(at, list):
        return ", ".join(str(x) for x in at)
    return str(at)


def report_nulls(rows: list[PhysicsProblem]) -> int:
    """(a) Null/empty field counts per dataset."""
    print("\n" + "=" * 70)
    print("(a) NULL / EMPTY FIELD COUNTS")
    print("=" * 70)

    fields = ["problem", "answer", "answer_type"]
    source_counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))

    for r in rows:
        for f in fields:
            val = getattr(r, f)
            if val is None or val == "" or val == [] or (isinstance(val, list) and not any(val)):
                source_counts[r.source][f] += 1

    total_flagged = 0
    for src, counts in sorted(source_counts.items()):
        for f, n in counts.items():
            print(f"  {src:<20} {f:<15} {n} rows")
            total_flagged += n

    if not any(source_counts.values()):
        print("  [OK] No null/empty fields found.")
    return total_flagged


def report_answer_types(rows: list[PhysicsProblem]) -> int:
    """(b) answer_type distribution with 3 sample rows per rare type."""
    print("\n" + "=" * 70)
    print("(b) ANSWER TYPE DISTRIBUTION")
    print("=" * 70)

    by_source: dict[str, Counter] = defaultdict(Counter)
    samples: dict[str, dict[str, list[str]]] = defaultdict(lambda: defaultdict(list))

    for r in rows:
        at = _at_str(r.answer_type)
        by_source[r.source][at] += 1
        if len(samples[r.source][at]) < 3:
            samples[r.source][at].append(r.problem_id)

    for src, counts in sorted(by_source.items()):
        print(f"\n  [{src}]")
        for at, n in counts.most_common():
            rare_marker = " ← RARE" if n <= 5 else ""
            print(f"    {at:<40} {n:>5}{rare_marker}")
            if n <= 5:
                for pid in samples[src][at]:
                    print(f"      example: {pid}")

    return 0  # informational only


def report_answer_lengths(rows: list[PhysicsProblem]) -> int:
    """(c) Answer length histogram — long answers often contain prose."""
    print("\n" + "=" * 70)
    print("(c) ANSWER LENGTH HISTOGRAM (long answers may be prose)")
    print("=" * 70)

    flagged = 0
    long_examples: list[tuple[str, str, str]] = []

    for r in rows:
        if isinstance(r.answer, list):
            lengths = [len(a) for a in r.answer]
            max_len = max(lengths) if lengths else 0
        else:
            max_len = len(str(r.answer))

        if max_len > 200:
            flagged += 1
            if len(long_examples) < 5:
                ans_preview = (
                    r.answer[0] if isinstance(r.answer, list) else r.answer
                )[:100]
                long_examples.append((r.source, r.problem_id, ans_preview))

    buckets = Counter()
    for r in rows:
        if isinstance(r.answer, list):
            max_len = max((len(a) for a in r.answer), default=0)
        else:
            max_len = len(str(r.answer))
        if max_len == 0:
            buckets["0"] += 1
        elif max_len <= 20:
            buckets["1-20"] += 1
        elif max_len <= 100:
            buckets["21-100"] += 1
        elif max_len <= 200:
            buckets["101-200"] += 1
        else:
            buckets[">200"] += 1

    for bucket in ["0", "1-20", "21-100", "101-200", ">200"]:
        print(f"  {bucket:<10} {buckets[bucket]:>5} rows")

    if long_examples:
        print(f"\n  Flagged (>200 chars): {flagged} rows. Examples:")
        for src, pid, preview in long_examples:
            print(f"    [{src}] {pid}: {preview!r}...")

    print("  (informational — long LaTeX equations are expected for multi-part problems)")
    return 0  # informational only; long equations are not a QC problem


def report_boxed_rate(rows: list[PhysicsProblem]) -> int:
    """(d) \\boxed{} presence rate per dataset."""
    print("\n" + "=" * 70)
    print(r"(d) \boxed{} PRESENCE RATE")
    print("=" * 70)

    by_source_total: Counter = Counter()
    by_source_boxed: Counter = Counter()

    for r in rows:
        by_source_total[r.source] += 1
        ans_str = (
            " ".join(r.answer) if isinstance(r.answer, list) else str(r.answer)
        )
        if r"\\boxed{" in ans_str or r"\boxed{" in ans_str:
            by_source_boxed[r.source] += 1

    for src in sorted(by_source_total):
        total = by_source_total[src]
        n_boxed = by_source_boxed[src]
        pct = 100 * n_boxed / total if total else 0
        print(f"  {src:<25} {n_boxed:>5} / {total:>5}  ({pct:.1f}%)")

    return 0  # informational only


def _passes_invariants(r: PhysicsProblem) -> list[str]:
    """Return list of invariant violations for a row."""
    violations = []

    # answer must be str or list[str]
    if isinstance(r.answer, list):
        if not r.answer:
            violations.append("empty answer list")
        elif not all(isinstance(a, str) for a in r.answer):
            violations.append(f"answer list contains non-str: {[type(a).__name__ for a in r.answer]}")
        # check for doubly-nested
        if any(isinstance(a, list) for a in r.answer):
            violations.append("answer is doubly nested List[List[str]]")
    elif r.answer is None:
        violations.append("answer is None")

    # answer must be non-empty
    if r.answer == "" or r.answer == []:
        violations.append("answer is empty string or empty list")

    # split must be valid
    valid_splits = {"train_candidate", "eval_tier1", "eval_tier2", "eval_tier3"}
    if r.split not in valid_splits:
        violations.append(f"invalid split: {r.split!r}")

    # problem must be non-empty
    if not r.problem or not r.problem.strip():
        violations.append("problem is empty")

    return violations


def report_invariants(rows: list[PhysicsProblem]) -> int:
    """(e) Rows failing core invariants."""
    print("\n" + "=" * 70)
    print("(e) INVARIANT VIOLATIONS")
    print("=" * 70)

    total_flagged = 0
    by_source: dict[str, list[tuple[str, list[str]]]] = defaultdict(list)

    for r in rows:
        violations = _passes_invariants(r)
        if violations:
            total_flagged += 1
            if len(by_source[r.source]) < 5:
                by_source[r.source].append((r.problem_id, violations))

    if total_flagged == 0:
        print("  [OK] All invariants pass.")
    else:
        print(f"  TOTAL FLAGGED: {total_flagged} rows")
        for src, examples in sorted(by_source.items()):
            print(f"\n  [{src}]")
            for pid, viols in examples:
                print(f"    {pid}: {viols}")

    return total_flagged


def report_ugphysics_dirty_labels(rows: list[PhysicsProblem]) -> int:
    """Check UGPhysics for remaining dirty answer_type labels."""
    print("\n" + "=" * 70)
    print("(f) UGPHYSICS DIRTY ANSWER_TYPE LABELS")
    print("=" * 70)

    flagged = 0
    examples: list[tuple[str, str]] = []

    for r in rows:
        if r.source != "UGPhysics":
            continue
        at = str(r.answer_type)
        if "\n" in at or "`" in at:
            flagged += 1
            if len(examples) < 5:
                examples.append((r.problem_id, repr(at)))

    if flagged == 0:
        print("  [OK] No dirty answer_type labels.")
    else:
        print(f"  FLAGGED: {flagged} rows")
        for pid, at in examples:
            print(f"    {pid}: {at}")

    return flagged


def report_phybench_prose(rows: list[PhysicsProblem]) -> int:
    """Check PHYBench for prose answers (not SymPy-verifiable)."""
    print("\n" + "=" * 70)
    print("(g) PHYBENCH PROSE ANSWERS (estimate)")
    print("=" * 70)

    # Detect prose: \text{...} in LaTeX (wraps English words in math mode),
    # or English sentences (2+ common words not preceded by a LaTeX backslash).
    # Exclude LaTeX commands like \Delta, \frac etc. by requiring no preceding backslash.
    _prose_re = re.compile(r'\\text\{|(?<!\\)[A-Z][a-z]{3,} [a-z]{3,} ')

    flagged = 0
    examples: list[tuple[str, str]] = []

    for r in rows:
        if r.source != "PHYBench":
            continue
        ans = str(r.answer)
        if _prose_re.search(ans):
            flagged += 1
            if len(examples) < 5:
                examples.append((r.problem_id, ans[:80]))

    total_phybench = sum(1 for r in rows if r.source == "PHYBench")
    print(f"  PHYBench total: {total_phybench}")
    print(f"  Estimated prose: {flagged}  ({100*flagged/total_phybench:.1f}% if total > 0)")
    if examples:
        print("  Examples:")
        for pid, ans in examples:
            print(f"    {pid}: {ans!r}")

    return flagged


def save_parquet(rows: list[PhysicsProblem], path: str) -> None:
    import json
    import pandas as pd

    records = []
    for r in rows:
        d = r.to_dict()
        # Serialize list-valued fields to JSON strings for parquet compatibility
        if isinstance(d.get("answer"), list):
            d["answer"] = json.dumps(d["answer"])
        if isinstance(d.get("answer_type"), list):
            d["answer_type"] = json.dumps(d["answer_type"])
        records.append(d)

    df = pd.DataFrame(records)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    print(f"\nSaved {len(df)} rows to {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Exploratory QC for training candidates")
    parser.add_argument(
        "--sources", nargs="+",
        default=["physics", "ugphysics", "olympiad", "phybench", "scibench"],
        choices=["physics", "ugphysics", "olympiad", "phybench", "scibench"],
        help="Which sources to load",
    )
    parser.add_argument(
        "--cache_dir", default="data/hf_cache",
        help="HuggingFace cache directory",
    )
    parser.add_argument(
        "--save", default=None,
        help="Optional path to save raw candidates as parquet",
    )
    args = parser.parse_args()

    rows = load_all_training(args.cache_dir, args.sources)

    total_flagged = 0
    total_flagged += report_nulls(rows)
    report_answer_types(rows)
    total_flagged += report_answer_lengths(rows)
    report_boxed_rate(rows)
    total_flagged += report_invariants(rows)
    total_flagged += report_ugphysics_dirty_labels(rows)
    total_flagged += report_phybench_prose(rows)

    print("\n" + "=" * 70)
    print(f"TOTAL FLAGGED ROWS: {total_flagged}")
    if total_flagged == 0:
        print("Phase B complete — all QC checks pass.")
    else:
        print("Phase B incomplete — fix issues in loaders or clean.py, then re-run.")
    print("=" * 70)

    if args.save:
        save_parquet(rows, args.save)


if __name__ == "__main__":
    main()
