"""validate_loaders.py — raw-vs-parsed invariant checker for all dataset loaders.

For each source, loads both the raw HuggingFace/file data and the parsed
PhysicsProblem output, then checks a set of invariants designed to catch the
classes of silent data loss we have observed:

  1. No empty answers after parsing
  2. Answer-part count matches answer-type count (for multi-part rows)
  3. No silent truncation: parsed row count is within expected bounds
  4. For PHYSICS specifically: no multi-alternative rows leaked through

Usage:
    python scripts/validate_loaders.py [--sources physics ugphysics olympiadbench]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from phys_reasoner.data.loaders import (
    _DEFAULT_CACHE,
    _DEFAULT_RAW,
    load_physics,
    load_ugphysics,
    load_olympiadbench,
    load_phybench,
    load_scibench_rl,
)


# ---------------------------------------------------------------------------
# Generic invariant checks
# ---------------------------------------------------------------------------

def _check_no_empty_answers(rows, source):
    empty = [r.problem_id for r in rows if not r.answer and r.answer != 0]
    if empty:
        print(f"  FAIL  empty answers ({len(empty)}): {empty[:5]}")
    else:
        print(f"  ok    no empty answers")
    return len(empty)


def _check_part_type_alignment(rows, source):
    """answer parts count must equal answer_type parts count."""
    mismatches = []
    for r in rows:
        n_ans = len(r.answer) if isinstance(r.answer, list) else 1
        n_at  = len(r.answer_type) if isinstance(r.answer_type, list) else 1
        if n_ans != n_at:
            mismatches.append((r.problem_id, n_ans, n_at))
    if mismatches:
        print(f"  FAIL  answer/type length mismatch ({len(mismatches)} rows):")
        for pid, na, nt in mismatches[:5]:
            print(f"        {pid}: {na} answer parts, {nt} type parts")
    else:
        print(f"  ok    all answer/type lengths aligned")
    return len(mismatches)


def _check_row_count(n_parsed, expected_min, expected_max, source):
    if n_parsed < expected_min or n_parsed > expected_max:
        print(f"  FAIL  row count {n_parsed} outside expected [{expected_min}, {expected_max}]")
        return 1
    print(f"  ok    row count {n_parsed} in [{expected_min}, {expected_max}]")
    return 0


# ---------------------------------------------------------------------------
# Source-specific validators
# ---------------------------------------------------------------------------

def validate_physics(cache_dir):
    print("\n=== PHYSICS ===")
    from datasets import load_dataset
    import re

    CJK_RE = re.compile(r"[\u4e00-\u9fff]")
    raw_ds = load_dataset("desimfj/PHYSICS", cache_dir=cache_dir, split="test")

    # Count what the raw data has, applying our known drops
    raw_en = 0
    raw_multi_alt = 0
    raw_open_end = 0
    for row in raw_ds:
        if CJK_RE.search(row["question"]):
            continue
        raw_en += 1
        ans = row["answer"]
        if isinstance(ans, list) and ans and isinstance(ans[0], list):
            if any(len(lst) > 1 for lst in ans):
                raw_multi_alt += 1

    print(f"  Raw EN rows: {raw_en}")
    print(f"  Multi-alternative (to drop): {raw_multi_alt}")
    expected_min = raw_en - raw_multi_alt - 5   # small slack for open-end moved to eval
    expected_max = raw_en

    rows = load_physics(cache_dir=cache_dir)
    n = _check_row_count(len(rows), expected_min, expected_max, "PHYSICS")

    # Check no multi-alternative rows leaked through
    leaked = []
    for r in rows:
        ans = r.answer
        if isinstance(ans, list):
            # Each element should be a string, not a list
            if any(isinstance(a, list) for a in ans):
                leaked.append(r.problem_id)
    if leaked:
        print(f"  FAIL  multi-alternative rows leaked: {leaked[:5]}")
    else:
        print(f"  ok    no multi-alternative rows")

    n += _check_no_empty_answers(rows, "PHYSICS")
    n += _check_part_type_alignment(rows, "PHYSICS")

    # Spot-check: answer_type values are from the expected vocabulary
    # "T/F" is a variant spelling of "True/False" used in some PHYSICS rows.
    valid_at = {"Numerical", "Expression", "Equation", "MCQ", "True/False", "T/F",
                "Interval", "Open-end", "symbolic"}
    unexpected_at = set()
    for r in rows:
        types = r.answer_type if isinstance(r.answer_type, list) else [r.answer_type]
        for t in types:
            if t not in valid_at:
                unexpected_at.add(t)
    if unexpected_at:
        print(f"  WARN  unexpected answer_type values: {sorted(unexpected_at)[:10]}")
    else:
        print(f"  ok    all answer_type values in expected vocabulary")

    return n


def validate_ugphysics(cache_dir):
    print("\n=== UGPHYSICS ===")
    rows = load_ugphysics(cache_dir=cache_dir)
    n = _check_row_count(len(rows), 5000, 6000, "UGPhysics")
    n += _check_no_empty_answers(rows, "UGPhysics")

    # UGPhysics answer is always a plain string (not a list) — verify
    list_answers = [r.problem_id for r in rows if isinstance(r.answer, list)]
    if list_answers:
        print(f"  WARN  {len(list_answers)} rows have list answers (unexpected for UGPhysics)")
    else:
        print(f"  ok    all answers are plain strings")

    # Check cleaned answer_type values don't contain newlines or backticks
    dirty = [r.problem_id for r in rows
             if isinstance(r.answer_type, str) and ("\n" in r.answer_type or "`" in r.answer_type)]
    if dirty:
        print(f"  FAIL  dirty answer_type ({len(dirty)} rows): {dirty[:3]}")
    else:
        print(f"  ok    no dirty answer_type labels")

    # Spot-check: all answer_type codes are valid UGPhysics codes
    valid_codes = {"NV", "EX", "EQ", "MC", "TF", "IN"}
    invalid = set()
    for r in rows:
        # Multi-part: "NV, EX" style
        parts = [p.strip() for p in r.answer_type.split(",")]
        for p in parts:
            if p not in valid_codes:
                invalid.add(p)
    if invalid:
        print(f"  WARN  unexpected UGPhysics type codes: {sorted(invalid)[:10]}")
    else:
        print(f"  ok    all answer_type codes valid")

    return n


def validate_olympiadbench(cache_dir):
    print("\n=== OLYMPIADBENCH ===")
    from datasets import load_dataset

    config = "OE_TO_physics_en_COMP"
    raw_ds_all = load_dataset("lscpku/OlympiadBench-official", config, cache_dir=cache_dir)
    split_name = list(raw_ds_all.keys())[0]
    raw_rows = list(raw_ds_all[split_name])

    # Compute how many raw rows the loader will keep (same logic as loader):
    # drop when n_ans != n_at (5 known mismatched rows in this config).
    from phys_reasoner.data.loaders import _unwrap_single
    kept_raw = 0
    for r in raw_rows:
        raw_at = r.get("answer_type") or ""
        raw_answer = r.get("final_answer", [])
        answer = _unwrap_single(raw_answer) if isinstance(raw_answer, list) else raw_answer
        answer_type = _unwrap_single(raw_at.split(",")) if "," in str(raw_at) else raw_at
        n_ans = len(answer) if isinstance(answer, list) else 1
        n_at  = len(answer_type) if isinstance(answer_type, list) else 1
        if n_ans == n_at:
            kept_raw += 1

    rows = load_olympiadbench(config=config, cache_dir=cache_dir)
    n = _check_row_count(len(rows), kept_raw - 1, kept_raw, "OlympiadBench")

    n += _check_no_empty_answers(rows, "OlympiadBench")
    n += _check_part_type_alignment(rows, "OlympiadBench")
    return n


def validate_phybench(cache_dir):
    print("\n=== PHYBENCH ===")
    from datasets import load_dataset

    raw_ds = load_dataset("Eureka-Lab/PHYBench", cache_dir=cache_dir, split="train")
    raw_nonempty = sum(1 for r in raw_ds
                       if r.get("content", "").strip() and r.get("answer", "").strip())

    rows = load_phybench(cache_dir=cache_dir)
    n = _check_row_count(len(rows), raw_nonempty - 2, raw_nonempty, "PHYBench")
    n += _check_no_empty_answers(rows, "PHYBench")

    # PHYBench answers are mostly raw LaTeX without \boxed, but a small number
    # in the source dataset do include it — just report, not a failure.
    boxed = [r.problem_id for r in rows if "boxed" in r.answer]
    if boxed:
        print(f"  note  {len(boxed)} PHYBench answers contain \\boxed (source inconsistency, not a bug)")
    else:
        print(f"  ok    no \\boxed in PHYBench answers")
    return n


def validate_scibench(cache_dir):
    print("\n=== SCIBENCH_RL ===")
    from datasets import load_dataset

    raw_ds = load_dataset("Sihangli/scibench-rl", cache_dir=cache_dir, split="train")
    raw_physics = sum(1 for r in raw_ds
                      if str(r.get("source") or "") not in {"atkins", "chemmc"})

    rows = load_scibench_rl(cache_dir=cache_dir)
    n = _check_row_count(len(rows), raw_physics - 2, raw_physics, "SciBench_RL")
    n += _check_no_empty_answers(rows, "SciBench_RL")

    # SciBench answers are bare numbers — check none became "None"
    none_str = [r.problem_id for r in rows if r.answer.strip().lower() == "none"]
    if none_str:
        print(f"  FAIL  {len(none_str)} answers became literal 'None': {none_str[:5]}")
        n += len(none_str)
    else:
        print(f"  ok    no 'None' answers")
    return n


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

_ALL_SOURCES = ["physics", "ugphysics", "olympiadbench", "phybench", "scibench"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sources", nargs="+", choices=_ALL_SOURCES, default=_ALL_SOURCES,
        metavar="SOURCE", help="Which sources to validate (default: all)"
    )
    args = parser.parse_args()

    total_failures = 0
    for source in args.sources:
        if source == "physics":
            total_failures += validate_physics(_DEFAULT_CACHE)
        elif source == "ugphysics":
            total_failures += validate_ugphysics(_DEFAULT_CACHE)
        elif source == "olympiadbench":
            total_failures += validate_olympiadbench(_DEFAULT_CACHE)
        elif source == "phybench":
            total_failures += validate_phybench(_DEFAULT_CACHE)
        elif source == "scibench":
            total_failures += validate_scibench(_DEFAULT_CACHE)

    print(f"\n{'='*60}")
    if total_failures == 0:
        print("ALL CHECKS PASSED")
    else:
        print(f"FAILURES: {total_failures} check(s) failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
