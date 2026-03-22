"""Spot-check verifier on ABench-Physics (Phy_B) answers.

Tests gold round-trip on eval-set answers to catch notation issues
that differ from training corpus (e.g. plain numerics, sci notation,
Chinese-sourced answers without units).

Usage:
    python scripts/spot_check_eval.py
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from phys_reasoner.verifier.router import verify_answer

PHY_B = Path(__file__).parent.parent / "data/raw/abench/Phy_B_dynamic_100.csv"
PHY_A = Path(__file__).parent.parent / "data/raw/abench/Phy_A_fixed_400.csv"


def load_abench(csv_path: Path) -> list[dict]:
    rows = []
    with open(csv_path, encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def spot_check(rows: list[dict], label: str) -> None:
    false_negatives = []
    unverifiable = []
    total = 0

    for row in rows:
        gold_raw = row["standard_answer"].strip()
        if not gold_raw:
            continue
        total += 1

        # ABench answers are numerical strings (possibly sci notation / LaTeX)
        result = verify_answer(
            pred_text=gold_raw,
            gold_answer=gold_raw,
            answer_type="numerical",
            gold_unit="",
            tolerance=0.05,
            xverify_judge=None,
        )

        if result == 0.0:
            false_negatives.append({
                "id": row.get("mid", "?"),
                "answer": gold_raw,
            })
        elif result == -1.0:
            unverifiable.append({
                "id": row.get("mid", "?"),
                "answer": gold_raw,
            })

    fn_rate = len(false_negatives) / total if total else 0.0
    unv_rate = len(unverifiable) / total if total else 0.0

    print(f"\n{'='*60}")
    print(f"{label}: {total} answers")
    print(f"  False negatives (rule=0.0): {len(false_negatives)} ({fn_rate:.1%})")
    print(f"  Unverifiable   (rule=-1.0): {len(unverifiable)}  ({unv_rate:.1%})")
    print(f"  Pass rate:                  {1 - fn_rate - unv_rate:.1%}")

    if false_negatives:
        print(f"\n  Sample FNs (first 10):")
        for fn in false_negatives[:10]:
            print(f"    [mid={fn['id']}] {fn['answer']!r}")

    if unverifiable:
        print(f"\n  Sample unverifiable (first 5):")
        for u in unverifiable[:5]:
            print(f"    [mid={u['id']}] {u['answer']!r}")


if __name__ == "__main__":
    if PHY_B.exists():
        rows_b = load_abench(PHY_B)
        spot_check(rows_b, "ABench Phy_B (dynamic, 100 base problems)")
    else:
        print(f"Phy_B not found: {PHY_B}")

    if PHY_A.exists():
        rows_a = load_abench(PHY_A)
        spot_check(rows_a, "ABench Phy_A (fixed, 400 problems)")
    else:
        print(f"Phy_A not found: {PHY_A}")
