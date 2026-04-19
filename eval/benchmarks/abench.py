"""ABench Phy_A + Phy_B loader.

Source: inclusionAI/ABench, CSVs at data/raw/abench/:
    Phy_A_fixed_400.csv   — 400 fixed problems (cols: mid, standard_question, standard_answer)
    Phy_B_dynamic_100.csv — 400 rows = 100 base problems × 4 parametric variants
                            (cols: mid, subid, standard_question, standard_answer)

Answers are numerical (e.g. "2.97") — the official scorer uses 1% tolerance.
Phy_B scoring aggregates per `mid`: a base problem counts as correct only if
ALL 4 variants (subid 0..3) are correct.

Problems are in Chinese. Qwen3.5-4B is multilingual.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

PHY_A_SOURCE = "ABench_Phy_A"
PHY_B_SOURCE = "ABench_Phy_B"
DEFAULT_A_CSV = "data/raw/abench/Phy_A_fixed_400.csv"
DEFAULT_B_CSV = "data/raw/abench/Phy_B_dynamic_100.csv"


def _norm_answer(a) -> str:
    """Upstream `standard_answer` often has leading/trailing whitespace."""
    return str(a).strip()


def load(
    out_path: str,
    variant: str,
    csv_path: str | None = None,
    force: bool = False,
) -> str:
    """variant: 'phy_a' or 'phy_b'."""
    out = Path(out_path)
    if out.exists() and not force:
        n = len(pd.read_parquet(out))
        print(f"[abench:{variant}] reuse {out} ({n} rows)")
        return str(out)

    if variant == "phy_a":
        src_csv = csv_path or DEFAULT_A_CSV
        data_source = PHY_A_SOURCE
    elif variant == "phy_b":
        src_csv = csv_path or DEFAULT_B_CSV
        data_source = PHY_B_SOURCE
    else:
        raise ValueError(f"variant must be 'phy_a' or 'phy_b', got {variant!r}")

    csv_abs = Path(src_csv)
    if not csv_abs.is_absolute():
        csv_abs = Path.cwd() / csv_abs
    src = pd.read_csv(csv_abs, dtype=str)
    print(f"[abench:{variant}] loaded {len(src)} rows from {csv_abs} (cols={list(src.columns)})")

    rows: list[dict] = []
    for _, r in src.iterrows():
        problem = str(r["standard_question"])
        answer = _norm_answer(r["standard_answer"])
        gold_boxed = r"\boxed{" + answer + "}"
        mid = str(r["mid"])
        extra_info = {
            "problem": problem,
            "answer_type": "numerical",
            "tolerance": 0.01,   # ABench official: 1% tolerance
            "mid": mid,
            "raw_answer": answer,
        }
        if variant == "phy_b":
            extra_info["subid"] = str(r["subid"])
        rows.append({
            "data_source": data_source,
            "prompt": [{"role": "user", "content": problem}],
            "reward_model": {"ground_truth": gold_boxed, "style": "rule"},
            "extra_info": extra_info,
            "pool": "external",
        })

    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(out, index=False)
    print(f"[abench:{variant}] wrote {len(rows)} rows → {out}")
    return str(out)


def main() -> None:
    p = argparse.ArgumentParser(description="Build ABench Phy_A or Phy_B eval parquet")
    p.add_argument("--variant", required=True, choices=("phy_a", "phy_b"))
    p.add_argument("--csv", default=None, help="Override CSV path")
    p.add_argument("--out", required=True)
    p.add_argument("--force", action="store_true")
    args = p.parse_args()
    load(out_path=args.out, variant=args.variant, csv_path=args.csv, force=args.force)


if __name__ == "__main__":
    main()
