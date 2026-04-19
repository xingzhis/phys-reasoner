"""PHYBench loader — 1,000 competition physics problems.

Source: Eureka-Lab/PHYBench (HF), cached at
  data/hf_cache/Eureka-Lab___phy_bench/default/0.0.0/<hash>/phy_bench-train.arrow

Schema (upstream): id, tag, content, solution, answer.
- `content` is the problem statement
- `answer` is the raw LaTeX answer (NO \\boxed{} wrapping)
- `solution` is a worked solution (unused here)

Output schema matches rollout.py expectations:
    data_source   : 'PHYBench'
    prompt        : [{role:user, content: <problem>}]
    reward_model  : {ground_truth: <\\boxed{answer}>, style: 'rule'}
    extra_info    : {problem, answer_type='expression', id, tag,
                     raw_answer (unboxed, for EED scorer)}

The scorer (eval.scoring.phybench_eed) uses the vendored phybench-official EED
algorithm which runs on SymPy expression trees — \\boxed{} wrapping is only for
the rollout/verifier contract; the EED scorer strips it before comparison.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

DATA_SOURCE_TAG = "PHYBench"
DEFAULT_ARROW = (
    "data/hf_cache/Eureka-Lab___phy_bench/default/0.0.0/"
    "d6d91c787b7abb865eb2490a328bf85a9f5095f0/phy_bench-train.arrow"
)


def _read_arrow(path: Path) -> pd.DataFrame:
    import pyarrow as pa
    import pyarrow.ipc as ipc
    with pa.memory_map(str(path), "r") as src:
        try:
            rdr = ipc.RecordBatchStreamReader(src)
        except Exception:
            src.seek(0)
            rdr = ipc.open_file(src)
        tbl = rdr.read_all()
    return tbl.to_pandas()


def _strip_outer_brackets(s: str) -> str:
    """PHYBench answers are often wrapped in \\[ ... \\] display-math. Strip."""
    s = s.strip()
    if s.startswith(r"\[") and s.endswith(r"\]"):
        return s[2:-2].strip()
    if s.startswith("$$") and s.endswith("$$"):
        return s[2:-2].strip()
    if s.startswith("$") and s.endswith("$") and len(s) >= 2:
        return s[1:-1].strip()
    return s


def load(
    out_path: str,
    arrow_path: str = DEFAULT_ARROW,
    force: bool = False,
) -> str:
    out = Path(out_path)
    if out.exists() and not force:
        n = len(pd.read_parquet(out))
        print(f"[phybench] reuse {out} ({n} rows)")
        return str(out)

    arrow_abs = Path(arrow_path)
    if not arrow_abs.is_absolute():
        arrow_abs = Path.cwd() / arrow_abs
    src = _read_arrow(arrow_abs)
    print(f"[phybench] loaded {len(src)} rows from {arrow_abs}")

    rows: list[dict] = []
    for _, r in src.iterrows():
        problem = str(r["content"])
        raw_answer = _strip_outer_brackets(str(r["answer"]))
        gold_boxed = r"\boxed{" + raw_answer + "}"
        extra_info = {
            "problem": problem,
            "answer_type": "expression",
            "id": int(r["id"]) if pd.notna(r.get("id")) else -1,
            "tag": str(r.get("tag") or ""),
            "raw_answer": raw_answer,
        }
        rows.append({
            "data_source": DATA_SOURCE_TAG,
            "prompt": [{"role": "user", "content": problem}],
            "reward_model": {"ground_truth": gold_boxed, "style": "rule"},
            "extra_info": extra_info,
            "pool": "external",
        })

    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(out, index=False)
    print(f"[phybench] wrote {len(rows)} rows → {out}")
    return str(out)


def main() -> None:
    p = argparse.ArgumentParser(description="Build PHYBench eval parquet")
    p.add_argument("--arrow", default=DEFAULT_ARROW)
    p.add_argument("--out", required=True)
    p.add_argument("--force", action="store_true")
    args = p.parse_args()
    load(out_path=args.out, arrow_path=args.arrow, force=args.force)


if __name__ == "__main__":
    main()
