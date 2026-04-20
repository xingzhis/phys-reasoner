"""Per-cell breakdown of tool-use rate and correctness by path.

For every (benchmark, mode, model, tag) dir that has both rollouts.parquet
and scored.parquet, slices the rollouts into:
  - called_tool  (code is not None)
     - sandbox_ok (no error)
     - sandbox_err
  - skipped_tool (pure-CoT within TIR)

and reports counts + pass@1 in each slice. Useful for understanding whether
gains come from better tool use or stronger pure-reasoning.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def _scan(outputs_eval: Path):
    """Yield (benchmark, cell_dir) for every cell with both parquets."""
    for bench_dir in sorted(outputs_eval.iterdir()):
        if not bench_dir.is_dir() or bench_dir.name.startswith(("_", "diff")):
            continue
        for cell_dir in sorted(bench_dir.iterdir()):
            if not cell_dir.is_dir():
                continue
            roll = cell_dir / "rollouts.parquet"
            scored = cell_dir / "scored.parquet"
            if roll.exists() and scored.exists():
                yield bench_dir.name, cell_dir


def analyze(cell_dir: Path) -> dict:
    df_r = pd.read_parquet(cell_dir / "rollouts.parquet")
    df_s = pd.read_parquet(cell_dir / "scored.parquet")
    # Join on (problem_idx, rollout_idx)
    r = df_r.set_index(["problem_idx", "rollout_idx"])
    s = df_s.set_index(["problem_idx", "rollout_idx"])
    common = r.index.intersection(s.index)
    r = r.loc[common]
    s = s.loc[common]

    n = len(r)
    # Cell mode is cot or tir; CoT rollouts have code=None always (no tool)
    called = r["code"].apply(lambda c: c is not None and not isinstance(c, float))
    sandbox_err = r["sandbox_error"].astype(bool) if "sandbox_error" in r.columns else pd.Series([False] * n, index=r.index)
    correct = s["correct"].astype(bool) if "correct" in s.columns else pd.Series([False] * n, index=s.index)

    call_n = int(called.sum())
    skip_n = n - call_n
    call_ok = int((called & ~sandbox_err).sum())
    call_err = int((called & sandbox_err).sum())

    call_correct = int((called & correct).sum())
    skip_correct = int((~called & correct).sum())
    call_ok_correct = int((called & ~sandbox_err & correct).sum())
    call_err_correct = int((called & sandbox_err & correct).sum())

    def _rate(num, denom):
        return num / denom if denom else float("nan")

    return {
        "n": n,
        "called_pct": _rate(call_n, n) * 100,
        "skipped_pct": _rate(skip_n, n) * 100,
        "called_correct_pct": _rate(call_correct, call_n) * 100,
        "skipped_correct_pct": _rate(skip_correct, skip_n) * 100,
        "call_ok_n": call_ok,
        "call_err_n": call_err,
        "call_ok_correct_pct": _rate(call_ok_correct, call_ok) * 100,
        "call_err_correct_pct": _rate(call_err_correct, call_err) * 100,
        "overall_correct_pct": _rate(call_correct + skip_correct, n) * 100,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--outputs-eval", default="outputs/eval")
    args = p.parse_args()

    root = Path(args.outputs_eval)
    rows: list[dict] = []
    for bench, cell in _scan(root):
        try:
            a = analyze(cell)
        except Exception as e:
            print(f"[{bench}/{cell.name}] ERROR: {e}")
            continue
        a["benchmark"] = bench
        a["cell"] = cell.name
        rows.append(a)

    if not rows:
        print("No cells found.")
        return

    df = pd.DataFrame(rows)
    # Sort by cell then benchmark for readable blocks
    df = df.sort_values(["cell", "benchmark"]).reset_index(drop=True)

    # Print table
    header = (
        f"{'benchmark':<24} {'cell':<58} "
        f"{'n':>4}  "
        f"{'called%':>7} {'call_OK':>7} {'call_err':>8}  "
        f"{'skipped%':>8}  "
        f"{'call_OK_pass%':>13} {'call_err_pass%':>14} {'skip_pass%':>10} "
        f"{'overall%':>8}"
    )
    print(header)
    print("-" * len(header))
    for _, r in df.iterrows():
        print(
            f"{r['benchmark']:<24} {r['cell']:<58} "
            f"{r['n']:>4}  "
            f"{r['called_pct']:>6.1f}% {r['call_ok_n']:>7} {r['call_err_n']:>8}  "
            f"{r['skipped_pct']:>7.1f}%  "
            f"{r['call_ok_correct_pct']:>12.1f}% {r['call_err_correct_pct']:>13.1f}% "
            f"{r['skipped_correct_pct']:>9.1f}% "
            f"{r['overall_correct_pct']:>7.1f}%"
        )


if __name__ == "__main__":
    main()
