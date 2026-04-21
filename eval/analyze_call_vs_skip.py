"""Task 1 — call/skip-conditional pass@1 by benchmark and by answer type.

Produces three outputs (paths configurable):
  - outputs/eval/call_vs_skip_by_benchmark.csv
  - outputs/eval/call_vs_skip_by_benchmark_type.csv
  - outputs/eval/call_vs_skip_summary.txt

Run over all Qwen3-4B-Thinking-2507 zero-shot TIR + CoT cells (both train/qwen presets).
Applicable to post-RL rollouts by passing --cells via CELL_SPEC (not implemented —
just re-run with different --outputs-eval root).

Judge column values: xverify-7b (pool_v2), olympiad-math-verify (olympiad),
phybench-exact (phybench), abench-tol1pct (abench_phy_*).

For PHYBench, extra EED columns are emitted (eed_mean, call_eed_mean, etc.)
because the paper's headline PHYBench metric is mean EED, not binary exact-match.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

MODEL_SLUG = "Qwen-Qwen3-4B-Thinking-2507"

BENCHMARKS = [
    "pool_v2_drsci", "pool_v2_physics", "pool_v2_scibench", "pool_v2_ugphysics",
    "olympiad_oe_to_physics", "phybench", "abench_phy_a", "abench_phy_b",
]
MODES = ["tir", "cot"]
PRESETS = ["train", "qwen"]

JUDGE = {
    "pool_v2_drsci": "xverify-7b",
    "pool_v2_physics": "xverify-7b",
    "pool_v2_scibench": "xverify-7b",
    "pool_v2_ugphysics": "xverify-7b",
    "olympiad_oe_to_physics": "olympiad-math-verify",
    "phybench": "phybench-exact",
    "abench_phy_a": "abench-tol1pct",
    "abench_phy_b": "abench-tol1pct",
}


def _load_cell(root: Path, benchmark: str, mode: str, preset: str) -> pd.DataFrame | None:
    cell = root / benchmark / f"{mode}__{MODEL_SLUG}__{preset}"
    r_path = cell / "rollouts.parquet"
    s_path = cell / "scored.parquet"
    if not (r_path.exists() and s_path.exists()):
        return None
    r = pd.read_parquet(r_path)
    s = pd.read_parquet(s_path)
    r = r.set_index(["problem_idx", "rollout_idx"])
    s = s.set_index(["problem_idx", "rollout_idx"])
    common = r.index.intersection(s.index)
    r = r.loc[common]
    s = s.loc[common]
    r["called_tool"] = r["code"].apply(
        lambda c: c is not None and not isinstance(c, float)
    )
    r["correct"] = s["correct"].astype(bool)
    if benchmark == "phybench" and "eed_score" in s.columns:
        r["eed_score"] = s["eed_score"].astype(float)
    r["benchmark"] = benchmark
    r["mode"] = mode
    r["preset"] = preset
    r["condition"] = f"zero_shot_{mode.upper()}"
    return r.reset_index()


def _percent(x) -> float:
    return 100.0 * float(x)


def _rate(num: int, denom: int) -> float:
    return (100.0 * num / denom) if denom else float("nan")


def _aggregate(df: pd.DataFrame) -> dict:
    n = len(df)
    n_call = int(df["called_tool"].sum())
    n_skip = n - n_call
    call_correct = int((df["called_tool"] & df["correct"]).sum())
    skip_correct = int((~df["called_tool"] & df["correct"]).sum())

    call_pct = _rate(n_call, n)
    skip_pct = _rate(n_skip, n)
    call_pass = _rate(call_correct, n_call)
    skip_pass = _rate(skip_correct, n_skip)
    overall_pass = _rate(call_correct + skip_correct, n)
    cms = (call_pass - skip_pass) if (not np.isnan(call_pass) and not np.isnan(skip_pass)) else float("nan")

    row = {
        "n": n,
        "n_called": n_call,
        "n_skipped": n_skip,
        "call_pct": call_pct,
        "skip_pct": skip_pct,
        "call_pass_pct": call_pass,
        "skip_pass_pct": skip_pass,
        "overall_pass_pct": overall_pass,
        "call_minus_skip_pass": cms,
    }
    if "eed_score" in df.columns:
        row["eed_mean"] = float(df["eed_score"].mean())
        row["call_eed_mean"] = float(df.loc[df["called_tool"], "eed_score"].mean()) if n_call else float("nan")
        row["skip_eed_mean"] = float(df.loc[~df["called_tool"], "eed_score"].mean()) if n_skip else float("nan")
    return row


def _pretty_float(x) -> str:
    if isinstance(x, float) and np.isnan(x):
        return "   nan"
    if isinstance(x, (int, np.integer)):
        return f"{x:>6d}"
    return f"{x:>6.1f}"


def build_rows(root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    agg_rows, type_rows = [], []
    for bench in BENCHMARKS:
        for mode in MODES:
            for preset in PRESETS:
                df = _load_cell(root, bench, mode, preset)
                if df is None:
                    continue
                judge = JUDGE[bench]
                agg = {
                    "condition": f"zero_shot_{mode.upper()}",
                    "preset": preset,
                    "benchmark": bench,
                    "judge": judge,
                }
                agg.update(_aggregate(df))
                agg_rows.append(agg)
                for atype, g in df.groupby("answer_type", dropna=False):
                    row = {
                        "condition": f"zero_shot_{mode.upper()}",
                        "preset": preset,
                        "benchmark": bench,
                        "judge": judge,
                        "answer_type": str(atype) if pd.notna(atype) else "None",
                    }
                    row.update(_aggregate(g))
                    type_rows.append(row)

    agg_df = pd.DataFrame(agg_rows)
    type_df = pd.DataFrame(type_rows)

    col_order = [
        "condition", "preset", "benchmark", "judge", "n",
        "n_called", "n_skipped", "call_pct", "skip_pct",
        "call_pass_pct", "skip_pass_pct", "overall_pass_pct", "call_minus_skip_pass",
        "eed_mean", "call_eed_mean", "skip_eed_mean",
    ]
    for c in col_order:
        if c not in agg_df.columns:
            agg_df[c] = np.nan
        if c not in type_df.columns:
            type_df[c] = np.nan
    agg_df = agg_df[col_order]
    type_col_order = col_order[:4] + ["answer_type"] + col_order[4:]
    type_df = type_df[type_col_order]
    return agg_df, type_df


def format_summary(agg_df: pd.DataFrame, type_df: pd.DataFrame) -> str:
    lines: list[str] = []
    lines.append("# Task 1 — Call vs Skip conditional pass@1 (zero-shot, Qwen3-4B-Thinking-2507)")
    lines.append("")
    lines.append("Notes:")
    lines.append("- Zero-shot rollout.n = 1 per problem (verified).")
    lines.append("- 'called_tool' = rollouts that emitted a <tool_call> with parseable code (matches analyze_tool_use semantics).")
    lines.append("- For PHYBench, eed_mean is the paper's headline metric; call_pass_pct/skip_pass_pct are exact-match.")
    lines.append("")

    for preset in PRESETS:
        lines.append(f"## Preset: {preset}  (train = RL-matching, qwen = Qwen-team-recommended)")
        lines.append("")
        sub = agg_df[agg_df["preset"] == preset].copy()
        if sub.empty:
            continue
        sub = sub.sort_values(["benchmark", "condition"]).reset_index(drop=True)
        header = (
            f"{'benchmark':<24} {'cond':<15} "
            f"{'n':>4}  {'call%':>6} {'skip%':>6} "
            f"{'call_pass%':>10} {'skip_pass%':>10} {'overall%':>8} "
            f"{'C-S':>6}  {'eed_mean':>8} {'call_eed':>8} {'skip_eed':>8}  {'judge':<22}"
        )
        lines.append(header)
        lines.append("-" * len(header))
        for _, r in sub.iterrows():
            lines.append(
                f"{r['benchmark']:<24} {r['condition']:<15} "
                f"{int(r['n']):>4}  {_pretty_float(r['call_pct'])} {_pretty_float(r['skip_pct'])} "
                f"{_pretty_float(r['call_pass_pct']):>10} {_pretty_float(r['skip_pass_pct']):>10} "
                f"{_pretty_float(r['overall_pass_pct']):>8} "
                f"{_pretty_float(r['call_minus_skip_pass']):>6}  "
                f"{_pretty_float(r['eed_mean']):>8} {_pretty_float(r['call_eed_mean']):>8} "
                f"{_pretty_float(r['skip_eed_mean']):>8}  {r['judge']:<22}"
            )
        lines.append("")

    lines.append("## Per-type zoom-in (in-dist pool_v2, TIR, train preset)")
    lines.append("")
    sub = type_df[
        (type_df["preset"] == "train")
        & (type_df["condition"] == "zero_shot_TIR")
        & (type_df["benchmark"].str.startswith("pool_v2_"))
    ].copy()
    sub = sub.sort_values(["benchmark", "answer_type"])
    header = (
        f"{'benchmark':<20} {'answer_type':<60} "
        f"{'n':>4} {'call%':>6} {'call_pass%':>10} {'skip_pass%':>10} "
        f"{'C-S':>6} {'overall%':>8}"
    )
    lines.append(header)
    lines.append("-" * len(header))
    for _, r in sub.iterrows():
        lines.append(
            f"{r['benchmark']:<20} {str(r['answer_type'])[:60]:<60} "
            f"{int(r['n']):>4} {_pretty_float(r['call_pct'])} "
            f"{_pretty_float(r['call_pass_pct']):>10} {_pretty_float(r['skip_pass_pct']):>10} "
            f"{_pretty_float(r['call_minus_skip_pass']):>6} "
            f"{_pretty_float(r['overall_pass_pct']):>8}"
        )
    lines.append("")
    return "\n".join(lines)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--outputs-eval", default="outputs/eval")
    p.add_argument("--out-agg", default="outputs/eval/call_vs_skip_by_benchmark.csv")
    p.add_argument("--out-type", default="outputs/eval/call_vs_skip_by_benchmark_type.csv")
    p.add_argument("--out-summary", default="outputs/eval/call_vs_skip_summary.txt")
    args = p.parse_args()

    root = Path(args.outputs_eval)
    agg_df, type_df = build_rows(root)

    for p_ in (args.out_agg, args.out_type, args.out_summary):
        Path(p_).parent.mkdir(parents=True, exist_ok=True)
    agg_df.to_csv(args.out_agg, index=False)
    type_df.to_csv(args.out_type, index=False)
    summary = format_summary(agg_df, type_df)
    Path(args.out_summary).write_text(summary)
    print(summary)
    print(f"\nWrote {args.out_agg}")
    print(f"Wrote {args.out_type}")
    print(f"Wrote {args.out_summary}")


if __name__ == "__main__":
    main()
