"""Task 1b — Per-problem paired TIR-mode vs CoT-mode comparison.

For each problem_id evaluated in both zero-shot TIR and zero-shot CoT
(same model, same preset), classify as:
    Win     : TIR correct AND CoT wrong
    Loss    : TIR wrong    AND CoT correct
    Tie-pass: both correct
    Tie-fail: both wrong

Outputs (paths configurable):
  - outputs/eval/paired_tir_vs_cot_by_benchmark.csv
  - outputs/eval/paired_tir_vs_cot_by_type.csv
  - outputs/eval/paired_tir_vs_cot_summary.txt
  - outputs/eval/paired_tir_vs_cot_by_benchmark_permid.csv (ABench-B only)

Zero-shot rollout.n = 1 per problem (verified), so no per-problem aggregation
is needed. If the same framework is re-run on multi-rollout parquets, aggregate
to problem-level pass@1 first (majority vote on `correct`, then re-classify).

Judge column values: xverify-7b (pool_v2), olympiad-math-verify (olympiad),
phybench-exact (phybench), abench-tol1pct (abench_phy_*).
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


def _load_correct(root: Path, benchmark: str, mode: str, preset: str) -> pd.DataFrame | None:
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

    out = pd.DataFrame(index=r.index)
    out["correct"] = s["correct"].astype(bool)
    out["answer_type"] = r["answer_type"].astype(str) if "answer_type" in r.columns else "unknown"
    out["gold_answer"] = r["gold_answer"] if "gold_answer" in r.columns else None
    out["called_tool"] = r["code"].apply(
        lambda c: c is not None and not isinstance(c, float)
    )
    if benchmark == "phybench" and "eed_score" in s.columns:
        out["eed_score"] = s["eed_score"].astype(float)
    if benchmark == "abench_phy_b" and {"mid", "subid"}.issubset(s.columns):
        out["mid"] = s["mid"]
        out["subid"] = s["subid"]
    return out


def _classify(tir_correct: bool, cot_correct: bool) -> str:
    if tir_correct and not cot_correct:
        return "win"
    if cot_correct and not tir_correct:
        return "loss"
    if tir_correct and cot_correct:
        return "tie_pass"
    return "tie_fail"


def _pair_problems(tir: pd.DataFrame, cot: pd.DataFrame, include_eed: bool) -> pd.DataFrame:
    """Join TIR and CoT rows on problem_idx (one row per problem, zero-shot)."""
    tir = tir.reset_index()
    cot = cot.reset_index()
    # Reduce to rollout_idx=0 (the only rollout zero-shot) — defensive against multi-rollout inputs.
    tir = tir.sort_values(["problem_idx", "rollout_idx"]).drop_duplicates("problem_idx", keep="first")
    cot = cot.sort_values(["problem_idx", "rollout_idx"]).drop_duplicates("problem_idx", keep="first")

    cols_tir = ["problem_idx", "answer_type", "correct", "called_tool"]
    cols_cot = ["problem_idx", "correct"]
    if include_eed:
        cols_tir.append("eed_score")
        cols_cot.append("eed_score")
    if "mid" in tir.columns:
        cols_tir += ["mid", "subid"]

    merged = tir[cols_tir].merge(
        cot[cols_cot].rename(columns={"correct": "cot_correct", "eed_score": "cot_eed_score"})
        if include_eed else cot[cols_cot].rename(columns={"correct": "cot_correct"}),
        on="problem_idx", how="inner",
    )
    merged = merged.rename(columns={
        "correct": "tir_correct",
        "called_tool": "tir_called_tool",
    })
    if include_eed:
        merged = merged.rename(columns={"eed_score": "tir_eed_score"})
    merged["outcome"] = [
        _classify(t, c) for t, c in zip(merged["tir_correct"], merged["cot_correct"])
    ]
    return merged


def _pct(x, n) -> float:
    return (100.0 * x / n) if n else float("nan")


def _summarize(merged: pd.DataFrame, include_eed: bool, extras: dict | None = None) -> dict:
    n = len(merged)
    counts = merged["outcome"].value_counts().to_dict()
    n_win = counts.get("win", 0)
    n_loss = counts.get("loss", 0)
    n_tp = counts.get("tie_pass", 0)
    n_tf = counts.get("tie_fail", 0)
    row = {
        "n_paired": n,
        "n_win": int(n_win),
        "n_loss": int(n_loss),
        "n_tie_pass": int(n_tp),
        "n_tie_fail": int(n_tf),
        "win_rate": _pct(n_win, n),
        "loss_rate": _pct(n_loss, n),
        "tie_pass_rate": _pct(n_tp, n),
        "tie_fail_rate": _pct(n_tf, n),
        "win_minus_loss": (_pct(n_win, n) - _pct(n_loss, n)) if n else float("nan"),
        "tir_pass_pct": _pct(int(merged["tir_correct"].sum()), n),
        "cot_pass_pct": _pct(int(merged["cot_correct"].sum()), n),
        "tir_call_pct": _pct(int(merged["tir_called_tool"].sum()), n),
    }
    if include_eed:
        row["tir_eed_mean"] = float(merged["tir_eed_score"].mean()) if n else float("nan")
        row["cot_eed_mean"] = float(merged["cot_eed_score"].mean()) if n else float("nan")
        row["eed_tir_minus_cot_mean"] = (
            float((merged["tir_eed_score"] - merged["cot_eed_score"]).mean()) if n else float("nan")
        )
    if extras:
        row.update(extras)
    return row


def _per_mid_collapse(merged: pd.DataFrame) -> pd.DataFrame:
    """For ABench-B: collapse 4 subids per mid to a single 'per-mid pass' = all-4-correct."""
    def _agg(g):
        return pd.Series({
            "tir_correct": bool(g["tir_correct"].all()),
            "cot_correct": bool(g["cot_correct"].all()),
            "tir_called_tool": bool(g["tir_called_tool"].any()),  # any of 4 called → "called"
            "n_subids": int(len(g)),
            "answer_type": g["answer_type"].iloc[0],
        })
    per_mid = merged.groupby("mid", dropna=False).apply(_agg).reset_index()
    per_mid["outcome"] = [
        _classify(t, c) for t, c in zip(per_mid["tir_correct"], per_mid["cot_correct"])
    ]
    return per_mid


def build_tables(root: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    agg_rows, type_rows, permid_rows = [], [], []
    for bench in BENCHMARKS:
        include_eed = (bench == "phybench")
        judge = JUDGE[bench]
        for preset in PRESETS:
            tir = _load_correct(root, bench, "tir", preset)
            cot = _load_correct(root, bench, "cot", preset)
            if tir is None or cot is None:
                continue

            n_tir = len(tir.reset_index().drop_duplicates("problem_idx"))
            n_cot = len(cot.reset_index().drop_duplicates("problem_idx"))
            merged = _pair_problems(tir, cot, include_eed=include_eed)
            n_paired = len(merged)

            extras = {
                "n_tir_only": n_tir - n_paired,
                "n_cot_only": n_cot - n_paired,
            }
            agg_row = {
                "benchmark": bench,
                "preset": preset,
                "judge": judge,
            }
            agg_row.update(_summarize(merged, include_eed=include_eed, extras=extras))
            agg_rows.append(agg_row)

            for atype, g in merged.groupby("answer_type", dropna=False):
                row = {
                    "benchmark": bench,
                    "preset": preset,
                    "judge": judge,
                    "answer_type": str(atype) if pd.notna(atype) else "None",
                }
                row.update(_summarize(g, include_eed=include_eed))
                type_rows.append(row)

            if bench == "abench_phy_b" and "mid" in merged.columns:
                per_mid = _per_mid_collapse(merged)
                pm_row = {
                    "benchmark": "abench_phy_b_permid",
                    "preset": preset,
                    "judge": judge,
                    "per_mid_pass_def": "all-4-subid-correct",
                    "per_mid_call_def": "any-of-4-subid-called",
                }
                pm_row.update(_summarize(per_mid, include_eed=False))
                permid_rows.append(pm_row)

    agg_df = pd.DataFrame(agg_rows)
    type_df = pd.DataFrame(type_rows)
    permid_df = pd.DataFrame(permid_rows)

    base_cols = [
        "benchmark", "preset", "judge",
        "n_paired", "n_tir_only", "n_cot_only",
        "n_win", "n_loss", "n_tie_pass", "n_tie_fail",
        "win_rate", "loss_rate", "tie_pass_rate", "tie_fail_rate", "win_minus_loss",
        "tir_pass_pct", "cot_pass_pct", "tir_call_pct",
        "tir_eed_mean", "cot_eed_mean", "eed_tir_minus_cot_mean",
    ]
    for c in base_cols:
        if c not in agg_df.columns:
            agg_df[c] = np.nan
    agg_df = agg_df[base_cols]

    type_cols = base_cols[:3] + ["answer_type"] + base_cols[3:]
    # type_df lacks n_tir_only / n_cot_only; keep NaN placeholder
    for c in type_cols:
        if c not in type_df.columns:
            type_df[c] = np.nan
    type_df = type_df[type_cols]

    if not permid_df.empty:
        permid_cols = ["benchmark", "preset", "judge", "per_mid_pass_def", "per_mid_call_def"] + [
            c for c in base_cols[3:] if c in permid_df.columns and c not in ("tir_eed_mean", "cot_eed_mean", "eed_tir_minus_cot_mean")
        ]
        permid_df = permid_df.reindex(columns=permid_cols)
    return agg_df, type_df, permid_df


def _pf(x) -> str:
    if isinstance(x, float) and np.isnan(x):
        return "   nan"
    if isinstance(x, (int, np.integer)):
        return f"{x:>6d}"
    return f"{x:>6.1f}"


def format_summary(agg_df: pd.DataFrame, type_df: pd.DataFrame, permid_df: pd.DataFrame) -> str:
    lines: list[str] = []
    lines.append("# Task 1b — Paired TIR-mode vs CoT-mode (zero-shot, Qwen3-4B-Thinking-2507)")
    lines.append("")
    lines.append("Win = TIR correct AND CoT wrong; Loss = CoT correct AND TIR wrong;")
    lines.append("Tie-pass = both correct; Tie-fail = both wrong.")
    lines.append("")
    lines.append("Zero-shot rollout.n = 1 per problem (verified).")
    lines.append("Join key: problem_idx (same benchmark dataset across TIR/CoT cells).")
    lines.append("")
    lines.append("For PHYBench, eed_mean columns are included (paper-headline metric);")
    lines.append("win/loss on PHYBench use binary exact-match (phybench-official EED==100).")
    lines.append("")

    for preset in PRESETS:
        sub = agg_df[agg_df["preset"] == preset]
        if sub.empty:
            continue
        lines.append(f"## Preset: {preset}")
        lines.append("")
        header = (
            f"{'benchmark':<24} {'n_pair':>6} "
            f"{'W':>4} {'L':>4} {'TP':>4} {'TF':>4} "
            f"{'win%':>5} {'loss%':>5} {'W-L':>5} "
            f"{'TIR%':>5} {'CoT%':>5} {'tir_call%':>9} "
            f"{'tir_eed':>7} {'cot_eed':>7} {'Δeed':>6}  {'judge':<22}"
        )
        lines.append(header)
        lines.append("-" * len(header))
        for _, r in sub.sort_values("benchmark").iterrows():
            lines.append(
                f"{r['benchmark']:<24} {int(r['n_paired']):>6} "
                f"{int(r['n_win']):>4} {int(r['n_loss']):>4} "
                f"{int(r['n_tie_pass']):>4} {int(r['n_tie_fail']):>4} "
                f"{_pf(r['win_rate'])} {_pf(r['loss_rate'])} {_pf(r['win_minus_loss'])} "
                f"{_pf(r['tir_pass_pct'])} {_pf(r['cot_pass_pct'])} {_pf(r['tir_call_pct']):>9} "
                f"{_pf(r['tir_eed_mean']):>7} {_pf(r['cot_eed_mean']):>7} "
                f"{_pf(r['eed_tir_minus_cot_mean']):>6}  {r['judge']:<22}"
            )
        lines.append("")

    lines.append("## Per-answer-type (train preset, in-dist pool_v2)")
    lines.append("")
    sub = type_df[
        (type_df["preset"] == "train")
        & (type_df["benchmark"].str.startswith("pool_v2_"))
    ]
    header = (
        f"{'benchmark':<20} {'answer_type':<50} {'n_pair':>6} "
        f"{'W':>4} {'L':>4} {'TP':>4} {'TF':>4} "
        f"{'win%':>5} {'loss%':>5} {'W-L':>5} {'tir%':>5} {'cot%':>5}"
    )
    lines.append(header)
    lines.append("-" * len(header))
    for _, r in sub.sort_values(["benchmark", "answer_type"]).iterrows():
        lines.append(
            f"{r['benchmark']:<20} {str(r['answer_type'])[:50]:<50} "
            f"{int(r['n_paired']):>6} "
            f"{int(r['n_win']):>4} {int(r['n_loss']):>4} "
            f"{int(r['n_tie_pass']):>4} {int(r['n_tie_fail']):>4} "
            f"{_pf(r['win_rate'])} {_pf(r['loss_rate'])} "
            f"{_pf(r['win_minus_loss'])} {_pf(r['tir_pass_pct'])} {_pf(r['cot_pass_pct'])}"
        )
    lines.append("")

    if not permid_df.empty:
        lines.append("## ABench-B per-mid (all-4-subid-correct as pass; any-of-4 called as 'called')")
        lines.append("")
        for preset in PRESETS:
            sub = permid_df[permid_df["preset"] == preset]
            if sub.empty:
                continue
            lines.append(f"### Preset: {preset}")
            for _, r in sub.iterrows():
                lines.append(
                    f"  n_mids={int(r['n_paired']):>3}  "
                    f"W={int(r['n_win']):>3}  L={int(r['n_loss']):>3}  "
                    f"TP={int(r['n_tie_pass']):>3}  TF={int(r['n_tie_fail']):>3}  "
                    f"win%={_pf(r['win_rate'])}  loss%={_pf(r['loss_rate'])}  "
                    f"W-L={_pf(r['win_minus_loss'])}  "
                    f"TIR%={_pf(r['tir_pass_pct'])}  CoT%={_pf(r['cot_pass_pct'])}"
                )
            lines.append("")
    return "\n".join(lines)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--outputs-eval", default="outputs/eval")
    p.add_argument("--out-agg", default="outputs/eval/paired_tir_vs_cot_by_benchmark.csv")
    p.add_argument("--out-type", default="outputs/eval/paired_tir_vs_cot_by_type.csv")
    p.add_argument("--out-permid", default="outputs/eval/paired_tir_vs_cot_by_benchmark_permid.csv")
    p.add_argument("--out-summary", default="outputs/eval/paired_tir_vs_cot_summary.txt")
    args = p.parse_args()

    root = Path(args.outputs_eval)
    agg_df, type_df, permid_df = build_tables(root)

    for path in (args.out_agg, args.out_type, args.out_permid, args.out_summary):
        Path(path).parent.mkdir(parents=True, exist_ok=True)

    agg_df.to_csv(args.out_agg, index=False)
    type_df.to_csv(args.out_type, index=False)
    if not permid_df.empty:
        permid_df.to_csv(args.out_permid, index=False)
    summary = format_summary(agg_df, type_df, permid_df)
    Path(args.out_summary).write_text(summary)
    print(summary)
    print(f"\nWrote {args.out_agg}")
    print(f"Wrote {args.out_type}")
    if not permid_df.empty:
        print(f"Wrote {args.out_permid}")
    print(f"Wrote {args.out_summary}")


if __name__ == "__main__":
    main()
