#!/usr/bin/env python3
"""Aggregate probe rollouts into per-problem stats + honest-metric summary.

Reads the scored rollouts parquet (one row per rollout) and emits:
  1. probe_per_problem.parquet  — one row per problem with n_correct, pass_fraction,
     stratification keys. This is the direct input for Strategy B weight computation.
  2. probe_summary.txt          — human-readable breakdown with:
        - overall correctness rate (= unbiased pass@1) and pass@8
        - per-problem pass-fraction histogram (0/8..8/8)
        - pass@k unbiased curve
        - stratified breakdowns (answer_type, difficulty bin, from)
        - Goldilocks zone fractions (pass ∈ [goldilocks_lo/n, goldilocks_hi/n])
        - MCQ sanity check (guessing signal at ~25% random baseline)

pass@k unbiased estimator (Chen et al. 2021):
    pass@k = 1 - C(n-c, k) / C(n, k)   if n-c >= k else 1
where n is total rollouts and c the correct count.
"""
from __future__ import annotations

import argparse
import json
from math import comb
from pathlib import Path

import numpy as np
import pandas as pd


SINGLE_TYPES = ("numerical", "expression", "equation", "mcq", "true_false", "interval")


def _safe_difficulty(x):
    if not isinstance(x, dict):
        return None
    v = x.get("difficulty")
    if v in (None, ""):
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None  # corpus uses text labels like "Undergraduate/Postgraduate"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--input",
        type=Path,
        default=Path("outputs/probe_v5_scored/rollouts_scored.parquet"),
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/probe_v5_scored"),
    )
    p.add_argument("--goldilocks-lo", type=int, default=1,
                   help="min correct rollouts (inclusive) to count as Goldilocks")
    p.add_argument("--goldilocks-hi", type=int, default=7,
                   help="max correct rollouts (inclusive) to count as Goldilocks")
    return p.parse_args()


def difficulty_bin(d: float | None) -> str:
    if d is None or (isinstance(d, float) and np.isnan(d)):
        return "unknown"
    if d < 0.2:
        return "0.0-0.2"
    if d < 0.4:
        return "0.2-0.4"
    if d < 0.6:
        return "0.4-0.6"
    if d < 0.8:
        return "0.6-0.8"
    return "0.8-1.0"


def _pass_at_k(n: int, c: int, k: int) -> float:
    if n - c < k:
        return 1.0
    return 1.0 - comb(n - c, k) / comb(n, k)


def pass_at_k_series(correct_by_problem: np.ndarray, n: int, k: int) -> float:
    """Unbiased pass@k across problems; each problem had n rollouts, c correct."""
    return float(np.mean([_pass_at_k(n, int(c), k) for c in correct_by_problem]))


def fmt_line(label: str, n_prob: int, n_roll: int,
             corr_rate: float, pass8: float,
             gold_frac: float, too_easy_frac: float,
             too_hard_frac: float) -> str:
    return (
        f"  {label:<28s} n_prob={n_prob:<5d} n_roll={n_roll:<6d} "
        f"corr={corr_rate*100:5.1f}%  pass@8={pass8*100:5.1f}%  "
        f"goldilocks={gold_frac*100:5.1f}%  "
        f"too_easy={too_easy_frac*100:5.1f}%  too_hard={too_hard_frac*100:5.1f}%"
    )


def build_per_problem(df: pd.DataFrame) -> pd.DataFrame:
    ei = df["extra_info"]
    df = df.assign(
        _from=ei.apply(lambda x: x.get("from") if isinstance(x, dict) else None),
        _source=ei.apply(lambda x: x.get("source") if isinstance(x, dict) else None),
        _difficulty=ei.apply(_safe_difficulty),
        _domain_coarse=ei.apply(
            lambda x: x.get("domain_coarse") if isinstance(x, dict) else None
        ),
    )
    df["_is_multi"] = df["answer_type"].astype(str).str.startswith("[")
    df["_single_type"] = np.where(df["_is_multi"], "multi-part", df["answer_type"])
    df["_diff_bin"] = df["_difficulty"].apply(difficulty_bin)
    df["_dataset"] = np.where(df["_from"].notna(), "drsci", "corpus")

    agg = (
        df.groupby("problem_idx", as_index=False)
        .agg(
            n_rollouts=("score", "size"),
            n_correct=("score", "sum"),
            answer_type=("answer_type", "first"),
            single_type=("_single_type", "first"),
            is_multi=("_is_multi", "first"),
            difficulty=("_difficulty", "first"),
            diff_bin=("_diff_bin", "first"),
            from_=("_from", "first"),
            source=("_source", "first"),
            domain_coarse=("_domain_coarse", "first"),
            dataset=("_dataset", "first"),
            gold_answer=("gold_answer", "first"),
        )
    )
    agg["n_correct"] = agg["n_correct"].astype(int)
    agg["pass_fraction"] = agg["n_correct"] / agg["n_rollouts"]
    return agg


def section(title: str) -> str:
    return f"\n=== {title} ===\n"


def breakdown_block(title: str, pp: pd.DataFrame, key: str,
                    gold_lo: int, gold_hi: int,
                    order: list[str] | None = None) -> str:
    lines = [section(title)]
    groups = pp.groupby(key, dropna=False)
    keys = list(order) if order is not None else sorted(
        [str(k) for k in groups.groups.keys()], key=lambda s: (s == "unknown", s)
    )
    for k in keys:
        if k not in groups.groups and k != "unknown":
            continue
        if k == "unknown" and k not in groups.groups:
            continue
        g = groups.get_group(k)
        n_prob = len(g)
        n_roll = int(g["n_rollouts"].sum())
        corr = g["n_correct"].sum() / n_roll if n_roll else 0.0
        pass8 = float((g["n_correct"] >= 1).mean())
        gold = float(((g["n_correct"] >= gold_lo) & (g["n_correct"] <= gold_hi)).mean())
        too_easy = float((g["n_correct"] > gold_hi).mean())
        too_hard = float((g["n_correct"] < gold_lo).mean())
        lines.append(fmt_line(str(k), n_prob, n_roll, corr, pass8, gold, too_easy, too_hard))
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(args.input)
    n_total_rollouts = len(df)

    pp = build_per_problem(df)
    out_parquet = args.output_dir / "probe_per_problem.parquet"
    pp.to_parquet(out_parquet, index=False)

    n_problems = len(pp)
    n = int(pp["n_rollouts"].iloc[0])
    assert (pp["n_rollouts"] == n).all(), "non-uniform rollout count per problem"

    # --- overall metrics
    total_correct = int(pp["n_correct"].sum())
    correctness_rate = total_correct / n_total_rollouts
    pass_at_8 = float((pp["n_correct"] >= 1).mean())
    mean_pass_fraction = float(pp["pass_fraction"].mean())

    # --- pass-fraction histogram (0..n)
    hist_counts = np.bincount(pp["n_correct"].to_numpy(), minlength=n + 1)

    # --- pass@k unbiased curve
    correct_arr = pp["n_correct"].to_numpy()
    ks = [1, 2, 4, 8] if n >= 8 else list({1, 2, 4, n})
    pass_at_k = {k: pass_at_k_series(correct_arr, n, k) for k in ks}

    # --- Goldilocks zone
    lo, hi = args.goldilocks_lo, args.goldilocks_hi
    gold_mask = (pp["n_correct"] >= lo) & (pp["n_correct"] <= hi)
    gold_frac = float(gold_mask.mean())
    too_easy_frac = float((pp["n_correct"] > hi).mean())
    too_hard_frac = float((pp["n_correct"] < lo).mean())

    # --- write text summary
    out_txt = args.output_dir / "probe_summary.txt"
    lines: list[str] = []
    lines.append(f"Probe summary — {n_problems} problems × {n} rollouts = {n_total_rollouts}")
    lines.append(f"Source: {args.input}")

    lines.append(section("HEADLINE METRICS"))
    lines.append(f"  correctness rate (pass@1, unbiased)  : {correctness_rate*100:.2f}%  "
                 f"[{total_correct}/{n_total_rollouts} rollouts]")
    lines.append(f"  pass@8 (any correct in 8)            : {pass_at_8*100:.2f}%  "
                 f"[{int((pp['n_correct']>=1).sum())}/{n_problems} problems]")
    lines.append(f"  mean per-problem pass fraction       : {mean_pass_fraction*100:.2f}%")

    lines.append(section("PASS@k UNBIASED CURVE"))
    for k in ks:
        lines.append(f"  pass@{k:<2d}                               : {pass_at_k[k]*100:.2f}%")

    lines.append(section("PER-PROBLEM PASS-FRACTION HISTOGRAM"))
    lines.append(f"  bin   c/n      n_problems   frac     zone")
    for c, cnt in enumerate(hist_counts):
        frac = cnt / n_problems
        if c == 0:
            zone = "too-hard"
        elif c == n:
            zone = "too-easy"
        elif lo <= c <= hi:
            zone = "GOLDILOCKS"
        else:
            zone = "-"
        lines.append(f"  {c}/{n}            {cnt:6d}    {frac*100:5.1f}%   {zone}")

    lines.append(section("GOLDILOCKS ZONE (Strategy B input)"))
    lines.append(f"  zone definition     : n_correct in [{lo}, {hi}]  "
                 f"(pass fraction in [{lo/n:.3f}, {hi/n:.3f}])")
    lines.append(f"  Goldilocks fraction : {gold_frac*100:.2f}%  "
                 f"[{int(gold_mask.sum())}/{n_problems} problems]")
    lines.append(f"  too-easy  (>{hi}/{n})    : {too_easy_frac*100:.2f}%")
    lines.append(f"  too-hard  (<{lo}/{n})    : {too_hard_frac*100:.2f}%")

    # --- stratified breakdowns
    type_order = list(SINGLE_TYPES) + ["multi-part"]
    lines.append(breakdown_block(
        "BY ANSWER TYPE (single types + multi-part combined)",
        pp, "single_type", lo, hi, order=type_order))

    diff_order = ["0.0-0.2", "0.2-0.4", "0.4-0.6", "0.6-0.8", "0.8-1.0", "unknown"]
    lines.append(breakdown_block(
        "BY DIFFICULTY (0.0 = hardest per Dr. SCI Qwen3-32B proxy; 1.0 = easiest)",
        pp, "diff_bin", lo, hi, order=diff_order))

    lines.append(breakdown_block(
        "BY DATASET",
        pp, "dataset", lo, hi, order=["drsci", "corpus"]))

    drsci = pp[pp["dataset"] == "drsci"]
    if len(drsci):
        lines.append(breakdown_block(
            "BY DR. SCI SOURCE (`from`)",
            drsci, "from_", lo, hi,
            order=sorted(drsci["from_"].dropna().unique().tolist())))

    corpus = pp[pp["dataset"] == "corpus"]
    if len(corpus) and corpus["source"].notna().any():
        lines.append(breakdown_block(
            "BY CORPUS SOURCE",
            corpus, "source", lo, hi,
            order=sorted(corpus["source"].dropna().unique().tolist())))

    # --- MCQ sanity check
    mcq = pp[pp["single_type"] == "mcq"]
    if len(mcq):
        lines.append(section("MCQ GUESSING SANITY CHECK"))
        lines.append(f"  MCQ problems: {len(mcq)}   (random-guess baseline for 4-option MCQ: pass@1 = 25%)")
        mcq_hist = np.bincount(mcq["n_correct"].to_numpy(), minlength=n + 1)
        lines.append(f"  Pass-fraction histogram (a random guesser would peak near {int(round(0.25*n))}/{n}):")
        for c, cnt in enumerate(mcq_hist):
            marker = " <-- random-guess mode" if c == int(round(0.25 * n)) else ""
            lines.append(f"    {c}/{n}   {cnt:4d}  ({cnt/len(mcq)*100:5.1f}%){marker}")
        mcq_corr = mcq["n_correct"].sum() / (len(mcq) * n)
        lines.append(f"  MCQ correctness rate: {mcq_corr*100:.2f}%   (vs 25% pure-guess baseline)")

    # --- SFT decision (unchanged gate)
    lines.append(section("SFT DECISION (CLAUDE.md: skip if zero-shot TIR correctness rate >= 15%)"))
    lines.append(f"  correctness rate: {correctness_rate*100:.2f}%  "
                 f"-> {'SKIP SFT' if correctness_rate >= 0.15 else 'RUN SFT'}")

    lines.append("")
    out_txt.write_text("\n".join(lines))

    # tiny JSON sidecar for downstream programmatic reads
    (args.output_dir / "probe_summary.json").write_text(json.dumps({
        "n_problems": n_problems,
        "n_rollouts_per_problem": n,
        "n_total_rollouts": n_total_rollouts,
        "correctness_rate": correctness_rate,
        "pass_at_8": pass_at_8,
        "pass_at_k": pass_at_k,
        "goldilocks": {"lo": lo, "hi": hi, "fraction": gold_frac,
                       "too_easy": too_easy_frac, "too_hard": too_hard_frac},
        "hist_counts": hist_counts.tolist(),
    }, indent=2))

    print(f"wrote {out_parquet}")
    print(f"wrote {out_txt}")
    print(f"wrote {args.output_dir / 'probe_summary.json'}")


if __name__ == "__main__":
    main()
