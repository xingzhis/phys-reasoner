"""Score a rollouts.parquet with the official OpenBMB AutoScoringJudge.

Vendored judge in `_olympiad_judge.py` (kept verbatim from OpenBMB/OlympiadBench
main branch as of 2026-04-19). This wrapper only handles the I/O contract:

- input  : rollouts.parquet (one row per rollout, raw phase texts)
- output : scored.parquet (verdict + correct bool per row) and scored.summary.txt

`gold_answer` and `extra_info.precision` come from the OlympiadBench loader and
are passed through to AutoScoringJudge.judge(GT, prediction, precision). The
judge itself does the \\boxed{} extraction from both gold and prediction strings,
splits on commas, handles \\pm expansion, and runs interval/numerical/expression/
equation equivalence checks in that order.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any

# The vendored AutoScoringJudge calls sympy.parsing.latex.parse_latex, which needs
# antlr4-python3-runtime >= 4.11. The SIF base ships 4.9.3, so we keep a 4.11
# wheel in eval/_pkgs/ (installed via `pip install --no-deps --target eval/_pkgs
# antlr4-python3-runtime==4.11`) and prepend it here before any sympy LaTeX call.
_PKGS = Path(__file__).resolve().parents[1] / "_pkgs"
if _PKGS.is_dir() and str(_PKGS) not in sys.path:
    sys.path.insert(0, str(_PKGS))


def _concat_pred_text(row) -> str:
    def _safe(v) -> str:
        if v is None:
            return ""
        if isinstance(v, float):  # NaN
            return ""
        return str(v)
    return _safe(row["phase1_text"]) + _safe(row["phase1b_text"]) + _safe(row["phase2_text"])


def _normalize_precision(p: Any) -> float | list[float]:
    """Pandas may round-trip a list as a numpy ndarray; AutoScoringJudge wants
    a Python list (it pops items)."""
    if p is None:
        return 1e-8
    if isinstance(p, (list, tuple)):
        return [float(x) for x in p]
    try:
        # numpy arrays
        return [float(x) for x in p]  # type: ignore[arg-type]
    except TypeError:
        try:
            return float(p)
        except (TypeError, ValueError):
            return 1e-8


def score_rollouts(rollouts_path: str, out_path: str) -> None:
    import pandas as pd  # noqa: PLC0415

    from eval.scoring._olympiad_judge import AutoScoringJudge  # noqa: PLC0415

    judge = AutoScoringJudge()
    df = pd.read_parquet(rollouts_path)
    n = len(df)
    print(f"Loaded {n} rollouts from {rollouts_path}")

    counts = {"correct": 0, "wrong": 0, "judge_error": 0}
    records: list[dict] = []
    for i, row in df.iterrows():
        pred_text = _concat_pred_text(row)
        gold = str(row["gold_answer"])
        ei = row.get("extra_info") or {}
        if not isinstance(ei, dict):
            try:
                ei = dict(ei)
            except Exception:
                ei = {}
        precision = _normalize_precision(ei.get("precision"))

        try:
            ok = bool(judge.judge(gold, pred_text, precision))
            err = ""
        except Exception as e:
            ok = False
            err = f"{type(e).__name__}: {e}"
            counts["judge_error"] += 1

        if ok:
            counts["correct"] += 1
        elif not err:
            counts["wrong"] += 1

        records.append({
            "problem_idx": row["problem_idx"],
            "rollout_idx": row["rollout_idx"],
            "gold_answer": gold,
            "answer_type": ei.get("answer_type", "unknown"),
            "unit": ei.get("unit", ""),
            "is_multiple_answer": bool(ei.get("is_multiple_answer", False)),
            "subfield": ei.get("subfield", ""),
            "pred_text_len": len(pred_text),
            "verdict": 1.0 if ok else 0.0,
            "correct": ok,
            "judge_error": err,
        })

        if (i + 1) % 50 == 0 or (i + 1) == n:
            print(f"  scored {i + 1}/{n}  (correct={counts['correct']}, "
                  f"wrong={counts['wrong']}, judge_err={counts['judge_error']})")

    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    pd.DataFrame(records).to_parquet(out_path, index=False)

    pass_rate = counts["correct"] / n if n else float("nan")
    summary_path = os.path.splitext(out_path)[0] + ".summary.txt"
    with open(summary_path, "w") as f:
        f.write(f"rollouts     : {rollouts_path}\n")
        f.write(f"scorer       : OlympiadBench AutoScoringJudge (official, vendored)\n")
        f.write(f"n            : {n}\n")
        f.write(f"correct      : {counts['correct']}\n")
        f.write(f"wrong        : {counts['wrong']}\n")
        f.write(f"judge_error  : {counts['judge_error']}\n")
        f.write(f"pass@1       : {pass_rate:.4f}\n")

    print(f"\nSaved → {out_path}")
    print(f"Summary → {summary_path}")
    print(f"pass@1={pass_rate:.4f}  (correct={counts['correct']}, wrong={counts['wrong']}, "
          f"judge_err={counts['judge_error']})")


def main() -> None:
    p = argparse.ArgumentParser(description="Score OlympiadBench rollouts with the official judge")
    p.add_argument("--rollouts", required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args()
    score_rollouts(args.rollouts, args.out)


if __name__ == "__main__":
    main()
