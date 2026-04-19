"""Score a rollouts.parquet with the phybench-official EED (Expression Edit
Distance) scorer.

The vendored phybench EED code is at eval/scoring/_phybench/ — it's kept
verbatim from github.com/phybench-official/phybench/tree/main/EED (as of
2026-04-19). `EED(gold_latex, pred_latex) -> (score, rel_dist, tree_size, dist)`
where score ∈ [0, 100]; 100 = exact symbolic match, 0 = completely different.

We report:
    score       — continuous [0, 100] per-row
    pass@1      — exact-match rate (score == 100)
    mean_EED    — average continuous score (the headline number in PHYBench
                  paper Table 2)

The EED implementation needs antlr4-python3-runtime>=4.11 (for sympy LaTeX)
plus timeout_decorator and zss — all installed under eval/_pkgs/.
"""
from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
# antlr4-python3-runtime 4.11, timeout_decorator, zss
_PKGS = _ROOT / "eval" / "_pkgs"
# Vendored phybench EED files (bare imports `from extended_zss import ext_distance`)
_PHYBENCH_DIR = Path(__file__).resolve().parent / "_phybench"
for p in (str(_PKGS), str(_PHYBENCH_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)


def _concat_pred_text(row) -> str:
    def _safe(v) -> str:
        if v is None:
            return ""
        if isinstance(v, float):
            return ""
        return str(v)
    return _safe(row["phase1_text"]) + _safe(row["phase1b_text"]) + _safe(row["phase2_text"])


def _extract_boxed(text: str) -> str | None:
    idx = text.rfind(r"\boxed{")
    if idx == -1:
        return None
    depth = 0
    start = idx + len(r"\boxed{")
    for i, ch in enumerate(text[start:], start):
        if ch == "{":
            depth += 1
        elif ch == "}":
            if depth == 0:
                return text[start:i].strip()
            depth -= 1
    return None


def _strip_gold_wrapping(gold: str) -> str:
    """Gold stored as \\boxed{...}; EED works on the inner LaTeX."""
    g = str(gold).strip()
    if g.startswith(r"\boxed{") and g.endswith("}"):
        inner = g[len(r"\boxed{"):-1].strip()
        return inner
    return g


def score_rollouts(rollouts_path: str, out_path: str) -> None:
    import pandas as pd  # noqa: PLC0415
    from EED import EED  # noqa: PLC0415  # vendored, from _phybench/

    df = pd.read_parquet(rollouts_path)
    n = len(df)
    print(f"Loaded {n} rollouts from {rollouts_path}")

    records: list[dict] = []
    counts = {"exact": 0, "non_exact_but_scored": 0, "no_boxed": 0, "eed_error": 0}
    total_eed = 0.0
    for i, row in df.iterrows():
        pred_text = _concat_pred_text(row)
        gold_latex = _strip_gold_wrapping(row["gold_answer"])
        pred_latex = _extract_boxed(pred_text)

        if pred_latex is None:
            eed_score = 0.0
            eed_error = False
            counts["no_boxed"] += 1
        else:
            try:
                s, _rel, _tsize, _dist = EED(gold_latex, pred_latex, debug_mode=False)
                eed_score = float(s)
                eed_error = False
            except Exception as e:
                eed_score = 0.0
                eed_error = True
                counts["eed_error"] += 1

        total_eed += eed_score
        is_exact = eed_score >= 100.0 - 1e-9
        if is_exact:
            counts["exact"] += 1
        elif pred_latex is not None and not eed_error:
            counts["non_exact_but_scored"] += 1

        records.append({
            "problem_idx": row["problem_idx"],
            "rollout_idx": row["rollout_idx"],
            "gold_answer": gold_latex,
            "pred_latex": pred_latex,
            "pred_text_len": len(pred_text),
            "eed_score": eed_score,
            "correct": bool(is_exact),
            "eed_error": eed_error,
        })

        if (i + 1) % 50 == 0 or (i + 1) == n:
            mean_eed = total_eed / (i + 1)
            print(f"  scored {i + 1}/{n}  (exact={counts['exact']}, "
                  f"no_boxed={counts['no_boxed']}, eed_err={counts['eed_error']}, "
                  f"running mean_EED={mean_eed:.2f})")

    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    pd.DataFrame(records).to_parquet(out_path, index=False)

    pass_rate = counts["exact"] / n if n else float("nan")
    mean_eed = total_eed / n if n else float("nan")
    summary_path = os.path.splitext(out_path)[0] + ".summary.txt"
    with open(summary_path, "w") as f:
        f.write(f"rollouts     : {rollouts_path}\n")
        f.write(f"scorer       : PHYBench EED (official, vendored)\n")
        f.write(f"n            : {n}\n")
        f.write(f"exact (EED=100) : {counts['exact']}\n")
        f.write(f"partial (0<EED<100) : {counts['non_exact_but_scored']}\n")
        f.write(f"no_boxed     : {counts['no_boxed']}\n")
        f.write(f"eed_error    : {counts['eed_error']}\n")
        f.write(f"pass@1       : {pass_rate:.4f}  (exact match rate)\n")
        f.write(f"mean_EED     : {mean_eed:.4f}  (continuous score, 0-100)\n")

    print(f"\nSaved → {out_path}")
    print(f"Summary → {summary_path}")
    print(f"pass@1={pass_rate:.4f}  mean_EED={mean_eed:.2f}")


def main() -> None:
    p = argparse.ArgumentParser(description="Score PHYBench rollouts with official EED")
    p.add_argument("--rollouts", required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args()
    score_rollouts(args.rollouts, args.out)


if __name__ == "__main__":
    main()
