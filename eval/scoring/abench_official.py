"""Score a rollouts.parquet for ABench Phy_A / Phy_B using the official 1%
tolerance numerical comparison.

Official ABench scoring (inclusionAI/ABench src/eval.py):
- Extract model's final numerical answer
- Compare against gold with relative tolerance 1% (i.e. |pred-gold| <= 0.01 * |gold|)
- Phy_A: pass@1 = per-row correct rate
- Phy_B: pass@1 = per-mid ALL-4-subid-correct rate (dynamic robustness check)

This wrapper extracts the model's \\boxed{...} content, tries to parse as float
(with light LaTeX cleanup), then compares. Ambiguous/unparseable preds count
as wrong for Phy_A; for Phy_B they poison the whole mid.
"""
from __future__ import annotations

import argparse
import os
import re


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


_LATEX_CLEANUP = [
    (r"\\mathrm\{[^}]*\}", ""),
    (r"\\text\{[^}]*\}", ""),
    (r"\\,", ""),
    (r"\\;", ""),
    (r"\\\\", ""),
    (r"\\\$", ""),
    (r"\s+", ""),
]
_UNIT_SUFFIX_RE = re.compile(r"\s*[a-zA-ZµμΩ°]+\s*$")


def _parse_float(s: str) -> float | None:
    """Very forgiving: strip common LaTeX wrappers and a trailing unit token,
    then float()."""
    if s is None:
        return None
    v = str(s).strip()
    v = v.replace("$", "").replace("{", "").replace("}", "")
    for pat, repl in _LATEX_CLEANUP:
        v = re.sub(pat, repl, v)
    v = v.strip()
    # Strip trailing unit letters like '2.97uA' → '2.97'
    m = re.match(r"^([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)", v)
    if m:
        try:
            return float(m.group(1))
        except ValueError:
            return None
    return None


def _strip_gold_wrapping(gold: str) -> str:
    g = str(gold).strip()
    if g.startswith(r"\boxed{") and g.endswith("}"):
        return g[len(r"\boxed{"):-1].strip()
    return g


def score_rollouts(rollouts_path: str, out_path: str, tolerance: float = 0.01) -> None:
    import pandas as pd  # noqa: PLC0415

    df = pd.read_parquet(rollouts_path)
    n = len(df)
    print(f"Loaded {n} rollouts from {rollouts_path}")

    records: list[dict] = []
    counts = {"correct": 0, "wrong": 0, "unparseable": 0}
    is_phy_b = False
    for i, row in df.iterrows():
        pred_text = _concat_pred_text(row)
        gold_str = _strip_gold_wrapping(row["gold_answer"])
        pred_raw = _extract_boxed(pred_text)
        gold_val = _parse_float(gold_str)
        pred_val = _parse_float(pred_raw) if pred_raw is not None else None

        ei = row.get("extra_info") or {}
        if not isinstance(ei, dict):
            try:
                ei = dict(ei)
            except Exception:
                ei = {}
        mid = str(ei.get("mid", ""))
        subid = str(ei.get("subid", "")) if "subid" in ei else None
        if subid is not None:
            is_phy_b = True

        if gold_val is None or pred_val is None:
            is_correct = False
            counts["unparseable"] += 1
        else:
            denom = abs(gold_val) if gold_val != 0 else 1.0
            is_correct = abs(pred_val - gold_val) <= tolerance * denom
            counts["correct" if is_correct else "wrong"] += 1

        records.append({
            "problem_idx": row["problem_idx"],
            "rollout_idx": row["rollout_idx"],
            "gold_answer": gold_str,
            "gold_val": gold_val,
            "pred_raw": pred_raw,
            "pred_val": pred_val,
            "pred_text_len": len(pred_text),
            "mid": mid,
            "subid": subid,
            "correct": bool(is_correct),
        })

        if (i + 1) % 50 == 0 or (i + 1) == n:
            print(f"  scored {i + 1}/{n}  (correct={counts['correct']}, "
                  f"wrong={counts['wrong']}, unparseable={counts['unparseable']})")

    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    scored = pd.DataFrame(records)
    scored.to_parquet(out_path, index=False)

    per_row_rate = counts["correct"] / n if n else float("nan")

    # Phy_B: per-mid aggregation (all 4 subid must be correct)
    per_mid_rate = None
    n_mids = None
    if is_phy_b:
        mids = scored.groupby("mid")["correct"].all()
        per_mid_rate = float(mids.sum() / len(mids)) if len(mids) else float("nan")
        n_mids = len(mids)

    summary_path = os.path.splitext(out_path)[0] + ".summary.txt"
    with open(summary_path, "w") as f:
        f.write(f"rollouts        : {rollouts_path}\n")
        f.write(f"scorer          : ABench official (1% tolerance numerical)\n")
        f.write(f"n               : {n}\n")
        f.write(f"correct         : {counts['correct']}\n")
        f.write(f"wrong           : {counts['wrong']}\n")
        f.write(f"unparseable     : {counts['unparseable']}\n")
        f.write(f"pass@1 per-row  : {per_row_rate:.4f}\n")
        if is_phy_b:
            f.write(f"n_mids          : {n_mids}\n")
            f.write(f"pass@1 per-mid  : {per_mid_rate:.4f}  (Phy_B ALL-4-subid-correct)\n")

    print(f"\nSaved → {out_path}")
    print(f"Summary → {summary_path}")
    print(f"pass@1 per-row = {per_row_rate:.4f}"
          + (f"   pass@1 per-mid = {per_mid_rate:.4f}" if is_phy_b else ""))


def main() -> None:
    p = argparse.ArgumentParser(description="Score ABench Phy_A/Phy_B rollouts")
    p.add_argument("--rollouts", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--tolerance", type=float, default=0.01)
    args = p.parse_args()
    score_rollouts(args.rollouts, args.out, tolerance=args.tolerance)


if __name__ == "__main__":
    main()
