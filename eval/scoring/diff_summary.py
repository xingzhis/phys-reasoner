"""Compact side-by-side of TIR/CoT disagreement cases (last 400 chars each)."""
from __future__ import annotations

import argparse
import os


def _safe(v) -> str:
    if v is None:
        return ""
    if isinstance(v, float):
        return ""
    return str(v)


def _concat(row) -> str:
    return _safe(row.get("phase1_text")) + _safe(row.get("phase1b_text")) + _safe(row.get("phase2_text"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tir_dir", required=True)
    ap.add_argument("--cot_dir", required=True)
    ap.add_argument("--direction", choices=("cot_wins", "tir_wins"), default="cot_wins")
    ap.add_argument("--tail", type=int, default=400)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    import pandas as pd  # noqa: PLC0415

    tir_r = pd.read_parquet(os.path.join(args.tir_dir, "rollouts.parquet")).set_index(["problem_idx", "rollout_idx"])
    cot_r = pd.read_parquet(os.path.join(args.cot_dir, "rollouts.parquet")).set_index(["problem_idx", "rollout_idx"])
    tir_s = pd.read_parquet(os.path.join(args.tir_dir, "scored.parquet")).set_index(["problem_idx", "rollout_idx"])
    cot_s = pd.read_parquet(os.path.join(args.cot_dir, "scored.parquet")).set_index(["problem_idx", "rollout_idx"])

    if args.direction == "cot_wins":
        mask = (~tir_s["correct"]) & (cot_s["correct"])
    else:
        mask = tir_s["correct"] & (~cot_s["correct"])
    idxs = tir_s[mask].index.tolist()
    if args.limit > 0:
        idxs = idxs[: args.limit]

    for k, key in enumerate(idxs):
        t = tir_r.loc[key]
        c = cot_r.loc[key]
        prob = (t.get("extra_info") or {}).get("problem", "") if isinstance(t.get("extra_info"), dict) else ""
        gold = str(t["gold_answer"])
        at = str(tir_s.loc[key, "answer_type"])
        tir_text = _concat(t)
        cot_text = _concat(c)
        tir_interrupted = bool(t["interrupted"])
        tir_has_code = t["code"] is not None
        tir_has_sandbox = bool(t["sandbox_stdout"])
        cot_interrupted = bool(c["interrupted"])

        print("=" * 100)
        print(f"CASE {k:03d}  prob_idx={key[0]}  answer_type={at}")
        print(f"  gold: {gold[:120]}")
        print(f"  prob: {prob[:200].strip()}")
        print(f"  TIR: interrupted={tir_interrupted}  code={tir_has_code}  sandbox_out={tir_has_sandbox}")
        print(f"  COT: interrupted={cot_interrupted}")
        print("-" * 100)
        print(f"TIR last {args.tail} chars:")
        print(tir_text[-args.tail:] if tir_text else "(empty)")
        print("-" * 100)
        print(f"COT last {args.tail} chars:")
        print(cot_text[-args.tail:] if cot_text else "(empty)")
        print()


if __name__ == "__main__":
    main()
