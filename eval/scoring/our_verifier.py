"""Score a rollouts.parquet with the project's rule+xVerify verifier.

Takes the raw text from a rollouts.parquet (phase1_text + phase1b_text + phase2_text
concatenated), passes it to `src/phys_reasoner/verifier/router.py::verify_answer`
together with an in-process XVerifyJudge, and writes scored.parquet with one row
per rollout.

Design notes
------------
- Uses `XVerifyJudge` directly (no HTTP). Bit-identical to the training reward
  since the HTTP server (scripts/serve_xverify.py) is a thin wrapper around the
  same class.
- `verify_answer` already does `\\boxed{}` extraction from the full text — we
  concatenate the raw phase texts and hand the whole string over, matching how
  the training reward is computed.
- Verdict semantics (from router.py):
    1.0   correct
    0.0   confirmed wrong
   -1.0   unverifiable (rule couldn't parse and no xVerify, or open_end/code)
  The `correct` bool column is True only when verdict == 1.0. Unverifiable rows
  are counted separately in the summary.

Usage
-----
    python3 eval/scoring/our_verifier.py \\
        --rollouts outputs/xxx/rollouts.parquet \\
        --out outputs/xxx/scored.parquet
"""
from __future__ import annotations

import argparse
import os


def _concat_pred_text(row) -> str:
    """Concatenate raw phase texts into the full model output for extraction.

    phase1b_text and phase2_text may be None (for not-interrupted / CoT / TIR
    no-tool-call rollouts respectively). Pandas may store None as NaN (float)
    after a parquet round-trip — treat both as empty.
    """
    def _safe(v) -> str:
        if v is None:
            return ""
        if isinstance(v, float):  # NaN
            return ""
        return str(v)

    return _safe(row["phase1_text"]) + _safe(row["phase1b_text"]) + _safe(row["phase2_text"])


def _extract_meta(row) -> dict:
    """Pull unit / tolerance / problem / answer_type from extra_info."""
    ei = row.get("extra_info") or {}
    if not isinstance(ei, dict):
        try:
            ei = dict(ei)
        except Exception:
            ei = {}
    return {
        "unit": str(ei.get("unit", "") or ""),
        "tolerance": float(ei.get("tolerance", 0.05) or 0.05),
        "problem": str(ei.get("problem", "") or ""),
        "answer_type": str(
            ei.get("answer_type", row.get("answer_type", "unknown")) or "unknown"
        ),
    }


def score_rollouts(
    rollouts_path: str,
    out_path: str,
    xverify_model: str = "IAAR-Shanghai/xVerify-7B-I",
    xverify_device: str = "cuda",
    no_xverify: bool = False,
) -> None:
    import pandas as pd  # noqa: PLC0415

    from phys_reasoner.verifier.router import verify_answer  # noqa: PLC0415

    df = pd.read_parquet(rollouts_path)
    n = len(df)
    print(f"Loaded {n} rollouts from {rollouts_path}")

    # Load xVerify (unless disabled — rule-only scoring for smoke tests)
    xverify_judge = None
    if not no_xverify:
        from phys_reasoner.verifier.xverify_judge import XVerifyJudge  # noqa: PLC0415
        print(f"Loading xVerify: {xverify_model} on {xverify_device} ...")
        xverify_judge = XVerifyJudge(model_name=xverify_model, device=xverify_device)
        print("xVerify ready.")

    records: list[dict] = []
    counts = {"correct": 0, "wrong": 0, "unverifiable": 0}
    for i, row in df.iterrows():
        pred_text = _concat_pred_text(row)
        gold = row["gold_answer"]
        meta = _extract_meta(row)
        verdict = verify_answer(
            pred_text=pred_text,
            gold_answer=gold,
            answer_type=meta["answer_type"],
            gold_unit=meta["unit"],
            tolerance=meta["tolerance"],
            xverify_judge=xverify_judge,
            problem_text=meta["problem"],
        )
        if verdict == 1.0:
            counts["correct"] += 1
        elif verdict == 0.0:
            counts["wrong"] += 1
        else:
            counts["unverifiable"] += 1

        records.append({
            "problem_idx": row["problem_idx"],
            "rollout_idx": row["rollout_idx"],
            "gold_answer": gold,
            "answer_type": meta["answer_type"],
            "unit": meta["unit"],
            "pred_text_len": len(pred_text),
            "verdict": verdict,
            "correct": bool(verdict == 1.0),
            "unverifiable": bool(verdict == -1.0),
        })

        if (i + 1) % 50 == 0 or (i + 1) == n:
            print(f"  scored {i + 1}/{n}  (correct={counts['correct']}, "
                  f"wrong={counts['wrong']}, unverifiable={counts['unverifiable']})")

    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    scored = pd.DataFrame(records)
    scored.to_parquet(out_path, index=False)

    # pass@1: denominator excludes unverifiable by default; also report inclusive form
    verifiable = counts["correct"] + counts["wrong"]
    exclusive_rate = counts["correct"] / verifiable if verifiable else float("nan")
    inclusive_rate = counts["correct"] / n if n else float("nan")

    summary_path = os.path.splitext(out_path)[0] + ".summary.txt"
    with open(summary_path, "w") as f:
        f.write(f"rollouts         : {rollouts_path}\n")
        f.write(f"scorer           : rule+xverify ({'disabled' if no_xverify else xverify_model})\n")
        f.write(f"n                : {n}\n")
        f.write(f"correct          : {counts['correct']}\n")
        f.write(f"wrong            : {counts['wrong']}\n")
        f.write(f"unverifiable     : {counts['unverifiable']}\n")
        f.write(f"pass@1 exclusive : {exclusive_rate:.4f}  (correct / (correct+wrong))\n")
        f.write(f"pass@1 inclusive : {inclusive_rate:.4f}  (correct / n; unverifiable counted as miss)\n")

    print(f"\nSaved scored parquet → {out_path}")
    print(f"Summary → {summary_path}")
    print(f"pass@1 exclusive={exclusive_rate:.4f}  inclusive={inclusive_rate:.4f}  "
          f"(correct={counts['correct']}, wrong={counts['wrong']}, "
          f"unverifiable={counts['unverifiable']})")


def main() -> None:
    p = argparse.ArgumentParser(description="Score a rollouts.parquet with rule+xVerify")
    p.add_argument("--rollouts", required=True, help="Path to rollouts.parquet")
    p.add_argument("--out", required=True, help="Path to scored.parquet output")
    p.add_argument("--xverify_model", default="IAAR-Shanghai/xVerify-7B-I")
    p.add_argument("--xverify_device", default="cuda")
    p.add_argument("--no_xverify", action="store_true",
                   help="Skip xVerify (rule-only). Useful for smoke tests without GPU.")
    args = p.parse_args()
    score_rollouts(
        rollouts_path=args.rollouts,
        out_path=args.out,
        xverify_model=args.xverify_model,
        xverify_device=args.xverify_device,
        no_xverify=args.no_xverify,
    )


if __name__ == "__main__":
    main()
