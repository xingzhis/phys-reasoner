"""Dump side-by-side diagnostics for cases where CoT and TIR disagree.

Joins two scored parquets + their rollouts on problem_idx, and writes one txt
per disagreement case for easy human spot-checking. Useful for sanity-checking
the eval harness: does the CoT-correct-TIR-wrong rollout actually look like a
clean win for CoT, or is it a verifier quirk?

Usage:
    python3 eval/scoring/diff_tir_vs_cot.py \\
        --tir_dir outputs/eval/corpus_test/tir_qwen35_4b_zeroshot \\
        --cot_dir outputs/eval/corpus_test/cot_qwen35_4b_zeroshot \\
        --out_dir outputs/eval/corpus_test/diff_cot_wins \\
        --direction cot_wins \\
        --max 20
"""
from __future__ import annotations

import argparse
import os


_INTERRUPT_PHRASE = "\nOkay, I've thought enough. Time to write my response.\n</think>\n"


def _safe(v) -> str:
    if v is None:
        return ""
    if isinstance(v, float):
        return ""
    return str(v)


def _concat_pred_text(row) -> str:
    return _safe(row.get("phase1_text")) + _safe(row.get("phase1b_text")) + _safe(row.get("phase2_text"))


def _annotated_trajectory(row, is_cot: bool) -> str:
    """Phase-annotated view of a rollout: [PHASE 1] / [INTERRUPT] / [PHASE 1B] /
    [CODE] / [SANDBOX] / [PHASE 2] markers inserted between the raw texts.
    """
    p1 = _safe(row.get("phase1_text"))
    p1b = _safe(row.get("phase1b_text"))
    p2 = _safe(row.get("phase2_text"))
    interrupted = bool(row.get("interrupted"))
    code = row.get("code")
    if isinstance(code, float):  # NaN
        code = None
    so = _safe(row.get("sandbox_stdout"))
    se = bool(row.get("sandbox_error"))

    parts = [f"----- [PHASE 1]  len={len(p1)}  interrupted={interrupted} -----", p1]
    if interrupted:
        parts += [
            "----- [THINK-INTERRUPT injected, mask=0 in training] -----",
            _INTERRUPT_PHRASE.strip(),
        ]
    if p1b:
        parts += [f"----- [PHASE 1B]  len={len(p1b)} -----", p1b]
    if not is_cot:
        parts += [
            f"----- [CODE EXTRACTED]  len={len(code) if code else 0}  code_is_none={code is None} -----",
            code if code else "(none extracted — no <tool_call>...</tool_call> found)",
            f"----- [SANDBOX]  error={se}  stdout_len={len(so)} -----",
            so if so else "(no stdout)",
        ]
    if p2:
        parts += [f"----- [PHASE 2]  len={len(p2)} -----", p2]
    elif not is_cot:
        parts.append("----- [PHASE 2 skipped — no tool call to execute] -----")
    return "\n".join(parts)


def _extract_problem(row) -> str:
    ei = row.get("extra_info")
    if isinstance(ei, dict):
        p = ei.get("problem")
        if p:
            return str(p)
    return "(unknown problem)"


def dump_diffs(
    tir_dir: str,
    cot_dir: str,
    out_dir: str,
    direction: str,
    max_cases: int,
) -> None:
    import pandas as pd  # noqa: PLC0415

    tir_roll = pd.read_parquet(os.path.join(tir_dir, "rollouts.parquet"))
    cot_roll = pd.read_parquet(os.path.join(cot_dir, "rollouts.parquet"))
    tir_score = pd.read_parquet(os.path.join(tir_dir, "scored.parquet"))
    cot_score = pd.read_parquet(os.path.join(cot_dir, "scored.parquet"))

    # Join scored rows on problem_idx + rollout_idx
    tir_s = tir_score.set_index(["problem_idx", "rollout_idx"])
    cot_s = cot_score.set_index(["problem_idx", "rollout_idx"])
    common = tir_s.index.intersection(cot_s.index)
    tir_s = tir_s.loc[common]
    cot_s = cot_s.loc[common]

    if direction == "cot_wins":
        mask = (~tir_s["correct"]) & (cot_s["correct"])
    elif direction == "tir_wins":
        mask = (tir_s["correct"]) & (~cot_s["correct"])
    elif direction == "both_wrong":
        mask = (~tir_s["correct"]) & (~cot_s["correct"])
    else:
        raise ValueError(f"unknown direction {direction!r}")

    idxs = tir_s[mask].index.tolist()
    print(f"Found {len(idxs)} cases with direction={direction}")
    if max_cases > 0:
        idxs = idxs[:max_cases]
        print(f"Dumping first {len(idxs)}")

    os.makedirs(out_dir, exist_ok=True)

    # Index rollouts by (problem_idx, rollout_idx) for fast lookup
    tir_roll_idx = tir_roll.set_index(["problem_idx", "rollout_idx"])
    cot_roll_idx = cot_roll.set_index(["problem_idx", "rollout_idx"])

    sep = "=" * 78
    thin = "-" * 78

    manifest_lines = [
        f"{'case':<4}  {'problem_idx':<12}  {'answer_type':<20}  "
        f"{'tir_verdict':<11}  {'cot_verdict':<11}  gold (truncated)",
        "-" * 120,
    ]

    for k, key in enumerate(idxs):
        prob_idx, roll_idx = key
        tir_r = tir_roll_idx.loc[key]
        cot_r = cot_roll_idx.loc[key]
        tir_v = tir_s.loc[key]
        cot_v = cot_s.loc[key]
        problem = _extract_problem(tir_r) or _extract_problem(cot_r)
        gold = str(tir_r.get("gold_answer", ""))
        answer_type = str(tir_v["answer_type"])

        tir_text = _annotated_trajectory(tir_r, is_cot=False)
        cot_text = _annotated_trajectory(cot_r, is_cot=True)

        # `unverifiable` is only produced by our_verifier (pool_v2); the
        # OlympiadBench scorer writes only {correct, verdict, judge_error}.
        tir_unverif = bool(tir_v.get("unverifiable", False)) if hasattr(tir_v, "get") else False
        cot_unverif = bool(cot_v.get("unverifiable", False)) if hasattr(cot_v, "get") else False
        tir_v_str = ("correct" if tir_v["correct"] else "wrong") + (" (unverif)" if tir_unverif else "")
        cot_v_str = ("correct" if cot_v["correct"] else "wrong") + (" (unverif)" if cot_unverif else "")

        manifest_lines.append(
            f"{k:<4}  {str(prob_idx):<12}  {answer_type[:20]:<20}  "
            f"{tir_v_str:<11}  {cot_v_str:<11}  {gold[:60]!r}"
        )

        out_lines = [
            sep,
            f"DIFF CASE {k:03d}  (direction={direction})",
            sep,
            f"problem_idx : {prob_idx}",
            f"rollout_idx : {roll_idx}",
            f"answer_type : {answer_type}",
            f"gold_answer : {gold}",
            f"tir_verdict : {tir_v_str}",
            f"cot_verdict : {cot_v_str}",
            f"tir_interrupted : {bool(tir_r['interrupted'])}  "
            f"code_extracted={tir_r['code'] is not None and not isinstance(tir_r['code'], float)}  "
            f"phase2={tir_r['phase2_text'] is not None and not isinstance(tir_r['phase2_text'], float)}",
            f"cot_interrupted : {bool(cot_r['interrupted'])}",
            "",
            sep,
            "PROBLEM",
            sep,
            problem.strip(),
            "",
            sep,
            "TIR TRAJECTORY (phase1 + phase1b + phase2 concatenated)",
            sep,
            tir_text.strip() if tir_text.strip() else "(empty)",
            "",
            sep,
            "COT TRAJECTORY (phase1 + phase1b concatenated)",
            sep,
            cot_text.strip() if cot_text.strip() else "(empty)",
            "",
            sep,
        ]
        out_path = os.path.join(out_dir, f"case_{k:03d}_p{prob_idx}.txt")
        with open(out_path, "w") as f:
            f.write("\n".join(out_lines))

    with open(os.path.join(out_dir, "_manifest.txt"), "w") as f:
        f.write(f"direction: {direction}\n")
        f.write(f"n_found  : {len(tir_s[mask])}\n")
        f.write(f"n_dumped : {len(idxs)}\n\n")
        f.write("\n".join(manifest_lines) + "\n")

    print(f"Wrote {len(idxs)} txt files to {out_dir}/")
    print(f"Manifest: {out_dir}/_manifest.txt")


def main() -> None:
    p = argparse.ArgumentParser(description="Dump TIR/CoT disagreement cases for spot-checking")
    p.add_argument("--tir_dir", required=True)
    p.add_argument("--cot_dir", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--direction", choices=("cot_wins", "tir_wins", "both_wrong"), default="cot_wins")
    p.add_argument("--max", type=int, default=20, help="Max cases to dump (0 = all)")
    args = p.parse_args()
    dump_diffs(args.tir_dir, args.cot_dir, args.out_dir, args.direction, args.max)


if __name__ == "__main__":
    main()
