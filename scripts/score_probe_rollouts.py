"""Score probe rollout parquets using the exact training reward function.

Calls compute_score() from phys_reasoner.training.reward — the same function
VeRL invokes during GRPO training (dapo.py line 78-90). Uses XVerifyHTTPClient
to reach the companion serve_xverify server, matching the production reward path
exactly (rule verifier → xVerify HTTP fallback → -1.0 clips to 0.0).

VeRL constructs solution_str by:
    tokenizer.decode(valid_response_ids, skip_special_tokens=True)
which yields the full multi-turn text. extract_answer() inside verify_answer
scans for the last \\boxed{} in that text. We reconstruct the equivalent by
concatenating phase1_text + phase1b_text + phase2_text (tool output excluded —
it never contains \\boxed{} so cannot affect extract_answer's result).

Requires a running xVerify server:
    sbatch scripts/serve_xverify.sbatch
The scorer discovers the URL via outputs/xverify_endpoints/current.url
(same rendezvous mechanism as training).

Outputs:
  - rollouts_scored.parquet  with per-rollout score and score_raw columns
  - score_summary.txt        hit rates by answer_type, difficulty

Usage:
  python scripts/score_probe_rollouts.py \\
      --inputs outputs/probe_calib_A/rollouts.parquet outputs/probe_calib_B/rollouts.parquet \\
      --output outputs/probe_scored/
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import pandas as pd


def _reconstruct_solution(row: pd.Series) -> str:
    """Reconstruct solution_str matching VeRL's tokenizer.decode(response_ids)."""
    parts = []
    p1 = row.get("phase1_text")
    if p1:
        parts.append(str(p1))
    if row.get("interrupted", False):
        p1b = row.get("phase1b_text")
        if p1b:
            parts.append(str(p1b))
    p2 = row.get("phase2_text")
    if p2:
        parts.append(str(p2))
    return "\n".join(parts)


def score_rollouts(df: pd.DataFrame, xverify_url: str) -> pd.DataFrame:
    from phys_reasoner.training.reward import compute_score
    from phys_reasoner.verifier.xverify_client import XVerifyHTTPClient

    client = XVerifyHTTPClient(xverify_url)
    if not client.health_check():
        print(f"ERROR: xVerify server at {xverify_url} failed health check", file=sys.stderr)
        sys.exit(1)
    print(f"xVerify server healthy at {xverify_url}")

    scores = []
    raw_scores = []
    n = len(df)
    t_start = time.time()

    for i, (_, row) in enumerate(df.iterrows()):
        solution_str = _reconstruct_solution(row)
        extra_info = row.get("extra_info")
        if not isinstance(extra_info, dict):
            extra_info = {}
        gold = row.get("gold_answer", "")

        # compute_score: exact VeRL training reward function.
        # Extracts answer_type/unit/tolerance from extra_info,
        # calls verify_answer with xverify_judge,
        # clips -1.0 (unverifiable) → 0.0.
        score = compute_score(
            solution_str=solution_str,
            ground_truth=gold,
            extra_info=extra_info,
            xverify_judge=client,
        )
        scores.append(score)

        # Raw score for analysis (before -1 → 0 clipping)
        from phys_reasoner.verifier.router import verify_answer

        raw = verify_answer(
            pred_text=solution_str,
            gold_answer=gold,
            answer_type=extra_info.get("answer_type", "numerical"),
            gold_unit=extra_info.get("unit", ""),
            tolerance=float(extra_info.get("tolerance", 0.05)),
            xverify_judge=client,
            problem_text=extra_info.get("problem", ""),
        )
        raw_scores.append(raw)

        if (i + 1) % 500 == 0 or i == n - 1:
            elapsed = time.time() - t_start
            rate = (i + 1) / elapsed
            eta = (n - i - 1) / rate if rate > 0 else 0
            n_correct = sum(1 for s in scores if s == 1.0)
            print(f"  [{i+1}/{n}] {rate:.1f} rollouts/s, ETA {eta/60:.1f}min, "
                  f"hit_rate={n_correct}/{i+1} ({n_correct/(i+1)*100:.1f}%)")

    df = df.copy()
    df["score"] = scores
    df["score_raw"] = raw_scores
    return df


def summarize(df: pd.DataFrame, out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)

    df["answer_type"] = df["extra_info"].apply(
        lambda ei: ei.get("answer_type", "unknown") if isinstance(ei, dict) else "unknown"
    )
    df["difficulty"] = df["extra_info"].apply(
        lambda ei: ei.get("difficulty", "unknown") if isinstance(ei, dict) else "unknown"
    )

    lines: list[str] = []

    def log(msg: str = "") -> None:
        print(msg)
        lines.append(msg)

    n = len(df)
    n_correct = int((df["score"] == 1.0).sum())
    n_unverifiable = int((df["score_raw"] == -1.0).sum())
    n_rule_wrong = int(((df["score_raw"] == 0.0) & (df["score"] == 0.0)).sum())

    log("=== OVERALL (training-equivalent scoring) ===")
    log(f"  total rollouts     : {n}")
    log(f"  correct (1.0)      : {n_correct}  ({n_correct/n*100:.2f}%)")
    log(f"  wrong (0.0)        : {n - n_correct}  ({(n-n_correct)/n*100:.2f}%)")
    log(f"    of which unverif : {n_unverifiable}  (raw=-1.0, clipped to 0.0 in training)")
    log(f"    confirmed wrong  : {n_rule_wrong}")
    log()

    # Per problem pass@1
    prob_scores = df.groupby("problem_idx")["score"].agg(["max", "mean", "count"])
    n_problems = len(prob_scores)
    n_problems_solved = int((prob_scores["max"] == 1.0).sum())
    mean_acc = prob_scores["mean"].mean()
    log("=== PER-PROBLEM ===")
    log(f"  total problems     : {n_problems}")
    log(f"  pass@1 (any ok)    : {n_problems_solved}  ({n_problems_solved/n_problems*100:.2f}%)")
    log(f"  mean rollout acc   : {mean_acc*100:.2f}%")
    log()

    # By answer_type
    log("=== BY ANSWER TYPE ===")
    log(f"  {'type':20s}  {'n_roll':>7}  {'hit%':>6}  {'pass@1':>14}  {'unverif':>8}")
    for atype, grp in sorted(df.groupby("answer_type"), key=lambda x: -len(x[1])):
        nc = int((grp["score"] == 1.0).sum())
        nu = int((grp["score_raw"] == -1.0).sum())
        probs = grp.groupby("problem_idx")["score"].max()
        ps = int((probs == 1.0).sum())
        log(f"  {atype:20s}  {len(grp):>7}  {nc/len(grp)*100:5.1f}%  "
            f"{ps:>5}/{len(probs):<5} ({ps/len(probs)*100:.1f}%)  {nu:>8}")
    log()

    # By difficulty bucket
    log("=== BY DIFFICULTY BUCKET ===")
    try:
        df["difficulty_f"] = pd.to_numeric(df["difficulty"], errors="coerce")
        df["diff_bucket"] = pd.cut(
            df["difficulty_f"],
            bins=[0, 0.2, 0.4, 0.6, 0.8, 1.0],
            labels=["0-0.2", "0.2-0.4", "0.4-0.6", "0.6-0.8", "0.8-1.0"],
            include_lowest=True,
        )
        log(f"  {'bucket':12s}  {'n_roll':>7}  {'hit%':>6}  {'pass@1':>14}")
        for bucket, grp in df.groupby("diff_bucket", observed=True):
            nc = int((grp["score"] == 1.0).sum())
            probs = grp.groupby("problem_idx")["score"].max()
            ps = int((probs == 1.0).sum())
            log(f"  {str(bucket):12s}  {len(grp):>7}  {nc/len(grp)*100:5.1f}%  "
                f"{ps:>5}/{len(probs):<5} ({ps/len(probs)*100:.1f}%)")
    except Exception as e:
        log(f"  (could not bucket difficulty: {e})")
    log()

    # SFT decision
    log("=== SFT DECISION (per CLAUDE.md: skip SFT if zero-shot TIR hit rate >= 15%) ===")
    log(f"  overall hit rate   : {n_correct/n*100:.2f}%")
    log(f"  pass@1 rate        : {n_problems_solved/n_problems*100:.2f}%")
    if n_correct / n >= 0.15:
        log("  -> SKIP SFT (hit rate >= 15%)")
    else:
        log("  -> SFT NEEDED (hit rate < 15%)")
    log()

    summary_path = os.path.join(out_dir, "score_summary.txt")
    with open(summary_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\nSummary saved to {summary_path}")


def main() -> None:
    p = argparse.ArgumentParser(
        description="Score probe rollout parquets with training-equivalent verification"
    )
    p.add_argument("--inputs", nargs="+", required=True,
                   help="Rollout parquet(s) from dump_rollouts")
    p.add_argument("--output", default="outputs/probe_scored/",
                   help="Output directory")
    p.add_argument("--xverify_url", default=None,
                   help="xVerify server URL. If omitted, reads from "
                        "outputs/xverify_endpoints/current.url (same as training)")
    args = p.parse_args()

    # Discover xVerify URL (same mechanism as training reward.py)
    url = args.xverify_url
    if not url:
        url_file = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "..",
            "outputs", "xverify_endpoints", "current.url"
        )
        url_file = os.path.normpath(url_file)
        if not os.path.isfile(url_file):
            print(f"ERROR: No xVerify URL provided and rendezvous file not found: {url_file}",
                  file=sys.stderr)
            print("Start the server first:  sbatch scripts/serve_xverify.sbatch", file=sys.stderr)
            sys.exit(1)
        with open(url_file) as f:
            url = f.readline().strip()
        print(f"Discovered xVerify URL from rendezvous: {url}")

    # Load and concatenate
    dfs = []
    for path in args.inputs:
        print(f"Loading {path}")
        df = pd.read_parquet(path)
        print(f"  {len(df)} rollouts")
        dfs.append(df)
    df = pd.concat(dfs, ignore_index=True)
    print(f"Total: {len(df)} rollouts")

    # Score
    df = score_rollouts(df, xverify_url=url)

    # Save
    os.makedirs(args.output, exist_ok=True)
    scored_path = os.path.join(args.output, "rollouts_scored.parquet")
    df.to_parquet(scored_path, index=False)
    print(f"Scored parquet saved to {scored_path}")

    # Summary
    summarize(df, args.output)


if __name__ == "__main__":
    main()
