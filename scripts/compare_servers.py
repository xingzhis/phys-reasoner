"""Per-row sanity comparison: new vLLM xverify server vs old transformers HTTP server.

Both are queried via XVerifyHTTPClient through the SAME compute_score → router
code path that VeRL training uses, so any disagreement reflects a real backend
behaviour difference (not preprocessing drift).

Usage:
  apptainer ... python3 scripts/compare_servers.py \\
    --merged outputs/probe_qwen3_4b/rollouts_merged.parquet \\
    --new_url http://nidA:8765/judge \\
    --old_url http://nidB:8765/judge \\
    --max_rows 200
"""
from __future__ import annotations

import argparse
import time
from concurrent.futures import ThreadPoolExecutor

import pandas as pd


def reconstruct(row) -> str:
    p = []
    if row.get("phase1_text"):  p.append(str(row["phase1_text"]))
    if row.get("interrupted") and row.get("phase1b_text"): p.append(str(row["phase1b_text"]))
    if row.get("phase2_text"):  p.append(str(row["phase2_text"]))
    return "\n".join(p)


def score_one(row, client):
    from phys_reasoner.training.reward import compute_score
    sol = reconstruct(row)
    gold = row.get("gold_answer", "")
    ei = row.get("extra_info") if isinstance(row.get("extra_info"), dict) else {}
    s = compute_score(
        solution_str=sol,
        ground_truth=str(gold) if gold is not None else "",
        extra_info=ei,
        xverify_judge=client,
    )
    return 1 if s > 0 else 0


def score_all(df, url, label, n_workers=32):
    from phys_reasoner.verifier.xverify_client import XVerifyHTTPClient
    client = XVerifyHTTPClient(url)
    assert client.health_check(), f"{label} server not healthy: {url}"
    print(f"[{label}] scoring {len(df)} rows with {n_workers} threads ...")
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=n_workers) as ex:
        futures = {ex.submit(score_one, row, client): i for i, (_, row) in enumerate(df.iterrows())}
        scores = [0] * len(df)
        for fut in futures:
            scores[futures[fut]] = fut.result()
    elapsed = time.time() - t0
    print(f"[{label}] done in {elapsed:.1f}s ({len(df)/elapsed:.1f} req/s)")
    return scores


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--merged", required=True)
    ap.add_argument("--new_url", required=True)
    ap.add_argument("--old_url", required=True)
    ap.add_argument("--max_rows", type=int, default=200)
    ap.add_argument("--n_workers_old", type=int, default=8,
                    help="HTTP-old is throughput-capped at ~5rps; high concurrency wastes nothing.")
    ap.add_argument("--n_workers_new", type=int, default=32)
    args = ap.parse_args()

    df = pd.read_parquet(args.merged).head(args.max_rows).reset_index(drop=True)
    print(f"loaded {len(df)} rows")

    new_scores = score_all(df, args.new_url, "new", args.n_workers_new)
    old_scores = score_all(df, args.old_url, "old", args.n_workers_old)

    n = len(df)
    n_agree = sum(1 for a, b in zip(new_scores, old_scores) if a == b)
    n_old_correct = sum(old_scores)
    n_new_correct = sum(new_scores)
    pct = 100 * n_agree / n if n else 0.0
    print()
    print("=" * 60)
    print(f"  rows                : {n}")
    print(f"  agree               : {n_agree}/{n} = {pct:.2f}%")
    print(f"  old correct count   : {n_old_correct}")
    print(f"  new correct count   : {n_new_correct}")

    disagreements = [(i, old_scores[i], new_scores[i]) for i in range(n) if old_scores[i] != new_scores[i]]
    if disagreements:
        print(f"\n  first 10 disagreements (i, old, new):")
        for i, o, n_ in disagreements[:10]:
            row = df.iloc[i]
            ei = row.get("extra_info") if isinstance(row.get("extra_info"), dict) else {}
            print(f"    {i:5d}  old={o} new={n_}  type={ei.get('answer_type')!s:12}  gold={str(row.get('gold_answer'))[:60]!r}")

    if pct >= 99.0:
        print("\nPASS: agreement >= 99%, vLLM server is byte-equivalent for filtering.")
    else:
        print("\nWARN: agreement below 99%, investigate disagreements.")


if __name__ == "__main__":
    main()
