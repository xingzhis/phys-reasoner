"""Per-row sanity comparison: vLLM xVerify (offline batched) vs HTTP xVerify.

For each row in the sample-scored parquet that required xverify (router returned -1
on rule pass), call XVerifyHTTPClient and compare to the vLLM-batched judgment.
Reports per-row agreement %, disagreement examples.

If agreement >= 99% we trust the vLLM path for the full 131k run.

Usage:
  apptainer exec ... python3 scripts/compare_vllm_vs_http_xverify.py \\
    --merged outputs/probe_qwen3_4b/rollouts_merged.parquet \\
    --vllm_scored outputs/probe_qwen3_4b/rollouts_scored.parquet \\
    --xverify_url_file outputs/xverify_endpoints/current.url \\
    --max_rows 2000
"""
from __future__ import annotations

import argparse
import time

import pandas as pd


def reconstruct_solution(row) -> str:
    parts = []
    p1 = row.get("phase1_text")
    if p1: parts.append(str(p1))
    if row.get("interrupted", False):
        p1b = row.get("phase1b_text")
        if p1b: parts.append(str(p1b))
    p2 = row.get("phase2_text")
    if p2: parts.append(str(p2))
    return "\n".join(parts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--merged", default="outputs/probe_qwen3_4b/rollouts_merged.parquet")
    ap.add_argument("--vllm_scored", default="outputs/probe_qwen3_4b/rollouts_scored.parquet")
    ap.add_argument("--xverify_url_file", default="outputs/xverify_endpoints/current.url")
    ap.add_argument("--max_rows", type=int, default=2000,
                    help="Compare only the first N rows of merged (matches the vLLM sample)")
    args = ap.parse_args()

    from phys_reasoner.verifier.router import verify_answer
    from phys_reasoner.verifier.xverify_client import XVerifyHTTPClient

    url = open(args.xverify_url_file).readline().strip()
    print(f"[http] xverify URL: {url}")
    client = XVerifyHTTPClient(url)
    assert client.health_check(), "xverify HTTP server not healthy"

    df_merged = pd.read_parquet(args.merged).head(args.max_rows).reset_index(drop=True)
    df_vllm = pd.read_parquet(args.vllm_scored).reset_index(drop=True)
    assert len(df_merged) == len(df_vllm), \
        f"size mismatch: merged head {len(df_merged)} vs vllm scored {len(df_vllm)}"

    # Identify rows that needed xverify (rule returned -1)
    print("[rule] running rule-only pass to identify xverify-routed rows ...")
    needs_xv: list[int] = []
    rule_only_correct: list[int] = []
    for i, (_, row) in enumerate(df_merged.iterrows()):
        sol = reconstruct_solution(row)
        gold = row.get("gold_answer", "")
        ei = row.get("extra_info") if isinstance(row.get("extra_info"), dict) else {}
        s = verify_answer(
            pred_text=sol,
            gold_answer=gold if gold is not None else "",
            answer_type=ei.get("answer_type", "numerical"),
            gold_unit=ei.get("unit", "") or "",
            tolerance=float(ei.get("tolerance", 0.05)),
            xverify_judge=None,
            problem_text=ei.get("problem", "") or "",
        )
        if s == -1.0:
            needs_xv.append(i)
        else:
            rule_only_correct.append(int(s == 1.0))
    print(f"[rule] {len(needs_xv)}/{len(df_merged)} rows routed to xverify")

    # Score those rows via HTTP xverify
    print(f"[http] querying xverify HTTP for {len(needs_xv)} rows ...")
    http_correct: dict[int, int] = {}
    t0 = time.time()
    for k, idx in enumerate(needs_xv):
        row = df_merged.iloc[idx]
        sol = reconstruct_solution(row)
        gold = row.get("gold_answer", "")
        ei = row.get("extra_info") if isinstance(row.get("extra_info"), dict) else {}
        # Mirror what verify_answer's xverify path does:
        # call client(pred_str, gold_str, problem_str) → bool
        # But to maximize fidelity, use compute_score so the routing matches
        # exactly what the new vLLM pass will see post-rule-fallback.
        from phys_reasoner.training.reward import compute_score
        s = compute_score(
            solution_str=sol,
            ground_truth=str(gold) if gold is not None else "",
            extra_info=ei,
            xverify_judge=client,
        )
        http_correct[idx] = 1 if s > 0 else 0
        if (k + 1) % 50 == 0:
            print(f"[http] {k+1}/{len(needs_xv)}  {(k+1)/(time.time()-t0):.1f} req/s")

    # Compare with vLLM scored values for those same indices
    n_agree = 0
    n_disagree = 0
    disagreements = []
    for idx in needs_xv:
        h = http_correct[idx]
        v = int(df_vllm.iloc[idx]["correct"])
        if h == v:
            n_agree += 1
        else:
            n_disagree += 1
            if len(disagreements) < 10:
                row = df_merged.iloc[idx]
                ei = row.get("extra_info") if isinstance(row.get("extra_info"), dict) else {}
                disagreements.append({
                    "idx": idx,
                    "answer_type": ei.get("answer_type"),
                    "gold": str(row.get("gold_answer"))[:80],
                    "http": h,
                    "vllm": v,
                })

    n_total_xv = len(needs_xv)
    pct = 100 * n_agree / n_total_xv if n_total_xv else 100.0
    print()
    print("=" * 60)
    print(f"  rows total                : {len(df_merged)}")
    print(f"  rule-decided (no xverify) : {len(df_merged) - n_total_xv}")
    print(f"  routed to xverify         : {n_total_xv}")
    print(f"  HTTP vs vLLM agreement    : {n_agree}/{n_total_xv} = {pct:.2f}%")
    print(f"  HTTP=correct count        : {sum(http_correct.values())}")
    print(f"  vLLM=correct count        : {sum(int(df_vllm.iloc[i]['correct']) for i in needs_xv)}")
    if disagreements:
        print("\n  first disagreements:")
        for d in disagreements:
            print(f"    idx={d['idx']:5d}  type={d['answer_type']:12s}  http={d['http']}  vllm={d['vllm']}  gold={d['gold']!r}")


if __name__ == "__main__":
    main()
