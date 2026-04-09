"""Benchmark the verifier under simulated production load.

Hits a live xVerify HTTP server (set XVERIFY_URL=...) with synthesized
predictions against real golds from a training parquet, in two modes:

  Mode A: 1 thread, calls compute_score sequentially (matches what one
          VeRL trainer worker actually does today).

  Mode B: N threads in parallel (matches what would happen if VeRL ran
          multiple trainer workers, OR if a future reward manager ran
          per-sample compute_score concurrently inside one worker).

For each mode, prints wall time, number of xVerify calls actually made,
and per-call latency percentiles. The delta A vs B tells you whether the
server's GPU lock is the bottleneck (no speedup => yes; speedup => the
server can already absorb concurrency and batching would help further).

--judge-mode compare (default) runs both logprob and generate sequentially
on the same workload and prints latency + agreement stats so you can decide
whether the logprob shortcut is worth keeping.

Usage:
  XVERIFY_URL=http://misha00:8765/judge \\
    python3 scripts/bench_xverify.py \\
      --parquet data/processed/drsci_physics_clean.parquet \\
      --n 300 --threads 8 --judge-mode compare
"""

from __future__ import annotations

import argparse
import http.client
import json
import os
import random
import time
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import urlparse

import pandas as pd

# Make src importable so this script works when run from the repo root.
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from phys_reasoner.training.reward import compute_score, _get_xverify_judge


# ---------------------------------------------------------------------------
# Sample synthesis
# ---------------------------------------------------------------------------

def _make_pred(gold: str, mode: str) -> str:
    """Synthesize a prediction string for a given gold answer.

    mode = 'correct'  : exact gold inside \\boxed
    mode = 'perturb'  : symbolic perturbation that should fail rule but
                        is "structurally close" so xVerify is asked
    mode = 'wrong'    : completely unrelated string
    """
    g = str(gold).strip()
    if mode == "correct":
        return f"final answer is \\boxed{{{g}}}"
    if mode == "perturb":
        # Append "+ 1" or flip a sign — both rule-fail but parseable
        if g and g[0] == "-":
            return f"final answer is \\boxed{{{g[1:]}}}"
        return f"final answer is \\boxed{{{g} + 1}}"
    return "the final answer is \\boxed{42}"


def build_workload(parquet: str, n: int, seed: int) -> list[dict]:
    df = pd.read_parquet(parquet)
    rng = random.Random(seed)

    # Pick a realistic mix: ~half numerical (rule short-circuit dominant) +
    # half expression/equation (xVerify dominant). MCQ excluded — different code path.
    by_type = {}
    for t in ("numerical", "expression", "equation"):
        sub = df[df["inferred_answer_type"] == t]
        by_type[t] = sub.sample(min(n, len(sub)), random_state=seed).to_dict("records")

    target = {"numerical": n // 3, "expression": n // 3, "equation": n - 2 * (n // 3)}
    rows = []
    for t, k in target.items():
        rows.extend(by_type[t][:k])
    rng.shuffle(rows)

    workload = []
    for r in rows:
        gold = r.get("reward_model.ground_truth")
        if gold is None or (isinstance(gold, float) and pd.isna(gold)):
            continue
        atype = r.get("inferred_answer_type", "numerical")
        problem = r.get("extra_info.question", "")
        # Even split across the three pred flavors so we exercise all paths.
        flavor = rng.choice(["correct", "perturb", "wrong"])
        workload.append({
            "pred": _make_pred(gold, flavor),
            "gold": gold,
            "extra_info": {
                "answer_type": atype,
                "unit": "",
                "tolerance": 0.05,
                "problem": str(problem),
            },
            "_flavor": flavor,
            "_atype": atype,
        })
    return workload


# ---------------------------------------------------------------------------
# Per-call wrapper that records timing and whether xVerify was hit
# ---------------------------------------------------------------------------

class _CountingJudge:
    """Wraps the real xVerify client to count and time each call."""

    def __init__(self, inner):
        self._inner = inner
        self.calls = 0
        self.latencies_ms: list[float] = []

    def __call__(self, pred_str, gold_str, problem_str=""):
        t0 = time.perf_counter()
        result = self._inner(pred_str, gold_str, problem_str)
        self.latencies_ms.append((time.perf_counter() - t0) * 1000)
        self.calls += 1
        return result


def _run_one(work: dict, judge) -> tuple[float, float]:
    """Returns (score, wall_ms_for_this_compute_score_call)."""
    t0 = time.perf_counter()
    score = compute_score(
        solution_str=work["pred"],
        ground_truth=work["gold"],
        extra_info=work["extra_info"],
        xverify_judge=judge,  # explicit so the recording wrapper sticks
    )
    return score, (time.perf_counter() - t0) * 1000


def run_mode(workload: list[dict], threads: int, base_judge) -> dict:
    judge = _CountingJudge(base_judge) if base_judge is not None else None
    scores = [0.0] * len(workload)
    per_call_ms: list[float] = []

    t_start = time.perf_counter()
    if threads <= 1:
        for i, w in enumerate(workload):
            scores[i], ms = _run_one(w, judge)
            per_call_ms.append(ms)
    else:
        with ThreadPoolExecutor(max_workers=threads) as ex:
            futs = [ex.submit(_run_one, w, judge) for w in workload]
            for i, f in enumerate(futs):
                scores[i], ms = f.result()
                per_call_ms.append(ms)
    wall = time.perf_counter() - t_start

    return {
        "wall_s": wall,
        "n_samples": len(workload),
        "n_xverify_calls": judge.calls if judge else 0,
        "xverify_latencies_ms": judge.latencies_ms if judge else [],
        "per_compute_score_ms": per_call_ms,
        "score_distribution": {
            "1.0": sum(1 for s in scores if s == 1.0),
            "0.0": sum(1 for s in scores if s == 0.0),
        },
    }


def _pct(xs, p):
    if not xs:
        return float("nan")
    xs = sorted(xs)
    k = max(0, min(len(xs) - 1, int(round(p / 100 * (len(xs) - 1)))))
    return xs[k]


def _print_report(name: str, r: dict) -> None:
    n = r["n_samples"]
    nx = r["n_xverify_calls"]
    xv_lat = r["xverify_latencies_ms"]
    cs_lat = r["per_compute_score_ms"]
    print(f"\n=== {name} ===")
    print(f"  samples       : {n}")
    print(f"  wall          : {r['wall_s']:.2f} s   ({r['wall_s']/n*1000:.1f} ms/sample)")
    print(f"  xverify calls : {nx}  ({nx/n*100:.0f}% of samples hit xverify)")
    print(f"  scores        : 1.0={r['score_distribution']['1.0']}  0.0={r['score_distribution']['0.0']}")
    if xv_lat:
        print(f"  xverify ms    : p50={_pct(xv_lat,50):.0f}  p95={_pct(xv_lat,95):.0f}  max={max(xv_lat):.0f}")
    if cs_lat:
        print(f"  compute_score : p50={_pct(cs_lat,50):.0f}  p95={_pct(cs_lat,95):.0f}  max={max(cs_lat):.0f}")


class _ModeClient:
    """Minimal HTTP client that sends a specific judge mode to the server.

    Exists only for benchmarking — production code uses XVerifyHTTPClient
    which sends no mode field (server uses its configured default). Quacks
    like XVerifyJudge: callable as (pred, gold, problem).
    """

    def __init__(self, url: str, mode: str, timeout: float = 30.0):
        parsed = urlparse(url)
        self._host = parsed.hostname
        self._port = parsed.port or 80
        self._path = parsed.path or "/judge"
        self._mode = mode
        self._timeout = timeout

    def __call__(self, pred_str: str, gold_str: str, problem_str: str = "") -> bool:
        body = json.dumps(
            {"pred": pred_str, "gold": gold_str, "problem": problem_str, "mode": self._mode}
        ).encode("utf-8")
        conn = http.client.HTTPConnection(self._host, self._port, timeout=self._timeout)
        try:
            conn.request("POST", self._path, body=body,
                         headers={"Content-Type": "application/json"})
            resp = conn.getresponse()
            payload = json.loads(resp.read().decode("utf-8"))
            return bool(payload.get("correct", False))
        finally:
            conn.close()


def run_compare(workload: list[dict], url: str) -> None:
    """Run the same workload with logprob and generate, print agreement + latency."""
    modes = ("logprob", "generate")
    results = {}

    for mode in modes:
        client = _ModeClient(url, mode=mode)
        # warm-up
        client("\\boxed{1}", "1", "")

        latencies_ms: list[float] = []
        verdicts: list[bool] = []
        t_start = time.perf_counter()
        for w in workload:
            t0 = time.perf_counter()
            v = client(w["pred"], w["gold"], w.get("extra_info", {}).get("problem", ""))
            latencies_ms.append((time.perf_counter() - t0) * 1000)
            verdicts.append(v)
        wall = time.perf_counter() - t_start

        results[mode] = {"wall_s": wall, "latencies_ms": latencies_ms, "verdicts": verdicts}
        print(f"\n=== {mode} ===")
        print(f"  wall          : {wall:.2f} s   ({wall/len(workload)*1000:.1f} ms/call)")
        print(f"  p50/p95/max   : {_pct(latencies_ms,50):.0f} / {_pct(latencies_ms,95):.0f} / {max(latencies_ms):.0f} ms")
        print(f"  verdicts      : correct={sum(verdicts)}  incorrect={sum(1 for v in verdicts if not v)}")

    lp = results["logprob"]["verdicts"]
    gn = results["generate"]["verdicts"]
    agree = sum(a == b for a, b in zip(lp, gn))
    n = len(lp)
    disagree_lp_yes = sum(1 for a, b in zip(lp, gn) if a and not b)
    disagree_gn_yes = sum(1 for a, b in zip(lp, gn) if not a and b)
    speedup = results["generate"]["wall_s"] / results["logprob"]["wall_s"]

    print(f"\n=== logprob vs generate ===")
    print(f"  n samples     : {n}")
    print(f"  agreement     : {agree}/{n} ({agree/n*100:.1f}%)")
    print(f"  disagreements : logprob=correct & generate=incorrect: {disagree_lp_yes}")
    print(f"                  generate=correct & logprob=incorrect: {disagree_gn_yes}")
    print(f"  speedup       : logprob is {speedup:.2f}x faster than generate")
    print()
    if speedup > 2.0 and agree / n >= 0.98:
        print("  VERDICT: keep logprob — >2x faster, ≥98% agreement")
    elif speedup < 1.5:
        print("  VERDICT: marginal speedup (<1.5x) — revert to generate for safety")
    else:
        print(f"  VERDICT: borderline ({speedup:.2f}x speedup, {agree/n*100:.1f}% agreement) — review disagreements")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", required=True)
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--judge-mode",
        choices=["logprob", "generate", "compare"],
        default="logprob",
        help="logprob: production path (default); generate: original __call__; "
             "compare: run both and print agreement + speedup",
    )
    args = ap.parse_args()

    if not os.environ.get("XVERIFY_URL"):
        raise SystemExit("XVERIFY_URL must be set, e.g. http://misha00:8765/judge")

    xverify_url = os.environ["XVERIFY_URL"]

    print(f"loading workload from {args.parquet} (n={args.n}) ...")
    workload = build_workload(args.parquet, args.n, args.seed)
    print(f"  built {len(workload)} samples; flavor mix: "
          f"{ {f: sum(1 for w in workload if w['_flavor']==f) for f in ['correct','perturb','wrong']} }")

    if args.judge_mode == "compare":
        print(f"\n--- logprob vs generate comparison (sequential, n={len(workload)}) ---")
        run_compare(workload, xverify_url)
        return

    # Single-mode path: throughput bench (Mode A sequential + Mode B concurrent)
    if args.judge_mode == "generate":
        judge = _ModeClient(xverify_url, mode="generate")
        mode_label = "generate"
    else:
        # Production client: sends no mode field, server uses its configured default.
        judge = _get_xverify_judge()
        if judge is None:
            raise SystemExit("xverify judge could not be initialized — is XVERIFY_URL reachable?")
        mode_label = "server-default"

    # Warm-up: one call so the server's first-prompt overhead doesn't pollute Mode A.
    print("warm-up call ...")
    judge("\\boxed{1}", "1", "")

    print(f"\n--- Mode A: 1 thread (sequential) [{mode_label}] ---")
    rA = run_mode(workload, threads=1, base_judge=judge)
    _print_report(f"Mode A — sequential [{mode_label}]", rA)

    print(f"\n--- Mode B: {args.threads} threads (concurrent) [{mode_label}] ---")
    rB = run_mode(workload, threads=args.threads, base_judge=judge)
    _print_report(f"Mode B — {args.threads} threads [{mode_label}]", rB)

    speedup = rA["wall_s"] / rB["wall_s"] if rB["wall_s"] > 0 else float("nan")
    print(f"\n--- A vs B speedup: {speedup:.2f}x ---")
    if speedup < 1.2:
        print("  → server is the bottleneck (GPU lock serializes). To go faster,")
        print("    add server-side request coalescing (one forward pass per N requests).")
    else:
        print("  → concurrency helps. Server can absorb parallel clients;")
        print("    additional batching would still pay off but is less urgent.")


if __name__ == "__main__":
    main()
