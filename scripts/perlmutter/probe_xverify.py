"""Standalone diagnostic: poke the xVerify server with known-answer pairs.

Run this BEFORE submitting a training job to catch dead/misconfigured servers
early. Covers numerical, expression, MCQ, and intentional-junk cases.

Usage:
    python3 scripts/perlmutter/probe_xverify.py                          # reads rendezvous file
    python3 scripts/perlmutter/probe_xverify.py --url http://host:8765/judge

Pass criteria (printed at the end):
    - All "gold == pred" cases return correct=True
    - All "gold != pred" cases return correct=False
    - p95 latency < 2 s (warmed up)

Exit code non-zero if any criterion fails.
"""
import argparse
import json
import os
import statistics
import sys
import time
import urllib.request
import urllib.error

CASES = [
    # (problem, gold, pred, expected_correct, kind)
    ("Compute 2+2.", "4", "4", True, "numerical/exact"),
    ("Compute 2+2.", "4", "5", False, "numerical/wrong"),
    ("What is pi to 3 decimals?", "3.142", "3.141", True, "numerical/tolerance"),
    ("Energy of photon with frequency f.", "h*f", "h f", True, "expression/equiv-spacing"),
    ("Solve x^2=4 for positive x.", "2", "\\sqrt{4}", True, "expression/simplify"),
    ("Kinetic energy formula.", "\\frac{1}{2}mv^2", "0.5 m v^2", True, "expression/format"),
    ("Which option is correct?", "B", "B", True, "mcq/exact"),
    ("Which option is correct?", "B", "C", False, "mcq/wrong"),
    ("Nonsense question.", "42", "asdfghjkl", False, "junk"),
    ("Compute 1/0.", "undefined", "", False, "empty-pred"),
]


def load_url(url_arg: str | None) -> str:
    if url_arg:
        return url_arg
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    path = os.path.join(root, "outputs", "xverify_endpoints", "current.url")
    if not os.path.exists(path):
        sys.exit(f"no --url given and rendezvous file missing: {path}")
    with open(path) as f:
        return f.readline().strip()


def call(url: str, problem: str, gold: str, pred: str, timeout: float = 30.0):
    body = json.dumps({"problem": problem, "gold": gold, "pred": pred}).encode()
    req = urllib.request.Request(
        url, data=body, headers={"Content-Type": "application/json"}, method="POST"
    )
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        raw = resp.read()
    dt = time.perf_counter() - t0
    return json.loads(raw), dt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", help="override xverify URL (e.g. http://host:8765/judge)")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    url = load_url(args.url)
    health_url = url.rsplit("/judge", 1)[0] + "/health"
    print(f"xverify URL   : {url}")
    print(f"xverify health: {health_url}")
    try:
        with urllib.request.urlopen(health_url, timeout=5) as r:
            print(f"  /health -> {r.status}")
    except Exception as e:
        print(f"  /health FAILED: {e}")

    # Warm up.
    try:
        call(url, *CASES[0][:3])
    except Exception as e:
        sys.exit(f"warmup failed, server unreachable: {e}")

    latencies = []
    failures = []
    print()
    print(f"{'kind':<28} {'expect':<7} {'got':<7} {'latency':<9} ok")
    print("-" * 70)
    for problem, gold, pred, expected, kind in CASES:
        try:
            result, dt = call(url, problem, gold, pred)
        except Exception as e:
            failures.append((kind, f"error: {e}"))
            print(f"{kind:<28} {str(expected):<7} {'ERR':<7} {'-':<9} ✗")
            continue
        latencies.append(dt)
        got = bool(result.get("correct"))
        ok = got == expected
        if not ok:
            failures.append((kind, f"expected {expected}, got {got}; raw={result}"))
        marker = "✓" if ok else "✗"
        print(f"{kind:<28} {str(expected):<7} {str(got):<7} {dt*1000:>6.0f}ms  {marker}")

    print()
    if latencies:
        p50 = statistics.median(latencies)
        p95 = sorted(latencies)[int(0.95 * len(latencies)) - 1]
        print(f"latency: p50={p50*1000:.0f}ms  p95={p95*1000:.0f}ms  n={len(latencies)}")
    print(f"failures: {len(failures)}")
    for kind, msg in failures:
        print(f"  - {kind}: {msg}")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
