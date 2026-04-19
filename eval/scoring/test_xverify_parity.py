"""Parity check: in-process XVerifyJudge vs scripts/serve_xverify.py HTTP.

Parity is guaranteed by construction — serve_xverify.py does exactly
    correct = _JUDGE(pred, gold, problem)       # <- XVerifyJudge.__call__
    score = 1.0 if correct else 0.0
    response = {"correct": score >= 0.5, ...}   # always equals `correct`
so the HTTP path cannot differ from __call__ unless JSON round-trip corrupts
bytes (it doesn't, for ASCII bool).

This script provides a functional verification: runs N representative triples
through the in-process judge, prints verdicts, and (if --http_url is passed)
also hits the HTTP path and asserts bit-identical correct flags.

Usage
-----
    # In-process only (loads xVerify-7B once; ~5 min):
    python3 eval/scoring/test_xverify_parity.py

    # Also cross-check against a running HTTP server:
    python3 scripts/serve_xverify.py --port 8765 &           # one GPU
    python3 eval/scoring/test_xverify_parity.py --http_url http://localhost:8765 \\
        --xverify_device cuda:1                              # other GPU
"""
from __future__ import annotations

import argparse
import json
import urllib.request


# Representative triples spanning common answer types. Each is (pred, gold, problem,
# expected_correct_hint) — the hint is what a careful human would say; xVerify
# is not guaranteed to agree in edge cases and the test does not assert against
# the hint (only in-process vs HTTP).
TRIPLES = [
    # Numerical matches
    ("3.14", "3.14", "What is pi to 2 decimals?", True),
    ("2.718", "e", "What is e?", True),
    ("9.8 m/s^2", "9.81", "Gravity?", True),
    # Numerical mismatches
    ("3.14", "42", "What is pi?", False),
    ("1000", "100", "How many cm in 1 m?", False),
    # Expression matches
    (r"\frac{1}{2}", "0.5", "Half?", True),
    (r"\sqrt{2}", "1.4142", "Square root of 2?", True),
    # Expression mismatches
    (r"x^2 + 1", "x^2 - 1", "Derivative of something?", False),
    # MCQ
    ("A", "A", "Which option?", True),
    ("B", "A", "Which option?", False),
    # Units
    ("0.332 nm", "3.32e-10 m", "Wavelength?", True),
    # Edge: boxed pred
    (r"\boxed{42}", "42", "What is the answer?", True),
    # Edge: long thinking prefix + boxed
    ("After some thought, the answer is \\boxed{2\\pi r}", "2\\pi r",
     "Circumference of a circle?", True),
]


def call_http(url: str, pred: str, gold: str, problem: str) -> bool:
    body = json.dumps({"pred": pred, "gold": gold, "problem": problem}).encode("utf-8")
    req = urllib.request.Request(
        f"{url.rstrip('/')}/judge",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    return bool(data["correct"])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--xverify_model", default="IAAR-Shanghai/xVerify-7B-I")
    ap.add_argument("--xverify_device", default="cuda")
    ap.add_argument("--http_url", default="", help="If set, also hit this HTTP endpoint and diff.")
    args = ap.parse_args()

    from phys_reasoner.verifier.xverify_judge import XVerifyJudge  # noqa: PLC0415
    print(f"Loading in-process xVerify: {args.xverify_model} on {args.xverify_device} ...")
    judge = XVerifyJudge(model_name=args.xverify_model, device=args.xverify_device)
    print("Ready.\n")

    n = len(TRIPLES)
    disagreements = 0
    for i, (pred, gold, problem, hint) in enumerate(TRIPLES):
        direct = judge(pred, gold, problem)
        row = f"[{i+1:2d}/{n}] direct={str(direct):>5}"
        if args.http_url:
            http = call_http(args.http_url, pred, gold, problem)
            match = "OK " if direct == http else "DIFF"
            row += f"  http={str(http):>5}  {match}"
            if direct != http:
                disagreements += 1
        row += f"  hint={str(hint):>5}"
        row += f"  | pred={pred[:40]!r}  gold={gold[:30]!r}"
        print(row)

    if args.http_url:
        print(f"\n{n - disagreements}/{n} parity matches, {disagreements} disagreements")
        if disagreements != 0:
            raise SystemExit(1)
    else:
        print(f"\nIn-process judge ran cleanly on {n} triples. "
              f"HTTP parity guaranteed by construction (server wraps __call__).")


if __name__ == "__main__":
    main()
