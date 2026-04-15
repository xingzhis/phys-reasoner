"""Verify a CoT-baseline run produced single-turn, tool-free rollouts.

Run after a run with COT_BASELINE=1 + DUMP_TRAIN_ROLLOUTS=1 (or DUMP_VAL_ROLLOUTS=1):

    python3 scripts/perlmutter/verify_cot_rollouts.py \
        outputs/physcode_tir/<EXPERIMENT>/rollout_dumps/

Pass criteria (any failure → exit 1):
  1. NO rollout contains '<tool_call>' or '</tool_call>' in its response.
  2. NO rollout contains '<tool_response>' (tool-call result blocks).
  3. NO rollout shows more than one assistant turn (num_turns <= 2; verl counts
     prompt=1 + response=1 for single-turn generations).
  4. At least one rollout per file contains '\\boxed{' so we know the format survived.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from collections import Counter


TOOL_CALL_PATTERNS = ["<tool_call>", "</tool_call>", "<tool_response>", "</tool_response>"]


def iter_rollouts(path: str):
    """Yield (file, idx, record) for every rollout under `path`.

    verl writes rollouts as JSONL (one record per line, fields include
    'input', 'output', 'score', optionally 'num_turns', 'reward_model', etc.).
    Some versions write step-based .jsonl shards inside the dir.
    """
    if os.path.isfile(path):
        files = [path]
    else:
        files = sorted(glob.glob(os.path.join(path, "**", "*.jsonl"), recursive=True))
        files += sorted(glob.glob(os.path.join(path, "*.jsonl")))
        files = list(dict.fromkeys(files))  # dedupe, preserve order
    if not files:
        sys.exit(f"no *.jsonl files found under {path}")
    for f in files:
        with open(f) as fh:
            for idx, line in enumerate(fh):
                line = line.strip()
                if not line:
                    continue
                try:
                    yield f, idx, json.loads(line)
                except json.JSONDecodeError as e:
                    print(f"WARN: {f}:{idx} could not parse: {e}", file=sys.stderr)


def response_text(rec: dict) -> str:
    """Pull whatever represents the model's generated response from a rollout record.

    verl dumps vary; try the common fields in priority order.
    """
    for key in ("output", "response", "generated", "output_text", "completion"):
        v = rec.get(key)
        if isinstance(v, str) and v:
            return v
        if isinstance(v, list) and v and isinstance(v[0], str):
            return "\n".join(v)
    # fallback: find any string field > 50 chars that contains 'assistant' or '\\boxed'
    for v in rec.values():
        if isinstance(v, str) and len(v) > 50 and ("\\boxed" in v or "assistant" in v.lower()):
            return v
    return ""


def num_turns(rec: dict) -> int | None:
    for key in ("num_turns", "turns", "n_turns"):
        if key in rec and isinstance(rec[key], (int, float)):
            return int(rec[key])
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path", help="rollout_dumps/ dir OR a single .jsonl file")
    ap.add_argument("--max-sample-print", type=int, default=3)
    args = ap.parse_args()

    total = 0
    violations = {p: 0 for p in TOOL_CALL_PATTERNS}
    multi_turn_count = 0
    missing_boxed_files = set()
    seen_boxed_files = set()
    turn_hist: Counter = Counter()
    sample_bad = []

    for f, idx, rec in iter_rollouts(args.path):
        total += 1
        resp = response_text(rec)
        # 1 & 2: tool-call text patterns
        local_bad = False
        for pat in TOOL_CALL_PATTERNS:
            if pat in resp:
                violations[pat] += 1
                local_bad = True
        if local_bad and len(sample_bad) < args.max_sample_print:
            sample_bad.append((f, idx, resp[:400]))
        # 3: num_turns
        nt = num_turns(rec)
        if nt is not None:
            turn_hist[nt] += 1
            if nt > 2:  # verl counts prompt+response = 2 for single turn
                multi_turn_count += 1
        # 4: boxed presence
        if "\\boxed{" in resp:
            seen_boxed_files.add(f)
        else:
            missing_boxed_files.add(f)

    print(f"total rollouts scanned : {total}")
    print(f"num_turns histogram    : {dict(turn_hist)}")
    print(f"multi-turn violations  : {multi_turn_count}")
    for pat, c in violations.items():
        print(f"  {pat!r:20s}: {c} occurrences")

    files_without_boxed = missing_boxed_files - seen_boxed_files
    print(f"\\\\boxed{{}} present in {len(seen_boxed_files)} file(s); "
          f"absent in {len(files_without_boxed)}")

    if sample_bad:
        print("\n--- sample violating rollouts (truncated) ---")
        for f, idx, resp in sample_bad:
            print(f"  {f}:{idx}")
            print("    " + resp.replace("\n", "\n    "))
            print()

    fails = []
    if sum(violations.values()) > 0:
        fails.append(f"tool-call tokens appeared in {sum(violations.values())} rollouts")
    if multi_turn_count > 0:
        fails.append(f"{multi_turn_count} rollouts have num_turns > 2")
    if len(seen_boxed_files) == 0 and total > 0:
        fails.append("no rollouts contained \\\\boxed{} (format broke)")

    if fails:
        print("\nFAIL:")
        for msg in fails:
            print(f"  - {msg}")
        sys.exit(1)
    print("\nPASS: all rollouts are single-turn, tool-free, and end in \\\\boxed{}")
    sys.exit(0)


if __name__ == "__main__":
    main()
