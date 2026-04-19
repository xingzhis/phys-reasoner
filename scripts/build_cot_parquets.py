"""Rewrite TIR training parquets into a CoT (no-tool) variant for the baseline ablation.

Reads an input dir (default: `data/processed_tir/data/`) containing
`{train,validation,test}.parquet`, replaces every row's system message with a
CoT-only prompt (no mention of Python/tools/code), and writes a matching
parquet triple to the output dir. Every other column is byte-preserved.

Usage:
    python3 scripts/build_cot_parquets.py \
        --in-dir  data/processed_tir/data \
        --out-dir data/processed_cot/data

Then push to HF (same pattern as phys-tir):
    HF_TOKEN=hf_... python3 scripts/push_dataset.py \
        --repo-id <user>/phys-cot \
        --in-dir data/processed_cot/data

The script HALTS before writing if any sanity check fails — never produces
half-baked data.
"""
from __future__ import annotations

import argparse
import hashlib
import os
import sys
from pathlib import Path

import pandas as pd

# Canonical CoT system prompt lives alongside TIR_SYSTEM_PROMPT for single-source truth.
from phys_reasoner.tir.prompts import COT_SYSTEM_PROMPT


SPLITS = ("train", "validation", "test")


def _swap_system(msgs):
    """Return a new message list with msgs[0].content replaced by the CoT prompt.

    Works with both list-of-dict and numpy arrays of dicts (pyarrow-loaded).
    Raises if the first message isn't a system message (invariant violation).
    """
    msgs = list(msgs)
    if not msgs:
        raise ValueError("empty prompt message list")
    first = dict(msgs[0])
    if first.get("role") != "system":
        raise ValueError(f"first message role is {first.get('role')!r}, expected 'system'")
    first["content"] = COT_SYSTEM_PROMPT
    msgs[0] = first
    return msgs


def _hash_nonprompt_row(row: pd.Series) -> str:
    """Hash everything except the prompt column for byte-preservation checks."""
    items = []
    for k in sorted(row.index):
        if k == "prompt":
            continue
        v = row[k]
        # pyarrow gives us numpy scalars / arrays of dicts; str() is stable enough
        # for a byte-preservation check (we compare before vs after the swap).
        items.append((k, str(v)))
    return hashlib.sha256(repr(items).encode()).hexdigest()


def rewrite_one(src: Path, dst: Path) -> tuple[int, list[str]]:
    print(f"\n=== {src.name} ===")
    df = pd.read_parquet(src)
    n_in = len(df)
    cols_in = list(df.columns)
    print(f"  rows: {n_in}")
    print(f"  cols: {cols_in}")

    if "prompt" not in cols_in:
        raise SystemExit(f"FAIL: {src} has no 'prompt' column")

    # Capture the pre-swap hash for every row's non-prompt fields. After the
    # swap these must be IDENTICAL — otherwise we accidentally touched something.
    pre_hashes = df.apply(_hash_nonprompt_row, axis=1).tolist()

    # Capture original user questions so we can confirm they survive intact.
    def user_text(msgs):
        for m in list(msgs):
            if dict(m).get("role") == "user":
                return dict(m).get("content", "")
        return ""
    pre_user = df["prompt"].apply(user_text).tolist()

    # Also confirm pre-swap the system prompt actually matches the TIR prompt
    # (i.e. this is a TIR parquet, not something already rewritten).
    def sys_text(msgs):
        first = dict(list(msgs)[0])
        return first.get("content", "") if first.get("role") == "system" else ""
    pre_sys = df["prompt"].apply(sys_text).tolist()
    unique_pre_sys = set(pre_sys)
    print(f"  unique system prompts before: {len(unique_pre_sys)}")
    if len(unique_pre_sys) > 1:
        print("  (more than one system prompt detected — rewrite will UNIFY them to CoT)")

    # ---- swap ----
    df["prompt"] = df["prompt"].apply(_swap_system)

    # ---- sanity checks (halt before writing if anything is off) ----
    errs: list[str] = []

    # 1. row count preserved
    if len(df) != n_in:
        errs.append(f"row count drift: {n_in} -> {len(df)}")

    # 2. columns preserved exactly
    if list(df.columns) != cols_in:
        errs.append(f"column set drift: {cols_in} -> {list(df.columns)}")

    # 3. non-prompt columns byte-identical
    post_hashes = df.apply(_hash_nonprompt_row, axis=1).tolist()
    mismatch = sum(1 for a, b in zip(pre_hashes, post_hashes) if a != b)
    if mismatch:
        errs.append(f"{mismatch}/{n_in} rows had non-prompt column drift")

    # 4. user-message content unchanged
    post_user = df["prompt"].apply(user_text).tolist()
    user_mismatch = sum(1 for a, b in zip(pre_user, post_user) if a != b)
    if user_mismatch:
        errs.append(f"{user_mismatch}/{n_in} rows had user-message drift")

    # 5. every row now has EXACTLY the CoT system prompt
    post_sys = df["prompt"].apply(sys_text).tolist()
    cot_mismatch = sum(1 for s in post_sys if s != COT_SYSTEM_PROMPT)
    if cot_mismatch:
        errs.append(f"{cot_mismatch}/{n_in} rows don't carry the new CoT system prompt")

    # 6. no tool keywords in the new system prompt (paranoia)
    banned = ["python", "tool", "code", "interpreter", "execute"]
    hits = [w for w in banned if w.lower() in COT_SYSTEM_PROMPT.lower()]
    if hits:
        errs.append(f"CoT prompt contains tool-related words: {hits}")

    if errs:
        print("  SANITY FAIL:")
        for e in errs:
            print(f"    - {e}")
        raise SystemExit(1)

    # ---- write ----
    dst.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(dst, index=False)
    print(f"  wrote {len(df)} rows to {dst}")

    # Spot-check round-trip.
    rt = pd.read_parquet(dst)
    rt_sys = rt["prompt"].apply(sys_text).tolist()
    if any(s != COT_SYSTEM_PROMPT for s in rt_sys):
        raise SystemExit(f"FAIL: round-trip read of {dst} does not carry the CoT prompt")

    return n_in, []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dir", default="data/processed_tir/data",
                    help="dir containing {train,validation,test}.parquet (TIR variant)")
    ap.add_argument("--out-dir", default="data/processed_cot/data",
                    help="dir to write the CoT variant; created if missing")
    args = ap.parse_args()

    in_dir = Path(args.in_dir).resolve()
    out_dir = Path(args.out_dir).resolve()
    if in_dir == out_dir:
        raise SystemExit("refuse to overwrite TIR parquets in place — use a separate --out-dir")

    print(f"CoT system prompt (will be applied to every row):")
    print("---")
    print(COT_SYSTEM_PROMPT)
    print("---\n")

    total = 0
    for split in SPLITS:
        src = in_dir / f"{split}.parquet"
        dst = out_dir / f"{split}.parquet"
        if not src.exists():
            raise SystemExit(f"FAIL: input missing: {src}")
        n, _ = rewrite_one(src, dst)
        total += n

    print(f"\n=== done. {total} rows rewritten to {out_dir} ===")
    print(f"Next: HF_TOKEN=hf_... python3 scripts/push_dataset.py "
          f"--repo-id <user>/phys-cot --in-dir {out_dir}")


if __name__ == "__main__":
    main()
