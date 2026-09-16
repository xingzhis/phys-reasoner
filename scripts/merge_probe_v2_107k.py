#!/usr/bin/env python3
"""Merge probe1 (17,827-row subset) + probe_v2 (91,531-row remainder) rollouts
into a single ``rollouts_merged_107k.parquet`` covering the full 107,977-row
xingzhi0/phys-tir @ 39600de2 revision.

Dedup rule: group by md5(problem+'|||'+gold); if >8 rollouts exist for a given
hash (happens when refill rounds overlapped), keep the first 8 by arrival order
from the source-scan (stable ordering: probe1 first, then v2 fleet, then
v2 refill rounds 2/3/4). This matches training's GRPO group size n_rollouts=8.

Output includes a ``problem_hash`` column so downstream joins are cheap.

Runs in the project container (needs pyarrow; CPU-only, no GPU):

    source env.sh
    PYTHONNOUSERSITE=1 apptainer exec --overlay "$OVERLAY:ro" \\
      --bind /etc/pki:/etc/pki --env "PYTHONPATH=/opt/phys-extras/" "$SIF" \\
      python3 scripts/merge_probe_v2_107k.py
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import os
import sys
import time

import pyarrow as pa
import pyarrow.parquet as pq


ROOT = '/pscratch/sd/b/bwhou/17-phy-reasoner/phys-reasoner'


def _hash(problem: str | None, gold: str | None) -> str:
    s = (problem or '') + '|||' + (gold or '')
    return hashlib.md5(s.encode('utf-8')).hexdigest()[:16]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=f'{ROOT}/outputs/probe_qwen3_4b_v2_107k/rollouts_merged_107k.parquet')
    ap.add_argument('--source_parquet',
                    default=f'{ROOT}/data/probe_v2_phys_tir_39600de2/data/train.parquet',
                    help='Full 107k parquet for coverage verification.')
    ap.add_argument('--include_probe1', action='store_true', default=True,
                    help='Also include outputs/probe_qwen3_4b/rollouts_merged.parquet (the 131,568-row probe1 merged parquet).')
    ap.add_argument('--n_rollouts', type=int, default=8)
    args = ap.parse_args()

    if os.path.exists(args.out):
        print(f'ERROR: {args.out} already exists — refusing to overwrite', file=sys.stderr)
        return 1

    # Scan directories in priority order: probe1 (fleet), then v2 fleet, then refills.
    # Earlier sources win for the first-8 dedup policy.
    scan_roots = []
    if args.include_probe1:
        scan_roots.append(f'{ROOT}/outputs/probe_qwen3_4b/fleet')
    scan_roots += [
        f'{ROOT}/outputs/probe_qwen3_4b_v2_107k/fleet',
        f'{ROOT}/outputs/probe_qwen3_4b_v2_107k/refill_round2',
        f'{ROOT}/outputs/probe_qwen3_4b_v2_107k/refill_round3',
        f'{ROOT}/outputs/probe_qwen3_4b_v2_107k/refill_round4',
    ]

    files: list[str] = []
    for r in scan_roots:
        if not os.path.isdir(r):
            continue
        files += sorted(glob.glob(f'{r}/**/rollouts_chunk_*.parquet', recursive=True))
    print(f'[merge] found {len(files)} slot parquets across {len(scan_roots)} source trees')

    # One pass: read tables, assign problem_hash, keep first N rollouts per hash.
    counts: dict[str, int] = {}
    kept_batches: list[pa.Table] = []
    t0 = time.time()
    total_read = 0
    total_kept = 0
    for i, fp in enumerate(files):
        try:
            t = pq.read_table(fp)
        except Exception as exc:
            print(f'[merge] skip {fp}: {exc}', file=sys.stderr)
            continue
        total_read += t.num_rows
        ei = t.column('extra_info').to_pylist()
        gold = t.column('gold_answer').to_pylist()
        # Build mask: keep row iff hash count still < n_rollouts
        mask = []
        hashes = []
        for e, g in zip(ei, gold):
            prob = e.get('problem') if isinstance(e, dict) else None
            h = _hash(prob, g)
            hashes.append(h)
            if counts.get(h, 0) < args.n_rollouts:
                counts[h] = counts.get(h, 0) + 1
                mask.append(True)
            else:
                mask.append(False)
        if not any(mask):
            continue
        # Attach hash column first (before filter so we don't recompute)
        t = t.append_column('problem_hash', pa.array(hashes, type=pa.string()))
        t = t.filter(pa.array(mask))
        kept_batches.append(t)
        total_kept += t.num_rows
        if (i + 1) % 500 == 0:
            print(f'[merge] {i+1}/{len(files)} files scanned '
                  f'({total_read} read → {total_kept} kept, {time.time()-t0:.0f}s)')

    print(f'[merge] done: read {total_read} rollouts → kept {total_kept} '
          f'across {len(counts)} unique prompt hashes ({time.time()-t0:.0f}s)')

    # Coverage check
    src_tbl = pq.read_table(args.source_parquet)
    src_rows = src_tbl.to_pylist()
    src_h = {_hash(r['extra_info']['problem'], r['reward_model']['ground_truth'])
             for r in src_rows}
    full = {h for h, c in counts.items() if c >= args.n_rollouts}
    covered = len(full & src_h)
    partial = sum(1 for h in src_h if 0 < counts.get(h, 0) < args.n_rollouts)
    missing = len(src_h - set(counts.keys()))
    print(f'[merge] source (107k) coverage: {covered}/{len(src_h)} full, '
          f'{partial} partial, {missing} missing')

    if covered < len(src_h):
        print(f'[merge] WARNING: {len(src_h) - covered} prompts under-covered',
              file=sys.stderr)

    # Write merged parquet
    if not kept_batches:
        print('[merge] ERROR: no kept batches', file=sys.stderr)
        return 1
    # Drop any __-prefixed dataset metadata columns so schema is clean
    def _clean(t: pa.Table) -> pa.Table:
        keep_cols = [c for c in t.column_names if not c.startswith('__')]
        return t.select(keep_cols)
    kept_batches = [_clean(t) for t in kept_batches]
    merged = pa.concat_tables(kept_batches, promote_options='default')
    print(f'[merge] writing {merged.num_rows} rows × {merged.num_columns} cols → {args.out}')
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    pq.write_table(merged, args.out)
    size_mb = os.path.getsize(args.out) / 1e6
    print(f'[merge] wrote {size_mb:.1f} MB')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
