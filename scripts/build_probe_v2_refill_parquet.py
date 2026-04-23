#!/usr/bin/env python3
"""Build a refill parquet for probe_qwen3_4b_v2_107k.

Scans every slot's saved intermediate parquets under
outputs/probe_qwen3_4b_v2_107k/{fleet,refill_round*}/**/rollouts_chunk_*.parquet,
identifies prompts (keyed by md5(problem+'|||'+ground_truth)) that have < 8
rollouts, and writes a new parquet containing just those un-done rows, preserving
the original row order of remainder_vs_probe1.parquet.

Run inside the project container (has pyarrow). Example:

    source env.sh
    PYTHONNOUSERSITE=1 apptainer exec --overlay "$OVERLAY:ro" \\
      --bind /etc/pki:/etc/pki --env "PYTHONPATH=/opt/phys-extras/" "$SIF" \\
      python3 scripts/build_probe_v2_refill_parquet.py --round 2

Outputs (with --round N):
    data/probe_v2_phys_tir_39600de2/refill_round<N>.parquet
    data/probe_v2_phys_tir_39600de2/refill_round<N>.json  (diagnostics)
"""

from __future__ import annotations
import argparse
import glob
import hashlib
import json
import os
import sys

import pyarrow as pa
import pyarrow.parquet as pq


ROOT = '/pscratch/sd/b/bwhou/17-phy-reasoner/phys-reasoner'
SOURCE_PARQUET = f'{ROOT}/data/probe_v2_phys_tir_39600de2/remainder_vs_probe1.parquet'
OUTPUT_BASE = f'{ROOT}/outputs/probe_qwen3_4b_v2_107k'


def _hash_row(problem: str | None, gt: str | None) -> str:
    s = (problem or '') + '|||' + (gt or '')
    return hashlib.md5(s.encode('utf-8')).hexdigest()[:16]


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument('--round', type=int, required=True,
                   help='Refill round number (e.g. 2 after initial fleet).')
    p.add_argument('--min_rollouts', type=int, default=8,
                   help='Rows with >= this many rollouts are considered done.')
    p.add_argument('--source', default=SOURCE_PARQUET)
    p.add_argument('--scan_roots', nargs='+',
                   default=[f'{OUTPUT_BASE}/fleet',
                            f'{OUTPUT_BASE}/refill_round1',
                            f'{OUTPUT_BASE}/refill_round2',
                            f'{OUTPUT_BASE}/refill_round3'],
                   help='Directories to scan for rollouts_chunk_*.parquet.')
    p.add_argument('--out_dir', default=f'{ROOT}/data/probe_v2_phys_tir_39600de2')
    args = p.parse_args()

    source_tbl = pq.read_table(args.source)
    n_source = source_tbl.num_rows
    print(f'[refill] source: {args.source}  rows={n_source}')

    # Source hashes, row-ordered
    src_rows = source_tbl.to_pylist()
    src_hashes = [_hash_row(r['extra_info']['problem'], r['reward_model']['ground_truth'])
                  for r in src_rows]

    # Count completed rollouts per hash across every saved slot parquet.
    counts: dict[str, int] = {}
    files_scanned = 0
    rollouts_scanned = 0
    for root_dir in args.scan_roots:
        if not os.path.isdir(root_dir):
            continue
        for fp in glob.glob(f'{root_dir}/**/rollouts_chunk_*.parquet', recursive=True):
            try:
                t = pq.read_table(fp, columns=['extra_info', 'gold_answer'])
            except Exception as exc:
                print(f'[refill] skip {fp}: {exc}', file=sys.stderr)
                continue
            ei = t.column('extra_info').to_pylist()
            gold = t.column('gold_answer').to_pylist()
            for e, g in zip(ei, gold):
                h = _hash_row(e.get('problem') if isinstance(e, dict) else None, g)
                counts[h] = counts.get(h, 0) + 1
            rollouts_scanned += t.num_rows
            files_scanned += 1
    print(f'[refill] scanned {files_scanned} parquet files, {rollouts_scanned} rollouts total')

    done = {h for h, c in counts.items() if c >= args.min_rollouts}
    print(f'[refill] done hashes (>={args.min_rollouts} rollouts): {len(done)} / {n_source}')

    # Keep row-order from source; mask undone rows
    mask = [h not in done for h in src_hashes]
    n_out = sum(mask)
    print(f'[refill] remaining: {n_out}  (coverage so far: {100*(n_source-n_out)/n_source:.1f}%)')

    if n_out == 0:
        print('[refill] nothing to refill — full coverage')
        return 0

    out_tbl = source_tbl.filter(pa.array(mask))
    assert out_tbl.num_rows == n_out

    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(args.out_dir, f'refill_round{args.round}.parquet')
    if os.path.exists(out_path):
        print(f'[refill] ERROR: {out_path} already exists — refusing to overwrite', file=sys.stderr)
        return 1
    pq.write_table(out_tbl, out_path)
    size_mb = os.path.getsize(out_path) / 1e6
    print(f'[refill] wrote {out_path} ({n_out} rows, {size_mb:.1f} MB)')

    diag_path = os.path.join(args.out_dir, f'refill_round{args.round}.json')
    diag = {
        'source_parquet': args.source,
        'source_rows': n_source,
        'scan_roots': args.scan_roots,
        'files_scanned': files_scanned,
        'rollouts_scanned': rollouts_scanned,
        'done_prompts': len(done),
        'refill_parquet': out_path,
        'refill_rows': n_out,
        'coverage_before_refill_pct': round(100*(n_source-n_out)/n_source, 3),
    }
    with open(diag_path, 'w') as f:
        json.dump(diag, f, indent=2)
    print(f'[refill] wrote diagnostics → {diag_path}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
