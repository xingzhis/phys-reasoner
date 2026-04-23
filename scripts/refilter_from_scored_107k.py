"""Build Goldilocks-filtered train.parquet from 107k scored rollouts.

Reads outputs/probe_qwen3_4b_v2_107k/rollouts_scored.parquet, groups by
problem_hash (canonical, each hash has exactly 8 rollouts), computes
Goldilocks keep set for ratio-based [lo/8, hi/8], then filters source
data/probe_v2_phys_tir_39600de2/data/train.parquet by hash.

Validation/test: copied unchanged from data/processed_tir_filtered/data/
(the original 8k Goldilocks-filtered set) since the 107k probe source has
no val/test splits.

Output: data/processed_tir_filtered_v2_{lo}{hi}/data/{train,validation,test}.parquet

Usage:
  python3 scripts/refilter_from_scored_107k.py --lo 1 --hi 7
"""
from __future__ import annotations

import argparse
import hashlib
import shutil
from pathlib import Path

import pandas as pd


ROOT = Path('/pscratch/sd/b/bwhou/17-phy-reasoner/phys-reasoner')


def _hash(problem: str | None, gold: str | None) -> str:
    s = (problem or '') + '|||' + (gold or '')
    return hashlib.md5(s.encode('utf-8')).hexdigest()[:16]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--scored',
                    default='outputs/probe_qwen3_4b_v2_107k/rollouts_scored.parquet')
    ap.add_argument('--source',
                    default='data/probe_v2_phys_tir_39600de2/data/train.parquet')
    ap.add_argument('--valtest_from',
                    default='data/processed_tir_filtered/data',
                    help='Directory to copy validation.parquet and test.parquet from')
    ap.add_argument('--lo', type=int, required=True, help='min n_correct (inclusive) out of 8')
    ap.add_argument('--hi', type=int, required=True, help='max n_correct (inclusive) out of 8')
    args = ap.parse_args()

    suffix = f'{args.lo}{args.hi}'
    dst_dir = ROOT / f'data/processed_tir_filtered_v2_{suffix}/data'
    if dst_dir.exists():
        raise FileExistsError(f'{dst_dir} exists — remove first')

    print(f'[refilter] reading {args.scored}')
    sc = pd.read_parquet(ROOT / args.scored)
    agg = sc.groupby('problem_hash').agg(n_tot=('correct', 'count'), n_cor=('correct', 'sum')).reset_index()
    print(f'[refilter] {len(agg)} unique problems in scored set')

    kept = agg[(agg['n_cor'] >= args.lo) & (agg['n_cor'] <= args.hi) & (agg['n_tot'] == 8)]
    keep_hashes = set(kept['problem_hash'].tolist())
    print(f'[refilter] kept {len(keep_hashes)}/{len(agg)} hashes (range [{args.lo},{args.hi}] / 8)')

    print(f'[refilter] reading source {args.source}')
    src = pd.read_parquet(ROOT / args.source)
    print(f'[refilter] source rows: {len(src)}')

    # Compute hash per source row
    hashes = []
    for _, r in src.iterrows():
        prob = r['extra_info'].get('problem') if isinstance(r['extra_info'], dict) else None
        gold = r['reward_model'].get('ground_truth') if isinstance(r['reward_model'], dict) else None
        hashes.append(_hash(prob, gold))
    src = src.assign(_h=hashes)
    kept_src = src[src['_h'].isin(keep_hashes)].drop(columns=['_h']).reset_index(drop=True)
    print(f'[refilter] kept {len(kept_src)} source rows (expected ~{len(keep_hashes)})')

    dst_dir.mkdir(parents=True, exist_ok=False)
    kept_src.to_parquet(dst_dir / 'train.parquet', index=False)
    print(f'[write] train -> {dst_dir / "train.parquet"}')

    valtest = ROOT / args.valtest_from
    for split in ('validation', 'test'):
        src_file = valtest / f'{split}.parquet'
        if src_file.exists():
            shutil.copy2(src_file, dst_dir / f'{split}.parquet')
            print(f'[copy ] {split} <- {src_file}')
        else:
            print(f'[warn ] missing {src_file}')

    print(f'[done ] suffix=_v2_{suffix}; submit with TRAIN_FILES=data/processed_tir_filtered_v2_{suffix}/data/train.parquet')


if __name__ == '__main__':
    main()
