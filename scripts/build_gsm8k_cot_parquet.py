"""Build GSM8K -> verl CoT parquet (infra sanity ablation arm).

Writes data/processed_gsm8k_cot/data/{train,validation,test}.parquet in the same
schema used by processed_cot (data_source, prompt, reward_model, extra_info, pool).

Reward style="rule"; ground_truth wrapped as "\\boxed{<number>}" so the rule
verifier latches on the standard \\boxed{} extraction. xVerify is skipped at
runtime for clean integer match (no FN noise), letting GSM8K act as a pure
infra sanity check.

Usage:
  python3 scripts/build_gsm8k_cot_parquet.py
"""
from __future__ import annotations

import re
from pathlib import Path

import datasets
import pandas as pd


ROOT = Path('/pscratch/sd/b/bwhou/17-phy-reasoner/phys-reasoner')
DST = ROOT / 'data/processed_gsm8k_cot/data'

COT_SYSTEM = (
    "You are a careful math problem solver.\n"
    "Solve the problem step by step. Show your reasoning concisely.\n\n"
    "End with your final answer as \\boxed{<value>}.\n"
)


def extract_gsm8k_answer(ans: str) -> str:
    m = re.search(r'####\s*([\-0-9\.\,]+)', ans)
    if m is None:
        return ans.strip()
    return m.group(1).replace(',', '').strip()


def to_verl_rows(split_rows, split_name: str) -> list[dict]:
    out = []
    for idx, row in enumerate(split_rows):
        q = row['question']
        gold = extract_gsm8k_answer(row['answer'])
        out.append({
            'data_source': 'openai/gsm8k',
            'prompt': [
                {'role': 'system', 'content': COT_SYSTEM},
                {'role': 'user', 'content': q},
            ],
            'reward_model': {
                'style': 'rule',
                'ground_truth': f'\\boxed{{{gold}}}',
            },
            'extra_info': {
                'answer_type': 'numerical',
                'split': split_name,
                'index': idx,
                'problem': q,
                'from': 'gsm8k',
            },
            'pool': 'gsm8k',
        })
    return out


def main() -> None:
    if DST.exists():
        raise FileExistsError(f'{DST} exists — remove first')
    ds = datasets.load_dataset('openai/gsm8k', 'main')
    print(f'[gsm8k] train={len(ds["train"])}  test={len(ds["test"])}')

    DST.mkdir(parents=True, exist_ok=False)

    # Train: full 7473
    train_rows = to_verl_rows(list(ds['train']), 'train')
    train_df = pd.DataFrame(train_rows)
    train_df.to_parquet(DST / 'train.parquet', index=False)
    print(f'[write] train: {len(train_df)} -> {DST / "train.parquet"}')

    # Val/test from gsm8k test split (1319): split 200 val / 1119 test
    test_all = list(ds['test'])
    val_rows = to_verl_rows(test_all[:200], 'validation')
    test_rows = to_verl_rows(test_all[200:], 'test')
    pd.DataFrame(val_rows).to_parquet(DST / 'validation.parquet', index=False)
    pd.DataFrame(test_rows).to_parquet(DST / 'test.parquet', index=False)
    print(f'[write] validation: 200 -> {DST / "validation.parquet"}')
    print(f'[write] test: {len(test_rows)} -> {DST / "test.parquet"}')


if __name__ == '__main__':
    main()
