"""Build DAPO-17k (BytedTsinghua-SIA) -> verl CoT parquet, 20k subsample.

DAPO-17k is a Goldilocks-filtered math set explicitly curated for GRPO
training; its size-class matches our physics [2,6] filter and the reward
is pure-numeric rule-verifiable (no xverify noise), making it the cleanest
"does our pipeline learn on known-good math" ablation arm.

Repo ships ~1.79M expanded candidates; we subsample 20k random rows
(seed=42) to fit our 150-step x batch=128 budget (~19k samples touched).

Writes data/processed_dapo17k_cot/data/{train,validation,test}.parquet in
the same 5-column schema used by processed_cot.

Usage:
  python3 scripts/build_dapo_math_cot_parquet.py
"""
from __future__ import annotations

from pathlib import Path

import datasets
import pandas as pd


ROOT = Path('/pscratch/sd/b/bwhou/17-phy-reasoner/phys-reasoner')
DST = ROOT / 'data/processed_dapo17k_cot/data'

COT_SYSTEM = (
    "You are a careful math problem solver.\n"
    "Solve the problem step by step. Show your reasoning concisely.\n\n"
    "End with your final answer as \\boxed{<value>}.\n"
)

N_TRAIN = 20000
N_VAL = 200
SEED = 42


def row_to_verl(row, idx: int, split: str) -> dict:
    # DAPO prompt is a chat array with a single user turn containing the problem
    # (system="Solve ... The last line ... Answer: $Answer"). We replace the
    # system with our \boxed{} instruction so our rule verifier works.
    original_prompt = row['prompt']
    if isinstance(original_prompt, list) and len(original_prompt) > 0:
        # Take the user-turn content (last in array)
        user_content = original_prompt[-1].get('content', '')
    else:
        user_content = str(original_prompt)

    # DAPO prepends boilerplate "Solve the following math problem...\n\nAnswer: $Answer...\n\n<PROBLEM>"
    # to every prompt; strip the leading boilerplate so model sees just the problem
    # under our own \boxed{} system prompt.
    if '\n\n' in user_content:
        parts = user_content.split('\n\n', 1)
        if parts[0].lower().startswith('solve the following math problem'):
            user_content = parts[1].strip()

    gold = row['reward_model']['ground_truth']
    return {
        'data_source': 'BytedTsinghua-SIA/DAPO-Math-17k',
        'prompt': [
            {'role': 'system', 'content': COT_SYSTEM},
            {'role': 'user', 'content': user_content},
        ],
        'reward_model': {
            'style': 'rule',
            'ground_truth': f'\\boxed{{{gold}}}',
        },
        'extra_info': {
            'answer_type': 'numerical',
            'split': split,
            'index': idx,
            'problem': user_content,
            'from': 'dapo17k',
            'orig_index': row['extra_info'].get('index', ''),
        },
        'pool': 'dapo17k',
    }


def main() -> None:
    if DST.exists():
        raise FileExistsError(f'{DST} exists — remove first')

    print('[dapo] loading BytedTsinghua-SIA/DAPO-Math-17k ...')
    ds = datasets.load_dataset('BytedTsinghua-SIA/DAPO-Math-17k')
    full = ds['train']
    print(f'[dapo] full train rows: {len(full)}')

    total_needed = N_TRAIN + N_VAL
    shuffled = full.shuffle(seed=SEED).select(range(total_needed))

    train_rows = [row_to_verl(r, i, 'train') for i, r in enumerate(shuffled.select(range(N_TRAIN)))]
    val_rows = [row_to_verl(r, i, 'validation')
                for i, r in enumerate(shuffled.select(range(N_TRAIN, total_needed)))]

    DST.mkdir(parents=True, exist_ok=False)
    pd.DataFrame(train_rows).to_parquet(DST / 'train.parquet', index=False)
    pd.DataFrame(val_rows).to_parquet(DST / 'validation.parquet', index=False)
    # No test split defined in DAPO; copy val as test placeholder
    pd.DataFrame(val_rows).to_parquet(DST / 'test.parquet', index=False)

    print(f'[write] train  : {N_TRAIN} -> {DST / "train.parquet"}')
    print(f'[write] val    : {N_VAL} -> {DST / "validation.parquet"}')
    print(f'[write] test   : {N_VAL} -> {DST / "test.parquet"}  (val placeholder)')


if __name__ == '__main__':
    main()
