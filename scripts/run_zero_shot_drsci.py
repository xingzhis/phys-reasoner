"""Zero-shot pass@1 inference on the Dr. SCI Goldilocks sample.

Mirrors run_zero_shot_nothink.py but reads from a Dr. SCI parquet whose
columns follow the Dr. SCI schema (extra_info.question, reward_model.ground_truth,
inferred_answer_type, extra_info.from, extra_info.difficulty).

Requires drsci_goldilocks_sample.parquet to exist (produced by
analyze_drsci_goldilocks.py --save_sample).

Usage
-----
  python scripts/run_zero_shot_drsci.py \\
      --input  data/processed/drsci_goldilocks_sample.parquet \\
      --model  Qwen/Qwen3.5-0.8B \\
      --output data/results/zero_shot_drsci_sample.parquet
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from phys_reasoner.verifier.router import verify_answer

_SYSTEM_PROMPT = (
    "Solve the physics problem step by step. Put your final answer in \\boxed{}."
)


# ---------------------------------------------------------------------------
# Load + validate Dr. SCI sample
# ---------------------------------------------------------------------------

def load_drsci_sample(parquet_path: str) -> pd.DataFrame:
    df = pd.read_parquet(parquet_path)
    # Ensure required columns
    required = ["extra_info.question", "reward_model.ground_truth"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in {parquet_path}: {missing}")
    # Normalize: add inferred_answer_type if absent
    if "inferred_answer_type" not in df.columns:
        print("  WARNING: inferred_answer_type not found — defaulting all to 'numerical'. "
              "Run drsci_audit.py --save_typed first for better accuracy.", flush=True)
        df = df.copy()
        df["inferred_answer_type"] = "numerical"
    return df


# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------

def build_prompt(problem: str, tokenizer) -> str:
    messages = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": problem},
    ]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,  # non-thinking mode (faster, cheaper for profiling)
    )


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def run_inference(
    df: pd.DataFrame,
    model_name: str,
    max_new_tokens: int,
    gpu_memory_utilization: float = 0.9,
) -> tuple[list[str], list[bool], list[int]]:
    from vllm import LLM, SamplingParams

    hf_home = os.environ.get("HF_HOME", "data/hf_cache")
    print(f"Loading model {model_name} (HF_HOME={hf_home})...", flush=True)

    llm = LLM(
        model=model_name,
        dtype="bfloat16",
        download_dir=hf_home,
        trust_remote_code=True,
        max_model_len=max_new_tokens + 4096,
        gpu_memory_utilization=gpu_memory_utilization,
    )
    tokenizer = llm.get_tokenizer()

    sampling_params = SamplingParams(
        temperature=0.7,
        top_p=0.8,
        top_k=20,
        max_tokens=max_new_tokens,
    )

    print(f"Building {len(df)} prompts...", flush=True)
    problems = df["extra_info.question"].tolist()
    prompts = [build_prompt(str(p), tokenizer) for p in problems]

    print(f"Running inference (thinking=OFF, max_new_tokens={max_new_tokens})...", flush=True)
    outputs = llm.generate(prompts, sampling_params)

    predictions: list[str] = []
    truncated: list[bool] = []
    token_counts: list[int] = []
    for out in outputs:
        text = out.outputs[0].text
        n_tok = len(out.outputs[0].token_ids)
        predictions.append(text)
        truncated.append(n_tok >= max_new_tokens)
        token_counts.append(n_tok)

    n_trunc = sum(truncated)
    if n_trunc:
        print(f"  Warning: {n_trunc}/{len(predictions)} outputs truncated at max_new_tokens={max_new_tokens}",
              flush=True)
    return predictions, truncated, token_counts


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def score_predictions(
    df: pd.DataFrame,
    predictions: list[str],
    truncated: list[bool],
    token_counts: list[int],
) -> pd.DataFrame:
    records = []
    for (_, row), pred_text, is_trunc, n_tok in zip(
        df.iterrows(), predictions, truncated, token_counts
    ):
        gold = str(row.get("reward_model.ground_truth") or "")
        at = str(row.get("inferred_answer_type") or "numerical")
        try:
            score = verify_answer(
                pred_text=pred_text,
                gold_answer=gold,
                answer_type=at,
                tolerance=0.05,
                xverify_judge=None,
            )
        except Exception:
            score = -1.0

        records.append({
            "drsci_from": row.get("extra_info.from", ""),
            "difficulty": row.get("extra_info.difficulty", None),
            "answer_type": at,
            "gold_answer": gold,
            "pred_text": pred_text,
            "n_tokens": n_tok,
            "score": score,
            "truncated": is_trunc,
        })
    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input",  default="data/processed/drsci_goldilocks_sample.parquet")
    parser.add_argument("--model",  default="Qwen/Qwen3.5-0.8B")
    parser.add_argument("--max_new_tokens", type=int, default=8192)
    parser.add_argument("--output", default="data/results/zero_shot_drsci_sample.parquet")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.9)
    args = parser.parse_args()

    print(f"Loading sample from {args.input}...", flush=True)
    df = load_drsci_sample(args.input)
    print(f"  Sample size: {len(df)} rows", flush=True)

    predictions, truncated, token_counts = run_inference(
        df, args.model, args.max_new_tokens,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )

    print("Scoring predictions...", flush=True)
    results_df = score_predictions(df, predictions, truncated, token_counts)

    print("\n=== Zero-shot Pass@1 on Dr. SCI Sample ===", flush=True)
    verifiable = results_df[results_df["score"] != -1.0]
    if len(verifiable):
        overall = verifiable["score"].mean()
        print(f"  Overall (rule-verifiable only): {overall:.1%}  (n={len(verifiable)})")
    else:
        print("  No verifiable rows — check inferred_answer_type distribution")

    # By source
    print("\n  By source (extra_info.from):")
    for src, grp in results_df.groupby("drsci_from", observed=True):
        verif = grp[grp["score"] != -1.0]
        acc = verif["score"].mean() if len(verif) else float("nan")
        print(f"    {str(src):<35} pass@1={acc:.1%}  n_verif={len(verif)}/{len(grp)}")

    # By difficulty bucket
    print("\n  By difficulty bucket:")
    results_df["diff_bucket"] = pd.cut(
        pd.to_numeric(results_df["difficulty"], errors="coerce").fillna(0.0),
        bins=[-0.001, 0.1, 0.25, 0.5, 0.75, 1.01],
        labels=["0.0", "0.1-0.25", "0.25-0.5", "0.5-0.75", "0.75+"],
    )
    for bucket, grp in results_df.groupby("diff_bucket", observed=True):
        verif = grp[grp["score"] != -1.0]
        acc = verif["score"].mean() if len(verif) else float("nan")
        print(f"    diff={bucket:<10} pass@1={acc:.1%}  n={len(grp)}")

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    results_df.to_parquet(args.output, index=False)
    print(f"\nSaved results to {args.output}", flush=True)


if __name__ == "__main__":
    main()
