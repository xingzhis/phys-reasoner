"""Zero-shot pass@k inference on a stratified sample of the 6.8k training corpus.

Mirrors run_zero_shot_drsci.py but reads from candidates_deduped.parquet whose
columns follow the corpus schema (problem, answer, answer_type, source, difficulty).

With --n_samples k (default 8), generates k completions per problem using
vLLM's native n= parameter and records per-problem pass_rate = correct/k.
This is the input for Goldilocks difficulty estimation on the 6.8k corpus.

Usage
-----
  # Pass@8 for Goldilocks profiling (default):
  python scripts/run_zero_shot_corpus_passk.py \\
      --input  data/processed/candidates_deduped.parquet \\
      --model  Qwen/Qwen3.5-4B \\
      --output data/results/zero_shot_corpus_passk.parquet

  # Quick accuracy check (pass@1):
  python scripts/run_zero_shot_corpus_passk.py --n_samples 1 ...
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from phys_reasoner.training.reward import compute_score

_SYSTEM_PROMPT = (
    "Solve the physics problem step by step. Put your final answer in \\boxed{}."
)


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------

def _simplify_type(at: str) -> str:
    """Bucket the 87 distinct answer_type values into 5 groups for stratification."""
    at = str(at).strip()
    if at.startswith("["):
        return "multi"
    return at if at in ("numerical", "expression", "equation", "mcq") else "other"


def build_sample(
    parquet_path: str,
    n_per_stratum: int = 20,
    seed: int = 42,
) -> pd.DataFrame:
    """Build a stratified sample by (source, simplified_answer_type).

    With 5 sources × up to 5 type groups × 20 per stratum → ~500 rows max.
    """
    df = pd.read_parquet(parquet_path)
    df = df.copy()
    df["_simple_type"] = df["answer_type"].apply(_simplify_type)

    parts = []
    for _, grp in df.groupby(["source", "_simple_type"], observed=True):
        parts.append(grp.sample(min(len(grp), n_per_stratum), random_state=seed))

    sample = pd.concat(parts).reset_index(drop=True)
    sample = sample.drop(columns=["_simple_type"])
    return sample


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
        enable_thinking=False,
    )


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def run_inference(
    df: pd.DataFrame,
    model_name: str,
    max_new_tokens: int,
    n_samples: int = 8,
    gpu_memory_utilization: float = 0.9,
) -> list[list[dict]]:
    """Run inference and return k outputs per problem."""
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
        n=n_samples,
        temperature=0.7,
        top_p=0.8,
        top_k=20,
        max_tokens=max_new_tokens,
    )

    print(f"Building {len(df)} prompts...", flush=True)
    prompts = [build_prompt(str(row["problem"]), tokenizer) for _, row in df.iterrows()]

    print(f"Running inference (n_samples={n_samples}, thinking=OFF, "
          f"max_new_tokens={max_new_tokens})...", flush=True)
    outputs = llm.generate(prompts, sampling_params)

    all_outputs: list[list[dict]] = []
    total_trunc = 0
    for out in outputs:
        samples = []
        for completion in out.outputs:
            n_tok = len(completion.token_ids)
            trunc = n_tok >= max_new_tokens
            if trunc:
                total_trunc += 1
            samples.append({"text": completion.text, "n_tokens": n_tok, "truncated": trunc})
        all_outputs.append(samples)

    if total_trunc:
        print(f"  Warning: {total_trunc}/{len(outputs) * n_samples} "
              f"completions truncated at max_new_tokens={max_new_tokens}", flush=True)
    return all_outputs


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def score_predictions(
    df: pd.DataFrame,
    all_outputs: list[list[dict]],
) -> pd.DataFrame:
    """Score k outputs per problem and record per-problem pass_rate."""
    records = []
    for (_, row), samples in zip(df.iterrows(), all_outputs):
        gold = row["answer"]
        # Deserialize JSON-list multi-part answers
        if isinstance(gold, str):
            try:
                parsed = json.loads(gold)
                if isinstance(parsed, list):
                    gold = parsed
            except Exception:
                pass

        at = str(row.get("answer_type") or "unknown")
        unit = str(row.get("unit") or "")

        scores: list[float] = []
        for s in samples:
            try:
                sc = compute_score(
                    solution_str=s["text"],
                    ground_truth=gold,
                    answer_type=at,
                    unit=unit,
                    tolerance=0.05,
                    xverify_judge=None,
                    problem=str(row.get("problem", "")),
                )
            except Exception:
                sc = -1.0
            scores.append(sc)

        n_correct = scores.count(1.0)
        n_verif = sum(1 for sc in scores if sc != -1.0)
        n_trunc = sum(1 for s in samples if s["truncated"])
        pass_rate = n_correct / len(scores)
        denom_nontrunc = len(scores) - n_trunc
        pass_rate_nontrunc = (n_correct / denom_nontrunc) if denom_nontrunc > 0 else float("nan")

        records.append({
            "problem_id": row.get("problem_id", ""),
            "source": row.get("source", ""),
            "answer_type": at,
            "difficulty": row.get("difficulty", None),
            "gold_answer": str(gold),
            "pred_text": samples[0]["text"],
            "n_tokens": samples[0]["n_tokens"],
            "truncated": samples[0]["truncated"],
            "pass_rate": pass_rate,
            "pass_rate_nontrunc": pass_rate_nontrunc,
            "n_correct": n_correct,
            "n_verifiable": n_verif,
            "n_truncated": n_trunc,
            "n_samples": len(scores),
            "scores": scores,
            "all_texts": [s["text"] for s in samples],
        })
    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input",   default="data/processed/candidates_deduped.parquet")
    parser.add_argument("--model",   default="Qwen/Qwen3.5-4B")
    parser.add_argument("--n_samples",      type=int, default=8)
    parser.add_argument("--n_per_stratum",  type=int, default=20,
                        help="Rows per (source × simplified_type) stratum (default 20 → ~500 total)")
    parser.add_argument("--max_new_tokens", type=int, default=8192)
    parser.add_argument("--output",  default="data/results/zero_shot_corpus_passk.parquet")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.9)
    parser.add_argument("--seed",    type=int, default=42)
    args = parser.parse_args()

    print(f"Building stratified sample from {args.input} "
          f"(n_per_stratum={args.n_per_stratum})...", flush=True)
    df = build_sample(args.input, n_per_stratum=args.n_per_stratum, seed=args.seed)
    print(f"  Sample size: {len(df)} rows  "
          f"(total inference calls: {len(df) * args.n_samples})", flush=True)
    print(f"  Source dist:\n{df['source'].value_counts().to_string()}", flush=True)

    all_outputs = run_inference(
        df, args.model, args.max_new_tokens,
        n_samples=args.n_samples,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )

    print("Scoring predictions...", flush=True)
    results_df = score_predictions(df, all_outputs)

    k = args.n_samples
    print(f"\n=== Zero-shot Pass@{k} on Corpus Sample ===", flush=True)
    print(f"  Overall pass@{k}: {results_df['pass_rate'].mean():.1%}  (n={len(results_df)})")

    n_unverif = (results_df["n_verifiable"] == 0).sum()
    print(f"  Rows with no verifiable outputs: {n_unverif} ({n_unverif/len(results_df):.1%})")

    total_trunc = results_df["n_truncated"].sum()
    total_completions = len(results_df) * k
    print(f"  Truncated completions: {total_trunc}/{total_completions} "
          f"({total_trunc/total_completions:.1%})")
    high_trunc = (results_df["n_truncated"] >= k // 2).sum()
    print(f"  Problems with ≥{k//2}/{k} truncated: {high_trunc} "
          f"({high_trunc/len(results_df):.1%}) — consider excluding from Goldilocks")

    gl_low, gl_high = 0.15, 0.85
    n_gl = ((results_df["pass_rate"] >= gl_low) & (results_df["pass_rate"] <= gl_high)).sum()
    print(f"\n  Goldilocks [{gl_low:.0%}–{gl_high:.0%}]: "
          f"{n_gl}/{len(results_df)} ({n_gl/len(results_df):.1%}) — estimated learnable problems")

    print(f"\n  By source:")
    for src, grp in results_df.groupby("source", observed=True):
        n_gl_src = ((grp["pass_rate"] >= gl_low) & (grp["pass_rate"] <= gl_high)).sum()
        print(f"    {str(src):<30} pass@{k}={grp['pass_rate'].mean():.1%}  "
              f"goldilocks={n_gl_src/len(grp):.1%}  n={len(grp)}")

    print(f"\n  By answer type (simplified):")
    results_df["_simple_type"] = results_df["answer_type"].apply(_simplify_type)
    for at, grp in results_df.groupby("_simple_type", observed=True):
        n_gl_at = ((grp["pass_rate"] >= gl_low) & (grp["pass_rate"] <= gl_high)).sum()
        print(f"    {str(at):<15} pass@{k}={grp['pass_rate'].mean():.1%}  "
              f"goldilocks={n_gl_at/len(grp):.1%}  n={len(grp)}")

    results_df = results_df.drop(columns=["_simple_type"])
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    results_df.to_parquet(args.output, index=False)
    print(f"\nSaved results → {args.output}", flush=True)


if __name__ == "__main__":
    main()
