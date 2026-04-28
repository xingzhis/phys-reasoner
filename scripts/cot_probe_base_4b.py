"""Quick CoT probe: pass@1 for base Qwen3-4B on a given parquet.

Goal: measure REAL CoT response length distribution, since the TIR probe
underestimates it by ~5x (tool delegates away inline math).

Output columns: problem_hash, correct, response_length_tokens, response
"""
import argparse
import hashlib
import os
import sys

import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Input parquet with 'prompt' and 'reward_model' cols")
    ap.add_argument("--output", required=True, help="Output parquet path")
    ap.add_argument("--model", default="Qwen/Qwen3-4B")
    ap.add_argument("--tp", type=int, default=4)
    ap.add_argument("--max_tokens", type=int, default=4096)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--subset", type=int, default=0, help="Subsample to first N rows (0 = all)")
    ap.add_argument("--start_idx", type=int, default=0, help="Slice start (for chunking)")
    ap.add_argument("--end_idx", type=int, default=0, help="Slice end (for chunking; 0 = to end)")
    ap.add_argument("--n_rollouts", type=int, default=1, help="Rollouts per problem (vLLM n=)")
    ap.add_argument("--gpu_mem", type=float, default=0.85)
    ap.add_argument("--max_model_len", type=int, default=6144)
    args = ap.parse_args()

    print(f"[probe] loading {args.input}")
    df = ds.dataset(args.input).to_table().to_pandas()
    if args.subset > 0 and args.subset < len(df):
        df = df.iloc[: args.subset].reset_index(drop=True)
    if args.end_idx > 0 or args.start_idx > 0:
        end = args.end_idx if args.end_idx > 0 else len(df)
        df = df.iloc[args.start_idx:end].reset_index(drop=True)
    print(f"[probe] {len(df)} problems (n_rollouts={args.n_rollouts}, slice=[{args.start_idx}:{args.end_idx or 'end'}])")

    def row_hash(row) -> str:
        prob = row["extra_info"].get("problem", "") if isinstance(row["extra_info"], dict) else ""
        gold = row["reward_model"].get("ground_truth", "") if isinstance(row["reward_model"], dict) else ""
        return hashlib.md5(f"{prob}|||{gold}".encode("utf-8")).hexdigest()[:16]
    df["problem_hash"] = df.apply(row_hash, axis=1)

    # apply chat template
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model)
    prompts = []
    for _, row in df.iterrows():
        msgs = list(row["prompt"])
        # apply_chat_template expects list of dicts
        text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        prompts.append(text)
    print(f"[probe] prompts built; len0={len(prompts[0])}")

    # vLLM
    from vllm import LLM, SamplingParams
    llm = LLM(
        model=args.model,
        tensor_parallel_size=args.tp,
        gpu_memory_utilization=args.gpu_mem,
        max_model_len=args.max_model_len,
        enforce_eager=False,
        enable_prefix_caching=True,
        dtype="bfloat16",
    )
    sp = SamplingParams(
        temperature=args.temperature,
        top_p=1.0,
        max_tokens=args.max_tokens,
        n=args.n_rollouts,
    )
    print(f"[probe] generating {len(prompts)} samples ...")
    outputs = llm.generate(prompts, sp)

    # score and record
    import re
    def extract_last_boxed(text: str) -> str:
        # find last \boxed{...}, handling nested braces
        out = None
        i = 0
        while True:
            idx = text.find("\\boxed{", i)
            if idx < 0:
                break
            # find matching brace
            depth = 1
            j = idx + 7
            while j < len(text) and depth > 0:
                if text[j] == "{":
                    depth += 1
                elif text[j] == "}":
                    depth -= 1
                j += 1
            if depth == 0:
                out = text[idx + 7 : j - 1]
            i = idx + 7
        return out or ""

    def normalize(s: str) -> str:
        s = s.strip().replace(" ", "").replace("\\,", "").replace("\\!", "")
        s = s.rstrip(".")
        return s

    results = []
    for row, out in zip(df.itertuples(index=False), outputs):
        gt_raw = row.reward_model.get("ground_truth", "")
        gt = extract_last_boxed(str(gt_raw)) or str(gt_raw)
        for rollout_idx, gen in enumerate(out.outputs):
            resp = gen.text
            resp_tok = len(gen.token_ids)
            pred = extract_last_boxed(resp)
            correct = int(normalize(pred) == normalize(gt)) if pred else 0
            results.append({
                "problem_hash": row.problem_hash,
                "rollout_idx": rollout_idx,
                "correct": correct,
                "response_length_tokens": resp_tok,
                "gt": str(gt_raw),
                "pred_boxed": pred,
                "response": resp,
            })

    out_df = pa.Table.from_pylist(results)
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    pq.write_table(out_df, args.output)

    # Summary
    import numpy as np
    import pandas as pd
    lens = [r["response_length_tokens"] for r in results]
    corrects = [r["correct"] for r in results]
    print(f"[probe] wrote {args.output} ({len(results)} rows = {len(df)} problems × {args.n_rollouts} rollouts)")
    # per-problem pass@n
    rdf = pd.DataFrame(results)
    pp = rdf.groupby("problem_hash").correct.mean()
    print(f"[probe] per-problem pass@{args.n_rollouts}: mean={pp.mean():.3f}, "
          f"frac_zero={(pp==0).mean():.3f}, frac_one={(pp==1).mean():.3f}, "
          f"frac_goldilocks(0,1)={((pp>0)&(pp<1)).mean():.3f}")
    print(f"[probe] pass@1 = {sum(corrects)/len(corrects):.3f}")
    print(f"[probe] response_length_tokens:")
    for p in [10, 50, 75, 90, 95, 99]:
        print(f"  p{p}={np.percentile(lens, p):.0f}")
    print(f"  max={max(lens)}  truncated(={args.max_tokens})={sum(1 for l in lens if l >= args.max_tokens)}")


if __name__ == "__main__":
    main()
