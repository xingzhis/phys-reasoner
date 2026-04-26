"""CoT probe with think-interrupt — mimics verl single_turn_agent_loop semantics.

Phase 1: generate up to thinking_budget tokens.
If </think> not emitted: append THINK_INTERRUPT_PHRASE then phase 2 with post_interrupt_budget.
Concatenated text scored as the final response.

Output schema: same as cot_probe_base_4b.py plus `interrupt_fired` and per-phase lengths.
"""
import argparse
import hashlib
import os

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

THINK_INTERRUPT_PHRASE = "\nOkay, I've thought enough. Time to write my response.\n</think>\n"


def extract_last_boxed(text: str) -> str:
    out = None
    i = 0
    while True:
        idx = text.find("\\boxed{", i)
        if idx < 0:
            break
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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--model", default="Qwen/Qwen3-4B")
    ap.add_argument("--tp", type=int, default=4)
    ap.add_argument("--thinking_budget", type=int, default=12288)
    ap.add_argument("--post_interrupt_budget", type=int, default=7184)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--n_rollouts", type=int, default=8)
    ap.add_argument("--gpu_mem", type=float, default=0.85)
    ap.add_argument("--max_model_len", type=int, default=20992,
                    help="prompt(1024) + thinking(12288) + interrupt(~16) + post(7184) + slack")
    ap.add_argument("--start_idx", type=int, default=0)
    ap.add_argument("--end_idx", type=int, default=0)
    args = ap.parse_args()

    print(f"[probe] loading {args.input}")
    df = pq.read_table(args.input).to_pandas()
    if args.end_idx > 0 or args.start_idx > 0:
        end = args.end_idx if args.end_idx > 0 else len(df)
        df = df.iloc[args.start_idx:end].reset_index(drop=True)
    print(f"[probe] {len(df)} problems × n={args.n_rollouts}, "
          f"thinking_budget={args.thinking_budget}, post_interrupt={args.post_interrupt_budget}")

    def row_hash(row):
        prob = row["extra_info"].get("problem", "") if isinstance(row["extra_info"], dict) else ""
        gold = row["reward_model"].get("ground_truth", "") if isinstance(row["reward_model"], dict) else ""
        return hashlib.md5(f"{prob}|||{gold}".encode("utf-8")).hexdigest()[:16]
    df["problem_hash"] = df.apply(row_hash, axis=1)
    df["answer_type"] = df["extra_info"].apply(
        lambda x: x.get("answer_type", "") if isinstance(x, dict) else "")

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model)
    interrupt_ids = tok.encode(THINK_INTERRUPT_PHRASE, add_special_tokens=False)
    think_end_id = tok.convert_tokens_to_ids("</think>")
    print(f"[probe] interrupt_len={len(interrupt_ids)}, </think>={think_end_id}")

    prompt_texts = []
    for _, row in df.iterrows():
        msgs = list(row["prompt"])
        text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        prompt_texts.append(text)
    prompt_token_ids = [tok.encode(t, add_special_tokens=False) for t in prompt_texts]

    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt
    llm = LLM(
        model=args.model,
        tensor_parallel_size=args.tp,
        gpu_memory_utilization=args.gpu_mem,
        max_model_len=args.max_model_len,
        enforce_eager=False,
        enable_prefix_caching=True,
        dtype="bfloat16",
    )
    sp1 = SamplingParams(
        temperature=args.temperature, top_p=1.0,
        max_tokens=args.thinking_budget, n=args.n_rollouts,
    )
    print(f"[probe] phase 1: generating {len(prompt_token_ids)} prompts × n={args.n_rollouts} ...")
    out1 = llm.generate(
        [TokensPrompt(prompt_token_ids=ids) for ids in prompt_token_ids],
        sampling_params=sp1,
    )

    # Build phase 2 prompts for rollouts that didn't finish thinking
    phase2_prompts = []  # list of (prompt_ids+phase1_ids+interrupt_ids)
    phase2_routing = []  # list of (problem_idx, rollout_idx) for re-attaching outputs
    for pi, output in enumerate(out1):
        for ri, gen in enumerate(output.outputs):
            tids = list(gen.token_ids)
            if think_end_id not in tids:
                phase2_prompts.append(list(prompt_token_ids[pi]) + tids + interrupt_ids)
                phase2_routing.append((pi, ri))
    print(f"[probe] phase 2 needed for {len(phase2_prompts)}/{len(prompt_token_ids)*args.n_rollouts} rollouts")

    sp2 = SamplingParams(
        temperature=args.temperature, top_p=1.0,
        max_tokens=args.post_interrupt_budget, n=1,
    )
    if phase2_prompts:
        out2 = llm.generate(
            [TokensPrompt(prompt_token_ids=ids) for ids in phase2_prompts],
            sampling_params=sp2,
        )
    else:
        out2 = []

    phase2_by_route = {phase2_routing[i]: out2[i].outputs[0] for i in range(len(out2))}

    results = []
    for pi, output in enumerate(out1):
        gt_raw = df.iloc[pi]["reward_model"].get("ground_truth", "")
        gt = extract_last_boxed(str(gt_raw)) or str(gt_raw)
        for ri, gen in enumerate(output.outputs):
            phase1_text = gen.text
            phase1_tokens = len(gen.token_ids)
            interrupt_fired = (pi, ri) in phase2_by_route
            if interrupt_fired:
                p2 = phase2_by_route[(pi, ri)]
                phase2_text = p2.text
                phase2_tokens = len(p2.token_ids)
                full_text = phase1_text + THINK_INTERRUPT_PHRASE + phase2_text
            else:
                phase2_text = ""
                phase2_tokens = 0
                full_text = phase1_text
            pred = extract_last_boxed(full_text)
            correct = int(normalize(pred) == normalize(gt)) if pred else 0
            results.append({
                "problem_hash": df.iloc[pi]["problem_hash"],
                "rollout_idx": ri,
                "correct": correct,
                "phase1_tokens": phase1_tokens,
                "phase2_tokens": phase2_tokens,
                "interrupt_fired": interrupt_fired,
                "answer_type": df.iloc[pi]["answer_type"],
                "gt": str(gt_raw),
                "pred_boxed": pred,
                "phase1_text": phase1_text,
                "phase2_text": phase2_text,
            })

    out_df = pa.Table.from_pylist(results)
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    pq.write_table(out_df, args.output)

    rdf = pd.DataFrame(results)
    pp = rdf.groupby("problem_hash").correct.mean()
    print(f"[probe] wrote {args.output} ({len(results)} rows = {len(df)} problems × {args.n_rollouts} rollouts)")
    print(f"[probe] interrupt_fired rate: {rdf['interrupt_fired'].mean():.3f}")
    print(f"[probe] per-problem pass@{args.n_rollouts}: mean={pp.mean():.3f}, "
          f"frac_zero={(pp==0).mean():.3f}, frac_one={(pp==1).mean():.3f}, "
          f"goldilocks(0,1)={((pp>0)&(pp<1)).mean():.3f}")
    print(f"[probe] phase1_tokens p50={np.percentile(rdf['phase1_tokens'], 50):.0f}, "
          f"p90={np.percentile(rdf['phase1_tokens'], 90):.0f}, "
          f"max={rdf['phase1_tokens'].max()}")
    if rdf["interrupt_fired"].any():
        ph2 = rdf[rdf["interrupt_fired"]]["phase2_tokens"]
        print(f"[probe] phase2_tokens (interrupted only) p50={ph2.median():.0f}, p90={ph2.quantile(0.9):.0f}, max={ph2.max()}")

    print("[probe] per (answer_type, band) Goldilocks:")
    bdf = rdf.groupby(["problem_hash"]).agg(
        answer_type=("answer_type", "first"),
        pass_rate=("correct", "mean"),
    ).reset_index()
    bins = [-0.001, 0.001, 0.124, 0.375, 0.624, 0.875, 1.001]
    labels = ["0/8","1/8","2-3/8","4-5/8","6-7/8","8/8"]
    bdf["band"] = pd.cut(bdf["pass_rate"], bins=bins, labels=labels)
    print(pd.crosstab(bdf["answer_type"], bdf["band"], margins=True))


if __name__ == "__main__":
    main()
