"""Side-by-side debug: feed identical xVerify prompts to vLLM and to transformers.

Goal: figure out WHY vLLM and HTTP-via-transformers disagree on ~8% of judgments.
Tries one row at a time, dumps:
  - exact prompt
  - vLLM raw output text + token ids
  - transformers raw output text + token ids
  - diff
"""
from __future__ import annotations

import argparse

import pandas as pd
import torch


def reconstruct(row) -> str:
    p = []
    if row.get("phase1_text"):  p.append(str(row["phase1_text"]))
    if row.get("interrupted") and row.get("phase1b_text"): p.append(str(row["phase1b_text"]))
    if row.get("phase2_text"):  p.append(str(row["phase2_text"]))
    return "\n".join(p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--merged", default="outputs/probe_qwen3_4b/rollouts_merged.parquet")
    ap.add_argument("--rows", nargs="+", type=int, default=[173, 313, 352, 459, 463, 537])
    ap.add_argument("--xverify_model", default="IAAR-Shanghai/xVerify-7B-I")
    args = ap.parse_args()

    from huggingface_hub import snapshot_download
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from vllm import LLM, SamplingParams
    from phys_reasoner.verifier.xverify_judge import _XVERIFY_PROMPT

    model_path = snapshot_download(args.xverify_model, local_files_only=True)

    df = pd.read_parquet(args.merged)
    print(f"loaded {len(df)} rows")

    # Build prompts for the disputed rows
    prompts = []
    contexts = []
    for idx in args.rows:
        row = df.iloc[idx]
        ei = row.get("extra_info") if isinstance(row.get("extra_info"), dict) else {}
        sol = reconstruct(row)
        gold = row.get("gold_answer", "")
        problem = ei.get("problem", "") or "(not provided)"
        prompt = _XVERIFY_PROMPT.format(problem=problem, pred=sol, gold=str(gold))
        prompts.append(prompt)
        contexts.append({"idx": idx, "gold": str(gold)[:60], "answer_type": ei.get("answer_type")})

    # ---- Pass 1: vLLM ----
    print("\n===== vLLM pass =====")
    llm = LLM(
        model=model_path, dtype="auto",
        gpu_memory_utilization=0.4,  # leave room for transformers in same process
        max_model_len=4096,
        enforce_eager=True,
    )
    sp = SamplingParams(temperature=0.0, max_tokens=10, top_p=1.0)
    vllm_outs = llm.generate(prompts, sp)
    vllm_results = []
    for c, o in zip(contexts, vllm_outs):
        text = o.outputs[0].text
        token_ids = list(o.outputs[0].token_ids)
        vllm_results.append({"text": text, "token_ids": token_ids})
        print(f"  idx={c['idx']:5d} gold={c['gold']!r:25s} vLLM_text={text!r:30s} ids={token_ids}")

    # Free vLLM GPU memory before loading transformers
    del llm
    import gc; gc.collect()
    torch.cuda.empty_cache()

    # ---- Pass 2: transformers ----
    print("\n===== transformers pass =====")
    tok = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path, dtype="auto", device_map="cuda")
    model.eval()
    tf_results = []
    for c, prompt in zip(contexts, prompts):
        inputs = tok(prompt, return_tensors="pt").to("cuda")
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=10,
                do_sample=False,
                pad_token_id=tok.eos_token_id,
            )
        new_tokens = out[0][inputs["input_ids"].shape[1]:]
        text = tok.decode(new_tokens, skip_special_tokens=True)
        token_ids = new_tokens.tolist()
        tf_results.append({"text": text, "token_ids": token_ids})
        print(f"  idx={c['idx']:5d} gold={c['gold']!r:25s} TF_text  ={text!r:30s} ids={token_ids}")

    # ---- Diff summary ----
    print("\n===== AGREEMENT =====")
    for c, v, t in zip(contexts, vllm_results, tf_results):
        v_correct = v["text"].strip().lower().startswith("correct")
        t_correct = t["text"].strip().lower().startswith("correct")
        agree = v_correct == t_correct
        print(f"  idx={c['idx']:5d}  vllm={'C' if v_correct else 'W'}  TF={'C' if t_correct else 'W'}  agree={agree}")

    # ---- Tokenization check ----
    print("\n===== TOKENIZATION CHECK (first prompt) =====")
    prompt0 = prompts[0]
    tf_ids = tok(prompt0, return_tensors="pt")["input_ids"][0].tolist()
    print(f"  TF input_ids[:8] = {tf_ids[:8]}")
    print(f"  TF input_ids[-8:]= {tf_ids[-8:]}")
    print(f"  bos_token_id = {tok.bos_token_id}")
    print(f"  eos_token_id = {tok.eos_token_id}")
    print(f"  add_bos_token attr = {getattr(tok, 'add_bos_token', None)}")
    # vLLM tokenization check
    from vllm.transformers_utils.tokenizer import get_tokenizer
    vtok = get_tokenizer(model_path)
    vllm_ids = vtok(prompt0)["input_ids"]
    print(f"  vLLM input_ids[:8] = {vllm_ids[:8]}")
    print(f"  vLLM input_ids[-8:]= {vllm_ids[-8:]}")
    if tf_ids == vllm_ids:
        print("  ✓ TOKENIZATIONS IDENTICAL")
    else:
        print(f"  ✗ TOKENIZATIONS DIFFER: TF len={len(tf_ids)}, vLLM len={len(vllm_ids)}")


if __name__ == "__main__":
    main()
