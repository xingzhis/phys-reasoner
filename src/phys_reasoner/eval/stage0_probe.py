"""Stage 0 probe: zero-shot TIR evaluation on a sample of physics problems.

Runs entirely on one GPU via vLLM.LLM (no Ray, no VeRL).
Measures: format adherence, exec success, verifier hit rate.

Decision gate: if verifier_hit_rate >= 0.15 on numerical problems → skip SFT.

The probe saves `p1_stop_reason` and `p2_stop_reason` to the output so you can
diagnose truncation vs format failures separately.

Usage:
    python -m phys_reasoner.eval.stage0_probe \\
        --model hf_cache/models--Qwen--Qwen3.5-4B/snapshots/... \\
        --parquet data/processed/drsci_physics_clean.parquet \\
        --n_samples 100 \\
        --output data/results/stage0_probe.parquet
"""

from __future__ import annotations

import argparse
import time

import pandas as pd

from phys_reasoner.tir.prompts import CODE_STOP, TIR_SYSTEM_PROMPT, extract_code

OUTPUT_PREFIX = "[output] "
OUTPUT_SUFFIX = "\n"


def build_tir_prompt(problem: str) -> list[dict]:
    """Return messages list for Qwen chat template."""
    return [
        {"role": "system", "content": TIR_SYSTEM_PROMPT},
        {"role": "user", "content": problem},
    ]


def _extract_boxed(text: str) -> str | None:
    """Extract content of the last \\boxed{...} with depth-aware brace tracking."""
    idx = text.rfind(r"\boxed{")
    if idx == -1:
        return None
    depth = 0
    start = idx + len(r"\boxed{")
    for i, ch in enumerate(text[start:], start):
        if ch == "{":
            depth += 1
        elif ch == "}":
            if depth == 0:
                return text[start:i].strip()
            depth -= 1
    return None


def run_probe(
    model_path: str,
    parquet_path: str,
    n_samples: int = 100,
    seed: int = 42,
    max_new_tokens: int = 4096,
    sandbox_timeout: float = 30.0,
    output_path: str | None = None,
) -> pd.DataFrame:
    """Run TIR Stage 0 probe. Returns DataFrame with per-problem results + diagnostics."""
    from vllm import LLM, SamplingParams  # noqa: PLC0415

    from phys_reasoner.tir.sandbox import execute_code  # noqa: PLC0415
    from phys_reasoner.verifier.router import verify_answer  # noqa: PLC0415

    # --- Load data ---
    df = pd.read_parquet(parquet_path)
    df = df.sample(n=min(n_samples, len(df)), random_state=seed).reset_index(drop=True)

    # --- Load model ---
    llm = LLM(
        model=model_path,
        dtype="bfloat16",
        enable_prefix_caching=True,
        max_model_len=max_new_tokens + 2048,  # prompt budget
        gpu_memory_utilization=0.90,
    )
    tokenizer = llm.get_tokenizer()

    # --- Build prompts ---
    messages_list = [build_tir_prompt(str(row["problem"])) for _, row in df.iterrows()]
    phase1_prompts = [
        tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        for msgs in messages_list
    ]

    # --- Phase 1: generate until [/code] ---
    phase1_params = SamplingParams(
        max_tokens=max_new_tokens,
        stop=[CODE_STOP],
        include_stop_str_in_output=True,
        temperature=0.7,
        top_p=0.8,
        top_k=20,
    )
    phase1_outputs = llm.generate(phase1_prompts, phase1_params)

    # --- Execute code + build phase 2 prompts ---
    phase2_prompts: list[str] = []
    phase1_texts: list[str] = []
    code_extracted_flags: list[bool] = []
    exec_success_flags: list[bool] = []
    exec_stdouts: list[str] = []
    p1_stop_reasons: list[str] = []
    tokens_used_p1: list[int] = []

    for i, out in enumerate(phase1_outputs):
        p1_out = out.outputs[0]
        p1_text = p1_out.text
        p1_stop = p1_out.finish_reason or "unknown"
        phase1_texts.append(p1_text)
        p1_stop_reasons.append(p1_stop)
        tokens_used_p1.append(len(p1_out.token_ids))

        code_str = extract_code(p1_text)
        code_extracted_flags.append(code_str is not None)

        if code_str and p1_stop == "stop":
            result = execute_code(code_str, timeout=sandbox_timeout)
            exec_ok = not result.error and bool(result.stdout)
            exec_success_flags.append(exec_ok)
            exec_stdouts.append(result.stdout if exec_ok else result.stderr[:300])
            output_injection = f"{OUTPUT_PREFIX}{result.stdout if exec_ok else '(execution error)'}{OUTPUT_SUFFIX}"
        else:
            exec_success_flags.append(False)
            exec_stdouts.append("")
            output_injection = f"{OUTPUT_PREFIX}(no code block){OUTPUT_SUFFIX}"

        phase2_prompts.append(phase1_prompts[i] + p1_text + output_injection)

    # --- Phase 2: no stop token — generate to EOS/max_tokens ---
    # Per-request SamplingParams to handle variable remaining budgets.
    phase2_params_list = [
        SamplingParams(
            max_tokens=max(64, max_new_tokens - used),
            temperature=0.7,
            top_p=0.8,
            top_k=20,
        )
        for used in tokens_used_p1
    ]
    phase2_outputs = llm.generate(phase2_prompts, phase2_params_list)

    # --- Collect results ---
    records = []
    wall_start = time.monotonic()

    for i, (_, row) in enumerate(df.iterrows()):
        p2_out = phase2_outputs[i].outputs[0]
        p2_text = p2_out.text
        p2_stop = p2_out.finish_reason or "unknown"

        # p2_text now includes "[/answer]" at the end (if stop was hit).
        # Extract \boxed{} from the full phase 2 text.
        boxed = _extract_boxed(p2_text)
        format_ok = code_extracted_flags[i] and (boxed is not None)

        # Truncation diagnostics
        p1_truncated = p1_stop_reasons[i] != "stop"
        p2_truncated = p2_stop != "stop"

        # Verify
        answer = row.get("answer", row.get("gold_answer", ""))
        answer_type = row.get("answer_type", row.get("inferred_answer_type", "numerical"))
        unit = row.get("unit", row.get("gold_unit", ""))

        if boxed is not None:
            score = verify_answer(
                pred_text=f"\\boxed{{{boxed}}}",
                gold_answer=answer,
                answer_type=answer_type,
                gold_unit=str(unit) if unit else "",
            )
        else:
            score = 0.0

        records.append({
            "problem_id": row.get("problem_id", str(i)),
            "answer_type": answer_type,
            "source": row.get("source", ""),
            "phase1_text": phase1_texts[i],
            "p1_stop_reason": p1_stop_reasons[i],
            "p1_tokens": tokens_used_p1[i],
            "p1_truncated": p1_truncated,
            "code_extracted": code_extracted_flags[i],
            "exec_success": exec_success_flags[i],
            "exec_stdout": exec_stdouts[i],
            "phase2_text": p2_text,
            "p2_stop_reason": p2_stop,
            "p2_truncated": p2_truncated,
            "boxed_answer": boxed,
            "verifier_score": score,
            "format_ok": format_ok,
            "wall_time_s": time.monotonic() - wall_start,
        })

    results = pd.DataFrame(records)

    _print_summary(results)

    if output_path:
        results.to_parquet(output_path, index=False)
        print(f"\nSaved to {output_path}")

    return results


def _print_summary(df: pd.DataFrame) -> None:
    n_total = len(df)
    p1_trunc = df["p1_truncated"].mean()
    p2_trunc = df["p2_truncated"].mean()
    print(f"\n=== Stage 0 Probe Summary (n={n_total}) ===")
    print(f"  Phase 1 truncated (no [/code]): {p1_trunc:.1%}")
    print(f"  Phase 2 truncated (no [/answer]): {p2_trunc:.1%}")

    for atype, grp in df.groupby("answer_type"):
        n = len(grp)
        exec_rate = grp["exec_success"].mean()
        fmt_rate = grp["format_ok"].mean()
        verified = grp[grp["verifier_score"] >= 0]
        hit_rate = (verified["verifier_score"] == 1.0).mean() if len(verified) > 0 else 0.0
        unverif_rate = (grp["verifier_score"] == -1.0).mean()
        print(
            f"  {atype:15s} n={n:3d}  exec={exec_rate:.2f}  fmt={fmt_rate:.2f}"
            f"  hit={hit_rate:.2f}  unverif={unverif_rate:.2f}"
        )

    # Decision gate
    numerical = df[df["answer_type"].str.contains("numerical", case=False, na=False)]
    if len(numerical) > 0:
        verified_num = numerical[numerical["verifier_score"] >= 0]
        hit_num = (
            (verified_num["verifier_score"] == 1.0).mean() if len(verified_num) > 0 else 0.0
        )
        gate = "PASS — skip SFT" if hit_num >= 0.15 else "FAIL — SFT recommended"
        print(f"\nDecision gate (numerical hit_rate >= 0.15): {hit_num:.3f} → {gate}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 0 TIR probe")
    parser.add_argument("--model", required=True)
    parser.add_argument("--parquet", required=True)
    parser.add_argument("--n_samples", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_new_tokens", type=int, default=4096)
    parser.add_argument("--sandbox_timeout", type=float, default=30.0)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    run_probe(
        model_path=args.model,
        parquet_path=args.parquet,
        n_samples=args.n_samples,
        seed=args.seed,
        max_new_tokens=args.max_new_tokens,
        sandbox_timeout=args.sandbox_timeout,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()
