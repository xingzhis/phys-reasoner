"""Dump TIR rollouts to human-readable txt files for trajectory inspection.

Runs the same 2-phase vLLM loop as stage0_probe (and VeRL's ToolAgentLoop during
training), but saves one txt file per rollout so you can read the full trajectory:
  - Problem + gold answer
  - Phase 1: model reasoning → <tool_call>
  - Extracted code + sandbox result
  - Phase 2: model final answer after <tool_response>
  - Verdict: boxed answer, verifier score

Usage (via dump_rollouts.sh):
    MODEL=Qwen/Qwen3-0.6B N=4 bash scripts/dump_rollouts.sh

Direct usage inside container:
    python3 scripts/dump_rollouts.py \\
        --model Qwen/Qwen3-0.6B \\
        --parquet data/processed/corpus_train.parquet \\
        --n 4 \\
        --out_dir outputs/rollouts_test \\
        --gpu_mem 0.38 \\
        --temperature 1.0 \\
        --top_p 0.9
"""

from __future__ import annotations

import argparse
import os
import textwrap

import pandas as pd


def _extract_boxed(text: str) -> str | None:
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


def _get_problem_text(row: pd.Series) -> str:
    # Training parquets store the raw problem text in extra_info["problem"].
    ei = row.get("extra_info")
    if isinstance(ei, dict) and ei.get("problem"):
        return str(ei["problem"])
    # Fallback: pull user content from the prompt message list.
    prompt = row.get("prompt")
    if isinstance(prompt, list):
        for msg in prompt:
            if isinstance(msg, dict) and msg.get("role") == "user":
                return str(msg.get("content", ""))
    return "(unknown problem)"


def _get_gold(row: pd.Series) -> str:
    rm = row.get("reward_model")
    if isinstance(rm, dict):
        return str(rm.get("ground_truth", ""))
    return ""


def _get_answer_type(row: pd.Series) -> str:
    ei = row.get("extra_info")
    if isinstance(ei, dict):
        return str(ei.get("answer_type", "unknown"))
    return "unknown"


def _format_rollout_txt(
    idx: int,
    problem: str,
    gold: str,
    answer_type: str,
    phase1_prompt: str,
    p1_text: str,
    p1_stop: str,
    code: str | None,
    sandbox_stdout: str,
    sandbox_err: str,
    sandbox_error: bool,
    injection_text: str,
    p2_text: str,
    p2_stop: str,
    boxed: str | None,
    verifier_score: float | None,
) -> str:
    sep = "=" * 72
    thin = "-" * 72

    def wrap(label: str, text: str) -> str:
        return f"{label}\n{text.strip()}\n"

    lines = [
        sep,
        f"ROLLOUT {idx:03d}",
        sep,
        f"ANSWER TYPE : {answer_type}",
        f"GOLD        : {gold}",
        "",
        wrap("PROBLEM", problem),
        thin,
        "PHASE 1 PROMPT  (rendered — check for <tools> schema injection)",
        thin,
        phase1_prompt,
        thin,
        f"PHASE 1  (stop_reason={p1_stop})",
        thin,
        p1_text.strip(),
        "",
        thin,
        "CODE EXTRACTED" if code is not None else "CODE EXTRACTED  [none]",
        thin,
    ]

    if code is not None:
        lines.append(code.strip())
    else:
        lines.append("(no <tool_call> block found)")

    lines += [
        "",
        thin,
        f"SANDBOX RESULT  (error={sandbox_error})",
        thin,
    ]
    if sandbox_stdout:
        lines.append(f"stdout: {sandbox_stdout}")
    if sandbox_err:
        lines.append(f"stderr: {sandbox_err[:300]}")
    if not sandbox_stdout and not sandbox_err:
        lines.append("(no output)")

    lines += [
        "",
        thin,
        "INJECTION  (what VeRL appends between phase 1 and phase 2)",
        thin,
        repr(injection_text),  # repr so whitespace/special tokens are visible
        "",
        thin,
        f"PHASE 2  (stop_reason={p2_stop})",
        thin,
        p2_text.strip() if p2_text else "(skipped — model answered in phase 1)",
        "",
        sep,
        "VERDICT",
        sep,
        f"  boxed_answer   : {boxed}",
        f"  verifier_score : {verifier_score}",
        "",
    ]

    return "\n".join(lines)


def run_dump(
    model_path: str,
    parquet_path: str,
    n: int,
    out_dir: str,
    seed: int = 42,
    gpu_mem: float = 0.38,
    max_tokens: int = 1536,
    temperature: float = 1.0,
    top_p: float = 0.9,
    enable_thinking: bool = True,
) -> None:
    from vllm import LLM, SamplingParams  # noqa: PLC0415

    from phys_reasoner.tir.prompts import (  # noqa: PLC0415
        PYTHON_TOOL_SCHEMA,
        TOOL_CALL_STOP,
        TIR_SYSTEM_PROMPT,
        extract_tool_call_code,
    )
    from phys_reasoner.tir.sandbox import execute_code  # noqa: PLC0415
    from phys_reasoner.verifier.router import verify_answer  # noqa: PLC0415

    os.makedirs(out_dir, exist_ok=True)

    df = pd.read_parquet(parquet_path)
    df = df.sample(n=min(n, len(df)), random_state=seed).reset_index(drop=True)
    print(f"Loaded {len(df)} rows from {parquet_path}")

    llm = LLM(
        model=model_path,
        dtype="bfloat16",
        gpu_memory_utilization=gpu_mem,
        max_model_len=max_tokens + 2048,  # prompt budget on top of response budget
    )
    tokenizer = llm.get_tokenizer()

    # Pre-compute the dummy-user prefix once.
    # verl.utils.chat_template.apply_chat_template handles Qwen3.5's "No user query found"
    # restriction by prepending an empty dummy user message and stripping its prefix from
    # the output.  We replicate that approach exactly here so the injection string is
    # identical to what VeRL injects during training.
    _dummy_user = [{"role": "user", "content": [{"type": "text", "text": ""}]}]
    _dummy_prefix = tokenizer.apply_chat_template(
        _dummy_user,
        add_generation_prompt=False,
        tokenize=False,
        enable_thinking=False,
    )

    def _make_tool_injection(output_text: str) -> str:
        """Produce the exact string VeRL injects between phase 1 and phase 2.

        Mirrors verl.utils.chat_template.apply_chat_template's Qwen3.5 fallback
        (verl/utils/chat_template.py lines 87-114): prepend dummy user, render
        [dummy_user + tool_msg] with add_generation_prompt=True, strip dummy prefix.
        Result is the tool response wrapped in hermes format + generation prompt:
            <|im_start|>user\\n<tool_response>\\n{output}\\n</tool_response><|im_end|>\\n
            <|im_start|>assistant\\n<think>\\n\\n</think>\\n\\n
        """
        full = tokenizer.apply_chat_template(
            _dummy_user + [{"role": "tool", "content": output_text}],
            add_generation_prompt=True,
            tokenize=False,
            enable_thinking=False,
        )
        return full[len(_dummy_prefix):]

    # Build phase 1 prompts — same path as stage0_probe + VeRL training.
    phase1_prompts = []
    problems, golds, answer_types = [], [], []
    for _, row in df.iterrows():
        prob = _get_problem_text(row)
        problems.append(prob)
        golds.append(_get_gold(row))
        answer_types.append(_get_answer_type(row))
        msgs = [
            {"role": "system", "content": TIR_SYSTEM_PROMPT},
            {"role": "user", "content": prob},
        ]
        phase1_prompts.append(
            tokenizer.apply_chat_template(
                msgs,
                tools=[PYTHON_TOOL_SCHEMA],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=enable_thinking,
            )
        )


    # Phase 1: stop at </tool_call>
    p1_params = SamplingParams(
        max_tokens=max_tokens,
        stop=[TOOL_CALL_STOP],
        include_stop_str_in_output=True,
        temperature=temperature,
        top_p=top_p,
    )
    print("Running phase 1...")
    p1_outputs = llm.generate(phase1_prompts, p1_params)

    # Execute code + build phase 2 prompts
    p1_texts, p1_stops, codes = [], [], []
    sandbox_stdouts, sandbox_errs, sandbox_errors = [], [], []
    injection_texts = []  # saved for the txt output so the reader can see what was injected
    phase2_prompts = []
    p1_tokens_used = []

    for i, out in enumerate(p1_outputs):
        p1 = out.outputs[0]
        p1_text = p1.text
        p1_stop = p1.finish_reason or "unknown"
        p1_texts.append(p1_text)
        p1_stops.append(p1_stop)
        p1_tokens_used.append(len(p1.token_ids))

        code = extract_tool_call_code(p1_text)
        codes.append(code)

        if code and p1_stop == "stop":
            res = execute_code(code)
            sandbox_stdouts.append(res.stdout)
            sandbox_errs.append(res.stderr)
            sandbox_errors.append(res.error)
            output_text = res.stdout if not res.error else "(execution error)"
        else:
            sandbox_stdouts.append("")
            sandbox_errs.append("")
            sandbox_errors.append(False)
            output_text = "(no code block)"

        # Use apply_chat_template(role="tool") to match VeRL's injection exactly.
        injection = _make_tool_injection(output_text)
        injection_texts.append(injection)
        phase2_prompts.append(phase1_prompts[i] + p1_text + injection)

    # Phase 2: no stop, run to EOS
    p2_params_list = [
        SamplingParams(
            max_tokens=max(64, max_tokens - used),
            temperature=temperature,
            top_p=top_p,
        )
        for used in p1_tokens_used
    ]
    print("Running phase 2...")
    p2_outputs = llm.generate(phase2_prompts, p2_params_list)

    # Save txt files
    summary_lines = [
        f"{'#':>4}  {'type':15s}  {'code?':6s}  {'exec?':6s}  {'boxed?':8s}  {'score':6s}  problem[:60]",
        "-" * 100,
    ]

    for i in range(len(df)):
        p2 = p2_outputs[i].outputs[0]
        p2_text = p2.text
        p2_stop = p2.finish_reason or "unknown"

        boxed = _extract_boxed(p2_text) or _extract_boxed(p1_texts[i])

        verifier_score = None
        if boxed is not None:
            verifier_score = verify_answer(
                pred_text=f"\\boxed{{{boxed}}}",
                gold_answer=golds[i],
                answer_type=answer_types[i],
                gold_unit="",
            )

        txt = _format_rollout_txt(
            idx=i,
            problem=problems[i],
            gold=golds[i],
            answer_type=answer_types[i],
            phase1_prompt=phase1_prompts[i],
            p1_text=p1_texts[i],
            p1_stop=p1_stops[i],
            code=codes[i],
            sandbox_stdout=sandbox_stdouts[i],
            sandbox_err=sandbox_errs[i],
            sandbox_error=sandbox_errors[i],
            injection_text=injection_texts[i],
            p2_text=p2_text,
            p2_stop=p2_stop,
            boxed=boxed,
            verifier_score=verifier_score,
        )

        out_path = os.path.join(out_dir, f"rollout_{i:03d}.txt")
        with open(out_path, "w") as f:
            f.write(txt)

        summary_lines.append(
            f"{i:>4}  {answer_types[i]:15s}  "
            f"{'yes' if codes[i] else 'no':6s}  "
            f"{'yes' if sandbox_stdouts[i] else 'no':6s}  "
            f"{str(boxed)[:8] if boxed else 'None':8s}  "
            f"{str(verifier_score) if verifier_score is not None else '-':6s}  "
            f"{problems[i][:60]!r}"
        )
        print(f"  [{i:03d}] saved → {out_path}")

    summary_path = os.path.join(out_dir, "summary.txt")
    with open(summary_path, "w") as f:
        f.write(f"model   : {model_path}\n")
        f.write(f"parquet : {parquet_path}\n")
        f.write(f"n       : {len(df)}\n")
        f.write(f"temp    : {temperature}  top_p={top_p}  enable_thinking={enable_thinking}\n\n")
        f.write("\n".join(summary_lines) + "\n")

    print(f"\nSaved {len(df)} rollouts + summary → {out_dir}/")
    print("\n".join(summary_lines))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen3-0.6B")
    p.add_argument("--parquet", default="data/processed/corpus_train.parquet")
    p.add_argument("--n", type=int, default=4)
    p.add_argument("--out_dir", default="outputs/rollouts")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--gpu_mem", type=float, default=0.38)
    p.add_argument("--max_tokens", type=int, default=1536)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top_p", type=float, default=0.9)
    p.add_argument("--enable_thinking", action="store_true", default=False)
    args = p.parse_args()

    run_dump(
        model_path=args.model,
        parquet_path=args.parquet,
        n=args.n,
        out_dir=args.out_dir,
        seed=args.seed,
        gpu_mem=args.gpu_mem,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        enable_thinking=args.enable_thinking,
    )


if __name__ == "__main__":
    main()
