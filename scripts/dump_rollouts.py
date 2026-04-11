"""Dump TIR rollouts to human-readable txt files and a structured parquet.

Runs the same 2-phase vLLM loop as VeRL's ToolAgentLoop (qwen3_coder format,
enable_thinking=True for phase 1 / False for phase 2, tool schema injected).

Supports think-interrupt: when --thinking_budget is set, phase 1 generation is
capped at thinking_budget tokens. If the model exceeds the budget without emitting
</think>, the interrupt phrase is injected and a second generation (phase 1b) runs
with max_tokens=tool_call_budget. This exactly mirrors the logic in
verl/verl/experimental/agent_loop/tool_agent_loop.py lines 294-333.

Output:
  - Per-rollout txt files for human inspection
  - rollouts.parquet with full trace and metadata (scoring is a separate step)

Usage (via dump_rollouts.sh):
    MODEL=Qwen/Qwen3.5-4B N=4 bash scripts/dump_rollouts.sh

    # With think-interrupt:
    THINKING_BUDGET=12288 TOOL_CALL_BUDGET=2048 ANSWER_BUDGET=4096 \\
        bash scripts/dump_rollouts.sh

Direct usage inside container:
    python3 scripts/dump_rollouts.py \\
        --model Qwen/Qwen3.5-4B \\
        --parquet data/processed/corpus_train.parquet \\
        --n 4 --n_rollouts 8 \\
        --thinking_budget 12288 --tool_call_budget 2048 --answer_budget 4096 \\
        --out_dir outputs/rollouts_test
"""

from __future__ import annotations

import argparse
import os

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
    ei = row.get("extra_info")
    if isinstance(ei, dict) and ei.get("problem"):
        return str(ei["problem"])
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
    interrupted: bool,
    p1b_text: str,
    p1b_stop: str,
    code: str | None,
    sandbox_stdout: str,
    sandbox_err: str,
    sandbox_error: bool,
    injection_text: str,
    p2_text: str,
    p2_stop: str,
    boxed: str | None,
    interrupt_phrase: str = "",
) -> str:
    sep = "=" * 72
    thin = "-" * 72

    lines = [
        sep,
        f"ROLLOUT {idx:03d}",
        sep,
        f"ANSWER TYPE : {answer_type}",
        f"GOLD        : {gold}",
        f"INTERRUPTED : {interrupted}",
        "",
        f"PROBLEM\n{problem.strip()}",
        thin,
        "PHASE 1 PROMPT  (rendered — check for <tools> schema injection)",
        thin,
        phase1_prompt,
        thin,
        f"PHASE 1  (stop_reason={p1_stop})",
        thin,
        p1_text.strip(),
        "",
    ]

    if interrupted:
        lines += [
            thin,
            "THINK-INTERRUPT  (injected — mask=0 in training)",
            thin,
            repr(interrupt_phrase),
            "",
            thin,
            f"PHASE 1B  (stop_reason={p1b_stop})",
            thin,
            p1b_text.strip() if p1b_text else "(empty)",
            "",
        ]

    lines += [
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
        repr(injection_text),
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
        "",
    ]

    return "\n".join(lines)


def run_dump(
    model_path: str,
    parquet_path: str,
    n: int,
    n_rollouts: int,
    out_dir: str,
    seed: int = 42,
    gpu_mem: float = 0.38,
    temperature: float = 1.0,
    top_p: float = 0.9,
    enable_thinking: bool = True,
    max_tokens: int = 8192,
    thinking_budget: int | None = None,
    tool_call_budget: int | None = None,
    answer_budget: int | None = None,
    max_prompt_len: int = 1024,
    max_tool_response_len: int = 512,
    dump_txt: bool = False,
) -> None:
    from transformers import AutoTokenizer  # noqa: PLC0415
    from vllm import LLM, SamplingParams  # noqa: PLC0415

    from phys_reasoner.tir.prompts import (  # noqa: PLC0415
        PYTHON_TOOL_SCHEMA,
        THINK_INTERRUPT_PHRASE,
        TOOL_CALL_STOP,
        TIR_SYSTEM_PROMPT,
        extract_tool_call_code,
    )
    from phys_reasoner.tir.sandbox import execute_code  # noqa: PLC0415

    # --- Validate budgets ---
    interrupt_enabled = thinking_budget is not None
    if interrupt_enabled:
        if tool_call_budget is None:
            raise ValueError("--tool_call_budget required when --thinking_budget is set")
        if answer_budget is None:
            raise ValueError("--answer_budget required when --thinking_budget is set")

    os.makedirs(out_dir, exist_ok=True)

    # --- Load tokenizer (needed before LLM for prompt building + interrupt setup) ---
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    if interrupt_enabled:
        interrupt_ids = tokenizer.encode(THINK_INTERRUPT_PHRASE, add_special_tokens=False)
        think_end_id = tokenizer.convert_tokens_to_ids("</think>")
        interrupt_len = len(interrupt_ids)
        # Match VeRL: response_length = thinking + interrupt + tool_call + tool_response + answer
        response_budget = thinking_budget + interrupt_len + tool_call_budget + max_tool_response_len + answer_budget
        print(f"Think-interrupt enabled: thinking={thinking_budget}, interrupt={interrupt_len}, "
              f"tool_call={tool_call_budget}, tool_response={max_tool_response_len}, answer={answer_budget}")
        print(f"  response_budget={response_budget}")
    else:
        # Phase 1 up to max_tokens + injection + phase 2 up to max_tokens
        response_budget = 2 * max_tokens + max_tool_response_len + 256
        print(f"Think-interrupt disabled: max_tokens={max_tokens}")

    # Match VeRL: max_model_len = max_prompt_len + response_budget
    max_model_len = max_prompt_len + response_budget

    # --- Load data ---
    df = pd.read_parquet(parquet_path)
    if n > 0 and n < len(df):
        df = df.sample(n=n, random_state=seed)
    # n <= 0 or n >= len(df): use all rows, preserve original order
    original_indices = df.index.tolist()
    df = df.reset_index(drop=True)
    print(f"Loaded {len(df)} problems from {parquet_path}")

    # --- Build phase 1 prompts ---
    problems, golds, answer_types, extra_infos = [], [], [], []
    phase1_prompts = []
    for _, row in df.iterrows():
        prob = _get_problem_text(row)
        problems.append(prob)
        golds.append(_get_gold(row))
        answer_types.append(_get_answer_type(row))
        extra_infos.append(row.get("extra_info", {}))
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

    # --- Expand for n_rollouts ---
    expanded_prompts = []
    expanded_map = []  # (problem_idx, rollout_idx)
    for i in range(len(df)):
        for r in range(n_rollouts):
            expanded_prompts.append(phase1_prompts[i])
            expanded_map.append((i, r))
    total = len(expanded_prompts)
    print(f"Expanded to {total} rollouts ({len(df)} problems x {n_rollouts} rollouts)")

    # --- Create vLLM engine ---
    print(f"Creating vLLM engine: max_model_len={max_model_len}, gpu_mem={gpu_mem}")
    llm = LLM(
        model=model_path,
        dtype="bfloat16",
        gpu_memory_utilization=gpu_mem,
        max_model_len=max_model_len,
    )

    # --- Tool injection helper (matches VeRL's Qwen3.5 dummy-user workaround) ---
    _dummy_user = [{"role": "user", "content": [{"type": "text", "text": ""}]}]
    _dummy_prefix = tokenizer.apply_chat_template(
        _dummy_user,
        add_generation_prompt=False,
        tokenize=False,
        enable_thinking=False,
    )

    def _make_tool_injection(output_text: str) -> str:
        full = tokenizer.apply_chat_template(
            _dummy_user + [{"role": "tool", "content": output_text}],
            add_generation_prompt=True,
            tokenize=False,
            enable_thinking=False,
        )
        return full[len(_dummy_prefix):]

    # =====================================================================
    # Phase 1: thinking + tool call (or just thinking if interrupt fires)
    # =====================================================================
    p1_max = thinking_budget if interrupt_enabled else max_tokens
    p1_params = SamplingParams(
        max_tokens=p1_max,
        stop=[TOOL_CALL_STOP],
        include_stop_str_in_output=True,
        temperature=temperature,
        top_p=top_p,
    )
    print(f"Phase 1: generating {total} rollouts, max_tokens={p1_max}")
    p1_outputs = llm.generate(expanded_prompts, p1_params)

    # =====================================================================
    # Check interrupt condition (mirrors tool_agent_loop.py lines 297-299)
    # =====================================================================
    p1_texts = []
    p1_stops = []
    interrupted_flags = []
    p1b_indices = []
    p1b_prompts = []

    for i, out in enumerate(p1_outputs):
        p1 = out.outputs[0]
        p1_texts.append(p1.text)
        p1_stops.append(p1.finish_reason or "unknown")

        needs_interrupt = (
            interrupt_enabled
            and len(p1.token_ids) >= thinking_budget
            and think_end_id not in p1.token_ids
        )
        interrupted_flags.append(needs_interrupt)

        if needs_interrupt:
            p1b_indices.append(i)
            p1b_prompts.append(expanded_prompts[i] + p1.text + THINK_INTERRUPT_PHRASE)

    # =====================================================================
    # Phase 1b: tool call generation after interrupt (only for interrupted)
    # =====================================================================
    p1b_data: dict[int, tuple[str, str]] = {}  # idx -> (text, stop_reason)
    if p1b_indices:
        p1b_params = SamplingParams(
            max_tokens=tool_call_budget,
            stop=[TOOL_CALL_STOP],
            include_stop_str_in_output=True,
            temperature=temperature,
            top_p=top_p,
        )
        n_interrupted = len(p1b_indices)
        print(f"Phase 1b: {n_interrupted}/{total} interrupted, max_tokens={tool_call_budget}")
        p1b_outputs = llm.generate(p1b_prompts, p1b_params)
        for j, idx in enumerate(p1b_indices):
            p1b_out = p1b_outputs[j].outputs[0]
            p1b_data[idx] = (p1b_out.text, p1b_out.finish_reason or "unknown")
    else:
        print("Phase 1b: no interrupts fired")

    # =====================================================================
    # Extract code + sandbox execution
    # =====================================================================
    codes: list[str | None] = []
    sandbox_stdouts: list[str] = []
    sandbox_errs: list[str] = []
    sandbox_errors: list[bool] = []

    for i in range(total):
        if interrupted_flags[i]:
            p1b_text, _ = p1b_data[i]
            code = extract_tool_call_code(p1b_text)
        else:
            code = extract_tool_call_code(p1_texts[i])
        codes.append(code)

        if code is not None:
            res = execute_code(code)
            sandbox_stdouts.append(res.stdout)
            sandbox_errs.append(res.stderr)
            sandbox_errors.append(res.error)
        else:
            sandbox_stdouts.append("")
            sandbox_errs.append("")
            sandbox_errors.append(False)

    # =====================================================================
    # Build phase 2 prompts with tool injection
    # =====================================================================
    injection_texts: list[str] = []
    phase2_prompts: list[str] = []

    for i in range(total):
        # Build tool response text (matches PythonSandboxTool.execute)
        if codes[i] is not None and not sandbox_errors[i]:
            output_text = sandbox_stdouts[i] or "(no output)"
        elif codes[i] is not None and sandbox_errors[i]:
            output_text = f"(execution error)\n{sandbox_errs[i][:300]}"
        else:
            output_text = "(no code block)"

        # Truncate tool response (matches VeRL: tool_response_truncate_side=right)
        if len(output_text) > max_tool_response_len:
            output_text = "(truncated)..." + output_text[-max_tool_response_len:]

        injection = _make_tool_injection(output_text)
        injection_texts.append(injection)

        if interrupted_flags[i]:
            p1b_text, _ = p1b_data[i]
            p2_prompt = expanded_prompts[i] + p1_texts[i] + THINK_INTERRUPT_PHRASE + p1b_text + injection
        else:
            p2_prompt = expanded_prompts[i] + p1_texts[i] + injection
        phase2_prompts.append(p2_prompt)

    # =====================================================================
    # Phase 2: final answer
    # =====================================================================
    p2_max = answer_budget if interrupt_enabled else max_tokens
    p2_params = SamplingParams(
        max_tokens=p2_max,
        temperature=temperature,
        top_p=top_p,
    )
    print(f"Phase 2: generating {total} rollouts, max_tokens={p2_max}")
    p2_outputs = llm.generate(phase2_prompts, p2_params)

    # =====================================================================
    # Write outputs: txt files + rollouts.parquet
    # =====================================================================
    records = []
    summary_lines = [
        f"{'#':>4}  {'prob':>4}  {'roll':>4}  {'type':15s}  {'intr':5s}  {'code?':6s}  "
        f"{'exec?':6s}  {'boxed?':8s}  problem[:50]",
        "-" * 110,
    ]

    for i in range(total):
        prob_idx, roll_idx = expanded_map[i]
        p2 = p2_outputs[i].outputs[0]
        p2_text = p2.text
        p2_stop = p2.finish_reason or "unknown"

        boxed = _extract_boxed(p2_text) or _extract_boxed(p1_texts[i])
        if interrupted_flags[i]:
            p1b_text, _ = p1b_data[i]
            boxed = boxed or _extract_boxed(p1b_text)

        p1b_text_for_output = p1b_data[i][0] if interrupted_flags[i] else ""
        p1b_stop_for_output = p1b_data[i][1] if interrupted_flags[i] else ""

        if dump_txt:
            txt = _format_rollout_txt(
                idx=i,
                problem=problems[prob_idx],
                gold=golds[prob_idx],
                answer_type=answer_types[prob_idx],
                phase1_prompt=phase1_prompts[prob_idx],
                p1_text=p1_texts[i],
                p1_stop=p1_stops[i],
                interrupted=interrupted_flags[i],
                p1b_text=p1b_text_for_output,
                p1b_stop=p1b_stop_for_output,
                code=codes[i],
                sandbox_stdout=sandbox_stdouts[i],
                sandbox_err=sandbox_errs[i],
                sandbox_error=sandbox_errors[i],
                injection_text=injection_texts[i],
                p2_text=p2_text,
                p2_stop=p2_stop,
                boxed=boxed,
                interrupt_phrase=THINK_INTERRUPT_PHRASE,
            )
            out_path = os.path.join(out_dir, f"rollout_{i:03d}.txt")
            with open(out_path, "w") as f:
                f.write(txt)

        summary_lines.append(
            f"{i:>4}  {prob_idx:>4}  {roll_idx:>4}  {answer_types[prob_idx]:15s}  "
            f"{'yes' if interrupted_flags[i] else 'no':5s}  "
            f"{'yes' if codes[i] else 'no':6s}  "
            f"{'yes' if sandbox_stdouts[i] else 'no':6s}  "
            f"{str(boxed)[:8] if boxed else 'None':8s}  "
            f"{problems[prob_idx][:50]!r}"
        )

        # Build parquet record
        records.append({
            "problem_idx": original_indices[prob_idx],
            "rollout_idx": roll_idx,
            "gold_answer": golds[prob_idx],
            "answer_type": answer_types[prob_idx],
            "phase1_text": p1_texts[i],
            "interrupted": interrupted_flags[i],
            "phase1b_text": p1b_text_for_output if interrupted_flags[i] else None,
            "code": codes[i],
            "sandbox_stdout": sandbox_stdouts[i],
            "sandbox_error": sandbox_errors[i],
            "phase2_text": p2_text,
            "extra_info": extra_infos[prob_idx],
        })

    # Write summary
    summary_path = os.path.join(out_dir, "summary.txt")
    with open(summary_path, "w") as f:
        f.write(f"model             : {model_path}\n")
        f.write(f"parquet           : {parquet_path}\n")
        f.write(f"n_problems        : {len(df)}\n")
        f.write(f"n_rollouts        : {n_rollouts}\n")
        f.write(f"total             : {total}\n")
        f.write(f"temperature       : {temperature}  top_p={top_p}  enable_thinking={enable_thinking}\n")
        if interrupt_enabled:
            f.write(f"thinking_budget   : {thinking_budget}\n")
            f.write(f"tool_call_budget  : {tool_call_budget}\n")
            f.write(f"answer_budget     : {answer_budget}\n")
            f.write(f"interrupt_len     : {interrupt_len}\n")
            n_interrupted = sum(interrupted_flags)
            f.write(f"interrupted       : {n_interrupted}/{total}\n")
        else:
            f.write(f"max_tokens        : {max_tokens}\n")
        f.write("\n")
        f.write("\n".join(summary_lines) + "\n")

    # Write parquet
    parquet_path_out = os.path.join(out_dir, "rollouts.parquet")
    pd.DataFrame(records).to_parquet(parquet_path_out, index=False)
    print(f"\nSaved {total} rollouts + summary → {out_dir}/")
    print(f"Parquet: {parquet_path_out}")
    print("\n".join(summary_lines))


def main() -> None:
    p = argparse.ArgumentParser(description="Dump TIR rollouts for trajectory inspection")
    p.add_argument("--model", default="Qwen/Qwen3-0.6B")
    p.add_argument("--parquet", default="data/processed/corpus_train.parquet")
    p.add_argument("--n", type=int, default=4,
                   help="Number of problems to sample; n<=0 uses all rows in parquet order")
    p.add_argument("--n_rollouts", type=int, default=1, help="Rollouts per problem")
    p.add_argument("--out_dir", default="outputs/rollouts")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--gpu_mem", type=float, default=0.38)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top_p", type=float, default=0.9)
    p.add_argument("--enable_thinking", action="store_true", default=False)
    # Budget args — simple mode
    p.add_argument("--max_tokens", type=int, default=8192,
                   help="Response budget per phase when interrupt is disabled")
    # Budget args — think-interrupt mode (all three required together)
    p.add_argument("--thinking_budget", type=int, default=None,
                   help="Phase 1 token cap; enables think-interrupt when set")
    p.add_argument("--tool_call_budget", type=int, default=None,
                   help="Phase 1b token cap after interrupt fires")
    p.add_argument("--answer_budget", type=int, default=None,
                   help="Phase 2 token cap for final answer")
    p.add_argument("--max_prompt_len", type=int, default=1024,
                   help="Prompt length budget for max_model_len (matches VeRL)")
    p.add_argument("--max_tool_response_len", type=int, default=512,
                   help="Tool response text cap in chars (matches VeRL)")
    p.add_argument("--dump_txt", action="store_true", default=False,
                   help="Also write per-rollout human-readable txt files (slow; parquet is always written)")
    args = p.parse_args()

    run_dump(
        model_path=args.model,
        parquet_path=args.parquet,
        n=args.n,
        n_rollouts=args.n_rollouts,
        out_dir=args.out_dir,
        seed=args.seed,
        gpu_mem=args.gpu_mem,
        temperature=args.temperature,
        top_p=args.top_p,
        enable_thinking=args.enable_thinking,
        max_tokens=args.max_tokens,
        thinking_budget=args.thinking_budget,
        tool_call_budget=args.tool_call_budget,
        answer_budget=args.answer_budget,
        max_prompt_len=args.max_prompt_len,
        max_tool_response_len=args.max_tool_response_len,
        dump_txt=args.dump_txt,
    )


if __name__ == "__main__":
    main()
