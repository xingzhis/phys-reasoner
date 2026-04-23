"""Generate rollouts (TIR or CoT) and save raw traces to a parquet.

Two modes share the same vLLM engine, tokenizer, chunking, and record schema;
the mode switch only changes prompt construction, stop tokens, and post-phase-1
handling.

TIR mode (--mode tir, default)
    Same 2-phase loop as VeRL's ToolAgentLoop (qwen3_coder format,
    enable_thinking=True for phase 1 / False for phase 2, tool schema injected
    via `tools=[...]`). Mirrors
    verl/experimental/agent_loop/tool_agent_loop.py lines 294-333.
    Phase 1 stops on </tool_call>; sandbox runs extracted code; tool response
    injected; phase 2 generates the final \\boxed{...}.

CoT mode (--mode cot)
    Mirrors VeRL's SingleTurnAgentLoop
    (verl/experimental/agent_loop/single_turn_agent_loop.py). No tool schema,
    no </tool_call> stop. Phase 1 runs up to thinking_budget. If interrupt
    fires (budget hit AND </think> not emitted), phase 2 runs with
    max_tokens = response_length - thinking_budget - interrupt_len, which
    matches the TIR budget identity:
        response_length = thinking + interrupt + tool_call + tool_response + answer
    If interrupt does NOT fire, no phase 2. Total response length is therefore
    identical between modes, making this a fair TIR-vs-CoT ablation.

Output:
  - rollouts.parquet with raw phase1/phase1b/phase2 text and metadata.
    Answer extraction is deliberately NOT done here — scorers own it.
  - Optional per-rollout txt files for human inspection (--dump_txt).

Usage (via rollout.sh):
    MODE=tir MODEL=Qwen/Qwen3.5-4B N=4 bash eval/inference/rollout.sh
    MODE=cot MODEL=Qwen/Qwen3.5-4B N=4 bash eval/inference/rollout.sh

    # With think-interrupt (training-matched budgets):
    THINKING_BUDGET=12288 TOOL_CALL_BUDGET=2048 ANSWER_BUDGET=4096 \\
        bash eval/inference/rollout.sh

Direct usage inside container:
    python3 eval/inference/rollout.py \\
        --mode tir \\
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
    top_p: float = 1.0,
    top_k: int = -1,
    presence_penalty: float = 0.0,
    repetition_penalty: float = 1.0,
    enable_thinking: bool = True,
    max_tokens: int = 8192,
    thinking_budget: int | None = None,
    tool_call_budget: int | None = None,
    answer_budget: int | None = None,
    max_prompt_len: int = 1024,
    max_tool_response_len: int = 1024,
    dump_txt: bool = False,
    start_idx: int | None = None,
    end_idx: int | None = None,
    chunk_size: int | None = None,
    mode: str = "tir",
) -> None:
    from transformers import AutoTokenizer  # noqa: PLC0415
    from vllm import LLM, SamplingParams  # noqa: PLC0415

    from phys_reasoner.tir.prompts import (  # noqa: PLC0415
        COT_SYSTEM_PROMPT,
        PYTHON_TOOL_SCHEMA,
        THINK_INTERRUPT_PHRASE,
        TOOL_CALL_STOP,
        TIR_SYSTEM_PROMPT,
        extract_tool_call_code,
    )
    from phys_reasoner.tir.sandbox import execute_code  # noqa: PLC0415

    if mode not in ("tir", "cot"):
        raise ValueError(f"--mode must be 'tir' or 'cot', got {mode!r}")
    is_cot = mode == "cot"

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
        # Match VeRL: response_length = thinking + interrupt + tool_call + tool_response + answer.
        # In CoT mode (single_turn_agent_loop.py), tool_call + tool_response + answer all
        # collapse into the post-interrupt answer budget so that TIR and CoT use identical
        # total response length.
        response_budget = thinking_budget + interrupt_len + tool_call_budget + max_tool_response_len + answer_budget
        cot_post_interrupt_budget = response_budget - thinking_budget - interrupt_len
        if is_cot:
            print(f"CoT mode: thinking={thinking_budget}, interrupt={interrupt_len}, "
                  f"post_interrupt={cot_post_interrupt_budget}")
        else:
            print(f"TIR mode: thinking={thinking_budget}, interrupt={interrupt_len}, "
                  f"tool_call={tool_call_budget}, tool_response={max_tool_response_len}, answer={answer_budget}")
        print(f"  response_budget={response_budget}")
    else:
        # Phase 1 up to max_tokens + injection + phase 2 up to max_tokens
        response_budget = 2 * max_tokens + max_tool_response_len + 256
        cot_post_interrupt_budget = None
        print(f"Think-interrupt disabled: max_tokens={max_tokens} (mode={mode})")

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

    # --- Apply shard slice (after sampling, before expansion) ---
    if start_idx is not None or end_idx is not None:
        s = start_idx if start_idx is not None else 0
        e = end_idx if end_idx is not None else len(df)
        s = max(0, s)
        e = min(len(df), e)
        df = df.iloc[s:e].reset_index(drop=True)
        original_indices = original_indices[s:e]
        print(f"Shard slice: rows [{s}:{e}) = {len(df)} problems")

    # --- Build phase 1 prompts ---
    # TIR: TIR_SYSTEM_PROMPT + tools=[PYTHON_TOOL_SCHEMA]
    # CoT: COT_SYSTEM_PROMPT + no tools (mirrors build_cot_parquets.py and VeRL's
    #      single_turn_agent — no tool schema visible to the model).
    system_prompt_text = COT_SYSTEM_PROMPT if is_cot else TIR_SYSTEM_PROMPT
    tools_arg = None if is_cot else [PYTHON_TOOL_SCHEMA]
    problems, golds, answer_types, extra_infos = [], [], [], []
    phase1_prompts = []
    for _, row in df.iterrows():
        prob = _get_problem_text(row)
        problems.append(prob)
        golds.append(_get_gold(row))
        answer_types.append(_get_answer_type(row))
        extra_infos.append(row.get("extra_info", {}))
        msgs = [
            {"role": "system", "content": system_prompt_text},
            {"role": "user", "content": prob},
        ]
        phase1_prompts.append(
            tokenizer.apply_chat_template(
                msgs,
                tools=tools_arg,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=enable_thinking,
            )
        )

    # --- Create vLLM engine (once; reused across chunks) ---
    # Qwen3-4B at 20k context needs ~2.81 GB KV per sequence. On 40G A100 this
    # is brutal. Settings tuned for 40G:
    #   max_num_seqs=16 — minimal activation buffer reservation
    #   max_num_batched_tokens=4096 — chunked prefill
    #   KV cache pool must be ≥ 2.81 GB × max_num_seqs ≈ 45 GB ideally, but
    #   we accept ~12 GB (4 concurrent) to fit within 40G.
    import os as _os
    max_num_seqs = int(_os.environ.get("VLLM_MAX_NUM_SEQS", "16"))
    max_num_batched_tokens = int(_os.environ.get("VLLM_MAX_NUM_BATCHED_TOKENS", "4096"))
    print(f"Creating vLLM engine: max_model_len={max_model_len}, gpu_mem={gpu_mem}, "
          f"max_num_seqs={max_num_seqs}, max_num_batched_tokens={max_num_batched_tokens}")
    llm = LLM(
        model=model_path,
        dtype="bfloat16",
        gpu_memory_utilization=gpu_mem,
        max_model_len=max_model_len,
        max_num_seqs=max_num_seqs,
        max_num_batched_tokens=max_num_batched_tokens,
        enable_chunked_prefill=True,
        enforce_eager=True,  # Match VeRL; CUDA graphs can destabilize Qwen3.5 GDN attention
    )

    # --- Tool injection helper (matches VeRL's Qwen3.5 dummy-user workaround) ---
    # VeRL's verl/utils/chat_template.py uses the same dummy-user trick and passes
    # enable_thinking from apply_chat_template_kwargs (=True in our config).
    # We must also prepend <|im_end|> to close the assistant turn that was cut short
    # by the </tool_call> stop string — in VeRL the model generates past </tool_call>
    # and naturally emits <|im_end|>, but here we stopped generation early.
    _im_end = tokenizer.eos_token  # <|im_end|> for Qwen3.5
    _dummy_user = [{"role": "user", "content": [{"type": "text", "text": ""}]}]
    _dummy_prefix = tokenizer.apply_chat_template(
        _dummy_user,
        add_generation_prompt=False,
        tokenize=False,
        enable_thinking=enable_thinking,
    )

    def _make_tool_injection(output_text: str) -> str:
        full = tokenizer.apply_chat_template(
            _dummy_user + [{"role": "tool", "content": output_text}],
            add_generation_prompt=True,
            tokenize=False,
            enable_thinking=enable_thinking,
        )
        return _im_end + full[len(_dummy_prefix):]

    # --- Shared sampling params ---
    p1_max = thinking_budget if interrupt_enabled else max_tokens
    p2_max = answer_budget if interrupt_enabled else max_tokens
    _common = dict(
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        presence_penalty=presence_penalty,
        repetition_penalty=repetition_penalty,
    )
    # Phase 1: TIR stops at </tool_call>; CoT has no stop (runs to EOS or budget).
    p1_params = SamplingParams(
        max_tokens=p1_max,
        stop=None if is_cot else [TOOL_CALL_STOP],
        include_stop_str_in_output=False if is_cot else True,
        **_common,
    )
    p2_params = SamplingParams(
        max_tokens=p2_max,
        **_common,
    )
    # Phase 1b:
    #   TIR: generates the tool call after interrupt — stops at </tool_call>,
    #        max_tokens=tool_call_budget.
    #   CoT: generates the final answer after interrupt — no stop, max_tokens is
    #        the post-interrupt budget = response_length - thinking - interrupt_len
    #        (absorbs TIR's tool_call + tool_response + answer share so total
    #        response length is identical across modes).
    p1b_params = None
    if interrupt_enabled:
        if is_cot:
            p1b_params = SamplingParams(
                max_tokens=cot_post_interrupt_budget,
                **_common,
            )
        else:
            p1b_params = SamplingParams(
                max_tokens=tool_call_budget,
                stop=[TOOL_CALL_STOP],
                include_stop_str_in_output=True,
                **_common,
            )

    # --- Chunk setup ---
    n_problems = len(df)
    chunk = chunk_size if (chunk_size is not None and chunk_size > 0) else n_problems
    n_chunks = (n_problems + chunk - 1) // chunk
    total_all = n_problems * n_rollouts
    print(f"Chunking: {n_problems} problems × {n_rollouts} rollouts = {total_all} "
          f"rollouts, chunk_size={chunk} ({n_chunks} chunks)")

    all_records: list[dict] = []
    all_summary_lines: list[str] = [
        f"{'#':>6}  {'prob':>5}  {'roll':>4}  {'type':15s}  {'intr':5s}  {'code?':6s}  "
        f"{'exec?':6s}  {'boxed?':8s}  problem[:50]",
        "-" * 112,
    ]
    global_i = 0

    for k in range(n_chunks):
        chunk_path = os.path.join(out_dir, f"rollouts_chunk_{k:04d}.parquet")
        if os.path.exists(chunk_path):
            print(f"[chunk {k+1}/{n_chunks}] resume: loading existing {chunk_path}")
            resumed = pd.read_parquet(chunk_path).to_dict("records")
            all_records.extend(resumed)
            global_i += len(resumed)
            continue

        lo = k * chunk
        hi = min((k + 1) * chunk, n_problems)
        print(f"\n[chunk {k+1}/{n_chunks}] problems [{lo}:{hi}) ({hi-lo} problems)")

        # --- Expand for this chunk ---
        expanded_prompts: list[str] = []
        expanded_map: list[tuple[int, int]] = []
        for i in range(lo, hi):
            for r in range(n_rollouts):
                expanded_prompts.append(phase1_prompts[i])
                expanded_map.append((i, r))
        total = len(expanded_prompts)

        # ===== Phase 1 =====
        print(f"  phase 1: {total} rollouts, max_tokens={p1_max}")
        p1_outputs = llm.generate(expanded_prompts, p1_params)

        p1_texts: list[str] = []
        p1_stops: list[str] = []
        interrupted_flags: list[bool] = []
        p1b_indices: list[int] = []
        p1b_prompts: list[str] = []
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

        # ===== Phase 1b =====
        # TIR: generates the tool call after interrupt (stops at </tool_call>).
        # CoT: generates the final answer after interrupt (post_interrupt_budget, no stop).
        p1b_data: dict[int, tuple[str, str]] = {}
        if p1b_indices:
            p1b_max = cot_post_interrupt_budget if is_cot else tool_call_budget
            print(f"  phase 1b: {len(p1b_indices)}/{total} interrupted, max_tokens={p1b_max}")
            p1b_outputs = llm.generate(p1b_prompts, p1b_params)
            for j, idx in enumerate(p1b_indices):
                p1b_out = p1b_outputs[j].outputs[0]
                p1b_data[idx] = (p1b_out.text, p1b_out.finish_reason or "unknown")

        # Defaults populated by either the TIR sandbox/phase-2 path or skipped in CoT.
        codes: list[str | None] = [None] * total
        sandbox_stdouts: list[str] = [""] * total
        sandbox_errs: list[str] = [""] * total
        sandbox_errors: list[bool] = [False] * total
        injection_texts: list[str] = [""] * total
        p2_results: dict[int, tuple[str, str]] = {}

        if is_cot:
            # CoT: no sandbox, no phase 2. Phase 1 (+ phase 1b when interrupted) is the
            # entire response — mirrors VeRL's single_turn_agent_loop.py.
            print(f"  cot: skipping sandbox/phase-2 (answer is phase1 + phase1b)")
        else:
            # ===== Sandbox (TIR) =====
            for i in range(total):
                if interrupted_flags[i]:
                    p1b_text, _ = p1b_data[i]
                    code = extract_tool_call_code(p1b_text)
                else:
                    code = extract_tool_call_code(p1_texts[i])
                codes[i] = code
                if code is not None:
                    res = execute_code(code)
                    sandbox_stdouts[i] = res.stdout
                    sandbox_errs[i] = res.stderr
                    sandbox_errors[i] = res.error

            # ===== Build phase 2 prompts (only for rollouts that produced a tool call) =====
            # Matches VeRL's tool_agent_loop.py: if no tool_calls found after phase 1,
            # the agent returns AgentState.TERMINATED — no tool response injection, no
            # phase 2 generation. The model's phase 1 output is the final answer.
            phase2_indices: list[int] = []
            phase2_prompts: list[str] = []
            for i in range(total):
                if codes[i] is None:
                    # No tool call → terminate (match VeRL behavior)
                    continue
                if not sandbox_errors[i]:
                    output_text = sandbox_stdouts[i] or "(no output)"
                else:
                    # Match python_sandbox_tool.py: append reminder on error so the
                    # model writes \boxed{} instead of retrying with a 2nd tool call.
                    from phys_reasoner.tir.prompts import TOOL_ERROR_REMINDER  # noqa: PLC0415
                    output_text = f"(execution error)\n{sandbox_errs[i][:300]}{TOOL_ERROR_REMINDER}"
                if len(output_text) > max_tool_response_len:
                    output_text = "(truncated)..." + output_text[-max_tool_response_len:]
                injection = _make_tool_injection(output_text)
                injection_texts[i] = injection
                if interrupted_flags[i]:
                    p1b_text, _ = p1b_data[i]
                    p2_prompt = expanded_prompts[i] + p1_texts[i] + THINK_INTERRUPT_PHRASE + p1b_text + injection
                else:
                    p2_prompt = expanded_prompts[i] + p1_texts[i] + injection
                phase2_indices.append(i)
                phase2_prompts.append(p2_prompt)

            # ===== Phase 2 (only for rollouts with tool calls) =====
            if phase2_prompts:
                print(f"  phase 2: {len(phase2_prompts)}/{total} rollouts (skipped {total - len(phase2_prompts)} with no tool call), max_tokens={p2_max}")
                p2_outputs = llm.generate(phase2_prompts, p2_params)
                for j, idx in enumerate(phase2_indices):
                    p2_out = p2_outputs[j].outputs[0]
                    p2_results[idx] = (p2_out.text, p2_out.finish_reason or "unknown")
            else:
                print(f"  phase 2: 0/{total} rollouts needed phase 2 (all terminated without tool call)")

        # ===== Build chunk records =====
        chunk_records: list[dict] = []
        for i in range(total):
            prob_idx, roll_idx = expanded_map[i]
            if i in p2_results:
                p2_text, p2_stop = p2_results[i]
            else:
                p2_text = None
                p2_stop = "cot_no_phase2" if is_cot else "terminated_no_tool_call"
            boxed_candidates = _extract_boxed(p1_texts[i])
            if interrupted_flags[i]:
                p1b_text, _ = p1b_data[i]
                boxed_candidates = boxed_candidates or _extract_boxed(p1b_text)
            if p2_text:
                boxed_candidates = _extract_boxed(p2_text) or boxed_candidates
            boxed = boxed_candidates[-1] if boxed_candidates else None
            p1b_text_for_output = p1b_data[i][0] if interrupted_flags[i] else ""
            p1b_stop_for_output = p1b_data[i][1] if interrupted_flags[i] else ""

            if dump_txt:
                txt = _format_rollout_txt(
                    idx=global_i,
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
                    p2_text=p2_text or "",
                    p2_stop=p2_stop,
                    boxed=boxed,
                    interrupt_phrase=THINK_INTERRUPT_PHRASE,
                )
                out_path = os.path.join(out_dir, f"rollout_{global_i:06d}.txt")
                with open(out_path, "w") as f:
                    f.write(txt)

            all_summary_lines.append(
                f"{global_i:>6}  {prob_idx:>5}  {roll_idx:>4}  {answer_types[prob_idx]:15s}  "
                f"{'yes' if interrupted_flags[i] else 'no':5s}  "
                f"{'yes' if codes[i] else 'no':6s}  "
                f"{'yes' if sandbox_stdouts[i] else 'no':6s}  "
                f"{str(boxed)[:8] if boxed else 'None':8s}  "
                f"{problems[prob_idx][:50]!r}"
            )
            global_i += 1

            chunk_records.append({
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

        # Flush chunk parquet (crash safety).
        # pyarrow rejects lone UTF-16 surrogates in strings; the model occasionally
        # emits them (e.g. '\udcXX'). Scrub string fields before serialization.
        def _scrub(v):
            if isinstance(v, str):
                # encode with 'replace' turns unencodable surrogates into '?'
                return v.encode("utf-8", "replace").decode("utf-8")
            return v
        for rec in chunk_records:
            for k, v in rec.items():
                rec[k] = _scrub(v)
        pd.DataFrame(chunk_records).to_parquet(chunk_path, index=False)
        print(f"  flushed {len(chunk_records)} rollouts → {chunk_path}")
        all_records.extend(chunk_records)

    # ===== Final outputs =====
    total_rollouts = len(all_records)
    n_interrupted_total = sum(1 for r in all_records if r["interrupted"])

    summary_path = os.path.join(out_dir, "summary.txt")
    with open(summary_path, "w") as f:
        f.write(f"mode              : {mode}\n")
        f.write(f"model             : {model_path}\n")
        f.write(f"parquet           : {parquet_path}\n")
        f.write(f"n_problems        : {n_problems}\n")
        f.write(f"n_rollouts        : {n_rollouts}\n")
        f.write(f"total             : {total_rollouts}\n")
        f.write(f"chunk_size        : {chunk}  (n_chunks={n_chunks})\n")
        f.write(f"temperature       : {temperature}  top_p={top_p}  top_k={top_k}  "
                f"presence_penalty={presence_penalty}  enable_thinking={enable_thinking}\n")
        if interrupt_enabled:
            f.write(f"thinking_budget   : {thinking_budget}\n")
            f.write(f"tool_call_budget  : {tool_call_budget}\n")
            f.write(f"answer_budget     : {answer_budget}\n")
            f.write(f"interrupt_len     : {interrupt_len}\n")
            f.write(f"response_budget   : {response_budget}\n")
            if is_cot:
                f.write(f"cot_post_interrupt: {cot_post_interrupt_budget}\n")
            f.write(f"interrupted       : {n_interrupted_total}/{total_rollouts}\n")
        else:
            f.write(f"max_tokens        : {max_tokens}\n")
        f.write("\n")
        f.write("\n".join(all_summary_lines) + "\n")

    parquet_path_out = os.path.join(out_dir, "rollouts.parquet")
    pd.DataFrame(all_records).to_parquet(parquet_path_out, index=False)
    print(f"\nSaved {total_rollouts} rollouts + summary → {out_dir}/")
    print(f"Parquet: {parquet_path_out}")


def main() -> None:
    p = argparse.ArgumentParser(description="Generate TIR or CoT rollouts via vLLM")
    p.add_argument("--mode", choices=("tir", "cot"), default="tir",
                   help="tir: 2-phase tool-integrated rollout (VeRL ToolAgentLoop parity). "
                        "cot: single-turn CoT with think-interrupt (VeRL SingleTurnAgentLoop parity).")
    p.add_argument("--model", default="Qwen/Qwen3-0.6B")
    p.add_argument("--parquet", default="data/processed/corpus_train.parquet")
    p.add_argument("--n", type=int, default=4,
                   help="Number of problems to sample; n<=0 uses all rows in parquet order")
    p.add_argument("--n_rollouts", type=int, default=1, help="Rollouts per problem")
    p.add_argument("--out_dir", default="outputs/rollouts")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--gpu_mem", type=float, default=0.38)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top_p", type=float, default=1.0)
    p.add_argument("--top_k", type=int, default=-1)
    p.add_argument("--presence_penalty", type=float, default=0.0)
    p.add_argument("--repetition_penalty", type=float, default=1.0,
                   help="1.0 = no penalty (Qwen3.5 thinking-mode default). vLLM's internal "
                        "default is also 1.0; we surface the flag so it's visible in configs.")
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
    p.add_argument("--max_tool_response_len", type=int, default=1024,
                   help="Tool response text cap in chars (matches VeRL)")
    p.add_argument("--dump_txt", action="store_true", default=False,
                   help="Also write per-rollout human-readable txt files (slow; parquet is always written)")
    p.add_argument("--start_idx", type=int, default=None,
                   help="Shard slice start (applied after sampling). Use with --end_idx to split across GPUs.")
    p.add_argument("--end_idx", type=int, default=None,
                   help="Shard slice end (exclusive). Use with --start_idx to split across GPUs.")
    p.add_argument("--chunk_size", type=int, default=None,
                   help="Problems per chunk; parquet is flushed after each chunk for crash safety. "
                        "Resume works: re-running skips chunks whose parquet already exists.")
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
        top_k=args.top_k,
        presence_penalty=args.presence_penalty,
        repetition_penalty=args.repetition_penalty,
        enable_thinking=args.enable_thinking,
        max_tokens=args.max_tokens,
        thinking_budget=args.thinking_budget,
        tool_call_budget=args.tool_call_budget,
        answer_budget=args.answer_budget,
        max_prompt_len=args.max_prompt_len,
        max_tool_response_len=args.max_tool_response_len,
        dump_txt=args.dump_txt,
        start_idx=args.start_idx,
        end_idx=args.end_idx,
        chunk_size=args.chunk_size,
        mode=args.mode,
    )


if __name__ == "__main__":
    main()
