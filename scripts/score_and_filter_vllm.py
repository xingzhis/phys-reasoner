"""Score merged rollouts and build filtered parquets — vLLM-batched xVerify path.

Replaces the HTTP-server path with a single-process vLLM batch inference so that
scoring finishes in ~10-15 min instead of ~7 hr.

Correctness guarantee:
  - xVerify uses greedy decoding (temperature=0, do_sample=False). vLLM with
    temperature=0 and max_tokens=10 yields the same argmax sequence as
    transformers.generate modulo tiny kernel-precision flips at the margin.
    The decoded response is classified via response.lower().startswith("correct").
    So results match 1:1 with the HTTP path on virtually every row.
  - Rule-based routing (MCQ/TF exact match; numerical unit-check + sympy; etc.)
    is untouched — we call verify_answer(xverify_judge=None) first to decide
    which rows need xverify. This is identical to training behavior for
    rule-decidable rows.

Usage:
  apptainer exec ... python3 scripts/score_and_filter_vllm.py \\
      --merged outputs/probe_qwen3_4b/rollouts_merged.parquet \\
      --out_dir outputs/probe_qwen3_4b \\
      --xverify_model IAAR-Shanghai/xVerify-7B-I \\
      --gpu_mem 0.85
"""
from __future__ import annotations

import argparse
import os
import shutil
import time
from pathlib import Path

import pandas as pd


def reconstruct_solution(row) -> str:
    parts = []
    p1 = row.get("phase1_text")
    if p1:
        parts.append(str(p1))
    if row.get("interrupted", False):
        p1b = row.get("phase1b_text")
        if p1b:
            parts.append(str(p1b))
    p2 = row.get("phase2_text")
    if p2:
        parts.append(str(p2))
    return "\n".join(parts)


def _rule_worker_loop(in_q, out_q):
    """Long-lived worker: reads (idx, kwargs) from in_q, puts (idx, score) on out_q.

    Sympy/math_verify execute inside C extensions that ignore Python signal
    handlers (SIGALRM is only checked at Python bytecode boundaries), so a hung
    row cannot be interrupted in-process. We instead run verify_answer in a
    subprocess the parent can kill; that's the only reliable preemption.
    """
    from phys_reasoner.verifier.router import verify_answer  # noqa: PLC0415

    while True:
        msg = in_q.get()
        if msg is None:
            return
        idx, kwargs = msg
        try:
            s = verify_answer(**kwargs)
        except Exception:
            s = -1.0
        out_q.put((idx, s))


def rule_only_pass(df: pd.DataFrame, rule_timeout: int = 0) -> tuple[list[float], list[int], list[tuple[str, str, str]]]:
    """First pass: run rule-based verify_answer for every row.

    Returns:
        rule_scores: list len N with 1.0 / 0.0 / -1.0 per row (matches router semantics)
        xv_idx:      indices into df that returned -1.0 (need xverify)
        xv_inputs:   list of (pred_str, gold_str, problem_str) triples for each xv_idx

    rule_timeout > 0 runs each row in a long-lived worker subprocess with a
    per-row wall-clock deadline. On timeout, the worker is killed + respawned
    and the row is treated as rule-undecidable (score = -1.0) and forwarded to
    xVerify — the same branch training takes on rule=None (see
    router._compare_single: "xVerify fallback called for rule=False or rule=None").
    SIGALRM was tried first but does not interrupt sympy's C extensions.
    """
    from phys_reasoner.verifier.router import verify_answer  # noqa: F401  (preload for worker fork)

    n = len(df)
    rule_scores: list[float] = [0.0] * n
    xv_idx: list[int] = []
    xv_inputs: list[tuple[str, str, str]] = []

    use_subprocess = bool(rule_timeout and rule_timeout > 0)

    if use_subprocess:
        import multiprocessing as _mp
        import queue as _queue

        # fork context: worker inherits already-imported sympy/math_verify from
        # parent, avoiding ~1s of re-import per respawn.
        _ctx = _mp.get_context("fork")
        _in_q: _mp.Queue = _ctx.Queue()
        _out_q: _mp.Queue = _ctx.Queue()
        _worker = _ctx.Process(target=_rule_worker_loop, args=(_in_q, _out_q))
        _worker.start()
        print(f"[rule] subprocess-timeout enabled: {rule_timeout}s per row → "
              f"xverify fallback on timeout (matches training's rule=None branch)")

    def _call_with_timeout(kwargs: dict):
        """Returns (score, status) where status in {"ok", "timeout", "exception"}."""
        nonlocal _worker, _in_q, _out_q
        _in_q.put((0, kwargs))
        try:
            _idx, score = _out_q.get(timeout=rule_timeout)
            return score, "ok"
        except _queue.Empty:
            # worker is stuck in C code; kill and respawn
            try:
                _worker.kill()
                _worker.join(timeout=2)
            except Exception:
                pass
            # Fresh queues — old ones may contain a stale result that the
            # killed worker was mid-putting.
            _in_q = _ctx.Queue()
            _out_q = _ctx.Queue()
            _worker = _ctx.Process(target=_rule_worker_loop, args=(_in_q, _out_q))
            _worker.start()
            return -1.0, "timeout"

    n_timeouts = 0
    n_exceptions = 0
    t0 = time.time()
    for i, (_, row) in enumerate(df.iterrows()):
        solution = reconstruct_solution(row)
        gold = row.get("gold_answer", "")
        extra_info = row.get("extra_info")
        if not isinstance(extra_info, dict):
            extra_info = {}
        answer_type = extra_info.get("answer_type", "numerical")
        unit = extra_info.get("unit", "")
        tolerance = float(extra_info.get("tolerance", 0.05))
        problem = extra_info.get("problem", "")

        kwargs = dict(
            pred_text=solution,
            gold_answer=gold if gold is not None else "",
            answer_type=answer_type,
            gold_unit=unit or "",
            tolerance=tolerance,
            xverify_judge=None,  # rule-only pass
            problem_text=problem or "",
        )
        if use_subprocess:
            s, status = _call_with_timeout(kwargs)
            if status == "timeout":
                n_timeouts += 1
        else:
            try:
                s = verify_answer(**kwargs)
            except Exception:
                n_exceptions += 1
                s = -1.0
        rule_scores[i] = s
        if s == -1.0:
            # xverify will compare reconstructed solution (not just \boxed extract) —
            # matches how XVerifyJudge is called in router._compare_single:
            #   xverify_judge(pred_str, gold_for_xverify, problem_text)
            # gold_for_xverify is typically the unboxed gold; pred_str is from \boxed.
            # But since we're doing it at the top level (whole row), we use the full
            # reconstructed solution as pred and the raw gold string. The verifier
            # prompt includes the question so xVerify can compare semantically.
            gold_str = gold if isinstance(gold, str) else (gold[0] if isinstance(gold, list) and gold else "")
            xv_idx.append(i)
            xv_inputs.append((solution, str(gold_str), problem or ""))

        if (i + 1) % 10000 == 0:
            rate = (i + 1) / (time.time() - t0)
            print(f"[rule] {i+1}/{n}  {rate:.0f} rows/s  unverifiable_so_far={len(xv_idx)} "
                  f"timeouts={n_timeouts} exceptions={n_exceptions}",
                  flush=True)
    print(f"[rule] done in {time.time()-t0:.1f}s. rule-decided={n-len(xv_idx)} "
          f"need_xverify={len(xv_idx)} (of which timeouts={n_timeouts} exceptions={n_exceptions})")
    if use_subprocess:
        try:
            _in_q.put(None)
            _worker.join(timeout=2)
            if _worker.is_alive():
                _worker.kill()
                _worker.join(timeout=1)
        except Exception:
            pass
    return rule_scores, xv_idx, xv_inputs


def vllm_xverify_batch(
    prompts: list[tuple[str, str, str]],
    model_path: str,
    gpu_mem: float,
    max_model_len: int,
) -> list[bool]:
    """Run xVerify on a batch of (pred, gold, problem) triples using vLLM.

    Returns list[bool] of same length: True if judged correct.
    """
    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer
    from phys_reasoner.verifier.xverify_judge import _XVERIFY_PROMPT

    # Build prompts in the same format as XVerifyJudge._build_prompt
    prompt_texts = [
        _XVERIFY_PROMPT.format(
            problem=(problem or "(not provided)"),
            pred=pred,
            gold=gold,
        )
        for (pred, gold, problem) in prompts
    ]

    # Some physics rollouts produce very long reasoning traces; after
    # _XVERIFY_PROMPT formatting a handful of rows exceed max_model_len.
    # Truncate the LEFT of the overly long prompt text (i.e. drop the
    # reasoning front; keep the boxed answer at the end) so vLLM accepts it.
    # We reserve 16 tokens of headroom on top of the 10 `max_new_tokens` so the
    # vLLM length check passes comfortably.
    tok = AutoTokenizer.from_pretrained(model_path)
    budget = max_model_len - 16 - 10
    n_truncated = 0
    n_truncated_chars = 0
    for i, t in enumerate(prompt_texts):
        ids = tok.encode(t, add_special_tokens=False)
        if len(ids) > budget:
            n_truncated += 1
            before = len(t)
            trunc_ids = ids[-budget:]
            prompt_texts[i] = tok.decode(trunc_ids, skip_special_tokens=True)
            n_truncated_chars += before - len(prompt_texts[i])
    if n_truncated:
        print(f"[vllm] left-truncated {n_truncated}/{len(prompt_texts)} overlong prompts "
              f"(dropped {n_truncated_chars} chars total)")

    print(f"[vllm] loading {model_path}")
    t_load = time.time()
    llm = LLM(
        model=model_path,
        dtype="auto",
        gpu_memory_utilization=gpu_mem,
        max_model_len=max_model_len,
        enforce_eager=True,
        enable_prefix_caching=True,
    )
    print(f"[vllm] loaded in {time.time()-t_load:.1f}s")

    sp = SamplingParams(temperature=0.0, max_tokens=10, top_p=1.0)

    print(f"[vllm] generating {len(prompt_texts)} judgments...")
    t_gen = time.time()
    outs = llm.generate(prompt_texts, sp)
    print(f"[vllm] generated in {time.time()-t_gen:.1f}s ({len(prompt_texts)/(time.time()-t_gen):.0f} req/s)")

    results: list[bool] = []
    for o in outs:
        text = o.outputs[0].text.strip().lower()
        results.append(text.startswith("correct"))
    return results


def combine_scores(rule_scores: list[float], xv_idx: list[int], xv_results: list[bool]) -> list[int]:
    """Merge rule and xverify into final binary correct/not."""
    assert len(xv_idx) == len(xv_results)
    n = len(rule_scores)
    final = [0] * n
    for i, s in enumerate(rule_scores):
        if s == 1.0:
            final[i] = 1
        elif s == 0.0:
            final[i] = 0
        # -1.0 entries overwritten below
    for i, ok in zip(xv_idx, xv_results):
        final[i] = 1 if ok else 0
    return final


def filter_informative(df: pd.DataFrame, min_correct: int, max_correct: int) -> set[int]:
    agg = df.groupby("problem_idx")["correct"].agg(["sum", "count"]).reset_index()
    print(f"[filter] rollouts/prompt distribution: min={agg['count'].min()} max={agg['count'].max()} mode={agg['count'].mode().iloc[0]}")
    # Only keep problems where all 8 rollouts present — anything less means slot failure partial coverage
    full_n = agg["count"].mode().iloc[0]
    complete = agg[agg["count"] == full_n]
    print(f"[filter] {len(complete)}/{len(agg)} problems have full {full_n} rollouts")

    kept = complete[(complete["sum"] >= min_correct) & (complete["sum"] <= max_correct)]
    print(f"[filter] kept {len(kept)}/{len(complete)} complete problems in [{min_correct},{max_correct}] ({100*len(kept)/len(complete):.1f}%)")

    dist = complete["sum"].value_counts().sort_index()
    print(f"[filter] n_correct distribution on complete problems: {dict(dist)}")
    return set(kept["problem_idx"].astype(int).tolist())


def write_filtered(src: Path, dst: Path, keep: set[int]) -> None:
    df = pd.read_parquet(src)
    kept = df.iloc[sorted(keep)].reset_index(drop=True)
    dst.parent.mkdir(parents=True, exist_ok=True)
    kept.to_parquet(dst, index=False)
    print(f"[write] {src} ({len(df)}) → {dst} ({len(kept)})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--merged", default="outputs/probe_qwen3_4b/rollouts_merged.parquet")
    ap.add_argument("--out_dir", default="outputs/probe_qwen3_4b")
    ap.add_argument("--xverify_model", default="IAAR-Shanghai/xVerify-7B-I")
    ap.add_argument("--gpu_mem", type=float, default=0.85)
    ap.add_argument("--max_model_len", type=int, default=4096)
    ap.add_argument("--min_correct", type=int, default=1)
    ap.add_argument("--max_correct", type=int, default=7)
    ap.add_argument("--sample", type=int, default=-1, help="Score only first N rows for testing")
    ap.add_argument("--no_filter", action="store_true")
    ap.add_argument("--rule_timeout", type=int, default=0,
                    help="Per-row SIGALRM timeout (seconds) for rule pass. "
                         "0 disables (legacy). On timeout, row is treated as "
                         "rule-undecidable and forwarded to xVerify, matching "
                         "training's rule=None→xverify fallback branch.")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[load] reading {args.merged}")
    df = pd.read_parquet(args.merged)
    if args.sample > 0:
        df = df.head(args.sample).reset_index(drop=True)
    print(f"[load] {len(df)} rollouts")

    # Resolve model path from HF cache
    from huggingface_hub import snapshot_download
    model_path = snapshot_download(args.xverify_model, local_files_only=True)

    # Phase 1: rule-only pass — cache result to disk so vLLM crashes don't
    # force us to re-run the ~15-min CPU pass.
    import pickle
    rule_cache = out_dir / "_rule_pass_cache.pkl"
    if rule_cache.exists():
        print(f"[rule] loading cached result from {rule_cache}")
        with open(rule_cache, "rb") as f:
            rule_scores, xv_idx, xv_inputs = pickle.load(f)
        assert len(rule_scores) == len(df), (
            f"rule cache length {len(rule_scores)} != df length {len(df)}; "
            "delete the cache and re-run if the merged parquet changed"
        )
        print(f"[rule] loaded: rule-decided={len(df)-len(xv_idx)} need_xverify={len(xv_idx)}")
    else:
        rule_scores, xv_idx, xv_inputs = rule_only_pass(df, rule_timeout=args.rule_timeout)
        with open(rule_cache, "wb") as f:
            pickle.dump((rule_scores, xv_idx, xv_inputs), f)
        print(f"[rule] cached result to {rule_cache}")

    # Phase 2: vLLM batch xverify
    if xv_inputs:
        xv_results = vllm_xverify_batch(
            xv_inputs,
            model_path=model_path,
            gpu_mem=args.gpu_mem,
            max_model_len=args.max_model_len,
        )
    else:
        xv_results = []

    # Combine
    final = combine_scores(rule_scores, xv_idx, xv_results)
    df["correct"] = final
    n_correct = sum(final)
    print(f"[score] {n_correct}/{len(final)} = {100*n_correct/len(final):.2f}% correct")

    # Save scored parquet
    scored_path = out_dir / "rollouts_scored.parquet"
    df.to_parquet(scored_path, index=False)
    print(f"[save] {scored_path}")

    if args.no_filter or args.sample > 0:
        return

    # Filter
    keep = filter_informative(df, args.min_correct, args.max_correct)

    # Build TIR + CoT filtered parquets
    for mode in ["tir", "cot"]:
        src = Path(f"data/processed_{mode}/data/train.parquet")
        dst = Path(f"data/processed_{mode}_filtered/data/train.parquet")
        write_filtered(src, dst, keep)
        for split in ["validation", "test"]:
            sp = Path(f"data/processed_{mode}/data/{split}.parquet")
            dp = Path(f"data/processed_{mode}_filtered/data/{split}.parquet")
            if sp.exists():
                dp.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(sp, dp)
                print(f"[copy] {sp} → {dp}")

    print(f"\n=== DONE === kept {len(keep)} problems")
    print(f"  TIR: data/processed_tir_filtered/data/train.parquet")
    print(f"  CoT: data/processed_cot_filtered/data/train.parquet")


if __name__ == "__main__":
    main()
