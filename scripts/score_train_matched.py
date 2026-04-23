#!/usr/bin/env python3
"""Score rollouts with the EXACT control flow VeRL training uses.

Difference vs ``score_and_filter_vllm.py``:

    score_and_filter_vllm.py's xVerify branch passes the full reconstructed
    solution as ``pred`` and the raw ``gold_answer`` column (no unit) as
    ``gold``. Training (via ``phys_reasoner.training.reward.compute_score``)
    instead calls ``verify_answer(..., xverify_judge=<client>)`` which, inside
    ``_compare_single``, queries xVerify with ``pred = boxed extraction`` and
    ``gold = gold_part + " " + gold_unit``, and can call xVerify multiple times
    per rollout (once per gold_part/variant, with permutation matching via
    ``_match_parts``). The resulting scores can diverge substantially.

    This script collects every xVerify call ``verify_answer`` would make in
    training, batches them through vLLM once, and replays the answers back into
    a second ``verify_answer`` invocation. The final score is therefore
    bit-identical (up to vLLM-vs-transformers greedy-decode margin) to what
    DAPO training's reward manager would record for each rollout.

Flow::

    Pass 1 (collect): for each rollout, run verify_answer with a Recorder
    xverify_judge that appends every (pred, gold, problem) triple it receives
    and returns False (so control flow walks every variant). Subprocess-isolated
    with SIGKILL-after-timeout to survive pathological sympy hangs.

    Batch: deduplicate collected queries, run all through vLLM xVerify.

    Pass 2 (replay): for each rollout, run verify_answer with a Replayer
    xverify_judge that looks up the batch result for each (pred, gold, problem).
    Short-circuits naturally on first True. compute_score clips -1.0 -> 0.0.

Output: ``{out_dir}/rollouts_scored_trainmatched.parquet`` with a ``correct``
column (0 or 1 per rollout), distinct from the existing
``rollouts_scored.parquet`` so both remain on disk for comparison.
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import pickle
import queue as _queue
import time
from pathlib import Path

import pandas as pd


# ---------------------------------------------------------------------------
# Worker — runs in a forked subprocess so pathological sympy hangs can be
# killed without taking down the parent.
# ---------------------------------------------------------------------------

def _worker_loop(in_q: mp.Queue, out_q: mp.Queue, answers_dict: dict | None, mode: str):
    """Long-lived worker. Consumes (idx, kwargs) messages; emits (idx, score, queries)."""
    from phys_reasoner.verifier.router import verify_answer  # noqa: PLC0415

    class _Recorder:
        __slots__ = ("queries",)

        def __init__(self):
            self.queries: list[tuple[str, str, str]] = []

        def __call__(self, pred_str, gold_str, problem_text):
            self.queries.append((pred_str, gold_str, problem_text))
            return False  # walk every variant during collection

    class _Replayer:
        __slots__ = ("answers",)

        def __init__(self, d):
            self.answers = d

        def __call__(self, pred_str, gold_str, problem_text):
            # Training's compute_score maps xverify False/None to 0.0 via the
            # same code path; missing lookups (shouldn't happen) default to False.
            return self.answers.get((pred_str, gold_str, problem_text), False)

    while True:
        msg = in_q.get()
        if msg is None:
            return
        idx, kwargs = msg
        if mode == "collect":
            rec = _Recorder()
            kwargs["xverify_judge"] = rec
            try:
                s = verify_answer(**kwargs)
            except Exception:
                s = -1.0
            out_q.put((idx, s, rec.queries))
        else:  # replay
            kwargs["xverify_judge"] = _Replayer(answers_dict)
            try:
                s = verify_answer(**kwargs)
            except Exception:
                s = -1.0
            out_q.put((idx, s, None))


def _row_kwargs(row) -> dict:
    """Build verify_answer kwargs from a merged-parquet row.

    ``pred_text`` mirrors training's ``solution_str`` — full reconstructed
    response (phase1 + phase1b-if-interrupted + phase2). Training decodes the
    full response token slice including think+tool+answer, which in our
    on-disk representation is exactly those three fields concatenated.
    """
    parts = []
    p1 = row.get("phase1_text")
    if p1:
        parts.append(str(p1))
    if row.get("interrupted"):
        p1b = row.get("phase1b_text")
        if p1b:
            parts.append(str(p1b))
    p2 = row.get("phase2_text")
    if p2:
        parts.append(str(p2))
    solution = "\n".join(parts)

    gold = row.get("gold_answer")
    if gold is None:
        gold = ""
    extra_info = row.get("extra_info")
    if not isinstance(extra_info, dict):
        extra_info = {}

    return dict(
        pred_text=solution,
        gold_answer=gold,
        answer_type=extra_info.get("answer_type", "numerical"),
        gold_unit=extra_info.get("unit", "") or "",
        tolerance=float(extra_info.get("tolerance", 0.05)),
        problem_text=extra_info.get("problem", "") or "",
    )


def _run_pass(
    df: pd.DataFrame,
    row_timeout: int,
    mode: str,
    answers_dict: dict | None,
    rule_scores: list[float] | None = None,
    tag: str = "pass",
) -> tuple[list[float], list[tuple[str, str, str]]]:
    """Run verify_answer over every row in df via the subprocess worker.

    Returns (scores, collected_queries). ``collected_queries`` is empty when
    mode='replay'. Rows where rule_scores[i] == 1.0 are short-circuited to 1.0
    (matches verify_answer + compute_score behavior) without invoking the worker.
    """
    ctx = mp.get_context("fork")
    in_q: mp.Queue = ctx.Queue()
    out_q: mp.Queue = ctx.Queue()
    worker = ctx.Process(target=_worker_loop, args=(in_q, out_q, answers_dict, mode))
    worker.start()

    n = len(df)
    scores: list[float] = [0.0] * n
    queries: list[tuple[str, str, str]] = []
    n_timeouts = 0
    n_exceptions = 0
    n_skipped = 0
    t0 = time.time()

    rows = df.to_dict("records")
    for i in range(n):
        if rule_scores is not None and rule_scores[i] == 1.0:
            scores[i] = 1.0
            n_skipped += 1
            continue
        kwargs = _row_kwargs(rows[i])
        in_q.put((i, kwargs))
        try:
            idx, s, q = out_q.get(timeout=row_timeout)
            assert idx == i
            scores[i] = s
            if q is not None:
                queries.extend(q)
        except _queue.Empty:
            n_timeouts += 1
            scores[i] = -1.0
            try:
                worker.kill()
                worker.join(timeout=2)
            except Exception:
                pass
            in_q = ctx.Queue()
            out_q = ctx.Queue()
            worker = ctx.Process(target=_worker_loop, args=(in_q, out_q, answers_dict, mode))
            worker.start()

        if (i + 1) % 10000 == 0:
            r = (i + 1) / (time.time() - t0)
            print(f"[{tag}] {i+1}/{n}  {r:.0f} rows/s  queries_so_far={len(queries)} "
                  f"timeouts={n_timeouts} skipped={n_skipped}", flush=True)

    print(f"[{tag}] done in {time.time()-t0:.1f}s. "
          f"scores=[{sum(1 for s in scores if s==1.0)} true, "
          f"{sum(1 for s in scores if s==0.0)} false, "
          f"{sum(1 for s in scores if s==-1.0)} undecided], "
          f"queries={len(queries)}, timeouts={n_timeouts}, skipped={n_skipped}")

    in_q.put(None)
    worker.join(timeout=2)
    if worker.is_alive():
        worker.kill()
        worker.join(timeout=1)
    return scores, queries


# ---------------------------------------------------------------------------
# vLLM batch xverify — identical prompt template to XVerifyJudge._build_prompt
# and to the HTTP serve_xverify server.
# ---------------------------------------------------------------------------

def _vllm_xverify_batch(
    prompts: list[tuple[str, str, str]],
    model_path: str,
    gpu_mem: float,
    max_model_len: int,
) -> list[bool]:
    from vllm import LLM, SamplingParams

    from phys_reasoner.verifier.xverify_judge import _XVERIFY_PROMPT
    from transformers import AutoTokenizer

    prompt_texts = [
        _XVERIFY_PROMPT.format(
            problem=(problem or "(not provided)"),
            pred=pred,
            gold=gold,
        )
        for (pred, gold, problem) in prompts
    ]

    # Left-truncate any prompt longer than max_model_len budget.
    tok = AutoTokenizer.from_pretrained(model_path)
    budget = max_model_len - 16 - 10
    n_truncated = 0
    for i, t in enumerate(prompt_texts):
        ids = tok.encode(t, add_special_tokens=False)
        if len(ids) > budget:
            n_truncated += 1
            prompt_texts[i] = tok.decode(ids[-budget:], skip_special_tokens=True)
    if n_truncated:
        print(f"[vllm] left-truncated {n_truncated}/{len(prompt_texts)} overlong prompts")

    print(f"[vllm] loading {model_path}")
    t0 = time.time()
    llm = LLM(
        model=model_path,
        dtype="auto",
        gpu_memory_utilization=gpu_mem,
        max_model_len=max_model_len,
        enforce_eager=True,
        enable_prefix_caching=True,
    )
    print(f"[vllm] loaded in {time.time()-t0:.1f}s")

    sp = SamplingParams(temperature=0.0, max_tokens=10, top_p=1.0)
    print(f"[vllm] generating {len(prompt_texts)} judgments (temperature=0)")
    t0 = time.time()
    outs = llm.generate(prompt_texts, sp)
    print(f"[vllm] generated in {time.time()-t0:.1f}s")

    results = [o.outputs[0].text.strip().lower().startswith("correct") for o in outs]
    return results


# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--merged", required=True,
                    help="Merged rollouts parquet (one row per rollout).")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--out_name", default="rollouts_scored_trainmatched.parquet",
                    help="Final scored parquet filename (written inside --out_dir).")
    ap.add_argument("--rule_cache", default=None,
                    help="Path to _rule_pass_cache.pkl from score_and_filter_vllm.py; "
                         "used to short-circuit rule=1.0 rows. Optional but saves time.")
    ap.add_argument("--xverify_model", default="IAAR-Shanghai/xVerify-7B-I")
    ap.add_argument("--gpu_mem", type=float, default=0.85)
    ap.add_argument("--max_model_len", type=int, default=16384)
    ap.add_argument("--row_timeout", type=int, default=5,
                    help="Per-row subprocess wall-time (seconds) before kill/respawn.")
    ap.add_argument("--start_idx", type=int, default=None,
                    help="Process only rows [start_idx, end_idx). For parallel chunking.")
    ap.add_argument("--end_idx", type=int, default=None)
    ap.add_argument("--collect_cache", default=None,
                    help="Optional pickle to cache Pass-1 output for resumability.")
    ap.add_argument("--xv_cache", default=None,
                    help="Optional pickle to cache vLLM xverify results for resumability.")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    out_path = out_dir / args.out_name
    if out_path.exists():
        print(f"ERROR: {out_path} already exists — move/delete before rerunning", flush=True)
        return 1

    # Default cache paths alongside the output; use out_name stem so multiple
    # runs in the same out_dir don't collide.
    stem = Path(args.out_name).stem
    collect_cache = Path(args.collect_cache) if args.collect_cache else (out_dir / f"_{stem}_collect.pkl")
    xv_cache = Path(args.xv_cache) if args.xv_cache else (out_dir / f"_{stem}_xv.pkl")

    print(f"[load] reading {args.merged}")
    df = pd.read_parquet(args.merged)
    if args.start_idx is not None or args.end_idx is not None:
        s = args.start_idx if args.start_idx is not None else 0
        e = args.end_idx if args.end_idx is not None else len(df)
        s = max(0, s); e = min(len(df), e)
        df = df.iloc[s:e].reset_index(drop=True)
        print(f"[load] chunk slice [{s}:{e}) = {len(df)} rollouts")
    print(f"[load] {len(df)} rollouts")

    rule_scores = None
    if args.rule_cache and os.path.exists(args.rule_cache):
        print(f"[load] rule cache from {args.rule_cache}")
        with open(args.rule_cache, "rb") as f:
            rule_scores, _, _ = pickle.load(f)
        if len(rule_scores) != len(df):
            print(f"[load] rule cache length {len(rule_scores)} != df {len(df)} — ignoring")
            rule_scores = None

    # ---- Pass 1: collect ----
    if collect_cache.exists():
        print(f"[collect] loading cached Pass 1 output from {collect_cache}")
        with open(collect_cache, "rb") as f:
            pass1_scores, all_queries = pickle.load(f)
    else:
        print("[collect] running Pass 1 (verify_answer + Recorder, per-row subprocess)")
        pass1_scores, all_queries = _run_pass(
            df, args.row_timeout, mode="collect",
            answers_dict=None, rule_scores=rule_scores, tag="collect"
        )
        with open(collect_cache, "wb") as f:
            pickle.dump((pass1_scores, all_queries), f)
        print(f"[collect] cached Pass 1 to {collect_cache}")

    # Deduplicate queries for batch.
    unique_queries = list({q: None for q in all_queries}.keys())
    print(f"[collect] queries: {len(all_queries)} total, {len(unique_queries)} unique")

    # ---- Batch xverify via vLLM ----
    if xv_cache.exists():
        print(f"[vllm] loading cached xverify answers from {xv_cache}")
        with open(xv_cache, "rb") as f:
            answers_dict = pickle.load(f)
        # Sanity: all our unique queries must have answers; add missing with False.
        missing = [q for q in unique_queries if q not in answers_dict]
        if missing:
            print(f"[vllm] WARNING: {len(missing)} queries missing from cache; defaulting to False")
            for q in missing:
                answers_dict[q] = False
    else:
        from huggingface_hub import snapshot_download  # noqa: PLC0415
        model_path = snapshot_download(args.xverify_model, local_files_only=True)
        if unique_queries:
            answers = _vllm_xverify_batch(
                unique_queries,
                model_path=model_path,
                gpu_mem=args.gpu_mem,
                max_model_len=args.max_model_len,
            )
            answers_dict = dict(zip(unique_queries, answers))
        else:
            answers_dict = {}
        with open(xv_cache, "wb") as f:
            pickle.dump(answers_dict, f)
        print(f"[vllm] cached answers to {xv_cache}")

    # ---- Pass 2: replay ----
    print("[replay] running Pass 2 (verify_answer + Replayer, per-row subprocess)")
    pass2_scores, _ = _run_pass(
        df, args.row_timeout, mode="replay",
        answers_dict=answers_dict, rule_scores=rule_scores, tag="replay"
    )

    # compute_score clipping: -1.0 -> 0 (unverifiable treated as wrong)
    final = [1 if s == 1.0 else 0 for s in pass2_scores]
    n_correct = sum(final)
    print(f"[score] {n_correct}/{len(final)} = {100*n_correct/len(final):.2f}% correct")

    df = df.copy()
    df["correct"] = final
    df.to_parquet(out_path, index=False)
    print(f"[save] {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
