"""Top-level eval orchestrator: (model, benchmark, mode) → scored.parquet.

Three stages, each idempotent:

    1. LOAD    benchmark_loader(out_path) -> data/processed/eval/<benchmark>.parquet
    2. ROLLOUT eval/inference/rollout.run_dump(...)  -> <out_dir>/rollouts.parquet
    3. SCORE   <benchmark>_scorer(rollouts.parquet) -> <out_dir>/scored.parquet
                                                       <out_dir>/scored.summary.txt

Benchmarks
----------
The BENCHMARKS dict below is the single source of truth for benchmark name →
(loader callable, scorer name, loader_kwargs). To add a new benchmark, add a
loader under eval/benchmarks/, a scorer under eval/scoring/ (or reuse
our_verifier / olympiad_official), and a row here.

Skip-stage flags
----------------
--skip_load     reuse existing benchmark parquet at its canonical path
--skip_rollout  reuse existing rollouts.parquet under <out_dir>
--skip_score    don't run the scorer (just produce rollouts)

Default out_dir is `outputs/eval/<benchmark>/<mode>__<model_tag>__<timestamp>`.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval.benchmarks import abench, olympiad_bench, phybench, pool_v2  # noqa: E402

# data_source filter alias → kwargs for pool_v2.load
_POOL_V2_SOURCES = ("drsci", "ugphysics", "physics", "scibench")
# Canonical local mirror of xingzhi0/phys-tir test split (1,064 rows). Fetch via
# scripts/fetch_dataset.py if missing. We use a `_v2` suffix on the directory
# while the qwen3-thinking branch still ships the older 2,200-row split at
# data/processed_tir/data/test.parquet to keep both branches operable.
_POOL_V2_TEST_PARQUET = "data/processed_tir_v2/data/test.parquet"

BENCHMARKS: dict[str, dict] = {
    # In-distribution: pool_v2 test split filtered by source
    **{
        f"pool_v2_{s}": {
            "loader": pool_v2.load,
            "loader_kwargs": {"source": s, "src_parquet": _POOL_V2_TEST_PARQUET},
            "scorer": "our_verifier",
            "default_parquet": f"data/processed/eval/pool_v2_{s}.parquet",
        }
        for s in _POOL_V2_SOURCES
    },
    # External
    "olympiad_oe_to_physics": {
        "loader": olympiad_bench.load,
        "loader_kwargs": {"config": "OE_TO_physics_en_COMP"},
        "scorer": "olympiad_official",
        "default_parquet": "data/processed/eval/olympiad_oe_to_physics.parquet",
    },
    "phybench": {
        "loader": phybench.load,
        "loader_kwargs": {},
        "scorer": "phybench_eed",
        "default_parquet": "data/processed/eval/phybench.parquet",
    },
    "abench_phy_a": {
        "loader": abench.load,
        "loader_kwargs": {"variant": "phy_a"},
        "scorer": "abench_official",
        "default_parquet": "data/processed/eval/abench_phy_a.parquet",
    },
    "abench_phy_b": {
        "loader": abench.load,
        "loader_kwargs": {"variant": "phy_b"},
        "scorer": "abench_official",
        "default_parquet": "data/processed/eval/abench_phy_b.parquet",
    },
}

SCORERS = {"our_verifier", "olympiad_official", "phybench_eed", "abench_official"}


def _slugify(s: str) -> str:
    # Make a model id like 'Qwen/Qwen3.5-4B' filesystem-friendly
    return re.sub(r"[^A-Za-z0-9._-]+", "_", s).strip("_")


def _resolve_out_dir(benchmark: str, mode: str, model: str, override: str | None) -> Path:
    if override:
        return Path(override)
    ts = _dt.datetime.now().strftime("%Y%m%d.%H%M%S")
    tag = f"{mode}__{_slugify(model)}__{ts}"
    return ROOT / "outputs" / "eval" / benchmark / tag


def _stage_load(spec: dict, force: bool) -> str:
    parquet = spec["default_parquet"]
    Path(parquet).parent.mkdir(parents=True, exist_ok=True)
    return spec["loader"](out_path=parquet, force=force, **spec["loader_kwargs"])


def _stage_rollout(args, parquet: str, out_dir: Path) -> str:
    from eval.inference.rollout import run_dump  # local import: vLLM is heavy

    out_dir.mkdir(parents=True, exist_ok=True)
    rollouts_path = out_dir / "rollouts.parquet"
    if rollouts_path.exists() and not args.force_rollout:
        print(f"[rollout] reuse {rollouts_path}")
        return str(rollouts_path)

    run_dump(
        model_path=args.model,
        parquet_path=parquet,
        n=args.n,
        n_rollouts=args.n_rollouts,
        out_dir=str(out_dir),
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
    if not rollouts_path.exists():
        raise RuntimeError(f"rollout finished but {rollouts_path} not written")
    return str(rollouts_path)


def _stage_score(scorer: str, rollouts_path: str, out_dir: Path, args) -> str:
    """Run scoring as a subprocess so the previous stage's vLLM engine releases
    its GPU memory before the next CUDA consumer (e.g. xVerify-7B) loads. The
    in-process path leaks vLLM's CUDA allocator into the scorer's transformers
    load and OOMs on consumer-grade GPUs.
    """
    import subprocess  # noqa: PLC0415

    scored_path = out_dir / "scored.parquet"
    cmd: list[str] = [sys.executable]
    if scorer == "our_verifier":
        cmd += [
            "-m", "eval.scoring.our_verifier",
            "--rollouts", rollouts_path,
            "--out", str(scored_path),
            "--xverify_model", args.xverify_model,
            "--xverify_device", args.xverify_device,
        ]
        if args.no_xverify:
            cmd.append("--no_xverify")
    elif scorer in ("olympiad_official", "phybench_eed", "abench_official"):
        cmd += [
            "-m", f"eval.scoring.{scorer}",
            "--rollouts", rollouts_path,
            "--out", str(scored_path),
        ]
    else:
        raise ValueError(f"unknown scorer {scorer!r} (known: {sorted(SCORERS)})")

    print(f"[score] subprocess: {' '.join(cmd)}")
    env = os.environ.copy()
    # Make sure the subprocess sees the project root on PYTHONPATH so
    # `python -m eval.scoring.<name>` resolves; also keep eval/_pkgs visible.
    extra_paths = [str(ROOT), str(ROOT / "eval" / "_pkgs")]
    cur = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = ":".join([p for p in extra_paths if p] + ([cur] if cur else []))
    rc = subprocess.run(cmd, env=env, cwd=str(ROOT)).returncode
    if rc != 0:
        raise SystemExit(f"scorer subprocess exited {rc}")
    return str(scored_path)


def main() -> None:
    p = argparse.ArgumentParser(description="Run a single (model × benchmark × mode) eval.")
    p.add_argument("--benchmark", required=True, choices=sorted(BENCHMARKS.keys()))
    p.add_argument("--mode", choices=["tir", "cot"], default="tir")
    p.add_argument("--model", required=True, help="HF model id or local path")
    p.add_argument("--out_dir", default=None,
                   help="Override output directory (default: outputs/eval/<bench>/<mode>__<model>__<ts>)")
    # Sampling — defaults match training-time configuration
    p.add_argument("--n", type=int, default=-1, help="Sample size (<=0 → use all rows)")
    p.add_argument("--n_rollouts", type=int, default=1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--gpu_mem", type=float, default=0.7)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top_p", type=float, default=1.0)
    p.add_argument("--top_k", type=int, default=-1)
    p.add_argument("--presence_penalty", type=float, default=0.0)
    p.add_argument("--repetition_penalty", type=float, default=1.0)
    p.add_argument("--enable_thinking", action="store_true", default=True)
    p.add_argument("--no_enable_thinking", dest="enable_thinking", action="store_false")
    p.add_argument("--max_tokens", type=int, default=8192)
    p.add_argument("--thinking_budget", type=int, default=None)
    p.add_argument("--tool_call_budget", type=int, default=None)
    p.add_argument("--answer_budget", type=int, default=None)
    p.add_argument("--max_prompt_len", type=int, default=1024)
    p.add_argument("--max_tool_response_len", type=int, default=1024)
    p.add_argument("--dump_txt", action="store_true")
    p.add_argument("--start_idx", type=int, default=None)
    p.add_argument("--end_idx", type=int, default=None)
    p.add_argument("--chunk_size", type=int, default=None)
    # Scoring — only consumed by our_verifier
    p.add_argument("--xverify_model", default="IAAR-Shanghai/xVerify-7B-I")
    p.add_argument("--xverify_device", default="cuda")
    p.add_argument("--no_xverify", action="store_true")
    # Stage skip / force
    p.add_argument("--skip_load", action="store_true")
    p.add_argument("--skip_rollout", action="store_true")
    p.add_argument("--skip_score", action="store_true")
    p.add_argument("--force_load", action="store_true")
    p.add_argument("--force_rollout", action="store_true")
    args = p.parse_args()

    spec = BENCHMARKS[args.benchmark]
    out_dir = _resolve_out_dir(args.benchmark, args.mode, args.model, args.out_dir)
    print(f"=== run_eval: benchmark={args.benchmark} mode={args.mode} model={args.model}")
    print(f"    out_dir={out_dir}")

    # Stage 1: load
    parquet = spec["default_parquet"]
    if args.skip_load:
        if not Path(parquet).exists():
            raise SystemExit(f"--skip_load given but {parquet} does not exist")
        print(f"[load] skip — reuse {parquet}")
    else:
        parquet = _stage_load(spec, force=args.force_load)

    # Stage 2: rollout
    rollouts_path = None
    if args.skip_rollout:
        if args.skip_score:
            # Load-only mode: parquet was the goal, nothing more to do.
            print("[rollout] skip (score also skipped — load-only mode)")
            return
        rollouts_path = str(out_dir / "rollouts.parquet")
        if not Path(rollouts_path).exists():
            raise SystemExit(f"--skip_rollout given but {rollouts_path} does not exist")
        print(f"[rollout] skip — reuse {rollouts_path}")
    else:
        rollouts_path = _stage_rollout(args, parquet, out_dir)

    # Stage 3: score
    if args.skip_score:
        print("[score] skip")
        return
    scored = _stage_score(spec["scorer"], rollouts_path, out_dir, args)
    print(f"\nDone. scored → {scored}")


if __name__ == "__main__":
    main()
