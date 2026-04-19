# Eval harness

Top-level entrypoint: `eval/run_eval.py` (or the apptainer wrapper
`eval/run_eval.sh`). Runs three idempotent stages — load → rollout → score —
for a single `(model × benchmark × mode)` cell.

## Layout

```
eval/
  run_eval.py            # orchestrator (load → rollout → score)
  run_eval.sh            # apptainer wrapper around run_eval.py
  benchmarks/            # one loader per benchmark; output schema = rollout.py input
    pool_v2.py           # in-distribution: slice pool_v2 test by data_source
    olympiad_bench.py    # external: HF lscpku/olympiad_bench-official, OE_TO physics EN
  inference/
    rollout.py           # TIR + CoT rollout dumper (mirrors VeRL agent loops)
    rollout.{sh,sbatch}
  scoring/
    our_verifier.py      # rule + xVerify-7B (used for in-dist + reward parity)
    olympiad_official.py # vendored OpenBMB AutoScoringJudge wrapper
    _olympiad_judge.py   # verbatim AutoScoringJudge from OpenBMB/OlympiadBench
    diff_tir_vs_cot.py   # per-case disagreement dump (existing)
    diff_summary.py      # compact triage view (existing)
    test_xverify_parity.py
  _pkgs/                 # gitignored; antlr4-python3-runtime==4.11 for sympy LaTeX
                         # (run pip install --no-deps --target eval/_pkgs antlr4-...)
```

## Benchmarks registered

| Name | Loader | Scorer | Notes |
|---|---|---|---|
| `pool_v2_drsci` | `pool_v2.load(source="drsci")` | `our_verifier` | 503 rows (post-pool-v2) |
| `pool_v2_ugphysics` | `pool_v2.load(source="ugphysics")` | `our_verifier` | 217 rows |
| `pool_v2_physics` | `pool_v2.load(source="physics")` | `our_verifier` | 191 rows |
| `pool_v2_scibench` | `pool_v2.load(source="scibench")` | `our_verifier` | 153 rows |
| `olympiad_oe_to_physics` | `olympiad_bench.load(config=OE_TO_physics_en_COMP)` | `olympiad_official` | 236 rows, text-only |

To add: drop a loader in `benchmarks/` (must produce parquet with columns
`data_source / prompt / reward_model / extra_info`), drop a scorer in `scoring/`
(must take `rollouts.parquet → scored.parquet`), then add a row to the
`BENCHMARKS` dict in `run_eval.py`.

## Running

```bash
# In-distribution: pool_v2 drsci slice with TIR mode
BENCHMARK=pool_v2_drsci MODEL=Qwen/Qwen3.5-4B \
  THINKING_BUDGET=12288 TOOL_CALL_BUDGET=2048 ANSWER_BUDGET=4096 \
  bash eval/run_eval.sh

# CoT baseline on the same slice (use the matched COT_SYSTEM_PROMPT)
BENCHMARK=pool_v2_drsci MODE=cot MODEL=Qwen/Qwen3.5-4B \
  THINKING_BUDGET=12288 TOOL_CALL_BUDGET=2048 ANSWER_BUDGET=4096 \
  bash eval/run_eval.sh

# External: OlympiadBench OE_TO physics, official scorer
BENCHMARK=olympiad_oe_to_physics MODEL=Qwen/Qwen3.5-4B \
  THINKING_BUDGET=12288 TOOL_CALL_BUDGET=2048 ANSWER_BUDGET=4096 \
  bash eval/run_eval.sh
```

Outputs land at
`outputs/eval/<benchmark>/<mode>__<model_slug>__<timestamp>/{rollouts,scored}.parquet`
with a `scored.summary.txt` next to `scored.parquet`.

## Stage skip / resume

The orchestrator checks for existing outputs and reuses them by default — handy
for re-scoring without re-rolling out. Use `--force_load` / `--force_rollout`
to override, or `--skip_*` to require an existing artifact.

## Vendored OlympiadBench judge

`eval/scoring/_olympiad_judge.py` is a verbatim copy of
[OpenBMB/OlympiadBench/eval/auto_scoring_judge.py](https://github.com/OpenBMB/OlympiadBench)
(main branch as of 2026-04-19). Do not edit. It depends on sympy's LaTeX parser
which requires `antlr4-python3-runtime>=4.11`; the SIF ships 4.9.3, so we keep
4.11 under `eval/_pkgs/` and prepend it to `sys.path` from `olympiad_official.py`
(and to `PYTHONPATH` in `run_eval.sh`).
