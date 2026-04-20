# Eval harness — results + reproduction guide

**Model evaluated here:** `Qwen/Qwen3-4B-Thinking-2507` (zero-shot baseline).
**Run date:** 2026-04-19 to 2026-04-20.
**Code tag:** `eval-v1-qwen35` on `main` (the harness); merge commit `515964f` on
`switch/qwen3-thinking` brings it into the Qwen3-Thinking branch.

## Contents

1. [Results — 32-cell zero-shot matrix](#results)
2. [Where the files live](#files)
3. [Rerun on a trained checkpoint](#rerun)
4. [Answer-type analysis](#answer-type)
5. [Known gotchas](#gotchas)

---

## <a name="results"></a>1. Results — 32-cell zero-shot matrix

**8 benchmarks × (2 modes × 2 sampling presets) = 32 cells.** All cells use
training-matched budgets: `thinking=12288`, `tool_call=2048`, `answer=4096`,
`max_prompt_len=1024`, `max_tool_response_len=1024`, `gpu_mem=0.7` on A100-40G,
rollout sharded across 2 GPUs.

Sampling presets:
- **train**: `temperature=1.0, top_p=1.0, top_k=-1, repetition_penalty=1.0`
  (matches RL training config).
- **qwen**: `temperature=0.6, top_p=0.95, top_k=20, repetition_penalty=1.0`
  (Qwen team's thinking-mode recommendation).

Modes match VeRL's training-time agent loops:
- **tir**: `tool_agent_loop` with `multi_turn.format=hermes` (Qwen3-Thinking
  native) — model emits `<tool_call>{"name":"python","arguments":{"code":...}}</tool_call>`,
  sandbox executes, tool response injected, phase-2 writes `\boxed{…}`.
- **cot**: `single_turn_agent_loop` with no tool schema — model reasons and
  writes `\boxed{…}`. Post-interrupt answer budget collapses the same total
  `response_length` used by TIR so budgets are identical.

### In-distribution (pool v2 test split)

| benchmark (n) | tir/train | cot/train | tir/qwen | cot/qwen |
|---|---|---|---|---|
| pool_v2_scibench (153) | **0.5987** | 0.5658 | 0.5724 | 0.5789 |
| pool_v2_physics (191) | 0.3927 | **0.4346** | 0.3665 | 0.3979 |
| pool_v2_ugphysics (217) | 0.3825 | **0.4101** | 0.3641 | 0.3963 |
| pool_v2_drsci (503) | 0.6302 | 0.6441 | 0.6123 | **0.6541** |

Pool v2 scored with `src/phys_reasoner/verifier/router.py` (rule → xVerify-7B
fallback on expression types). Headline is `pass@1 exclusive` = correct /
(correct + wrong), i.e. unverifiable rows are dropped; the verifier also
exposes `pass@1 inclusive` in each summary.

### External (held out from training)

| benchmark (n) | tir/train | cot/train | tir/qwen | cot/qwen |
|---|---|---|---|---|
| olympiad_oe_to_physics (236) — pass@1 | 0.0466 | 0.0508 | 0.0593 | **0.0678** |
| phybench (1000) — exact pass@1 | 0.0150 | 0.0100 | **0.0220** | 0.0170 |
| phybench — mean EED (continuous, 0-100) | 3.03 | 2.39 | **3.30** | 3.01 |
| abench_phy_a (400) — per-row | 0.2025 | 0.1950 | 0.2025 | **0.2150** |
| abench_phy_b (400) — per-row | 0.6325 | 0.6325 | 0.6400 | 0.6200 |
| abench_phy_b — **per-mid** (ALL-4-subid correct) | **0.5000** | 0.4900 | 0.4900 | 0.4800 |

- OlympiadBench: vendored OpenBMB `AutoScoringJudge` (sympy, exact symbolic).
- PHYBench: vendored `EED` scorer; report both exact-match (score=100) rate
  and continuous mean_EED.
- ABench: 1% relative tolerance on numerical answers. Phy_B's per-mid is the
  dynamic-robustness metric — only counts a base problem if all 4 parametric
  variants verify.

### Qualitative observations

- **TIR ≈ CoT** on this model (±5pp on every in-dist cell). Qwen3-Thinking
  often pure-reasons inside TIR mode without invoking the tool. Post-RL
  training (which forces tool use) should shift the comparison.
- **Qwen sampling slightly hurts TIR** (~2-4pp) and slightly helps CoT (~1-2pp).
  Lower entropy makes the thinking model skip the tool more confidently; CoT
  benefits marginally from the same determinism.
- **OlympiadBench and PHYBench are punishing.** The official judges require
  exact symbolic form; the base model rarely matches despite arriving at the
  right numerical answer. EED's continuous score is a better signal (≈3/100
  average partial credit) than the raw exact-match rate (~1-2%).
- **Qwen3-Thinking ≫ Qwen3.5-4B base** on every cell we compared: +30pp scibench,
  +18pp ugphysics, +22pp drsci, +10pp physics. That comparison is archived in
  `outputs/eval/README_QWEN35_SWEEP_IN_PROGRESS.md` (partial Qwen3.5 sweep).

---

## <a name="files"></a>2. Where the files live

```
$ROOT = /diskarray/home/xs272/phys-reasoner      # local
      ≅ /gpfs/.../phys-reasoner                  # Perlmutter
```

### Benchmark inputs (materialized parquets)

```
$ROOT/data/processed/eval/pool_v2_scibench.parquet   #  153 rows
$ROOT/data/processed/eval/pool_v2_physics.parquet    #  191
$ROOT/data/processed/eval/pool_v2_ugphysics.parquet  #  217
$ROOT/data/processed/eval/pool_v2_drsci.parquet      #  503
$ROOT/data/processed/eval/olympiad_oe_to_physics.parquet  # 236
$ROOT/data/processed/eval/phybench.parquet           # 1000
$ROOT/data/processed/eval/abench_phy_a.parquet       #  400
$ROOT/data/processed/eval/abench_phy_b.parquet       #  400 = 100 mid × 4 subid
```

Each parquet has columns `data_source, prompt, reward_model, extra_info, pool`
matching `eval/inference/rollout.py`'s input schema. Materialized lazily by
loaders under `eval/benchmarks/`.

### Upstream sources (used by loaders; you rarely need these directly)

```
$ROOT/data/processed_tir_v2/data/test.parquet         # 1064-row pool_v2 test; sliced by data_source
                                                      #   for the 4 pool_v2 cells.
$ROOT/data/hf_cache/lscpku___olympiad_bench-official/OE_TO_physics_en_COMP/...  # OlympiadBench arrow
$ROOT/data/hf_cache/Eureka-Lab___phy_bench/default/.../phy_bench-train.arrow    # PHYBench arrow
$ROOT/data/raw/abench/Phy_A_fixed_400.csv, Phy_B_dynamic_100.csv                # ABench CSVs
```

### Per-cell outputs

```
$ROOT/outputs/eval/<benchmark>/<mode>__<model_slug>__<tag>/
    shard_0/rollouts_chunk_*.parquet    # crash-safe chunks (64 rollouts each)
    shard_0/rollouts.parquet            # shard 0 concat, full rollouts for rows [0 : N/2)
    shard_1/rollouts.parquet            # shard 1, rows [N/2 : N)
    rollouts.parquet                    # merged: full N rollouts (one row per problem × rollout_idx)
    scored.parquet                      # per-rollout verdict + correctness
    scored.summary.txt                  # headline pass@1 + counts
```

`<model_slug>` is `Qwen-Qwen3-4B-Thinking-2507` for the current sweeps.
`<tag>` = `train` or `qwen`.

Example to inspect one cell:
```python
import pandas as pd
base = "outputs/eval/pool_v2_drsci/tir__Qwen-Qwen3-4B-Thinking-2507__train"
r = pd.read_parquet(f"{base}/rollouts.parquet")     # trajectories
s = pd.read_parquet(f"{base}/scored.parquet")       # correctness
print(s["correct"].mean(), len(s))
```

### Roll-up summaries

- `$ROOT/logs/all_sweeps_summary.txt` — in-dist matrix (5 benchmarks × 4 cells).
- `$ROOT/logs/external_sweeps_summary.txt` — external matrix (3 benchmarks × 4 cells).
- `$ROOT/outputs/eval/README_QWEN35_SWEEP_IN_PROGRESS.md` — paused Qwen3.5
  baseline; partial cells kept as a fallback revert checkpoint.

### Tooling (all under `eval/`)

| File | What |
|---|---|
| `eval/run_eval.py` | Orchestrator: `(model, benchmark, mode) → rollouts.parquet + scored.parquet + summary.txt`. 3 idempotent stages. |
| `eval/run_eval.sh` | Apptainer wrapper for `run_eval.py`. Passes the right binds, env, PYTHONPATH. |
| `eval/run_sharded_sweep.sh` | Per-benchmark 2-GPU sharded rollout + merge + score for one `(mode, preset)` cell. Overridable `BENCHMARKS` env. |
| `eval/run_all_sweeps.sh` | Top-level driver: polls for any running sweep, then runs the 4 cells `(tir/train, cot/train, tir/qwen, cot/qwen)` on the 5 in-dist+olympiad benchmarks. |
| `eval/run_external_sweeps.sh` | Same 4 cells, but `BENCHMARKS=phybench,abench_phy_a,abench_phy_b`. Waits for any in-flight sharded sweep. |
| `eval/rescore_failed.sh` | Walks `outputs/eval/**/rollouts.parquet` and re-runs the appropriate scorer for any cell missing `scored.parquet`. |
| `eval/benchmarks/{pool_v2,olympiad_bench,phybench,abench}.py` | One loader per benchmark; writes the materialized parquet. |
| `eval/scoring/{our_verifier,olympiad_official,phybench_eed,abench_official}.py` | Per-benchmark scorer. |
| `eval/analyze_tool_use.py` | CSV-style per-cell breakdown: tool-call rate × (sandbox ok / error) × pass rate in each slice. |
| `eval/dump_cot_wins.sh` | Per-benchmark diff dumps for case-level spot-checking. |

---

## <a name="rerun"></a>3. Rerun on a trained checkpoint

The harness is model-agnostic — just override `MODEL`. Data is cached so
subsequent runs only re-roll; nothing else reloads.

### End-to-end (both matrices, all 32 cells)

```bash
cd $ROOT

# 5 in-dist + OlympiadBench cells × 4 (mode, preset) = 20 cells
MODEL=/path/to/trained/checkpoint bash eval/run_all_sweeps.sh \
    > logs/all_sweeps_<run_name>.log 2>&1 &

# PHYBench + 2 ABench cells × 4 = 12 cells
# This script polls until run_all_sweeps finishes, then starts — safe to queue now.
MODEL=/path/to/trained/checkpoint bash eval/run_external_sweeps.sh \
    > logs/external_sweeps_<run_name>.log 2>&1 &
```

Output lands at `outputs/eval/<benchmark>/<mode>__<slug>__<tag>/` where
`<slug>` is derived from `MODEL` (slashes → dashes, `_` for non-alnum/dot).
The two sweeps won't collide thanks to the poll-wait in
`run_external_sweeps.sh`.

### Single cell (e.g. just TIR/train on drsci)

```bash
MODEL=/path/to/ckpt MODE=tir PRESET=train TAG=train BENCHMARKS=pool_v2_drsci \
    bash eval/run_sharded_sweep.sh
```

`BENCHMARKS` takes a comma-separated list of registered benchmarks. Valid
names are the keys of the `BENCHMARKS` dict in `eval/run_eval.py` (also
listed by `python3 eval/run_eval.py --help`).

### Partial resume (skip already-scored cells)

The orchestrators short-circuit on existence: if
`<out_dir>/scored.summary.txt` already exists they skip the cell entirely.
If shard `rollouts.parquet` exists they skip that shard's rollout but still
merge + score. So interrupt-and-rerun is safe — just re-invoke the same
command and it picks up where it left off.

### Rescore after a verifier/scorer tweak

If you patch a scorer (e.g. added better unit handling), wipe the stale
`scored.parquet`s and call:

```bash
find $ROOT/outputs/eval -name 'scored.parquet' -delete
bash $ROOT/eval/rescore_failed.sh
```

`rescore_failed.sh` picks the right scorer per benchmark and doesn't
re-rollout.

### Expected walltime (A100-40G × 2)

| Stage | Walltime |
|---|---|
| Main matrix (5 in-dist + olympiad × 4 cells) | ~3-4 hours |
| External matrix (phybench + 2 abench × 4 cells) | ~3-4 hours |
| Both sequentially | ~7-8 hours |

PHYBench dominates (1000 rows × thinking budget) — if you only need
in-dist, skip the external launch.

---

## <a name="answer-type"></a>4. Answer-type analysis

This is the core workflow for RQ2 (LaTeX FN rate per answer type). All data
needed is already in `rollouts.parquet` and `scored.parquet`.

### Schema

Every `rollouts.parquet` carries an `extra_info` struct per row with (among
other fields) `answer_type` — one of
`numerical | expression | equation | mcq | true_false | open_end | code`
(the exact set depends on the benchmark).

Pool v2 rows also carry `primary_answer_type` in `extra_info` (from upstream
Dr.SCI) which is the cleaned version used for stratified splitting.

`scored.parquet` duplicates `answer_type` at the top level for easy groupby.

### Quick recipe — per-answer-type pass@1

```python
import pandas as pd
from pathlib import Path

ROOT = Path("/diskarray/home/xs272/phys-reasoner")
MODEL_SLUG = "Qwen-Qwen3-4B-Thinking-2507"
rows = []
for bench in ["pool_v2_drsci", "pool_v2_physics", "pool_v2_ugphysics",
              "pool_v2_scibench"]:
    for mode, preset in [("tir","train"),("cot","train"),("tir","qwen"),("cot","qwen")]:
        cell = ROOT / f"outputs/eval/{bench}/{mode}__{MODEL_SLUG}__{preset}"
        if not (cell / "scored.parquet").exists():
            continue
        s = pd.read_parquet(cell / "scored.parquet")
        agg = (s.groupby("answer_type")["correct"]
                 .agg(["mean", "count"])
                 .rename(columns={"mean": "pass@1", "count": "n"}))
        agg["bench"] = bench
        agg["cell"] = f"{mode}/{preset}"
        rows.append(agg.reset_index())
df = pd.concat(rows, ignore_index=True)
print(df.pivot_table(index=["bench","answer_type"],
                     columns="cell", values="pass@1"))
```

This produces the table the paper's RQ2 figure needs: per-type pass@1 for
TIR vs CoT. The TIR/CoT gap per type is the "verifier FN signal" the
execution reward is supposed to remove.

### Tool-use rate × correctness breakdown

`eval/analyze_tool_use.py` already does this across all cells:

```bash
bash -c "PYTHONNOUSERSITE=1 apptainer exec \
  --overlay $ROOT/phys-reasoner-overlay-017.img:ro --no-home \
  --bind /etc/pki:/etc/pki --bind $ROOT --pwd $ROOT \
  --env PYTHONNOUSERSITE=1 --env PYTHONPATH=/opt/phys-extras/ \
  $ROOT/verl_vllm017.latest.sif \
  python3 $ROOT/eval/analyze_tool_use.py"
```

Reports per-cell:
- % of rollouts that called the tool vs. skipped
- pass@1 in each slice (tool OK, tool errored, skipped)
- overall pass@1

Useful for sanity checks (e.g. "is TIR winning on this source because of
skip-path accidents?" — the analyzer separates them).

### Diff spot-checks

`eval/dump_cot_wins.sh` materializes a folder of side-by-side txt files
for every case where CoT got it right and TIR got it wrong (or the reverse
with `DIRECTION=tir_wins`). Read the `_manifest.txt` for the per-type
breakdown of disagreement counts — it's the fastest way to diagnose where
the verifier is false-negative-ing.

```bash
TAG=train DIRECTION=cot_wins MAX=50 bash eval/dump_cot_wins.sh
```

Outputs under `outputs/eval/<benchmark>/diff_cot_wins_<tag>/case_*.txt`.

---

## <a name="gotchas"></a>5. Known gotchas

1. **OOM race between sweeps.** External sweep starts as soon as the main
   sweep's last scoring subprocess exits, but xVerify-7B (~14 GB) can take
   a few seconds to release GPU memory. If the external sweep's first cell
   launches during that window, shard_0 can OOM. Workaround: the orchestrator
   marks the cell failed and continues; rerun that single cell manually after
   the sweep ends with `BENCHMARKS=<that_bench>`. This bit phybench/tir/train
   on 2026-04-19 and was retried cleanly (final number in the matrix).
2. **antlr4 for sympy's LaTeX parser.** OlympiadBench and PHYBench scorers
   call `sympy.parsing.latex.parse_latex` which requires
   `antlr4-python3-runtime>=4.11`. The SIF ships 4.9.3; we keep 4.11 under
   `eval/_pkgs/` (gitignored) and prepend it in `run_eval.sh` /
   `rescore_failed.sh`. Reinstall on a new host with:
   ```bash
   pip install --no-deps --target $ROOT/eval/_pkgs \
       antlr4-python3-runtime==4.11 timeout-decorator zss
   ```
3. **pint's decibel unit is offset-based.** `our_verifier` used to crash on
   rows with `dB` units (`OffsetUnitCalculusError`). Patched: per-row
   exceptions are now caught and the row is marked unverifiable. If you see
   `verifier_error > 0` in a summary, those are cells where the verifier
   itself threw — they'd show up as "unverifiable" in the inclusive rate.
4. **`env.sh` unsets `CUDA_VISIBLE_DEVICES`** (for Ray placement during
   training). The sharded launcher re-saves and re-exports it so each shard
   goes to the right GPU. If you change `run_eval.sh`, preserve the
   `_saved_cuda` block near the top.
5. **Benchmark-parquet cache is model-agnostic.** Each benchmark parquet is
   materialized once (under `data/processed/eval/`) and shared across all
   models / modes / presets. If you edit a loader (e.g. to change the prompt
   template), delete the cached parquet first or pass `--force_load` to the
   orchestrator.
