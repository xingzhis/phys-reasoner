# Probe Rollout Runtime Plan (Step 3 of data-pipeline.md)

**Status:** Ready to launch. Script + tests + calibration done. Not yet run at full scale.
**Last updated:** 2026-04-11 (cross-cluster handoff doc)
**Parent plan:** `.claude/plans/data-pipeline.md` Step 3
**Caller context:** `.claude/session-notes.md` session 6

---

## What this is

Zero-shot difficulty probe for Goldilocks data selection. Run `scripts/dump_rollouts.py`
on the probe subset with 8 rollouts per problem using Qwen/Qwen3.5-4B (the instruct model —
that's the HF naming convention for this family, NOT a base model), then aggregate pass rates
per stratum (Step 4).

## Inputs

| What | Path | Size |
|---|---|---|
| Probe subset | `data/processed/probe_subset.parquet` | 2,500 rows (2,000 drsci + 500 corpus) |
| Model | `Qwen/Qwen3.5-4B` (HF; cached in `$HF_HOME`) | 4B params bf16 |
| Train subset (for full-scan variant) | `data/processed/{drsci,corpus}_train_split.parquet` | 104,980 rows total |

## Configuration (matches training budgets)

```
THINKING_BUDGET=12288
TOOL_CALL_BUDGET=2048
ANSWER_BUDGET=4096
max_prompt_len=1024             # dump_rollouts default
max_tool_response_len=512       # dump_rollouts default
temperature=1.0, top_p=0.9, seed=42
GPU_MEM=0.8                     # A100-40G confirmed OK; bump to 0.85–0.9 on 80G
max_tool_calls=1                # (session notes §"Open design question") — diagnostic via analyze_rollouts.py
```

These MUST stay in sync with `scripts/train_smoke_interrupt.sbatch` so probe pass rates
reflect actual training conditions. If budgets change, recalibrate.

## Script features (added 2026-04-10/11)

`scripts/dump_rollouts.py` supports:
- `--start_idx / --end_idx`: shard slice applied **after** `df.sample`, so two shards can
  be launched on different GPUs and share the same global sample.
- `--chunk_size`: processes N problems per chunk, flushes `rollouts_chunk_{k:04d}.parquet`
  after each chunk. Crash loses at most one chunk. **Resume is automatic:** re-running
  the same command skips chunks whose parquet already exists.
- `OUT_DIR` env var in `scripts/dump_rollouts.sh` (override the default timestamp dir) —
  required when launching multiple shards concurrently so they don't collide.
- Final `rollouts.parquet` in each shard dir is the concat of all chunks in that shard.

All 30 tests in `tests/test_dump_rollouts.py` pass with the refactor, including the
`TestPromptFidelityVsVeRL` byte-identical assertion.

## Calibration (2026-04-10, 1×A100-PCIe-40GB)

Run: `N=50 N_ROLLOUTS=8 GPU_MEM=0.8 THINKING_BUDGET=12288 TOOL_CALL_BUDGET=2048 ANSWER_BUDGET=4096`
on `probe_subset.parquet`, chunk_size=25. Output: `outputs/rollouts_20260410.232904/`.

| Metric | Value |
|---|---|
| Wall time (incl. ~2 min vLLM engine init) | **3381 s (~56.4 min)** |
| Trajectories | 400 |
| **Per-rollout (amortized)** | **8.45 s** |
| Per-rollout (steady state, est.) | ~8.1 s |
| Think-interrupt rate | 91/400 = **22.8%** |
| Chunk 0 (33/200 interrupted) | ~1050 s gen time (5.25 s/roll) |
| Chunk 1 (58/200 interrupted) | ~1590 s gen time (7.95 s/roll) |

**Observation: per-rollout time has ±20% variance depending on interrupt count.**
Chunks dominated by long-thinkers run ~50% slower. Use **8.5 s/rollout** as the
A100-40G planning number with ±20% headroom.

Known harmless log spam: `OSError: [Errno 30] /.config: Read-only file system` from
vLLM usage telemetry under `--no-home`. Silence with `--env VLLM_NO_USAGE_STATS=1` in
the apptainer invocation if desired.

## Runtime extrapolations

**Throughput scaling assumption:** A100-80G and H200-141G have ~2× per-GPU throughput
vs A100-40G because they fit more concurrent sequences in KV cache at 20k context.
UNVERIFIED — recalibrate with a 50-problem run on the actual target hardware before
committing to a production schedule.

### (A) Stage-1 probe: 2,500 × 8 = **20,000 trajectories**

| Hardware | Wall time | Verdict |
|---|---|---|
| 1×A100-40G | ~47 h | too slow |
| **2×A100-40G** | **~24 h** | overnight, no queueing overhead — chosen for initial run |
| 4×A100-80G | ~6 h | overkill for 20k |
| 4×H200-141G | ~4 h | overkill |

### (B) Full-train probe (optional production scan): 104,980 × 8 = **~840,000 trajectories**

| Hardware | Wall time | Verdict |
|---|---|---|
| 2×A100-40G | ~41 days | infeasible |
| 4×A100-80G | ~10 days | tight |
| **8×A100-80G (2 nodes)** | **~5 days** | realistic production choice |
| 4×H200-141G | ~3–4 days | best single-node option |

### (C) Per-stage rescoring (Strategy D, offline DDC on ~5k rows): **40,000 trajectories**

| Hardware | Wall time | Verdict |
|---|---|---|
| 2×A100-40G | ~47 h | too slow between stages |
| **4×A100-80G** | **~12 h** | fits between stages — required minimum |
| 4×H200-141G | ~6 h | comfortable |

Per-stage rescoring at 10k rows ≈ double the (C) numbers. Scale the sample to whatever
12–24h window fits your between-stage gap.

## Launch commands

### 2-GPU single-node template (used for Stage-1 probe on this cluster)

```bash
# Shard 0 — GPU 0
CUDA_VISIBLE_DEVICES=0 \
MODEL=Qwen/Qwen3.5-4B PARQUET=data/processed/probe_subset.parquet \
N=-1 N_ROLLOUTS=8 SEED=42 GPU_MEM=0.8 \
THINKING_BUDGET=12288 TOOL_CALL_BUDGET=2048 ANSWER_BUDGET=4096 \
START_IDX=0 END_IDX=1250 CHUNK_SIZE=50 \
OUT_DIR=outputs/probe_rollouts_shard0 \
bash scripts/dump_rollouts.sh > /tmp/probe_shard0.log 2>&1 &

# Shard 1 — GPU 1
CUDA_VISIBLE_DEVICES=1 \
MODEL=Qwen/Qwen3.5-4B PARQUET=data/processed/probe_subset.parquet \
N=-1 N_ROLLOUTS=8 SEED=42 GPU_MEM=0.8 \
THINKING_BUDGET=12288 TOOL_CALL_BUDGET=2048 ANSWER_BUDGET=4096 \
START_IDX=1250 END_IDX=2500 CHUNK_SIZE=50 \
OUT_DIR=outputs/probe_rollouts_shard1 \
bash scripts/dump_rollouts.sh > /tmp/probe_shard1.log 2>&1 &
```

Wait for both to finish, then concat:
```python
import pandas as pd
pd.concat([
    pd.read_parquet(f"outputs/probe_rollouts_shard{k}/rollouts.parquet")
    for k in (0, 1)
]).to_parquet("outputs/probe_rollouts.parquet", index=False)
```

### N-GPU generalization (for collaborator cluster)

Given `N_SHARDS` GPUs and `N_ROWS` (2500 for probe, 104980 for full train):

```bash
N_SHARDS=4          # or 8 for 2-node
N_ROWS=2500         # or 104980 for full-train probe
ROWS_PER_SHARD=$(( (N_ROWS + N_SHARDS - 1) / N_SHARDS ))
for k in $(seq 0 $((N_SHARDS-1))); do
    lo=$((k * ROWS_PER_SHARD))
    hi=$(( (k+1) * ROWS_PER_SHARD ))
    (( hi > N_ROWS )) && hi=$N_ROWS
    CUDA_VISIBLE_DEVICES=$k \
    MODEL=Qwen/Qwen3.5-4B PARQUET=data/processed/probe_subset.parquet \
    N=-1 N_ROLLOUTS=8 SEED=42 GPU_MEM=0.85 \
    THINKING_BUDGET=12288 TOOL_CALL_BUDGET=2048 ANSWER_BUDGET=4096 \
    START_IDX=$lo END_IDX=$hi CHUNK_SIZE=50 \
    OUT_DIR=outputs/probe_rollouts_shard${k} \
    bash scripts/dump_rollouts.sh > /tmp/probe_shard${k}.log 2>&1 &
done
wait
```

Notes for non-local clusters:
- **Always recalibrate** on the target hardware with `N=50 N_ROLLOUTS=8` first. A100-80G
  and H200 KV-cache headroom means throughput is memory-bandwidth bound, not compute bound,
  and the 8.5 s/rollout number above is A100-40G-specific.
- **GPU_MEM**: safe starting values — A100-40G 0.8, A100-80G 0.85, H200-141G 0.9.
- **CHUNK_SIZE=50** is appropriate for shards of ~600 problems. For bigger shards (full
  train) use `CHUNK_SIZE=100–200` so flushes happen every ~15–30 min of work, not every
  few minutes.
- **Resume**: if a shard crashes, rerunning the same command resumes from the last
  flushed chunk automatically. No `--resume` flag needed — the code checks for existing
  `rollouts_chunk_{k:04d}.parquet` files in `OUT_DIR` and skips them.
- **xVerify server NOT required** for this step. Scoring happens in Step 4
  (`aggregate_probe_scores.py`, not written yet) via the `phys_reasoner/verifier/`
  stack, which calls xVerify as a fallback. The parquet only contains raw traces +
  gold answers.
- **SLURM caveat**: the multi-shard bash loop above assumes you can pack N GPUs into
  one sbatch job. If your cluster prefers one GPU per job (common), make N sbatch
  submissions, each with `#SBATCH --gres=gpu:1` and different `START_IDX/END_IDX/OUT_DIR`
  values. All shards share `probe_subset.parquet` read-only, no coordination needed.

## Post-probe next steps (Step 4 and beyond)

After the rollout parquet exists:
1. **`scripts/analyze_rollouts.py`** (not written): bucket rollouts by
   `(sandbox_error, has_boxed, is_correct)` to decide whether `max_tool_calls=1` is
   sufficient. Decision gate in `.claude/session-notes.md` §"Open design question".
2. **`scripts/aggregate_probe_scores.py`** (Step 4, not written): compute per-problem
   pass rate over 8 rollouts, attach stratum metadata, write `probe_scores.parquet`.
   Use `src/phys_reasoner/verifier/` (rule + xVerify) for scoring — do NOT reinvent
   answer parsing.
3. Steps 5 & 6 (weights + hard bank): see `.claude/plans/data-pipeline.md`.

## Pointers for future sessions

- Plan: `.claude/plans/data-pipeline.md`
- Current state: `.claude/session-notes.md`
- Container + environment: `CLAUDE.md` § Environment
- Budget identity reference: `scripts/train_smoke_interrupt.sbatch`
- Think-interrupt phrase sync point: `src/phys_reasoner/tir/prompts.py` +
  `verl/verl/experimental/agent_loop/tool_agent_loop.py` (must match byte-for-byte)
- Calibration raw output: `outputs/rollouts_20260410.232904/` (keep for reference)
