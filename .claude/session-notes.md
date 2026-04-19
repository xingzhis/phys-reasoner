# Session Notes — for next session context

Last updated: 2026-04-19 (Perlmutter handoff prep)

---

## Status: Ready for Phase-0 Perlmutter smoke with Qwen3-4B-Thinking

**Completed since Apr 14:**
- **2026-04-17 Perlmutter baseline run** (Qwen3.5-4B): step-1 took 27 min, step-2 OOM'd. Root cause analysis at `perlmutter_debug/FINDINGS.md` — 12-tier speedup catalogue with rationale.
- **2026-04-18 Qwen3 switch executed** (`.claude/plans/qwen3-switch.md`): MODEL=Qwen/Qwen3-4B-Thinking-2507, multi_turn.format=hermes, Ulysses SP=2, staleness=1, `use_remove_padding=True`, `log_prob_max_token_len_per_gpu=81920` (4× actor), dynamic_bsz on ref+rollout, `PYTORCH_ALLOC_CONF=expandable_segments:True`, `PPO_MAX_TOKEN_LEN_PER_GPU=20480`. Qwen3.5-specific SDPA override + Qwen3_5DecoderLayer wrap policy removed (Qwen3 is vanilla transformer, flash-attn works natively).
- **2026-04-18 0.6B end-to-end verified locally** (commit `cb10403`): `train_smoke_async.sh` with `Qwen3-0.6B` completes 2 steps cleanly; 4B rollout `dump_rollouts` looks healthy.
- **2026-04-18 `[FullyAsyncTrainer] step=N ...` timing print landed** (verl `d822b76a`, parent `c230577`): per-phase timing (`timing_s/*`) + rollouter counters (`fully_async/*`) now emitted to stdout immediately at end of each `fit_step`, before the next step's `update_weights`. Fixes FINDINGS §9 caveat where a step-N+1 OOM ate step-N's decomposition. Also fixed pre-existing double-prefix bug (`timing_s/timing_s/param_sync` → `timing_s/param_sync`).
- **2026-04-19 Perlmutter-smoke prep cleanup** (commit `2932311`): `train_smoke_async.sh` hardcoded `VLLM_GPU_MEM_UTIL=0.8` no longer shadows env override; `TRAIN_SP=1` default for single-GPU smoke; NCCL tuning env vars (`NCCL_IB_TIMEOUT=32`, `NCCL_NVLS_ENABLE=1`, `NCCL_IBEXT_DISABLE=1`, `TORCH_NCCL_ENABLE_MONITORING=0`) added to `scripts/perlmutter/smoke_tir_het.sbatch`; CLAUDE.md Qwen3.5 section replaced with Qwen3-Thinking active overrides.

**Next work (start here):**
- **Phase-0 Perlmutter smoke** — submit `scripts/perlmutter/smoke_tir_het.sbatch` as-is on Perlmutter from parent `main` (verl submodule on `physcode`). Target: ≤5 min/step (vs 27 min baseline). Read new `[FullyAsyncTrainer] step=` lines from train.log for per-phase decomposition. Pass criteria in `perlmutter_debug/SMOKE_HANDOFF.md` §5.2.
- **The handoff package for a Perlmutter Claude session lives in `.claude/handoff/` (entry point: `perlmutter-onboarding.md`).**

---

## Key discovery: Qwen3.5 + vLLM B200 bug (read `docs/training-decisions.md`)

Only B200 (Blackwell SM_100) is affected. vLLM auto-selects the TRTLLM prefill attention kernel which has known correctness issues with long-context GDN models.

| GPU | Status | Fix needed? |
|-----|--------|-------------|
| B200 | **BROKEN** — drifts to 100% degeneration after ~10 chunks | `VLLM_USE_TRTLLM_ATTENTION=0` or avoid |
| H200 | Clean at full scale (10k rollouts, batch=400) | None |
| A100 / H100 | Clean | None |
| RTX 5000 Ada | Clean | None |

**For the planned 3×4×A100 async training setup: completely unaffected.** A100 uses FA2, not TRTLLM.

---

## How to rerun the full probe pipeline

### 1. Generate probe rollouts (2500 problems × 8 rollouts = 20000 on H200)

```bash
# Shard A: problems [0, 1250)
sbatch --partition=gpu_h200 --gres=gpu:h200:1 --time=24:00:00 --job-name=probe_X_A \
  --export=ALL,N=-1,N_ROLLOUTS=8,\
PARQUET=data/processed/probe_subset.parquet,\
THINKING_BUDGET=12288,TOOL_CALL_BUDGET=2048,ANSWER_BUDGET=4096,\
ENABLE_THINKING=1,TEMPERATURE=1.0,TOP_P=1.0,GPU_MEM=0.85,\
START_IDX=0,END_IDX=1250,CHUNK_SIZE=50,\
OUT_DIR=outputs/probe_X_A \
  scripts/dump_rollouts.sbatch

# Shard B: problems [1250, 2500)
sbatch --partition=gpu_h200 --gres=gpu:h200:1 --time=24:00:00 --job-name=probe_X_B \
  --export=ALL,N=-1,N_ROLLOUTS=8,\
PARQUET=data/processed/probe_subset.parquet,\
THINKING_BUDGET=12288,TOOL_CALL_BUDGET=2048,ANSWER_BUDGET=4096,\
ENABLE_THINKING=1,TEMPERATURE=1.0,TOP_P=1.0,GPU_MEM=0.85,\
START_IDX=1250,END_IDX=2500,CHUNK_SIZE=50,\
OUT_DIR=outputs/probe_X_B \
  scripts/dump_rollouts.sbatch
```

Runtime: each shard ~4-5h on H200 (single GPU). Chunks flush to parquet every 50 problems → automatic resume if cancelled.

**If forced to run on B200**: add `VLLM_USE_TRTLLM_ATTENTION=0` to the `--export` (already plumbed through apptainer via `dump_rollouts.sbatch`).

### 2. Score the rollouts (requires xVerify server)

```bash
# Step 2a: Start xVerify-7B server (GPU, 6h wall time, writes URL rendezvous file)
sbatch --partition=gpu_h200 --gres=gpu:h200:1 scripts/serve_xverify.sbatch

# Step 2b: Score (CPU, reads URL from outputs/xverify_endpoints/current.url)
sbatch --partition=day_amd \
  --export=ALL,PROBE_INPUTS="outputs/probe_X_A/rollouts.parquet outputs/probe_X_B/rollouts.parquet",\
PROBE_OUTPUT=outputs/probe_X_scored \
  scripts/score_probe.sbatch
```

Scoring runs at ~50-100 rollouts/sec → ~3-7 min for 20k rollouts. **Kill xVerify server after** (`scancel $JOB`).

Output: `rollouts_scored.parquet` (full rollouts + `score`/`score_raw`) + `score_summary.txt` (per-type/difficulty pass rates).

### 3. How the probe subset was built (already done, 2026-04-11)

The probe subset is a stratified sample of 2500 rows (2000 Dr. SCI + 500 corpus) from the train-only split. Full reproducibility chain from cleaned source data:

```bash
# Step 1: build enriched training parquets (extra_info metadata, row filtering)
python3 scripts/build_training_parquets.py
# Inputs (defaults): data/processed/drsci_physics_clean.parquet,
#                    data/processed/candidates_filtered.parquet
# Outputs:           data/processed/drsci_train.parquet,
#                    data/processed/corpus_train.parquet

# Step 2: stratified train/dev/test split
python3 scripts/split_train_dev_test.py --seed 42
# Inputs (defaults): drsci_train.parquet, corpus_train.parquet
# Outputs: {drsci,corpus}_{train_split,dev,test}.parquet
# Splits: Dr. SCI 2k dev + 2k test; Corpus 200 dev + 200 test; rest = train
# Stratification: (answer_type × from × difficulty_bin) for Dr. SCI,
#                 (primary_answer_type × source) for Corpus

# Step 3: probe subsample from train splits
python3 scripts/subsample_probe.py --seed 42
# Inputs (defaults): drsci_train_split.parquet, corpus_train_split.parquet
# Output:            data/processed/probe_subset.parquet  (2500 rows)
# --n-drsci 2000 --n-corpus 500 --min-per-stratum 10 (defaults)
# Add --report for stats-only dry run (no parquet written)
```

All three scripts default to `seed=42` — deterministic given the same cleaned upstream data (`drsci_physics_clean.parquet`, `candidates_filtered.parquet`). Run inside apptainer with the overlay if dependencies aren't on the host.

---

## Where things are

| What | Path |
|------|------|
| Probe rollouts (shard A) | `outputs/probe_v5_A/rollouts.parquet` (10000 rollouts) |
| Probe rollouts (shard B) | `outputs/probe_v5_B/rollouts.parquet` (10000 rollouts) |
| Scored rollouts | `outputs/probe_v5_scored/rollouts_scored.parquet` |
| Score summary | `outputs/probe_v5_scored/score_summary.txt` |
| Probe source data | `data/processed/probe_subset.parquet` |
| Old/diagnostic runs | `outputs/old/` (10+ exploratory probes from debugging) |
| xVerify server rendezvous | `outputs/xverify_endpoints/current.url` |

---

## Code changes made this session

Modified (all committed to `git status`):
- `docs/training-decisions.md` — corrected root cause diagnosis; documents the full debugging trail and B200 workaround
- `scripts/dump_rollouts.py` — `enforce_eager=True`, `top_p=1.0` default, `<|im_end|>` prepend + `enable_thinking=True` in tool injection (matches VeRL ToolAgentLoop behavior exactly)
- `scripts/dump_rollouts.sh` — default `TOP_P=1.0`
- `scripts/dump_rollouts.sbatch` — forwards `VLLM_USE_TRTLLM_ATTENTION` env var to apptainer, default `TOP_P=1.0`
- `scripts/train_async.sh` — B200 safety warning, rollout logging flags (`DUMP_VAL_ROLLOUTS=1` default), forwards `VLLM_USE_TRTLLM_ATTENTION`
- `scripts/merge_probe_shards.py` — **new** utility to merge shards into one parquet (not used here since score_probe.sbatch accepts multiple inputs, but available)

**No VeRL training behavior changed** — only logging/safety flags added. All Hydra overrides for actor/rollout/PPO config are unchanged.

---

## TODO before the actual training run

From `docs/training-decisions.md`:

1. **Update top_p in training scripts** (currently still at `0.9` from pre-bug-fix):
   - `scripts/smoke_tir_qwen35.sh:207`
   - `scripts/smoke_tir.sh:161`
   - `scripts/train_async.sh:199` (or wherever the Hydra override is)

   Change to `top_p=1.0` for raw on-policy RL sampling. Preserved as `0.9` for now so existing smoke test results remain reproducible.

2. **If using B200 for training rollouts**: set `VLLM_USE_TRTLLM_ATTENTION=0` via `sbatch --export`.

3. **Build Strategy B weights** from `probe_v5_scored` (Step 4 of data-pipeline.md).

---

## Environment state

- vLLM 0.17, transformers 5.3.0 in overlay `phys-reasoner-overlay-017.img`
- flash-attn 2.8.3 already in SIF (if we want to pin away from flashinfer TRTLLM, flag is `--attention-backend flash_attn`, but H200 path already works without)
- Model cached under `$HF_HOME` — Qwen/Qwen3.5-4B, xVerify-7B-I

## State of dump_rollouts.py (as of 2026-04-14)

Faithfully mirrors VeRL `ToolAgentLoop` for Qwen3.5 qwen3_coder format:
- Phase 1 (`enable_thinking=True`) → stop at `</tool_call>` → optional think-interrupt (phase 1b) → sandbox exec → phase 2 tool injection (`enable_thinking=True`, matching VeRL) → final answer
- Engine: `LLM(enforce_eager=True, ...)` — matches VeRL RolloutConfig
- Shard-aware (`--start_idx/--end_idx`), chunked parquet flush, resume-safe
- Supports `--dump_txt` / `DUMP_TXT=1` for per-rollout human-readable txt files (off by default)

Parquet columns: `problem_idx, rollout_idx, gold_answer, answer_type, phase1_text, interrupted, phase1b_text, code, sandbox_stdout, sandbox_error, phase2_text, extra_info`. Scoring is a separate step — the parquet does NOT contain a parsed `\boxed{}` or a correctness label.

---

## Archived session 7 notes

Previous sessions 1–7 notes (pre-debugging) are preserved in git history — the original content of this file is in the prior commit. Archiving was intentional: the old notes described the probe launch as an *upcoming* task; now it's done.
