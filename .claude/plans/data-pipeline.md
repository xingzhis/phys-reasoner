# Plan: Data Pipeline — Goldilocks Filtering for Training

**Status:** Active — Week 2
**Started:** 2026-04-10
**Reference:** `docs/training-decisions.md` → "Goldilocks Data Selection: Strategy B + D"

---

## Overview

This pipeline takes the cleaned training parquets (`drsci_train.parquet`, `corpus_train.parquet`)
and produces:
1. Enriched parquets with full metadata
2. Stratified train/dev/test splits
3. A probe subset for rollout scoring
4. Per-problem pass rates from 8 rollouts
5. `_train_weight` column on training rows (Strategy B)
6. A hard problem bank for progressive curriculum (Strategy D feed-in)

A separate reusable rescoring job (Strategy D) runs after each training stage using the
same scoring logic but with a stage checkpoint instead of the base model.

**Each step is independent enough to tackle in a separate session.** Read the step's
"Context" block and "Blockers" before starting.

---

## Key file locations

| What | Path |
|---|---|
| Dr. SCI training parquet | `data/processed/drsci_train.parquet` (102,563 rows) |
| Corpus training parquet | `data/processed/corpus_train.parquet` (6,817 rows) |
| Dr. SCI intermediate (rich metadata) | `data/processed/drsci_physics_clean.parquet` |
| Corpus intermediate (rich metadata) | `data/processed/candidates_deduped.parquet` |
| Train parquet builder | `scripts/build_training_parquets.py` |
| Rollout dump script | `scripts/dump_rollouts.py` |
| Verifier router | `src/phys_reasoner/verifier/router.py` |
| xVerify server | running on misha00 (see session notes) |

---

## Step 0 — Metadata enrichment + rebuild train parquets

**Status:** Done (2026-04-10)
**Blocker:** None

**What:** Update `scripts/build_training_parquets.py` to preserve richer metadata in
`extra_info`, then rebuild both train parquets.

**Dr. SCI changes:**
- Preserve `extra_info.from` (MegaScience / natural_reasoning / WebInstruct-Verified / NaN)
  from `drsci_physics_clean.parquet` into `extra_info` of the output parquet
- Fix `difficulty` to be stored as float (currently stringified "0.0")
- `inferred_answer_type` is already in `extra_info` as `answer_type` — no change needed

**Corpus changes:**
- Add `primary_answer_type`: normalize the 87 raw `answer_type` values down to 7 clean values.
  Rule: if the raw value is a JSON list → `"multi-part"`; otherwise use value as-is.
  Valid single-type values: `numerical`, `expression`, `equation`, `mcq`, `true_false`, `interval`.
- Add `domain_coarse`: map the 29 raw domain values to 6 coarse buckets:

  | Coarse bucket | Raw domain values |
  |---|---|
  | `mechanics` | ClassicalMechanics, Mechanics, MECHANICS, TheoreticalMechanics |
  | `em_electro` | ClassicalElectromagnetism, Electrodynamics, Electromagnetism, ELECTRICITY |
  | `quantum_modern` | QuantumMechanics, AtomicPhysics, Modern Physics, MODERN, ADVANCED, quan |
  | `thermo_stat` | StatisticalMechanics, Thermodynamics, thermo, stat, THERMODYNAMICS |
  | `optics` | WaveOptics, Optics, GeometricalOptics, OPTICS |
  | `other` | SemiconductorPhysics, Solid-StatePhysics, Relativity, OE_TO_physics_en_COMP, fund, calculus |

- Preserve `source` and `domain` (raw) in `extra_info` alongside `domain_coarse`

**Output:** Rebuilt `drsci_train.parquet` and `corpus_train.parquet` with enriched `extra_info`.
Also update `smoke.parquet` if needed (run `scripts/make_smoke_parquet.sh` and update
`BASE_PARQUET` in `train_smoke.sbatch`).

---

## Step 0b — Train/dev/test split

**Status:** Done (2026-04-10)
**Blocker:** None

**What:** New script `scripts/split_train_dev_test.py`. Stratified split of each dataset
into train/dev/test. Dev is used for GRPO eval logging; test is held out for paper numbers.

**Split sizes:**
- Dr. SCI: 2,000 dev + 2,000 test + ~101,729 train
- Corpus: 200 dev + 200 test + ~6,466 train

**Stratification keys:**
- Dr. SCI: `(inferred_answer_type × from × difficulty_bin)`
  where `difficulty_bin` = `low` (0.0) / `medium` (0.125–0.375) / `high` (0.5–0.75)
- Corpus: `(primary_answer_type × source)` — domain adds too many sparse cells for 6.8k rows

**Output:**
```
data/processed/drsci_{train,dev,test}.parquet
data/processed/corpus_{train,dev,test}.parquet
```

**Note:** Dev and test rows must be excluded from the probe subsample (Step 1).

---

## Step 1 — Stratified probe subsample

**Status:** Done (2026-04-10)
**Blocker:** None

**What:** New script `scripts/subsample_probe.py`. Sample a manageable subset of the
*train split only* for the rollout probe in Step 3.

**Target size:** ~2,500 rows (2,000 Dr. SCI + 500 corpus). Aim for ~40 rows per stratum
so per-stratum pass rates are stable.

**Stratification keys:** Same as Step 0b. Use proportional allocation within each stratum,
with a minimum floor (e.g., 10 rows per stratum even if the stratum is small).

**Output:** `data/processed/probe_subset.parquet` with all enriched metadata columns.

---

## Step 2 — Think-interrupt in dump_rollouts.py

**Status:** Deferred — pending think-interrupt budget decision (separate session)
**Blocker:** Think-interrupt session must finalize `thinking_budget` and `max_response_len`

**What:** Mirror the VeRL think-interrupt logic (already implemented in
`verl/verl/experimental/agent_loop/tool_agent_loop.py`) in `scripts/dump_rollouts.py`.

**New flags to add:**
- `--thinking_budget` (default=None, disabled): token count at which to inject interrupt phrase
- `--max_response_len` (total budget: think + tool call + answer)

**Logic:** After phase 1 generate, if `len(p1.token_ids) >= thinking_budget` AND
`think_end_id not in p1.token_ids`, inject interrupt phrase and re-generate phase 1b
(max_tokens=2048, same as VeRL sub-call 2). Then proceed to phase 2 as normal.

**Interrupt phrase** (must match VeRL exactly):
```
"\nOkay, time is up. Let me stop thinking and formulate a final answer\n</think>\n"
```

**Reference:** `.claude/plans/physcode.md` Week 2 "Think-interrupt patch" for the full VeRL spec.
The dump_rollouts.py implementation should be a direct translation.

---

## Step 3 — Run 8 rollouts per probe problem

**Status:** Not started
**Blocker:** Steps 1, 2 (think-interrupt must be settled before running)

**What:** Run `dump_rollouts.py` on `probe_subset.parquet` with 8 rollouts per problem.

**New feature needed in dump_rollouts.py:** `--n_rollouts N` flag (currently always 1 rollout).
Loop phase 1+2 N times per problem with different random seeds.

**Settings:**
- Model: Qwen3.5-4B instruct (base model for Stage 1 probe)
- temperature=1.0, top_p=0.9
- `--thinking_budget` and `--max_response_len` as decided in Step 2 session
- xVerify server must be running on misha00 before starting

**Output:** `outputs/probe_rollouts/` — per-problem txt files + `scores.parquet`
(columns: `problem_id`, `rollout_idx`, `score`, `boxed_answer`, `gold_answer`, metadata)

**Run via:** New `scripts/run_probe_rollouts.sbatch` targeting collaborator cluster.

---

## Step 4 — Aggregate pass rates

**Status:** Not started
**Blocker:** Step 3

**What:** New script `scripts/aggregate_probe_scores.py`. Read rollout scores,
compute pass_rate per problem, attach metadata.

```
pass_rate = sum(scores) / n_rollouts   (per problem)
```

**Output:** `data/processed/probe_scores.parquet`
Columns: `problem_id`, `pass_rate`, `answer_type`, `from` (Dr. SCI) or `source` (corpus),
`difficulty_bin`, `domain_coarse` (corpus), `dataset`.

Also write `outputs/probe_rollouts/goldilocks_report.txt`: per-stratum pass rate
distribution showing fraction in [0.05, 0.95], < 0.05, > 0.95.

---

## Step 5 — Goldilocks weights (Strategy B)

**Status:** Not started
**Blocker:** Step 4

**What:** New script `scripts/compute_goldilocks_weights.py`. For each training row,
assign a `_train_weight` based on its stratum's Goldilocks rate from the probe.

**Logic:**
1. From `probe_scores.parquet`, compute per-stratum Goldilocks rate
   = fraction of probe rows in that stratum with pass_rate ∈ [0.05, 0.95]
2. Map each training row to its stratum → look up Goldilocks rate → `_train_weight`
3. Strata with no probe rows: use nearest stratum by answer_type (fallback)

**Output:** Updated `data/processed/drsci_train_split.parquet` and
`data/processed/corpus_train_split.parquet` with `_train_weight` column added.

---

## Step 6 — Hard problem bank

**Status:** Not started
**Blocker:** Step 4

**What:** Collect problems with `pass_rate < 0.05` in the probe into a separate parquet.
These are too hard for the base model and excluded from Stage 1 training. Strategy D
(between-stage rescoring) re-evaluates them and graduates them into the active pool once
the model is strong enough.

**Output:** `data/processed/hard_bank.parquet`
Columns: same as train parquets + `pass_rate` + `probe_rollout_date`.

---

## Rescoring Pipeline (Strategy D — reusable between stages)

**Status:** Not started
**Blocker:** Stage 1 checkpoint must exist

**What:** After each training stage, rerun the scoring pipeline on a stratified sample
using the latest checkpoint. Update weights and hard bank.

**Script:** `scripts/rescore_goldilocks.py` — parameterized version of Steps 3+4.
- `--model`: path to stage checkpoint
- `--n`: number of rows to rescore (default 5000, scale up if compute permits)
- `--hard_bank`: path to current hard bank (will be updated in-place)
- `--output_weights`: output parquet with updated `_train_weight`

**Sbatch:** `scripts/rescore_goldilocks.sbatch`

**Run after each stage:**
```
Stage 1 ckpt → rescore → updated weights + hard bank → Stage 2 dataset
Stage 2 ckpt → rescore → updated weights + hard bank → Stage 3 dataset
```

**Hard bank graduation:** Any hard bank problem with pass_rate > 0.05 in the rescore
moves into the active training parquet with its new `_train_weight`.

**Analysis output:** Per-stage Goldilocks zone drift report (which strata gained/lost
problems, overall zone expansion). This is data for the paper's curriculum analysis.
