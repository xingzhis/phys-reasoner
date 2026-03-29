# Plan: Dr. SCI Dataset Investigation & Integration

**Status:** Active — started 2026-03-29
**Feeds into:** `adaptive-compute-routing.md` Week 1–2 (corpus finalization)
**Output:** Decision on corpus composition for SFT + RL training

---

## Background

Dr. SCI (MiniByte-666/Dr.SCI) has 890,505 rows total. After filtering `subject=physics` + `match_rule=True`:
- **115,497 rows** saved to `data/processed/drsci_physics.parquet`
- All have `reward_model.style = 'rule'` (rule-verifiable)
- Answers are bare numeric/symbolic strings (e.g. `6.48`, `8.2 \times 10^{-6}`)
- Sources: `MegaScience`, `WebInstruct-Verified`, `natural_reasoning`
- Difficulty: discretized 0–0.75, median = 0.0 (skewed easy)
- Key fields: `extra_info.question`, `extra_info.reference_answer`, `reward_model.ground_truth`, `extra_info.match_rule`, `extra_info.difficulty`, `extra_info.from`

Existing training corpus: `data/processed/candidates_deduped.parquet` (~6,866 rows, carefully curated from 5 sources).

---

## Phase 1 — Data Quality & Dedup

**Goal:** Know exactly how many clean, non-contaminated, English, parseable rows survive.
**Target completion:** 2026-04-01

### 1a. Eval contamination check (BLOCKING — do first)

Dr. SCI's sources (MegaScience, WebInstruct-Verified) are aggregators that almost certainly
contain OlympiadBench, PHYSICS (desimfj), and UGPhysics rows — our eval sets.

**Script to write:** `scripts/drsci_dedup.py`

Steps:
1. Load `data/processed/drsci_physics.parquet` (115,497 rows)
2. Load all eval sets via existing loaders in `run_dedup.py::load_eval_candidates()` — these cover Yale NLP, ABench Phy_A/B, CritPt
3. **Also load the training corpus** (`candidates_deduped.parquet`) and treat it as a dedup target — identifies exact overlaps between Dr. SCI and our existing 6.8k
4. Run existing 3-pass dedup logic (exact SHA-256 → near-dedup MinHash → contamination check) from `run_dedup.py` — reuse the functions, don't reinvent
5. Report:
   - Rows removed by each pass
   - Which eval sets contributed the most contamination
   - Overlap count with existing 6.8k corpus (informational, not necessarily removed)
6. Save clean set to `data/processed/drsci_physics_deduped.parquet`

**Critical threshold:** If >30% of the 115k is contaminated, the dataset is much smaller than expected and may not be usable as primary training corpus.

### 1b. Answer format audit

**Script:** add `--audit` flag to `scripts/drsci_dedup.py` or separate `scripts/drsci_audit.py`

Run on a 500-row stratified sample (by `extra_info.from`, `extra_info.difficulty`):

1. For each row, attempt to parse `reward_model.ground_truth` with our `extract.py` / `math_verify_wrapper.py`
   - Assign tentative `answer_type`: if parseable as float → `numerical`; if contains LaTeX operators → `expression`; otherwise → `unknown`
2. Run `verify_answer(gold, gold, answer_type)` — gold-gold round-trip — should return 1.0
3. Report:
   - Parse success rate by source and difficulty bucket
   - Fraction returning 1.0 / 0.0 / -1.0 on gold-gold
   - Sample of failures (the answer strings that break parsing)
4. Spot-check 20–30 failures manually — are they multi-part answers? LaTeX edge cases? Prose answers that slipped through `match_rule`?

**Key question:** Does `ground_truth` == `reference_answer` always? (Observed True in samples — verify at scale.)

**Critical threshold:** If gold-gold round-trip pass rate < 90%, we need answer-format normalization before using Dr. SCI — add to Phase 1c.

### 1c. Source quality stratification

Run the format audit separately per source (`extra_info.from`):
- `MegaScience` — expected highest quality (curated from textbooks/papers)
- `WebInstruct-Verified` — web-scraped, verified; likely good but noisier
- `natural_reasoning` — lower quality, more likely to have formatting issues

Decision rule:
- If `natural_reasoning` round-trip pass rate < 80% → drop it entirely; report count
- If `WebInstruct-Verified` < 85% → flag but keep with warning

### 1d. Language / English filter

Quick check: count rows where `extra_info.question` contains CJK characters.
If > 0.1% non-English → add filter. (Expect near-zero since `match_rule` filters are aggressive.)

---

## Phase 2 — Verifier Round-Trip & Answer Type Assignment

**Goal:** Confirm our verifier handles Dr. SCI answer formats with high fidelity, and assign answer_type labels.
**Target completion:** 2026-04-01 (parallel with Phase 1)

### 2a. Answer type inference

Dr. SCI has no explicit `answer_type` field — we need to assign one for our verifier.

**Logic (in order):**
1. Try parsing `ground_truth` as float (after stripping LaTeX notation like `\times 10^{-3}`) → `numerical`
2. If contains `=` at top level → `equation`
3. If contains LaTeX operators (`\frac`, `\sqrt`, `^`, `_`) or symbolic vars → `expression`
4. Otherwise → `unknown` (will go to xVerify fallback if needed)

Implement as `scripts/drsci_infer_answer_type.py` or fold into audit script.
Validate on 100 manually labeled rows before applying at scale.

### 2b. Tolerance calibration

Dr. SCI doesn't have a tolerance field. Our default is 5% relative.

Test: on 200 gold-gold pairs for `numerical` type, what's the distribution of
`abs(parsed_gold - parsed_gold) / parsed_gold` after LaTeX parsing? (Should be 0, but rounding
during parse could introduce error.) Confirm 5% tolerance isn't too tight.

### 2c. xVerify need assessment

For rows where rule tier returns -1.0 (unverifiable) on gold-gold:
- What fraction are `expression` type that need xVerify?
- If > 20%, we need xVerify in the reward loop for Dr. SCI too — same as existing corpus
- If < 5%, Dr. SCI is cleaner than our 6.8k (where xVerify was needed for ~30%)

### 2d. Write unit tests

Add `tests/test_drsci_verifier.py`:
- 20 gold-gold round-trip tests with real Dr. SCI rows spanning all answer types
- 10 near-correct answers (small perturbation) → should score 1.0 at 5% tolerance
- 10 clearly wrong answers → should score 0.0
- These serve as regression tests if verifier changes

---

## Phase 3 — Zero-Shot Goldilocks Profiling

**Goal:** Find the effective training-usable slice of Dr. SCI (pass@1 ≈ 20–80%).
**Target completion:** 2026-04-04 (early Week 2, requires sbatch)

### 3a. Stratified sample design

After dedup (Phase 1), sample 600 rows stratified by:
- `extra_info.difficulty`: [0.0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75] — 600/7 ≈ 85 per bucket
  - Oversample higher difficulties since they're rare: aim for ≥50 per bucket
- `extra_info.from`: maintain source proportions within each bucket

Save to `data/processed/drsci_goldilocks_sample.parquet`.

### 3b. Zero-shot pass@1 inference

**Script:** `scripts/run_zero_shot_drsci.py`
- Mirrors `run_zero_shot_nothink.py` but reads from Dr. SCI parquet
- Input columns: `extra_info.question` (problem), `reward_model.ground_truth` (answer), inferred `answer_type`
- No `\boxed{}` in ground truth — our verifier needs to extract from model output only
- System prompt: same as existing zero-shot (require `\boxed{}`)
- Model: Qwen/Qwen3.5-0.8B (fast, cheap for profiling); rerun with 4B if needed

**sbatch script:** `scripts/zero_shot_drsci_sample.sbatch`
- Single GPU job, ≤1h for 600 rows with 0.8B

Output: `data/results/zero_shot_drsci_sample.parquet` with `xv_score` or rule score per row.

### 3c. Goldilocks analysis

`scripts/analyze_drsci_goldilocks.py`:
1. Pass@1 by difficulty bucket → plot histogram
2. Pass@1 by source (`extra_info.from`)
3. Identify "Goldilocks slice": difficulty buckets where pass@1 ∈ [0.15, 0.85]
4. Estimate effective training set size = fraction of full 115k in Goldilocks slice × post-dedup count
5. Compare difficulty distribution to existing 6.8k corpus

### 3d. Pass@k on harder subset (optional, if time allows)

On the difficulty ≥ 0.25 slice (probably ~30% of sample = ~180 rows):
- Run 8 samples per problem with temperature=0.7
- Compute pass@1, pass@4, pass@8
- Gives a cleaner picture of the difficulty ceiling

---

## Phase 4 — Corpus Decision & Integration

**Goal:** Decide corpus composition and update training pipeline.
**Target completion:** 2026-04-06

### Decision framework

Based on Phase 1–3 results, apply this decision tree:

```
Post-dedup Dr. SCI count:
  < 30k → Use as supplement to 6.8k (original plan)
  30k–80k → Dr. SCI becomes primary; 6.8k becomes hard-problem supplement
  > 80k → Dr. SCI is primary; 6.8k moved to additional eval tier

Goldilocks effective fraction:
  < 30% → Filter Dr. SCI to Goldilocks slice only; mix with 6.8k
  ≥ 30% → Dr. SCI Goldilocks slice alone is sufficient (≥20k usable rows)

Verifier round-trip < 90% → Requires answer normalization step before use
xVerify needed for > 20% of Dr. SCI → Factor into reward latency / cost
```

### 4a. If Dr. SCI becomes primary corpus

1. Add `"DrSCI"` to `Source` literal in `src/phys_reasoner/data/schema.py`
2. Write `src/phys_reasoner/data/loaders.py::load_drsci()` — loads `drsci_physics_deduped.parquet`, maps columns to `PhysicsProblem`
3. Update `run_dedup.py` to include Dr. SCI in the priority ordering (decide priority vs existing sources)
4. Freeze train/dev/test split:
   - 80% train, 10% dev, 10% test (or all train if dev/test come from other sources)
   - Stratify by difficulty to ensure Goldilocks zone represented in dev/test

### 4b. If 6.8k is repurposed as eval

1. Check eval contamination of 6.8k vs Dr. SCI (already done in Phase 1c)
2. Add 6.8k or clean subset as `eval_tier1b` (in-domain physics, known provenance)
3. Update `run_dedup.py` to treat it as eval-only in future dedup runs

### 4c. Update plan and memory

- Update `adaptive-compute-routing.md` Week 2 tasks to reflect finalized corpus
- Update `project_corpus.md` memory with new verified counts
- Document decision rationale in `docs/datasets.md`

---

## Outputs (deliverables)

| File | Description |
|---|---|
| `data/processed/drsci_physics_deduped.parquet` | Post-dedup Dr. SCI (Phase 1) |
| `data/processed/drsci_goldilocks_sample.parquet` | 600-row stratified sample (Phase 3) |
| `data/results/zero_shot_drsci_sample.parquet` | Zero-shot scores on sample (Phase 3) |
| `scripts/drsci_dedup.py` | Dedup + contamination script |
| `scripts/drsci_audit.py` | Format audit + answer_type inference |
| `scripts/run_zero_shot_drsci.py` | Zero-shot inference on Dr. SCI format |
| `scripts/analyze_drsci_goldilocks.py` | Goldilocks analysis |
| `scripts/zero_shot_drsci_sample.sbatch` | Sbatch wrapper for Phase 3 inference |
| `tests/test_drsci_verifier.py` | Verifier round-trip tests on Dr. SCI rows |

---

## Key risks

| Risk | Likelihood | Mitigation |
|---|---|---|
| >50% eval contamination from MegaScience | Medium | Run dedup first; fall back to 6.8k if true |
| Goldilocks effective set < 10k after filtering | Medium | Mix with 6.8k; use pass@k filter to widen zone |
| Answer format issues in natural_reasoning | Medium-High | Drop natural_reasoning source if round-trip < 80% |
| xVerify needed for >30% of Dr. SCI expressions | Low | Already have xVerify in reward loop; add latency note |
| Dr. SCI covers narrow physics subfields | Low | Check subject distribution in Phase 1 audit |

---

## Non-negotiables

- Do NOT integrate Dr. SCI into training pipeline before Phase 1 (dedup) completes
- Do NOT drop existing 6.8k corpus until Phase 3 (Goldilocks) results are in
- Do NOT skip round-trip verifier test — silent verifier bugs are the worst kind of error
