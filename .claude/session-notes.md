# Session Notes — for next session context

Last updated: 2026-03-31

---

## Where things stand

### Data (DONE)
- **6.8k corpus**: `data/processed/candidates_deduped.parquet` — 6,866 rows, columns: `problem`, `answer`, `answer_type`, `source`, `difficulty`, `unit`
- **Dr. SCI**: `data/processed/drsci_physics_clean.parquet` — 107,158 rows (after all cleaning + prose-gold drop), columns include `inferred_answer_type`
- Both have been through full cleaning pipelines. Dr. SCI cleaning has 6 drop steps; prose-gold drop (step 6) is new as of 2026-03-30.

### Verifier (DONE)
- Full rule + xVerify-7B pipeline working
- `_sympy_numerical_equiv` numerical substitution tier implemented and tested
- Authoritative baseline: `data/results/rescore_7b_v3.parquet` (6,866 rows, think-ON, 39.7% accuracy with xVerify-7B)

### Pass@k baselines (DONE)
- Corpus sample (311 rows): `data/results/zero_shot_corpus_passk_xv7b.parquet` — 30.3% pass@8
- Dr. SCI sample (580 rows, prose-gold filtered): `data/results/zero_shot_drsci_sample_xv7b.parquet` — 24.3% pass@8
- Goldilocks [15%–85%]: ~17–18% of problems in both corpora
- Key doc: `docs/verifier-zero-shot-experiments.md` (has all tables, methodology, truncation discussion)

### Project direction (CHANGED 2026-03-31)
- **Old direction** (adaptive-compute-routing): 4-action routing policy — ABANDONED
- **New direction** (PhysCode): single-block TIR + RLVR, execution-based reward
- See `docs/physcode_proposal_v5.md` and `.claude/plans/physcode.md`

---

## What to do next (in order)

### 1. VeRL single-block injection — GATING ITEM
Implement and smoke-test the mid-sequence injection in VeRL rollout:
1. Model generates until `[/code]` stop token
2. Execute code in sandbox (30s timeout, subprocess + resource limits)
3. Inject `[output] {result}\n` as fixed continuation
4. Model resumes until `[answer]` stop token
5. Extract `\boxed{}` and verify with existing pipeline

Nothing else can proceed until this works end-to-end.

### 2. Stage 0 probe (100-problem TIR zero-shot)
- Run Qwen3.5-4B instruct (thinking OFF) with single-block TIR prompt on 100 problems from training corpus
- Measure: execution success rate, verifier hit rate, per-type breakdown (numerical/expression/MCQ)
- Decision threshold: hit rate ≥15% → skip SFT; <15% → generate 2–5k TIR demos

### 3. Execution sandbox
- Subprocess isolation, 30s timeout, resource limits
- Profile timeout distribution on Dr. SCI SymPy expressions

### 4. LaTeX FN rate measurement
- Run rule verifier on training corpus gold-vs-gold round-trip
- Measure per-type FN rate: numerical, expression, equation, MCQ
- This quantifies the reward noise for RQ2

### 5. Add `_train_weight` column to both parquets (Goldilocks B strategy)
Small script mapping `(source × answer_type)` → Goldilocks-rate-based weight.
See lookup table in `docs/training-decisions.md`.

---

## Important file locations

| What | Path |
|------|------|
| Dr. SCI clean parquet | `data/processed/drsci_physics_clean.parquet` |
| Corpus parquet | `data/processed/candidates_deduped.parquet` |
| Dr. SCI pass@k (xV-7B) | `data/results/zero_shot_drsci_sample_xv7b.parquet` |
| Corpus pass@k (xV-7B) | `data/results/zero_shot_corpus_passk_xv7b.parquet` |
| Authoritative corpus baseline | `data/results/rescore_7b_v3.parquet` |
| Training strategy doc | `docs/training-decisions.md` |
| Verifier experiments doc | `docs/verifier-zero-shot-experiments.md` |
| PhysCode proposal | `docs/physcode_proposal_v5.md` |
| PhysCode plan | `.claude/plans/physcode.md` |
| Dr. SCI cleaning script | `scripts/drsci_clean.py` |
| Pass@k inference (Dr. SCI) | `scripts/run_zero_shot_drsci.py` |
| Pass@k inference (corpus) | `scripts/run_zero_shot_corpus_passk.py` |
| xVerify rescore (pass@k) | `scripts/rescore_passk_xverify.py` |

## Sbatch templates
| Job | Script |
|-----|--------|
| Dr. SCI pass@k | `scripts/zero_shot_drsci_sample.sbatch` |
| Corpus pass@k | `scripts/zero_shot_corpus_passk.sbatch` |
| xVerify rescore (any pass@k) | `scripts/rescore_passk_xverify.sbatch` |

---

## Environment reminder
- SIF: `verl_vllm017.latest.sif`
- Overlay: `phys-reasoner-overlay-017.img` (use for all sbatch jobs)
- Always: `export PYTHONNOUSERSITE=1` before apptainer calls
- HF cache: `hf_cache/` — use `local_files_only=True` for xVerify on compute nodes
- GPU partition: `gpu`, qos `qos_nmi`, gres `h200:1`
