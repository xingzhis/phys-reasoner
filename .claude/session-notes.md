# Session Notes — for next session context

Last updated: 2026-03-30

---

## Where things stand

### Data (DONE)
- **6.8k corpus**: `data/processed/candidates_deduped.parquet` — 6,866 rows, columns: `problem`, `answer`, `answer_type`, `source`, `difficulty`, `unit`
- **Dr. SCI**: `data/processed/drsci_physics_clean.parquet` — 107,158 rows (after all cleaning + prose-gold drop), columns include `inferred_answer_type`
- Both have been through full cleaning pipelines. Dr. SCI cleaning has 6 drop steps; prose-gold drop (step 6) is new as of this session.

### Verifier (DONE)
- Full rule + xVerify-7B pipeline working
- `_sympy_numerical_equiv` numerical substitution tier implemented and tested — plan file `floating-percolating-thunder.md` is stale/complete
- Authoritative baseline: `data/results/rescore_7b_v3.parquet` (6,866 rows, think-ON, 39.7% accuracy with xVerify-7B)

### Pass@k baselines (DONE)
- Corpus sample (311 rows): `data/results/zero_shot_corpus_passk_xv7b.parquet` — 30.3% pass@8
- Dr. SCI sample (580 rows, prose-gold filtered): `data/results/zero_shot_drsci_sample_xv7b.parquet` — 24.3% pass@8
- Goldilocks [15%–85%]: ~17–18% of problems in both corpora
- Key doc: `docs/verifier-zero-shot-experiments.md` (has all tables, methodology, truncation discussion)

### Key decisions made this session
- **SFT skipped** — go directly to GRPO. See `docs/training-decisions.md`.
- **Goldilocks strategy B+C** — bucket weights at load time + online [0.05, 0.95] filter in GRPO. See `docs/training-decisions.md`.
- **Prose-gold drop** — 1,532 rows removed from Dr. SCI as step 6 in `drsci_clean.py` (computation narratives / definition clauses). Distribution impact < 1pp.

---

## What to do next (in order)

### 1. Add `_train_weight` column to both parquets
Small script that maps `(source × answer_type)` → Goldilocks-rate-based weight using the lookup table in `docs/training-decisions.md`. Write to both `candidates_deduped.parquet` and `drsci_physics_clean.parquet`.

### 2. Write four prompting templates
Files to create: `src/phys_reasoner/prompts/` (or similar).
- `answer.py` — direct response, ends with `\boxed{}`
- `check.py` — answer + structured verification + optional revision; required fields: ACTION, CANDIDATE_ANSWER, CHECK_TYPE, CHECK_WORK, REVISED, FINAL_ANSWER
- `think_deep.py` — extended CoT before answering; uses `enable_thinking=True`
- `tool_check.py` — answer + tool call + optional revision; required fields: ACTION, CANDIDATE_ANSWER, TOOL_NAME, TOOL_INPUT, TOOL_RESULT, REVISED, FINAL_ANSWER

### 3. Parser for Check / Tool-Check outputs
Validate structured fields, reject malformed outputs, log parse pass/fail.

### 4. Tool wrappers
`check_equation`, `check_units`, `plug_values`, `check_root`, `compare_expr` — restricted SymPy/pint wrappers.

### 5. GRPO setup with verl
- Reward: `R = R_correct − λ·C_action`
- Cost `C_action`: Answer=1, Check=2, Think-Deep=3, Tool-Check=4 (or token-based)
- Online filter: skip update if `pass_rate ∉ [0.05, 0.95]`
- Start with Qwen3.5-0.6B dev run on ~500 problems

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
| Implementation checklist | `docs/implementation_checklist_v2_5_2.md` |
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
- GPU partition: `gpu`, qos `qos_nmi`, gres `h200:1` (updated from a100)
