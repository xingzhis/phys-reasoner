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

### TIR rollout format: single-turn (decided 2026-03-31)
- **Decision**: single-block TIR, NOT multi-turn (no ToolAgentLoop / SGLang multi-turn)
- **Rationale**: (1) multi-turn adds ~1.5 weeks engineering (SGLang switch, ToolAgentLoop integration, sequence stitching bugs); (2) the paper's core claim (execution-based reward eliminates LaTeX FN noise) is orthogonal to number of turns; (3) Dr. SCI physics problems are overwhelmingly closed-form — 85–90% solvable in one block; (4) single-turn = cleaner experimental design (only variable is execution vs. not)
- **Re-evaluate if**: Stage 0 probe shows execution success <30% and failure mode is clearly "code error multi-turn would fix"
- **Implementation**: custom stop-string injection wrapper (~100 LOC) with vLLM, NOT ToolAgentLoop

### Model name
- Using **Qwen3.5-4B** (not Qwen3-4B). References to "Qwen3-4B" in `docs/grpo_training.md` are intentional — that doc describes POLARIS results which use the older Qwen3-4B.

---

## What to do next (in order)

### 1. TIR injection pipeline — IMPLEMENTED (2026-03-31), needs smoke test
Files written:
- `src/phys_reasoner/tir/sandbox.py` — subprocess execution, whitelist, timeout
- `src/phys_reasoner/tir/prompts.py` — TIR_SYSTEM_PROMPT, CODE_STOP, extract_code()
- `src/phys_reasoner/tir/tir_agent_loop.py` — @register("physcode_tir") VeRL subclass
- `src/phys_reasoner/eval/stage0_probe.py` — standalone vLLM probe (no Ray)
- `scripts/grpo_train.sh` + `scripts/train_physcode.py` — training launcher
- `scripts/stage0_probe.sbatch` — sbatch job for Stage 0
- `tests/test_tir.py` (43 tests, pass) + `tests/test_tir_verl.py` (needs -017.img)

Key design decisions:
- **No [/answer] stop token** — phase 2 runs to EOS/max_tokens; `_extract_boxed()` extracts answer
- **[answer] is a prompt-only marker** — not a stop signal
- **Model path**: use `Qwen/Qwen3.5-4B` HF ID; auto-downloads to HF_HOME if not cached
- **gold-vs-gold FN test**: already exists in `scripts/drsci_audit.py:run_round_trip()` (NOT in latex_fn_rate.py — that was deleted as redundant)

Parameters to tune before production run (see Stage 0 probe results):
- `max_response_length` (default 4096): raise if `p1_truncated > 5%` in probe output
- `sandbox_timeout` (default 30s): profile on SymPy-heavy Dr. SCI problems
- `temperature/top_p/top_k` (0.7/0.8/20): Qwen defaults, not tuned
- `lr` (1e-6): conservative; try 3e-6 in ablation
- Algorithm (GRPO): defer DAPO/GSPO/CiSPO until first smoke run

### 2. Stage 0 probe — NEXT ACTION
`sbatch scripts/stage0_probe.sbatch` — runs 100-problem TIR zero-shot
- Check: `p1_truncated` rate, `exec_success_rate`, `verifier_hit_rate` per type
- Decision: hit_rate ≥ 0.15 on numerical → skip SFT

### 3. gold-vs-gold FN rate on Dr. SCI (RQ2 baseline)
`drsci_audit.py` already has `run_round_trip()`. Run on `drsci_physics_clean.parquet`.
Was done on old 6.8k corpus; needs fresh run on Dr. SCI clean corpus.

### 4. Add `_train_weight` column (Goldilocks B strategy)

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
