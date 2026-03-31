# Plan: PhysCode — Tool-Integrated Reasoning for Physics via RLVR

**Status:** Active — Week 1 in progress
**Started:** 2026-03-31
**Target completion:** 2026-05-12
**Proposal:** `docs/physcode_proposal_v5.md`

---

## Project Summary

Train Qwen3.5-4B with RLVR to solve physics problems using Tool-Integrated Reasoning (TIR): the model reasons, executes exactly one Python/SymPy code block, receives the output, and reasons to a final \boxed{} answer. Core claim: execution-based reward eliminates the symbolic verification noise (LaTeX FN rates ~68% on expression types) that degrades CoT-GRPO learning, connecting empirically to the RLVεR theoretical framework.

---

## Key Decisions (Frozen)

| Item | Decision |
|---|---|
| Main model | Qwen3.5-4B instruct (thinking OFF for TIR format) |
| Debug model | Qwen3.5-0.8B |
| TIR format | Single code block: `[think]…[code]…[/code][output]…[think]…[answer]\boxed{}` |
| Primary training data | Dr. SCI clean (~65k numerical + expression + MCQ after Goldilocks filter) |
| Supplemental data | 6.8k curated corpus (`candidates_deduped.parquet`) |
| Reward | Binary: R_correct (λ=0 initially; token cost penalty in late ablation) |
| Curriculum | Numerical first → expression + MCQ once stable |
| SFT | Conditional on Stage 0 probe: skip if verifier hit rate ≥15%, else 2–5k TIR demos |
| Goldilocks strategy | B+C — bucket weights + online [0.05, 0.95] filter in GRPO |
| Secondary benchmark | MATH-500 hard subset |

---

## Week 1 — 2026-03-31 to 2026-04-06

**Goal:** Prove the VeRL injection mechanism works; measure LaTeX FN rates; decide on SFT.

### Gating item: VeRL single-block injection
Implement and smoke-test the mid-sequence injection in VeRL rollout:
1. Model generates until `[/code]` stop token
2. Sandbox executes code (30s timeout; subprocess + resource limits)
3. `[output] {result}\n` injected as fixed continuation
4. Model resumes until `[answer]` stop token
5. Final `\boxed{}` extracted and verified

**This must work before any training. Everything else is blocked on this.**

### Stage 0 probe (100-problem TIR zero-shot)
- Run Qwen3.5-4B instruct (thinking OFF) with single-block TIR prompt on 100 problems
- Measure: execution success rate, verifier hit rate, per-type accuracy (numerical/expression/MCQ)
- Decision: if verifier hit rate ≥15% → skip SFT; if <15% → proceed to SFT

### Execution sandbox
- 30s timeout, subprocess isolation, resource limits
- Test on representative physics SymPy expressions

### LaTeX FN rate measurement
- Run rule verifier on training corpus (Dr. SCI + 6.8k), gold-vs-gold round-trip
- Measure per-type FN rate: numerical, expression, equation, MCQ
- This is the baseline for RQ2

### Week 1 outputs
- Working VeRL injection (smoke-tested end-to-end)
- Working execution sandbox
- Stage 0 probe results (execution success rate, verifier hit rate by answer type)
- LaTeX FN rate table by answer type
- SFT decision (go/no-go)

---

## Week 2 — 2026-04-07 to 2026-04-13

**Goal:** SFT if needed; GRPO eval loop instrumented; training data ready.

### SFT (conditional on Stage 0 result)
- If Stage 0 hit rate < 15%: generate 2–5k TIR demonstrations via GPT-4o/Claude
  - Auto-exec filter: keep only demos where code executes and verifier hits
  - Fine-tune Qwen3.5-4B on these to stabilize format
- If Stage 0 hit rate ≥ 15%: skip SFT entirely

### GRPO eval loop instrumentation
- Instrument per-type accuracy logging at checkpoints 500 / 1000 / 2000 gradient steps
- Dev set: 1k stratified holdout balanced by answer type (fixed before any training)
- Needed for RQ2 correlation analysis

### Training data weights
- Add `_train_weight` column to both parquets (Goldilocks B strategy)
- Implement online [0.05, 0.95] filter in VeRL reward wrapper (Goldilocks C strategy)

### Week 2 outputs
- SFT checkpoint (if needed) with ≥30% execution success rate on dev
- Instrumented GRPO eval loop (per-type accuracy at checkpoints)
- Both parquets with `_train_weight` column
- Fixed dev/test splits

---

## Week 3 — 2026-04-14 to 2026-04-20

**Goal:** RL environment verified end-to-end; first 4B GRPO run launched.

Tasks:
- [ ] 0.8B GRPO smoke test on numerical-only curriculum (~subset of Dr. SCI numerical)
- [ ] Debug reward pipeline: execution errors, verifier integration, reward logging
- [ ] Confirm reward signal is not sparse (target: >10% of rollouts correct on numerical)
- [ ] Launch first 4B GRPO run on numerical curriculum once environment is stable

### Week 3 outputs
- Verified RL environment (reward signal confirmed non-sparse)
- First 4B GRPO learning curve on numerical curriculum

---

## Week 4 — 2026-04-21 to 2026-04-27

**Goal:** Full curriculum running; TIR-GRPO vs CoT-GRPO comparison at checkpoints.

Tasks:
- [ ] Expand 4B GRPO to expression + MCQ types
- [ ] Compare TIR-GRPO vs CoT-GRPO at 500 / 1000 / 2000 gradient step checkpoints
  - Per-type accuracy for both conditions
  - Compute Pearson correlation: per-type LaTeX FN rate vs per-type accuracy gap
- [ ] Start writing related work + methods sections

### Week 4 outputs
- Mid-project comparison table (TIR-GRPO vs CoT-GRPO by answer type + checkpoint)
- First RQ2 correlation plot (FN rate vs accuracy gap)

---

## Week 5 — 2026-04-28 to 2026-05-04

**Goal:** Ablations and OOD eval done; behavioral analysis; results frozen.

Tasks (run training jobs in parallel — GPU-abundant):
- [ ] Cost penalty ablation: λ > 0 vs λ = 0
- [ ] Curriculum ablation: numerical-first vs full mixed from start
- [ ] Scaling ablation: 0.8B vs 4B TIR-GRPO accuracy
- [ ] OOD eval: MATH-500 hard subset
- [ ] Behavioral analysis (100–150 sampled outputs per condition):
  - Strategy categorization: direct compute, unit conversion scaffolding, error-free single-shot
  - Failure modes: wrong physics setup / execution error / correct code + wrong interpretation
  - Truncation rate vs CoT baseline (target: below 22%)

### Week 5 outputs
- Ablation table
- OOD result table (MATH-500 hard)
- Behavioral analysis summary

---

## Week 6 — 2026-05-05 to 2026-05-11

**Goal:** Paper written; figures finalized.

Tasks:
- [ ] Freeze all results
- [ ] Finalize figures: per-type accuracy curves, FN rate vs accuracy gap correlation, truncation reduction
- [ ] Complete paper draft
- [ ] If TIR-GRPO does not outperform CoT-GRPO in aggregate: pivot to reward noise analysis framing (see proposal §11)

---

## Fallback Story

If TIR-GRPO does not outperform CoT-GRPO in aggregate accuracy:

> *"We measure how symbolic verifier reward noise varies across physics answer types and show this predicts differential GRPO learning failure at fixed compute budgets. Execution-based reward eliminates the noise on expression and MCQ types but not equation derivations, revealing a principled boundary for when TIR is necessary vs. sufficient for reliable RL training."*

This is a clean empirical contribution independent of aggregate accuracy improvement.

---

## Non-Negotiables

- Do not start GRPO training before VeRL injection is smoke-tested
- Do not skip the Stage 0 probe — it gates the SFT decision
- Do not skip per-type accuracy instrumentation — it is the core of RQ2
- Do not skip CoT-GRPO baseline run — it is the primary comparison
- Do not leave secondary benchmark (MATH-500 hard) unrun

---

## Relationship to Existing Infrastructure

- `src/phys_reasoner/verifier/` — answer verification pipeline (unchanged; used as GRPO reward)
- `data/processed/drsci_physics_clean.parquet` (107,158 rows) — primary training corpus
- `data/processed/candidates_deduped.parquet` (6,866 rows) — supplemental training corpus
- `data/results/rescore_7b_v3.parquet` — authoritative CoT baseline (39.7% pass@1 with xV-7B)
