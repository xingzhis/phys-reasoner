# Plan: Adaptive Compute Routing with Verification-Oriented Tool Use

**Status:** Active — Week 1 in progress
**Started:** 2026-03-27
**Target completion:** 2026-05-07
**Proposal:** `docs/standalone_proposal_v2_5_2.md`
**Checklist:** `docs/implementation_checklist_v2_5_2.md`

---

## Project Summary

Train a small open model (Qwen3.5-4B) to route each physics problem to one of four compute modes — **Answer**, **Check**, **Think-Deep**, **Tool-Check** — using SFT + RL with a correctness-minus-cost reward. Core claim: a learned routing policy achieves a better accuracy-cost frontier than fixed strategies or heuristic routers, and exhibits predictable structure (symbolic → Tool-Check, long derivations → Think-Deep, easy → Answer, medium → Check).

---

## Key Decisions (Frozen)

| Item | Decision |
|---|---|
| Main model | Qwen3.5-4B (base vs instruct TBD — decide before SFT; see proposal §5.1 for tradeoffs) |
| Debug model | Qwen3.5-0.6B |
| Primary dataset | Cleaned physics corpus (~6k, `data/processed/candidates_deduped.parquet`) |
| Supplemental data | Dr. SCI physics + rule-verifiable subset (MiniByte-666/Dr.SCI on HF) |
| Secondary benchmark | MATH-500 hard subset |
| Actions | Answer / Check / Think-Deep / Tool-Check |
| Reward | R = R_correct − λ·C_action |
| Tools | Restricted verification wrappers: check_equation, check_units, plug_values, check_root, compare_expr |

---

## Week 1 — 2026-03-27 to 2026-04-02

**Goal:** Dr. SCI investigation underway, corpus decision imminent, splits frozen.

**Dr. SCI investigation** (see `.claude/plans/drsci-investigation.md` for full detail):
- [x] Download and explore Dr. SCI — 115,497 physics + match_rule=True rows saved to `data/processed/drsci_physics.parquet`
- [ ] Phase 1: Dedup + eval contamination check → `data/processed/drsci_physics_deduped.parquet`
- [ ] Phase 1: Answer format audit (gold-gold round-trip, source stratification, English filter)
- [ ] Phase 2: Verifier round-trip + answer_type inference + unit tests (`tests/test_drsci_verifier.py`)

**Corpus decision gates on Phase 1–2 results** (target: complete by end of Week 1)

**Deferred to Week 2** (blocked by corpus decision):
- [ ] Freeze final train/dev/test splits
- [ ] Build answer extraction + correctness evaluator for all four action schemas
- [ ] Implement four fixed prompting templates (Answer, Check, Think-Deep, Tool-Check)
- [ ] Run fixed-action profiling on dev subset: accuracy, token cost, latency, failure modes
- [ ] Build heuristic router
- [ ] Decide cost definition

**Outputs:**
- Dr. SCI dedup report (contamination counts, post-dedup size)
- Verifier round-trip pass rate by source and answer type
- Preliminary corpus decision (or decision pending Phase 3 Goldilocks)

---

## Week 2 — 2026-04-03 to 2026-04-09

**Goal:** Corpus finalized, SFT data ready, parser working, tool wrappers working, SFT checkpoint stable.

**Dr. SCI investigation continued** (see `.claude/plans/drsci-investigation.md`):
- [ ] Phase 3: Zero-shot Goldilocks profiling — stratified 600-row sample, pass@1 by difficulty/source (sbatch)
- [ ] Phase 3: Analyze Goldilocks slice; estimate effective training set size
- [ ] Phase 4: Corpus decision — mixing ratio or Dr. SCI primary; freeze splits
- [ ] Phase 4: If Dr. SCI primary — add `DrSCI` source to schema, write loader, update dedup pipeline

**SFT pipeline:**
- [ ] Integrate finalized corpus (after Week 1/2 decision)
- [ ] Sample 200–500 verifier-friendly items for Check demonstrations
- [ ] Generate/template structured Check demonstrations with required schema fields
- [ ] Create Tool-Check demonstrations using only restricted wrapper calls
- [ ] Validate final-answer formatting across all four action types
- [ ] Implement Check parser (required fields, known check type, non-empty check work, reject generic reflective text)
- [ ] Implement restricted tool wrappers: check_equation, check_units, plug_values, check_root, [compare_expr]
- [ ] Train SFT checkpoint (Qwen3.5-4B, 10k–30k examples, action-conditioned formatting)
- [ ] Evaluate parse success on held-out examples before any RL

**Outputs:**
- Working Check parser
- Working restricted tool API (built on existing SymPy/pint verifier)
- SFT checkpoint with stable schemas

---

## Week 3 — 2026-04-10 to 2026-04-16

**Goal:** RL environment verified end-to-end; first 4B RL runs launched.

Tasks:
- [ ] Run 0.6B smoke-test RL (confirm reward logging, cost logging, parser logging, tool-call lifecycle)
- [ ] Debug malformed Check and Tool-Check outputs
- [ ] Launch first 4B RL runs after environment stability confirmed

**Outputs:**
- Verified RL environment
- First routing learning curves

---

## Week 4 — 2026-04-17 to 2026-04-23

**Goal:** RL vs baselines comparison; early routing pattern analysis.

Tasks:
- [ ] Compare RL router to fixed baselines and heuristic/classifier router
- [ ] Inspect action usage by problem subtype and difficulty bucket
- [ ] Check whether early routing trends match the falsifiable prediction (symbolic → Tool-Check, long → Think-Deep, easy → Answer, medium → Check)
- [ ] Build classifier router if not done in Week 1

**Outputs:**
- Mid-project comparison table
- Early routing-pattern plots

---

## Week 5 — 2026-04-24 to 2026-04-30

**Goal:** Ablations complete, secondary benchmark done, results frozen.

Tasks:
- [ ] Run ablations: no cost penalty, no Check, no Tool-Check, RL vs classifier router
- [ ] Run MATH-500 hard subset evaluation
- [ ] Freeze metrics and figures

**Outputs:**
- Final result tables
- Secondary benchmark result table
- Ablation table

---

## Week 6 — 2026-05-01 to 2026-05-07

**Goal:** Paper written.

Tasks:
- [ ] Write paper
- [ ] Create final figures (accuracy-cost tradeoff curve, routing distribution plots)
- [ ] Produce routing-analysis section centered on the falsifiable prediction
- [ ] If RL underperforms, pivot to systems paper framing (routing analysis + strong baselines)

---

## Fallback Story

If RL fails to beat heuristic routing: publish a careful systems study of adaptive routing for scientific reasoning — showing which extra-compute modes help which problem types, and why structure-aware heuristic routing is a strong baseline.

---

## Non-Negotiables

- Do not expand tool use beyond verification wrappers
- Do not add evolving rubrics to the main project
- Do not start serious 4B RL before parser and schemas are stable
- Do not treat routing analysis as optional
- Do not leave secondary benchmark unspecified

---

## Relationship to Existing Infrastructure

The existing verifier stack (`src/phys_reasoner/verifier/`) maps directly onto Tool-Check wrappers:
- `router.py` → orchestration logic
- `math_verify_wrapper.py` → check_equation / check_root
- `unit_check.py` → check_units
- `extract.py` → answer extraction for all four actions
- SymPy backend → compare_expr, plug_values

The `candidates_deduped.parquet` (~6,866 rows) is the primary training corpus.
