# Standalone Execution Checklist: Adaptive Compute Routing with Verification-Oriented Tool Use

## Immediate Decisions

- Main model: **Qwen3.5-4B** (instruct variant confirmed — see verifier experiments).
- Debug model: **Qwen3.5-0.6B**.
- Primary dataset: `candidates_deduped.parquet` (6,866) + `drsci_physics_clean.parquet` (107,158). Combined: ~114k.
- Secondary benchmark: **MATH-500 hard subset**.
- Actions: **Answer / Check / Think-Deep / Tool-Check**.
- Reward: \(R = R_{\text{correct}} - \lambda C_{\text{action}}\).
- Tools: restricted verification-only wrappers.
- **SFT stage: SKIPPED** — go directly to GRPO. See `docs/training-decisions.md`.
- **Goldilocks strategy: B + C** — bucket weights at load time + online [0.05, 0.95] filter in GRPO. See `docs/training-decisions.md`.

---

## Schemas to Finalize

### Action output schema

Every action should end with a single `FINAL_ANSWER` field.

### Check schema

Required fields:
- `ACTION: CHECK`
- `CANDIDATE_ANSWER`
- `CHECK_TYPE`
- `CHECK_WORK`
- `REVISED`
- `FINAL_ANSWER`

### Tool-Check schema

Required fields:
- `ACTION: TOOL_CHECK`
- `CANDIDATE_ANSWER`
- `TOOL_NAME`
- `TOOL_INPUT`
- `TOOL_RESULT`
- `REVISED`
- `FINAL_ANSWER`

---

## Week 1 Tasks

- Freeze data splits.
- Build answer extraction and correctness evaluator.
- Implement four fixed prompting templates.
- Run fixed-action profiling on a dev subset.
- Build heuristic router using basic structural features.
- Build classifier router if time allows.
- Decide cost definition: tokens only, latency only, or weighted combination.

### Week 1 outputs

- Baseline accuracy table.
- Baseline latency / token-cost table.
- List of top failure modes for each action.

---

## Week 2 Tasks

> **Note (2026-03-30): SFT stage removed.** No SFT data generation or SFT training.
> Week 2 focuses on templates, parser, tools, and training data weights — everything needed to
> launch GRPO directly. See `docs/training-decisions.md` for rationale.

### Training data weights

- Add `_train_weight` column to `candidates_deduped.parquet` and `drsci_physics_clean.parquet`
  using measured Goldilocks rates per `(source × answer_type)` bucket (Strategy B).
- Implement online [0.05, 0.95] pass-rate filter in verl reward wrapper (Strategy C).

### Prompting templates

- Write four fixed-action prompt templates: Answer / Check / Think-Deep / Tool-Check.
- Each template must produce a parseable structured output ending with `FINAL_ANSWER`.

### Parser

- Implement Check parser.
- Rules: required fields, known check type, non-empty check work, reject generic reflective text.
- Log parse pass/fail and reason codes.

### Tools

- Implement wrappers:
  - `check_equation`
  - `check_units`
  - `plug_values`
  - `check_root`
  - optional `compare_expr`
- Ensure deterministic outputs for logging.

### Week 2 outputs

- Both parquets with `_train_weight` column.
- Four working prompt templates.
- Working parser.
- Working restricted tool API.

---

## Week 3 Tasks

- Run 0.6B end-to-end environment tests.
- Confirm reward logging, cost logging, parser logging, and tool-call lifecycle.
- Debug malformed Check and Tool-Check outputs.
- Launch first 4B RL runs only after environment stability is confirmed.

### Week 3 outputs

- Verified RL environment.
- First routing-learning curves.

---

## Week 4 Tasks

- Compare RL router to fixed baselines and heuristic/classifier router.
- Inspect action usage by subtype.
- Check whether early routing trends match prediction:
  - symbolic -> Tool-Check
  - long derivations -> Think-Deep
  - easy -> Answer
  - medium with plausibility checks -> Check

### Week 4 outputs

- Mid-project comparison table.
- Early routing-pattern plots.

---

## Week 5 Tasks

- Run main ablations:
  - no cost penalty,
  - no Check,
  - no Tool-Check,
  - RL vs classifier router.
- Run MATH-500 hard subset evaluation.
- Freeze metrics and figures.

### Week 5 outputs

- Final result tables.
- Secondary benchmark result table.
- Ablation table.

---

## Week 6 Tasks

- Write paper.
- Create final figures.
- Produce routing-analysis section centered on the falsifiable prediction.
- Prepare fallback framing if RL does not beat heuristic routing.

---

## Non-Negotiables

- Do not expand tool use beyond verification wrappers.
- Do not add evolving rubrics to the main project.
- Do not start serious 4B RL before parser and schemas are stable.
- Do not treat routing analysis as optional.
- Do not leave the secondary benchmark unspecified.

---

## Fast Fallback Story

If RL fails to beat heuristic routing, the fallback paper is:

**A careful systems study of adaptive routing for scientific reasoning, showing which extra-compute modes help which problem types, and why heuristic structure-aware routing is a strong baseline.**
