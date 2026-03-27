# Standalone Execution Checklist: Adaptive Compute Routing with Verification-Oriented Tool Use

## Immediate Decisions

- Main model: **Qwen3-4B-Base**.
- Debug model: **Qwen3-0.6B**.
- Primary dataset: cleaned physics corpus (~6k).
- Secondary benchmark: **MATH-500 hard subset**.
- Actions: **Answer / Check / Think-Deep / Tool-Check**.
- Reward: \(R = R_{\text{correct}} - \lambda C_{\text{action}}\).
- Tools: restricted verification-only wrappers.

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

### SFT data

- Sample 200–500 verifier-friendly items for Check examples.
- Generate or template structured Check demonstrations.
- Create tool-check demonstrations using only wrapper calls.
- Validate final-answer formatting.

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

### SFT training

- Train a small SFT model on schema compliance and action-conditioned behavior.
- Evaluate parse success before any RL.

### Week 2 outputs

- Working parser.
- Working restricted tool API.
- SFT checkpoint with stable schemas.

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
