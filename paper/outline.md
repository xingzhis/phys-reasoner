# Outline — PhysCode, ICML 2026 AI4Physics workshop

**Status:** v1 — reframed around tool-use miscalibration finding (Apr 19)
**Page budget:** 8 pages ICML 2-column (references excluded)

---

## Thesis (one sentence, locked)

Zero-shot tool use on a reasoning-tuned base (Qwen3-4B-Thinking-2507) is miscalibrated: tools are invoked on problems where they hurt and skipped on problems where they would help. We show that tool-integrated RLVR recalibrates this decision, lifting call-conditional pass rates and outperforming a strict CoT-GRPO baseline matched in data, training budget, and reward stack — with gains concentrated on expression-type answers where zero-shot triage is worst.

## Contributions (3, in priority order)

1. **Empirical finding.** Base-model tool use is miscalibrated zero-shot (quantified by call-conditional vs. skip-conditional pass@1 across five physics benchmarks and six answer types); TIR-GRPO recalibrates it.
2. **Controlled comparison.** First apples-to-apples TIR-GRPO vs. CoT-GRPO at matched RL training budget on physics, across five benchmarks. Per-type decomposition identifies where tool use helps (expression, numerical-on-hard) and where it doesn't (short numerical, pure-reasoning equation derivation).
3. **Recipe.** Single-block TIR with ScaleRL-style think-interrupt inside VeRL for Qwen3-Thinking, plus the Dr.GRPO + DAPO-lite training config that made it stable. Reproducible artifact.

---

## Page budget (aim)

| § | Pages | Net content |
|---|---|---|
| Abstract | — | ~180 words |
| 1. Introduction | 1.0 | Miscalibration hook → RL as recalibration → contributions; Figure 1 hero. |
| 2. Related Work | 0.5 | RL for reasoning, TIR, physics benchmarks, verifiers. |
| 3. Method | 1.5 | Trajectory (conceptual), rollout mechanism, training algorithm. Figure 2. Details → appendix. |
| 4. Experimental Setup | 1.0 | Data, benchmarks, baselines, metrics, training config. |
| 5. Results | 2.5 | Zero-shot miscalibration table + main table + per-type + call-rate-shift + output characterization. |
| 6. Analysis | 0.75 | Qualitative strategies + failure modes. |
| 7. Discussion | 0.5 | Scope, limitations, reward-noise hypothesis as future work. |
| 8. Conclusion | 0.25 | — |
| — | **8.0** | — |
| References | — | On overflow page. |
| Appendix | — | Format specs, hyperparameters, ablations, extended prompts, failure examples. |

Pressure valve: collapse Analysis into Results; cut Related Work to ~0.4; drop Figure 2 if tight.

---

## §1 Introduction — beats

- **Hook:** Strong reasoning-tuned LLMs reach for symbolic-computation tools on ~30–80% of physics problems zero-shot, but on many benchmarks tool calls are as-or-less accurate than skipping the tool. Tool availability ≠ productive tool use.
- **Why this is a physics RL story:** physics answer types are heterogeneous (numerical, expression, equation, MCQ, interval). Tool usefulness is per-type. RL should teach the model *when* and *how* to use the tool.
- **What we do:** train Qwen3-4B-Thinking-2507 with single-block tool-integrated RLVR on a curriculum-filtered physics training pool; compare to strict CoT-GRPO at matched budget.
- **What we find:** (placeholder) call-conditional pass rates rise; overall accuracy improves across benchmarks; gains concentrate on the expression/equation types where zero-shot triage was worst.
- **Why it matters for the workshop:** tool-augmented agents is a named direction. The miscalibration finding is a concrete instance of a failure mode the CFP names (tool use reliability).
- **Contribution bullets** (3, as above).
- **Figure 1** (hero): three-panel sketch — (a) a representative problem where base calls tool and gets it wrong, (b) the same problem where base skips and gets it right, (c) TIR-GRPO correctly triages. Or a single "call-path vs skip-path pass@1" bar chart across benchmarks, zero-shot.

## §2 Related Work — beats

**Target length:** 0.5 page. Four short paragraphs, ~3–5 citations each.

- **RL for reasoning with verifiable rewards.** DeepSeek-R1, GRPO (DeepSeekMath), DAPO, Dr. GRPO, Dr. SCI. One sentence on each; emphasize math/code focus with physics under-studied.
- **Tool-integrated reasoning.** ToRA, MathCoder, PAL, SimpleTIR, GTPO. Most are SFT + inference-time tool use; few do RL with tool integration in the rollout; none report tool-use calibration dynamics during training on physics.
- **Physics reasoning benchmarks.** UGPhysics, PHYBench, OlympiadBench, SciBench, Dr. SCI corpus, (PhysReason, PHYSICS). One line each; note limited RL training work on these datasets.
- **Verification for open-ended answers.** math-verify, xVerify. Frame: we use rule + xVerify as the reward stack; this is tooling, not our contribution.

**Stylistic note:** workshop related-work is brief and orienting, not exhaustive.

## §3 Method — beats

### 3.1 Single-block TIR trajectory (conceptual)

- Schema: thinking → one tool call → tool output → interpretation → boxed answer. Exact token-level format in appendix.
- Design choice: single-block for budget discipline, token-length control, reproducibility. Multi-block deliberately out of scope.
- Base model: Qwen3-4B-Thinking-2507 with `enable_thinking=True` throughout; hermes tool-call format native to the chat template.

### 3.2 Rollout mechanism (conceptual)

- **Think-interrupt** (ScaleRL-style): if thinking budget is exhausted without closing `</think>`, inject a short interrupt phrase with mask=0 and resume generation for the tool call. Prevents runaway reasoning from starving the tool-call phase. Exact phrase and implementation details in appendix.
- **Sandbox:** subprocess-isolated Python with SymPy + NumPy + stdlib, 30 s timeout.
- **Figure 2:** rollout flow diagram (4–5 boxes: think → interrupt? → tool → output → interpret → boxed).

### 3.3 Training algorithm

- Dr. GRPO (no length norm, no std norm) + DAPO-lite (clip-higher, token-mean loss aggregation, overlong shaping). KL off so zero-advantage groups contribute zero to loss.
- Reward: binary `R_correct` on final boxed answer via rule + xVerify-7B.
- One sentence each on why these choices are defensible; cite Dr. GRPO and DAPO.

## §4 Experimental Setup — beats

### 4.1 Training data

- **TODO:** lock numbers — difficulty-filtered subset of Dr. SCI + curated corpus (numerical / expression / MCQ / equation / multi-part / interval).
- Stratified train/dev/test splits.

### 4.2 Benchmarks

- **In-distribution test slices (pool_v2):** drsci, physics, ugphysics, scibench. Drawn from the training pool; held out from training via the split.
- **External physics (held out entirely from training):** OlympiadBench (OE_TO subset), PHYBench, ABench-Phy A, ABench-Phy B. Each judged by the benchmark-provided scorer where available; rule+xVerify elsewhere.
- (Stretch) critpt — research-style physics problems. Only if eval pipeline finishes in time.

### 4.3 Baselines and models

- **Zero-shot CoT (Qwen3-4B-Thinking-2507):** thinking-on, no tool. Already evaluated.
- **Zero-shot TIR (Qwen3-4B-Thinking-2507):** thinking-on with tool available. Already evaluated — this is where we observe the miscalibration.
- **CoT-GRPO (strict baseline):** same VeRL setup, same pool, same steps, same rule+xVerify reward. Only difference: no code tool.
- **TIR-GRPO (ours):** same as CoT-GRPO + single-block tool rollout.

### 4.4 Metrics

- **Pass@1** (xVerify-7B as judge, or benchmark-native scorer when provided).
- **Call rate** — fraction of rollouts that invoke the tool (TIR only).
- **Call-conditional pass@1 / skip-conditional pass@1** — headline for §5.
- **Per-answer-type accuracy** on in-dist test.
- **Truncation rate** — % rollouts hitting response cap without a boxed answer.
- **Tool-execution success rate** (TIR only).

### 4.5 Training config

- Hardware: 4 A100-80G trainer nodes + 6 A100-80G rollout nodes + 1 xVerify GPU; VeRL `fully_async_policy` with `trigger_parameter_sync_step=1` (staleness-1).
- `rollout.n=8`, `ppo_mini_batch_size=128`, 1500 steps.
- Table with key hyperparameters in body; full config in appendix.

## §5 Results — beats

### 5.1 Zero-shot miscalibration (setup for the paper)

- **Table A (new, pre-RL):** per benchmark, {call%, call-conditional pass@1, skip-conditional pass@1, overall pass@1}. One row per benchmark × preset. Highlights the benchmarks where call-path is worse than skip-path.
- Per-type breakdown for pool_v2_drsci: numerical well-triaged (high call, high call-pass); expression/equation/MCQ under-called with low call-pass.
- One paragraph interpretation: tool use is already on the menu for this base model, but the decision to call is miscalibrated on most benchmarks.

### 5.2 Main benchmark results (headline)

- **Table 1:** rows = {zero-shot CoT, zero-shot TIR, CoT-GRPO, TIR-GRPO}; cols = {pool_v2_drsci, pool_v2_physics, pool_v2_ugphysics, pool_v2_scibench, OlympiadBench, PHYBench (EED), ABench-A, ABench-B per-mid, macro-avg}.
- TIR-GRPO vs CoT-GRPO delta is the claim-carrying number.

### 5.3 Recalibration: call rate and call-conditional accuracy shift

- **Figure 3 (new):** call rate × call-conditional pass@1 scatter, one point per (benchmark × type), before and after RL. Arrows from zero-shot to post-RL show the recalibration direction.
- Per-type analysis: which types saw call-rate increase, which saw call-path accuracy increase, which saw both.

### 5.4 Per-answer-type decomposition

- **Table 2:** in-dist accuracy by answer type for {zero-shot, CoT-GRPO, TIR-GRPO}. Column for TIR−CoT delta.
- **Figure 4:** learning curves per answer type (2×3 grid) for CoT-GRPO and TIR-GRPO over training steps on dev.

### 5.5 Output characterization

- Tool-execution success rate across training.
- Code complexity distribution (compact SymPy lines vs scaffolded numerical setup).
- Note: rule verifier coverage on TIR vs CoT outputs — brief; full table in appendix.

### 5.6 Truncation reduction (optional, cut if space tight)

- **Figure 5:** truncation rate vs step; response-length CDF at final checkpoint.

## §6 Analysis — beats

- 100–150 sampled outputs per {TIR-GRPO, CoT-GRPO}, labeled (LLM-assisted first pass + spot check).
- Solution strategies: direct-compute, unit-conversion scaffolding, multi-step derivation with partial code, error-free single-shot.
- Failure modes: wrong physics / execution error / correct code but wrong interpretation / truncation.
- **Table 3:** strategy and failure-mode frequencies by condition.
- 2–3 example walkthroughs in body.

## §7 Discussion — beats

- **Scope of the recalibration finding:** works where the solution is reachable by one computational step; pure symbolic derivation and geometric-intuition problems show smaller gains.
- **Limitations:** single-block constraint; SymPy/NumPy dependency; 4B scale; hermes tool-call format; training data source composition.
- **Future work / open hypothesis:** rule verifier coverage on TIR outputs may be higher than on CoT outputs, consistent with a cleaner reward signal for TIR. Isolating the reward-quality effect from problem difficulty requires a rule-only vs rule+xVerify reward ablation, which we leave to future work. (Park for the verifier-noise idea.)

## §8 Conclusion — beats

- Restate the miscalibration + recalibration claim.
- Position: reproducible recipe + honest evaluation of where tool use helps physics RLVR.
- One sentence on follow-up direction.

---

## Appendix plan (no page limit, but don't inflate)

- A. Training config + hyperparameter tables
- B. Data pipeline details (cleaning, splits, stratification, filter rules)
- C. TIR trajectory format, exact token-level markers, think-interrupt phrase
- D. VeRL think-interrupt patch summary
- E. Sandbox details (timeout, resource limits, allowlist)
- F. Per-benchmark × per-type tables (full expansion of Table 2)
- G. Rule verifier coverage table (TIR vs CoT)
- H. Extended failure-mode examples
- I. Prompt templates (TIR system prompt, CoT system prompt, tool schema)
- J. Ablations if any (cost penalty, think-interrupt on/off, 0.8B) — only if run

---

## Open decisions / TODO before prose drafting

- [ ] Lock training data size + composition (user to provide)
- [ ] Confirm final benchmark list and scorer per benchmark
- [ ] Paper title + author list + affiliations
- [ ] Pull 2–3 sample workshop papers → `bibliography/sample-abstracts/`
- [ ] Decide critpt inclusion (Apr 22 cut-off)
