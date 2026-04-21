# Outline — PhysCode, ICML 2026 AI4Physics workshop

**Status:** v2 — claim-1-led thesis (Apr 21)
**Page budget:** 8 pages ICML 2-column (references excluded)

---

## Thesis (one sentence, locked)

Tool availability alone is not enough: giving a reasoning-tuned LLM (Qwen3-4B-Thinking-2507) access to a symbolic-computation tool fails to consistently help on physics and, on four of five in-distribution benchmarks, hurts — despite call rates of 31–89% zero-shot. We show tool-integrated RLVR bridges this gap, with TIR-GRPO outperforming a strict CoT-GRPO baseline matched in data, training budget, and reward stack, and gains concentrated on answer types where zero-shot TIR underperforms most.

## Contributions (3, in priority order)

1. **Empirical finding (zero-shot characterization).** Paired comparison of TIR-mode vs. CoT-mode on the same problems shows tool availability does not translate to benefit on a reasoning-tuned 4B model — TIR ≤ CoT on 4 of 5 in-distribution physics benchmarks despite tool use being attempted on 31–89% of problems. We characterize this per-benchmark and per-answer-type.
2. **Controlled RL intervention.** First apples-to-apples TIR-GRPO vs. CoT-GRPO at matched RL training budget on physics. TIR-GRPO closes and reverses the zero-shot gap; per-type decomposition shows gains concentrate on answer types where zero-shot TIR underperformed most.
3. **Recipe.** Single-block TIR with ScaleRL-style think-interrupt inside VeRL for Qwen3-Thinking, plus the Dr.GRPO + DAPO-lite training configuration. Reproducible artifact.

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

- **Hook:** Equipping a reasoning-tuned LLM with a symbolic-computation tool doesn't obviously help — and on 4 of 5 physics benchmarks we test, it hurts. The model does invoke the tool (31–89% of problems), but invoking it does not translate to accuracy gains.
- **Why this is a physics RL story:** Physics is where tools should shine — it is precisely the domain with dimensional analysis, symbolic manipulation, and numerical evaluation that LLMs struggle to do reliably in chain-of-thought. That the gap *narrows or reverses* there is the surprise. RL must teach not just *when* but *how* to use the tool productively.
- **What we do:** train Qwen3-4B-Thinking-2507 with single-block tool-integrated RLVR on a curriculum-filtered physics training pool; compare against strict CoT-GRPO at matched training budget.
- **What we find:** (placeholder) TIR-GRPO closes the zero-shot TIR-vs-CoT gap and surpasses CoT-GRPO across benchmarks. Gains concentrate on answer types where zero-shot TIR was worst.
- **Why it matters for the workshop:** tool-augmented agents is a named direction. The zero-shot finding alone (tool availability isn't enough) quantifies a known-but-underreported failure mode for reasoning-tuned LLMs on physics.
- **Contribution bullets** (3, as above).
- **Figure 1** (hero): paired bar chart — for each in-dist benchmark, two bars (TIR-mode, CoT-mode) zero-shot, plus two matching bars after RL (TIR-GRPO, CoT-GRPO). The visual story: zero-shot pair shows TIR ≤ CoT; post-RL pair shows the reversal.

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

### 5.1 Zero-shot: tool availability ≠ tool benefit (sets up the paper)

- **Table A (headline):** paired TIR-mode vs CoT-mode zero-shot on each benchmark. Columns: benchmark, n, TIR pass@1, CoT pass@1, Δ(TIR−CoT), TIR call rate. One row per benchmark. Δ column is the story — negative on 4/5 in-dist slices.
- **Per-problem paired comparison** (once Task 1b output lands): for each problem, is tool mode "Win" (correct when CoT wrong), "Loss" (wrong when CoT correct), "Tie"? Report Win/Loss/Tie counts per benchmark and per answer type — **this is the strongest form of the evidence, no selection bias**.
- One paragraph interpretation: the model attempts tool use at 31–89% rates but on most benchmarks the attempts do not help or hurt aggregate accuracy.

### 5.2 Within-TIR zoom-in: call-path vs skip-path (with caveat)

- **Table 2a (secondary):** within TIR mode, per benchmark, {call%, call-path pass@1, skip-path pass@1, overall}. Caveat: the model's choice of when to call is not random, so call-path and skip-path populations differ in problem difficulty. The pass-rate difference therefore cannot be read as "tool harms" or "tool helps" per call; it characterizes the model's current triage against the reality of which problems it succeeds on.
- Per-type breakdown for pool_v2_drsci: where is call rate high and where is it low. Numerical has high call rate (76%) with high overall pass (82%); expression/equation/MCQ have lower call rates (25–35%) with lower-to-moderate overall pass (48–67%).

### 5.3 Main benchmark table with RL

- **Table 1:** rows = {base CoT zero-shot, base TIR zero-shot, CoT-GRPO, TIR-GRPO}; cols = in-dist slices + external benchmarks + macro-avg. This is the claim-carrying table.
- Interpretation: how much of the zero-shot TIR-vs-CoT gap does RL close? Does TIR-GRPO exceed CoT-GRPO?

### 5.4 Per-answer-type decomposition

- **Table 2:** in-dist accuracy by answer type for {zero-shot CoT, zero-shot TIR, CoT-GRPO, TIR-GRPO}. Columns include Δ(zero-shot TIR−CoT) and Δ(TIR-GRPO−CoT-GRPO). Predicts: types with most-negative zero-shot Δ become types with most-positive post-RL Δ.
- **Figure 3:** per-type learning curves on dev for CoT-GRPO and TIR-GRPO.

### 5.5 Tool-use dynamics during training

- **Figure 4:** TIR-GRPO's call rate and call-path pass rate over training steps, overlaid on dev set. Goal: show which axis moves — call rate, call-path pass, or both.
- Collapse check: was there a run where call rate → 0 or → 1? Discuss briefly.

### 5.6 Output characterization (short)

- Tool-execution success rate across training.
- Code complexity distribution (compact SymPy vs scaffolded numerical setup).
- Note: rule verifier coverage on TIR vs CoT outputs — one-line mention; full table in appendix.

### 5.7 Truncation reduction (optional, cut if space tight)

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
