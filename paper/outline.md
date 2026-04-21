# Outline — PhysCode, ICML 2026 AI4Physics workshop

**Status:** v0 — first pass, awaiting review
**Page budget:** 8 pages ICML 2-column (references excluded)

---

## Thesis (one sentence)

Training Qwen3.5-4B with single-block tool-integrated RLVR on ~108k physics problems yields higher accuracy on a suite of physics benchmarks than a strict CoT-GRPO baseline matched in data, training budget, and reward stack, with gains concentrated on expression-type answers and harder problems.

## Contributions (3, in priority order)

1. **Empirical.** First controlled TIR-GRPO vs. CoT-GRPO comparison for physics at RL training level, across five physics benchmarks. Matched data, steps, reward.
2. **Behavioral.** Per-answer-type decomposition showing where TIR gains concentrate (expression-type, hard problems) and where it does not (equation derivation remains open).
3. **Recipe.** Single-block TIR with ScaleRL-style think-interrupt implemented inside VeRL for Qwen3.5 hybrid attention, plus the Dr.GRPO + DAPO-lite training configuration that made it stable. Reproducible artifact.

---

## Page budget (aim)

| § | Pages | Net content |
|---|---|---|
| Abstract | — | ~180 words |
| 1. Introduction | 1.0 | Motivation → problem → approach → contributions; Figure 1 hero. |
| 2. Related Work | 0.5 | RL for reasoning, TIR, physics benchmarks, verifiers. |
| 3. Method | 1.75 | Trajectory, rollout mechanism, training algorithm. Figure 2. |
| 4. Experimental Setup | 1.0 | Data, benchmarks, baselines, metrics, training config. |
| 5. Results | 2.25 | Main benchmark table + per-type + output characterization + truncation. |
| 6. Analysis | 0.75 | Solution-strategy qualitative + failure modes. |
| 7. Discussion | 0.5 | Scope, limitations, reward-noise hypothesis as future work. |
| 8. Conclusion | 0.25 | — |
| — | **8.0** | — |
| References | — | On overflow page. |
| Appendix | — | Hyperparameters, ablations, extended prompts, failure examples. |

Pressure valve: if tight, collapse Analysis into Results; cut Related Work to ~0.4.

---

## §1 Introduction — beats

- **Hook:** Physics problems stress symbolic verifiers in ways that pure-math tasks do not — mixed answer types (equations, expressions with units, dimensional prose) make gold-vs-prediction comparison unreliable.
- **Why RLVR for physics is a story worth telling:** RLVR has driven recent progress in math reasoning (DeepSeek-R1, DAPO, Dr. GRPO, Dr. SCI). Physics lags. Part of the gap is that reliable reward signal on physics is harder to obtain.
- **What we do:** train Qwen3.5-4B with single-block tool-integrated RLVR, where the model reasons, executes one Python/SymPy block, and returns a final boxed answer. Compared to strict CoT-GRPO.
- **What we find:** (placeholder numbers) aggregate gain on benchmark suite; gains concentrated on expression and harder problems.
- **Why it matters for the workshop:** tool-augmented agents is a named direction. Our setup is a concrete, reproducible instance.
- **Contribution bullets** (3, as above).
- **Figure 1** (hero): side-by-side CoT vs TIR trajectory on one example problem with annotation.

## §2 Related Work — beats

**Target length:** 0.5 page. Four short paragraphs, ~3-5 citations each.

- **RL for reasoning with verifiable rewards.** DeepSeek-R1, GRPO, DAPO, Dr. GRPO, Dr. SCI. One sentence on each; emphasize that most work is on math or code, with physics under-studied.
- **Tool-integrated reasoning.** ToRA, MathCoder, SimpleTIR, GTPO, NeMo TIR. Note: most are SFT + inference-time tool use; few do RL with tool integration in the rollout. Ours is single-block TIR at RL training time.
- **Physics reasoning benchmarks.** UGPhysics, PHYBench, OlympiadBench, SciBench, PhysReason, Dr. SCI corpus. One line on each; note limited RL training work on these datasets.
- **Verification for open-ended answers.** math-verify, xVerify, rule-based symbolic checkers. Frame: we use rule + xVerify as the reward stack; this is standard, not our contribution.

**Stylistic note:** workshop related-work should be brief and orienting, not exhaustive.

## §3 Method — beats

### 3.1 Single-block TIR trajectory

- Schema: `<think>reasoning</think> <tool_call>code</tool_call> <tool_response>output</tool_response> <think>interpret</think> \boxed{answer}`
- Why single-block: execution-budget discipline, token-length control, reproducibility. Multi-block deliberately out of scope.
- Qwen3.5-4B specifics: `enable_thinking=True` in phase 1 (reasoning leaks into code comments without), `enable_thinking=True` also in phase 2 (matches VeRL's ToolAgentLoop). Brief.

### 3.2 Rollout mechanism (think-interrupt + sandbox)

- Think-interrupt (ScaleRL-style): if model is still inside `<think>` at thinking budget (12288 tokens), inject an interrupt phrase with mask=0 and resume generation for tool call.
- Sandbox: 30 s timeout, subprocess isolation, import allowlist (SymPy, NumPy, standard library).
- Format markers: `[code]…[/code]` as stop token; `[output]` as injected continuation marker; `[answer]` format marker only (not a stop token).
- **Figure 2:** rollout flow diagram.

### 3.3 Training algorithm

- Dr. GRPO (no length norm, no std norm) + DAPO-lite (clip-higher, token-mean loss aggregation, overlong-response shaping).
- KL off (DAPO canonical), so zero-advantage groups contribute exactly zero to the loss.
- Reward: binary `R_correct` on final `\boxed{}` via rule + xVerify-7B on dedicated GPU. No length penalty in main run (ablated later).
- One sentence each on why these choices are defensible (refer to papers).

## §4 Experimental Setup — beats

### 4.1 Training data

- Dr. SCI clean (~101k rows, Chinese-origin university physics translated) + curated corpus (~6.8k rows from SciBench, UGPhysics, OlympiadBench, PHYBench, PHYSICS).
- Train/dev/test splits stratified by (answer type × source × difficulty). Details in appendix.
- Total training pool: ~108k problems, ~2k dev, ~2k test in-domain.

### 4.2 Benchmarks

- **In-domain held-out:** Dr. SCI test + corpus test.
- **External physics:** UGPhysics, PHYBench, OlympiadBench, SciBench (physics subset). These were moved out of training pool to serve as held-out benchmarks.
- (Stretch) critpt — research-style physics problems. Included only if results arrive in time.

### 4.3 Baselines and models

- **Base Qwen3.5-4B** (instruct, thinking on): zero-shot reference.
- **TIR zero-shot Qwen3.5-4B:** single-block TIR prompt, no RL training.
- **CoT-GRPO (strict):** same VeRL setup, same pool, same steps, same rule+xVerify reward. Only difference: no code tool.
- **TIR-GRPO (ours):** same as CoT-GRPO + single-block tool rollout.

### 4.4 Metrics

- Accuracy (pass@1, xVerify-7B as judge) on each benchmark.
- Per-answer-type accuracy on in-domain test.
- Truncation rate (% rollouts hitting response cap without a boxed answer).
- Tool-execution success rate (TIR only).

### 4.5 Training config

- 4×A100-80G trainer + 5 A100 rollout nodes, VeRL `fully_async_policy`.
- `rollout.n=8`, `ppo_mini_batch_size=128`, 1500 steps (≈ 1.8 epochs of the pool).
- Table with key hyperparameters; full config in appendix.

## §5 Results — beats

### 5.1 Main benchmark results (headline)

- **Table 1:** rows = {base, TIR zero-shot, CoT-GRPO, TIR-GRPO}; cols = {in-domain, UGPhysics, PHYBench, OlympiadBench, SciBench, (critpt), macro-avg}.
- One paragraph interpretation: aggregate gain of TIR-GRPO over CoT-GRPO, consistency across benchmarks.

### 5.2 Per-answer-type decomposition

- **Table 2:** in-domain accuracy by answer type (numerical, expression, MCQ, equation, multi-part, interval) for {base, CoT-GRPO, TIR-GRPO}. Column for TIR−CoT delta.
- **Figure 3:** learning curves per answer type (2×2 or 2×3 panel) for CoT-GRPO and TIR-GRPO over training steps on dev set.
- Interpretation: where the gain concentrates.

### 5.3 Output characterization

- Rule verifier coverage: % of xVerify-correct outputs that rule verifier also accepts, per type, for TIR vs CoT.
- Tool-execution success rate across training.
- Code complexity (lines, SymPy usage) distribution.

### 5.4 Truncation reduction

- **Figure 4:** truncation rate vs training step for TIR and CoT; response length distribution.
- Target result: TIR uses shorter responses on hard problems, fewer truncated rollouts.

## §6 Analysis — beats

- 100–150 sampled outputs per {TIR-GRPO, CoT-GRPO}, labeled by hand.
- Solution strategy categories (qualitative): direct-compute, unit-conversion scaffolding, multi-step derivation with partial code, error-free single-shot.
- Failure modes: wrong physics / execution error / correct code but wrong interpretation / truncation.
- **Table 3:** strategy and failure-mode frequencies by condition.
- Representative example walkthroughs (2-3 in body, more in appendix).

## §7 Discussion — beats

- **Scope:** when is TIR necessary vs. sufficient? It works where the solution is *reachable by one computational step*; equation-derivation and geometric-intuition problems are still open.
- **Limitations:** single-block constraint; dependency on SymPy/NumPy capability; 4B scale; Chinese-origin training data translated to English.
- **Future work / open hypothesis (one paragraph):** rule verifier coverage on TIR outputs is substantially higher than on CoT outputs; this suggests the RL reward signal may be less noisy for TIR, but isolating the reward-quality effect from problem-difficulty requires a rule-only-vs-rule+xVerify ablation we leave to future work. (This is where the verifier-noise idea lives — honestly framed, not oversold.)

## §8 Conclusion — beats

- Restate the single claim.
- Position: a reproducible recipe + honest evaluation of where tool use helps physics RLVR.
- One sentence on follow-up direction.

---

## Appendix plan (no page limit, but don't inflate)

- A. Hyperparameter tables (training, sampling, reward)
- B. Data pipeline details (cleaning, splits, stratification)
- C. Think-interrupt implementation (VeRL patch summary)
- D. Sandbox details (timeout, resource limits, allowlist)
- E. Additional benchmark results (per-type per-benchmark)
- F. Ablations (cost penalty, think-interrupt on/off, 0.8B scaling) — run only if time permits
- G. Extended failure-mode examples
- H. Prompt templates

---

## Open decisions / TODO before prose drafting

- [ ] Confirm exact list of external benchmarks in main Table 1 (stretch: critpt)
- [ ] Confirm whether CoT-GRPO uses the same thinking budget as TIR-GRPO
- [ ] Decide whether 0.8B scaling ablation runs (compute-dependent)
- [ ] Resolve whether "equation-derivation" goes in main body or appendix (data-dependent on how many expression-type problems we actually have)
- [ ] Pull 2–3 sample workshop papers for style reference → `bibliography/sample-abstracts/`
