# Standalone Research Proposal: Adaptive Compute Routing with Verification-Oriented Tool Use for Physics Reasoning

## 0. One-Paragraph Summary

This project studies whether a small open language model can learn to allocate extra test-time compute more effectively than fixed prompting or heuristic routing on physics problems with verifiable answers. The model chooses among four actions per problem: **Answer**, **Check**, **Think-Deep**, and **Tool-Check**. The outer framing is adaptive compute allocation; the inner domain-specific mechanism is verification-oriented checking. The central scientific claim is not just that selective routing can improve the accuracy-cost tradeoff, but that the learned policy will exhibit predictable structure: symbolic equation and expression-validation problems should favor **Tool-Check**, long derivations should favor **Think-Deep**, short straightforward problems should favor **Answer**, and medium-difficulty problems with cheap plausibility tests should favor **Check**.

---

## 1. Project Framing

### 1.1 Core Idea

This proposal deliberately combines two perspectives that initially looked like competing directions:

- **Outer systems framing:** adaptive allocation of extra test-time compute.
- **Inner scientific framing:** in physics reasoning, one especially valuable use of extra compute is verification-oriented checking.

This resolves the earlier tension between "compute budget" and "self-verification." The project is not about cost for its own sake, and it is not a pure self-verification paper either. Instead, it asks whether a model can learn **which type of extra computation** is worth paying for on a given problem.

### 1.2 Why Physics

Physics is a strong domain for this question because many problems are partially or fully verifiable:

- equations can be checked by substitution,
- expressions can be simplified or compared symbolically,
- units can be validated,
- numerical answers can be sanity-checked,
- derivations can be assessed for intermediate consistency.

That makes it natural to study a routing policy that decides between direct answering, deeper reasoning, internal structured checking, and restricted external tool-based verification.

### 1.3 Why Verification-Oriented Tools Instead of General Tools

The tool action is intentionally restricted to **verification-oriented** operations rather than open-ended solving.

Reasons:

1. It keeps the project scientifically coherent: the tool is an extra-compute mechanism for grounded checking, not a separate general agent.
2. It makes the action space interpretable: one action is deeper reasoning, one is internal checking, and one is external grounded verification.
3. It keeps scope manageable for a 6-week effort.
4. It makes the evaluation cleaner, because the tool action is not simply a second unrestricted solver.

This restriction is a scope choice, not a claim that direct tool-assisted solving is unimportant. Direct tool-assisted solving is a reasonable future extension, but it is intentionally out of scope for the MVP.

---

## 2. Research Question and Hypotheses

### 2.1 Main Research Question

**Can a small open model, trained with verifiable rewards, learn to route each physics problem to the right computation mode—direct answer, internal structured check, deeper reasoning, or external verification-oriented tool use—better than fixed prompting or heuristic routing?**

### 2.2 Primary Hypothesis

A learned four-action policy can achieve a better accuracy-cost frontier than:

- always answering directly,
- always checking,
- always thinking deeply,
- always using tool-based verification,
- or using a strong classifier / heuristic router.

### 2.3 Secondary Hypothesis

The best action is not determined only by scalar difficulty. It also depends on **problem structure**:

- some hard problems benefit from deeper reasoning,
- some medium problems benefit from quick internal verification,
- some symbolic problems benefit from grounded tool-based checking,
- and some easy problems should be answered directly.

This is important because if action choice depends on more than generic difficulty, learned routing has a plausible chance to outperform static routing heuristics.

### 2.4 Falsifiable Routing Prediction

The proposal makes a concrete prediction about routing behavior:

- **Tool-Check** should dominate on symbolic equation-solving and expression-validation problems.
- **Think-Deep** should dominate on long multi-step derivations and conceptually dense problems.
- **Answer** should dominate on short and straightforward problems.
- **Check** should dominate on medium-difficulty problems where a cheap internal plausibility test can catch likely mistakes.

This prediction is a core scientific contribution. If the learned policy matches it, that supports the framing. If it systematically diverges in an interpretable way, that is still scientifically interesting and publishable.

---

## 3. Scope and Non-Goals

### 3.1 In Scope

- Instance-level routing among four discrete actions.
- Physics reasoning with verifiable answers.
- Restricted verification-oriented tools only.
- SFT + RL training pipeline.
- Cost-aware evaluation.
- Routing-pattern analysis as a main contribution.

### 3.2 Out of Scope

- Token-level adaptive routing.
- Open-ended agentic planning.
- Unrestricted tool use.
- Direct tool-assisted solving as a main action.
- Online evolving rubrics or meta-reward learning.
- Continual self-improving verifier loops.

### 3.3 Why the Scope Is Disciplined This Way

Earlier versions explored broader directions, including direct tool use and evolving verification rubrics. Those ideas are interesting, but they introduce too much engineering complexity for the current timeline and weaken the paper’s clarity. The final scope keeps the strongest elements while preserving a clean central claim.

---

## 4. Action Space

| Action | Meaning | Cost level | Typical use |
|---|---|---:|---|
| **Answer** | Produce final answer directly. | Lowest | Easy / high-confidence items. |
| **Check** | Produce answer, run structured internal verification, optionally revise, finalize. | Low-medium | Medium items with cheap plausibility tests. |
| **Think-Deep** | Use a longer bounded reasoning trace before answering. | Medium-high | Long derivations, conceptually difficult items. |
| **Tool-Check** | Use a restricted external symbolic/numeric/unit checker, optionally revise, finalize. | Highest | Symbolic equations, unit-sensitive problems, validation-heavy items. |

### 4.1 Check vs Tool-Check

This distinction must be crystal clear:

- **Check** is a **model-internal structured operation** with no external sandbox call.
- **Tool-Check** invokes an **external symbolic or numeric verifier** and should be used only when grounded checking cannot be done reliably enough inside the model alone.

This sentence should appear in the final paper because it preempts the most obvious reviewer objection.

### 4.2 Why "Check" Is Not Just More Chain-of-Thought

The Check action must include at least one concrete verification operation rather than vague reflection.

Allowed examples:

- substitute a proposed answer back into the governing equation,
- test unit consistency,
- check sign or magnitude plausibility,
- verify a boundary case,
- compare against a conserved quantity,
- confirm expression-form consistency.

Disallowed example:

- "Let me verify... this seems right."

### 4.3 Check Action Flow

The Check action follows this explicit flow:

1. Produce an initial candidate answer.
2. Produce at least one structured internal verification step.
3. Optionally revise the answer based on the check.
4. Emit one **final answer** field.

The system evaluates the **final answer**, not the pre-check candidate. This is important because a malformed or harmful check should be able to hurt performance, and a useful check should be able to rescue an initially incorrect answer.

If the Check output is malformed or lacks a valid structured verification step, it is still evaluated on the final answer for correctness, but it is counted as an invalid Check-format event in analysis logs.

### 4.4 Tool-Check Flow

The Tool-Check action follows this flow:

1. Produce an initial candidate answer.
2. Emit a restricted tool request using an allowed wrapper.
3. Receive tool output.
4. Optionally revise the answer based on the grounded tool result.
5. Emit one **final answer** field.

Again, evaluation is performed on the final answer.

---

## 5. Model Choice

### 5.1 Main Model

Use **Qwen3.5-4B** as the primary model. The base vs instruct variant is TBD and should be decided before SFT begins, informed by how SFT data synthesis goes.

Arguments for **base**:
1. **Cleaner routing study.** The instruct model was trained with a built-in think mode, giving it an existing prior over when to reason deeply. This could confound the routing policy — Think-Deep would not be a cleanly learned decision.
2. **Think-mode CoT traces are too long for RL training.** Techniques for premature truncation exist but add engineering complexity on the critical path.
3. **Cleaner scientific story.** Routing behavior that emerges from base is genuinely learned, not inherited from post-training.

Arguments for **instruct**:
1. Better instruction following and output formatting out of the box, reducing SFT burden.
2. If base requires extensive SFT to stabilize schemas, the cleanliness advantage may be offset.

The instruct fallback is allowed for the MVP if environment instability becomes a bottleneck, but should be explicitly framed as a compromise in the paper.

Why this model family:

- small enough to be trainable within budget,
- large enough to support meaningful reasoning behavior,
- Qwen3.5 architecture well-supported by VeRL/GRPO pipeline.

### 5.2 Debug Model

Use **Qwen3.5-0.6B** for environment debugging, schema validation, parser development, and RL smoke tests.

This is not the main scientific model; it is an engineering accelerator.

### 5.3 Fallback

If environment instability or formatting failure becomes a bottleneck, an instruct-style fallback checkpoint is allowed for the MVP. If that happens, the paper should explicitly frame it as a pragmatic compromise rather than the cleanest causal setup.

---

## 6. Data Plan

### 6.1 Primary Dataset

Use the existing cleaned physics corpus (~6k examples) as the main training and evaluation source.

This is a central asset because it already aligns with the domain, answer formats, and verifier assumptions.

### 6.2 Supplemental Data

Add a small filtered subset of scientific reasoning data in the style of Dr. SCI, focusing only on examples that are medium-to-hard and verifier-compatible.

If compatible data are available, add a small symbolic-derivation subset to improve coverage of derivation-heavy items.

### 6.3 Secondary Benchmark

The secondary benchmark is no longer generic. Use a **MATH-500 hard subset** as the required non-physics evaluation set.

Why this choice:

- small enough to run quickly,
- recognizable to reviewers,
- easy to integrate,
- provides a clean test of whether the routing method generalizes beyond physics.

If implementation constraints make MATH-500 inconvenient, the fallback is an **AMC/AIME-style curated subset**, but the default plan should be MATH-500 hard subset.

### 6.4 Filtering Rules

Retain examples that satisfy most of the following:

- final answer is extractable,
- answer supports at least one verification path,
- problem is medium or hard enough to make routing nontrivial,
- problem can plausibly benefit from one of the four actions,
- answer format is compatible with automatic evaluation.

---

## 7. SFT Plan

### 7.1 Purpose of SFT

SFT is not mainly for teaching the model to reason from scratch. It is for stabilizing the action-conditioned interface:

- output schemas,
- final-answer formatting,
- structured Check templates,
- restricted Tool-Check API calls,
- revision flow.

Without SFT, RL will waste effort learning syntax instead of routing.

### 7.2 SFT Mixture

Target total: **10k–30k examples**.

Suggested composition:

- **70–85%** regular answer / think-deep examples from the cleaned physics corpus and filtered supplemental data.
- **5–10%** Tool-Check examples showing valid restricted tool invocation and revision.
- **5–10%** Check examples showing answer + internal structured verification + optional revision.
- **remainder** formatting / parser-stability examples.

### 7.3 Where Check Examples Come From

Check examples are not expected to exist naturally in large quantity. They must be created deliberately.

Plan:

1. Sample 200–500 verifier-compatible problems from the main dataset.
2. Generate draft Check demonstrations using a stronger teacher or scripted templates.
3. Filter for examples that contain a real structured internal check.
4. Manually inspect a small subset for quality.
5. Use these as seeds to create a few hundred to a few thousand Check-format SFT examples.

This work should be explicitly budgeted at **1–2 days in Week 2**.

### 7.4 Tool-Check Demonstrations

Tool-Check SFT demonstrations should use only the restricted wrappers, not raw arbitrary code.

Preferred wrappers:

- `check_equation(eq, candidate)`
- `check_units(expr)`
- `plug_values(expr, values)`
- `check_root(eq, candidate)`
- optional `compare_expr(expr1, expr2)`

This reduces tool-format variance and makes RL easier.

---

## 8. Tooling and Verifiers

### 8.1 Restricted Tool Set

Expose only:

- **SymPy core**: simplify, solve, substitute, factor, symbolic equivalence, numeric evaluation.
- **SymPy units or Pint**: unit compatibility and conversion.
- **Python / NumPy**: numeric plugging and sanity checks.

### 8.2 Philosophy

The tool stack is for **verification-oriented grounding**, not for open-ended planning or arbitrary program synthesis.

### 8.3 Internal Check Schema

A Check output should have a simple schema such as:

```text
ACTION: CHECK
CANDIDATE_ANSWER: ...
CHECK_TYPE: substitution | units | sign | magnitude | boundary_case | conservation | expression_form
CHECK_WORK: ...
REVISED: yes | no
FINAL_ANSWER: ...
```

This schema can be simplified further if needed, but it should preserve:

- candidate answer,
- explicit check type,
- concrete check content,
- final answer.

### 8.4 Lightweight Check Parser

The Check parser is required even in the no-shaping-reward MVP.

The parser should verify:

- presence of required fields,
- recognized check type,
- non-empty check content,
- evidence that the check content is more than generic reflective prose.

A lightweight validity heuristic is enough for MVP. It does **not** need to grade reasoning quality. It only needs to detect whether the output contains a real structured verification attempt.

Possible parser rules:

- reject if check text contains only generic phrases and no equation / units / numeric comparison / named check type,
- accept if substitution syntax, units logic, numerical bound, or explicit consistency statement appears in a structured slot,
- log parse success / failure for later analysis.

---

## 9. Reward and Training Objective

### 9.1 Main Reward

Use the simplest workable reward in the first RL run:

\[
R = R_{\text{correct}} - \lambda C_{\text{action}}
\]

Where:

- \(R_{\text{correct}}\) is the correctness reward,
- \(C_{\text{action}}\) is measured action cost, using tokens and/or latency,
- \(\lambda\) controls the cost penalty.

### 9.2 Why This Simplicity Matters

Earlier versions included verification-validity shaping. That is now intentionally optional because it increases parser burden, tuning burden, and failure surface.

### 9.3 Optional Late Ablation

Only after the main system works, optionally try:

\[
R = R_{\text{correct}} + \alpha R_{\text{valid-check}} - \lambda C_{\text{action}}
\]

But this should not be on the critical path.

### 9.4 What Gets Evaluated During Training

- correctness of final answer,
- action chosen,
- action cost,
- parser success for Check,
- tool-call success for Tool-Check.

Parser success is logged even if it is not directly rewarded.

---

## 10. Baselines

### 10.1 Fixed-Action Baselines

1. **Answer Only**
2. **Always Check**
3. **Always Think-Deep**
4. **Always Tool-Check**

### 10.2 Routing Baselines

5. **Heuristic router**
6. **Classifier router**
7. **Learned one-step RL router**

### 10.3 Why the Router Baseline Matters Most

The strongest non-RL comparator is the heuristic / classifier router. If RL cannot beat it, the paper should narrow its claim. This baseline is therefore not optional.

### 10.4 Candidate Router Features

- question length,
- equation density,
- symbolic vs numeric structure,
- unit presence,
- source dataset,
- answer-format type,
- lexical signals of derivation vs calculation.

An approximate oracle label can be derived from whichever fixed action performs best on a training item.

---

## 11. Related-Work Positioning

This project should explicitly distinguish itself from nearby families of methods.

### 11.1 Not Self-Consistency

Self-consistency-style methods improve reasoning by sampling many trajectories or answers and aggregating them. This proposal instead asks whether the model can learn **when one targeted extra-compute action is enough**, which is potentially much cheaper.

### 11.2 Not Process Reward Modeling

PRM-style approaches supervise or score many intermediate reasoning steps. This proposal does not try to supervise every step. It studies coarse-grained routing between qualitatively different computation modes.

### 11.3 Not General Tool Agents

General tool-using agents may solve tasks with arbitrary external tools. This proposal deliberately restricts tools to verification-oriented use so that the action remains interpretable and the study remains focused.

---

## 12. Evaluation Plan

### 12.1 Primary Metrics

Report:

- final answer accuracy,
- average token cost,
- wall-clock latency,
- cost per correct answer,
- action distribution,
- tool-call success rate,
- Check-parser success rate,
- accuracy by problem subtype.

### 12.2 Main Scientific Analysis

Do not treat routing analysis as an afterthought. The paper should explicitly test the routing prediction from Section 2.4.

Required plots / tables:

- action distribution by problem subtype,
- action distribution by difficulty bucket,
- action distribution by symbolic vs numeric structure,
- accuracy-cost tradeoff curve,
- confusion between heuristic router and learned router,
- parser/tool success statistics.

### 12.3 Error Analysis Questions

- When does Check help versus hurt?
- When does Tool-Check rescue wrong initial answers?
- Are there classes where Think-Deep is overused?
- Does the policy learn interpretable selectivity or noisy routing?
- Does the routing prediction hold on the secondary benchmark?

---

## 13. Concrete Experimental Protocol

### 13.1 Train / Dev / Test Discipline

Use fixed train/dev/test splits and do not update the method based on final test outcomes.

### 13.2 Fixed-Action Profiling First

Before RL, profile all four fixed actions on a dev subset to estimate:

- cost,
- latency,
- raw effectiveness,
- parser/tool failure modes.

### 13.3 Router Baseline Before RL

Build the heuristic/classifier router before expensive RL runs. This protects the project from an RL-specific failure mode and gives a publishable fallback.

### 13.4 RL After Interface Stabilization

Only start serious 4B RL once:

- output schemas are stable,
- Check parser works,
- tool wrappers work,
- fixed-action baselines are profiled.

---

## 14. Risks and Mitigations

| Risk | Severity | Mitigation |
|---|---|---|
| RL router fails to beat heuristic router | High | Build router baseline early; retain fallback paper path. |
| Check collapses into fake reflection | High | Use SFT demonstrations + lightweight parser + format logging. |
| Tool interface is brittle | Medium | Restrict wrappers and avoid arbitrary code generation. |
| Model overuses expensive actions | Medium | Use cost penalty and monitor action entropy. |
| Check examples are too weak | Medium | Budget explicit synthetic-data creation in Week 2. |
| Secondary benchmark integration slips | Medium | Commit now to MATH-500 hard subset. |
| Scope drifts toward evolving-rubric meta-learning | High | Keep rubric evolution out of the main project. |

---

## 15. Timeline

### Week 1

- Freeze scope and schemas.
- Profile fixed-action baselines.
- Build first heuristic/classifier router.
- Audit dataset splits and evaluation scripts.

### Week 2

- Finalize SFT data.
- Create Check-format demonstrations (budget 1–2 days).
- Train small SFT for action-conditioned formatting.
- Implement restricted tool wrappers.
- Build lightweight Check parser.
- Validate end-to-end Answer / Check / Think-Deep / Tool-Check formats.

### Week 3

- Run 0.6B smoke-test RL.
- Debug reward logging, parser logging, and tool-call lifecycle.
- Start first 4B RL runs once the environment is stable.

### Week 4

- Compare RL router to fixed-action and heuristic/classifier baselines.
- Inspect routing distributions and failure cases.
- Tune only minimal essentials.

### Week 5

- Run core ablations.
- Run secondary benchmark evaluation on MATH-500 hard subset.
- Freeze main results.

### Week 6

- Produce figures and tables.
- Write paper draft.
- If RL underperforms, pivot to a strong routing-systems paper emphasizing analysis and baselines.

---

## 16. Minimal Deliverables

A successful MVP should include:

- four working actions,
- stable SFT-conditioned formatting,
- restricted verification tool stack,
- lightweight Check parser,
- fixed-action baselines,
- heuristic/classifier router baseline,
- at least one successful 4B RL run,
- main evaluation on physics + secondary benchmark,
- routing-pattern analysis versus the falsifiable prediction.

---

## 17. Suggested Intro Contribution Claims

The introduction can state the contributions roughly as:

1. A four-action adaptive-compute framework for scientific reasoning that distinguishes direct answering, internal checking, deeper reasoning, and external verification-oriented tool use.
2. A restricted verification-tool design that keeps the action space grounded and interpretable.
3. A cost-aware empirical test of whether learned routing improves over fixed strategies and heuristic routers.
4. A hypothesis-driven analysis of routing behavior that tests whether the learned policy matches predicted structure across problem types.

---

## 18. Abstract Draft

We study adaptive compute routing for physics reasoning with small language models under verifiable rewards. Instead of applying a fixed strategy to every problem, our model chooses among four actions per instance: direct answer, structured internal verification, deeper reasoning, or restricted tool-based verification. We combine a compact supervised fine-tuning stage, a simple correctness-minus-cost RL objective, and verification-oriented tool interfaces to test whether learned routing improves the accuracy-cost frontier over fixed strategies and heuristic routers. Beyond aggregate accuracy, we test whether the learned policy matches our prediction that symbolic validation problems favor tool-based checking, long derivations favor deeper reasoning, and medium-difficulty problems with cheap plausibility tests favor internal checking.
