# PhysCode: Tool-Integrated Reasoning for Physics Problem Solving via RLVR

## 0. Summary

We train a small open language model (Qwen3.5-4B) with RLVR to solve verifiable physics problems using tool-integrated reasoning (TIR): the model reasons, executes exactly one Python/SymPy code block, receives the output, and reasons to a final answer. Code execution provides a deterministic, low-noise reward signal — directly addressing the symbolic verification floor that degrades CoT-GRPO on physics problems with expression and equation answers. Unlike existing TIR-GRPO work (SimpleTIR, GTPO) which targets math benchmarks, we study how execution-based reward quality shapes RLVR learning in a domain with structurally noisy symbolic verification, connecting empirically to the RLVεR theoretical framework. We use a 107k-problem clean physics corpus plus a 6.8k curated corpus spanning numerical, expression, equation, and MCQ types.

---

## 1. Motivation

Two concrete problems motivate this project.

**Problem 1: CoT-RLVR on physics has a hard verification ceiling.** Zero-shot baselines show rule-only verification achieves only 1.9% on expression-type and 2.7% on equation-type answers; xVerify-7B recovers to 31–33%, but algebraic rearrangements and symbolic equivalences remain persistent false-negative sources. This is not marginal noise — LaTeX verifier FN rates are ~68% on expression types before xVerify rescue. The RLVεR theoretical framework predicts that this level of reward noise systematically degrades GRPO learning curve quality. Our switch to execution-based reward is directly motivated by quantifying and eliminating this noise.

**Problem 2: Chain-of-thought truncates on hard problems.** 22% of Qwen3.5-4B outputs hit the 32k token limit and score near-zero (1–7%), versus 48.9% accuracy on non-truncated outputs. TIR compresses long derivation chains into compact executable code, reducing the token budget required on hard problems.

---

## 2. Method

### 2.1 TIR Trajectory Format — Single Code Block MVP

Each solution follows a fixed single-execution structure:

```
[think] I need to find the resonant frequency. Let me set up the RLC circuit equations.
[code]
import sympy as sp
L, C = 1e-3, 1e-6
omega = 1 / sp.sqrt(L * C)
print(float(omega))
[/code]
[output] 31622.776
[think] So ω ≈ 3.16 × 10⁴ rad/s. This matches option C.
[answer] \boxed{C}
```

**The model generates exactly one code block per trajectory.** The sandbox executes it, injects the output, and the model reasons to a final \boxed{} answer. The verifier always checks the final answer — not the code output directly. Multi-block trajectories are explicitly out of scope for the MVP; this avoids the distributional drift and compounding error issues that multi-turn TIR rollouts introduce in VeRL.

### 2.2 Rollout Implementation

The single-code-block constraint makes VeRL integration straightforward:

1. Model generates text until `[/code]` stop token
2. Sandbox executes code block (30s timeout; subprocess + resource limits)
3. `[output] {result}\n` injected as a fixed continuation
4. Model resumes generation until `[answer]` stop token
5. Final \boxed{} extracted and verified via existing pipeline

This is a single logical rollout with one mid-sequence injection — not a multi-turn loop. No custom async rollout engine required. **Week 1 gating item: implement and smoke-test this injection mechanism before any training.**

### 2.3 Why Single-Block Is Not a Compromise

For physics problems with verifiable answers:
- Numerical problems: one SymPy/NumPy computation suffices
- Expression problems: one `sp.simplify()` or `sp.solve()` call produces the answer
- MCQ: one computation, model reads result and selects option
- Equation derivation: one SymPy setup expresses the relationship

The cases that genuinely need multiple code blocks are largely in the equation/derivation subset, which is Phase 2. Phase 1 data is selected precisely because single-block execution is sufficient.

### 2.4 Why TIR-GRPO for Physics Is Novel

Existing TIR-GRPO work (SimpleTIR NeurIPS 2025, GTPO Nov 2025) studies math benchmarks where LaTeX answer verification is reliable. The physics domain introduces a structurally different verification problem: symbolic expressions, equation equivalence classes, and unit-dimensional answers have high LaTeX FN rates (~68% on expression types) that do not exist in math benchmarks. We contribute:

1. The first measurement of how LaTeX verifier FN rate varies by physics answer type, and whether this predicts GRPO learning curve degradation (connecting to RLVεR)
2. Evidence that switching to execution-based reward eliminates this noise on exactly the answer types where LaTeX verification fails
3. Per-answer-type behavioral analysis of what TIR-GRPO learns differently from CoT-GRPO

### 2.5 Training Pipeline

**Stage 0 — Probe (before committing to SFT):**
Run 100-problem zero-shot check with Qwen3.5-4B instruct (thinking off), prompting for single-block TIR format. Measure execution success rate and verifier hit rate. If verifier hit rate ≥ 15%, skip SFT.

**Stage 1 — SFT (conditional):**
If zero-shot format compliance is poor (<15% verifier hit rate), fine-tune on 2–5k TIR demonstrations generated by GPT-4o/Claude with auto-execution filtering. Goal: stabilize code format, SymPy usage, and post-execution reasoning. Likely skippable with the instruct model.

**Stage 2 — GRPO:**
Train with binary reward on final answer using VeRL/GRPO:

$$R = R_{\text{correct}} - \lambda \cdot C_{\text{tokens}}$$

where $R_{\text{correct}}$ is 1 if the final \boxed{} answer matches gold within verifier tolerance, and $C_{\text{tokens}}$ is a cost penalty added only in a late ablation ($\lambda = 0$ initially).

**Curriculum:** Numerical problems first (highest zero-shot accuracy, cleanest execution path), then expression and MCQ once training is stable.

---

## 3. Research Questions

**RQ1 (Main):** Does TIR-GRPO improve accuracy over CoT-GRPO on physics problems, particularly on expression and equation types where LaTeX verification is noisy?

**RQ2 (Reward noise):** Does LaTeX verifier FN rate by answer type predict GRPO learning curve degradation, and does switching to execution-based reward improve learning on exactly those types?

Operationalized as: measure per-type accuracy at fixed compute checkpoints (500 / 1000 / 2000 gradient steps) for both CoT-GRPO and TIR-GRPO. Compute per-type LaTeX FN rate from the training corpus. Test the correlation between per-type FN rate and per-type accuracy gap (TIR-GRPO minus CoT-GRPO) at each checkpoint. A positive correlation confirms the RLVεR prediction: higher verifier noise → larger TIR-GRPO gain.

**RQ3 (Behavioral):** What solution strategies does TIR-GRPO learn compared to CoT-GRPO — and does it reduce truncation on hard problems?

---

## 4. Data Plan

| Split | Source | n | Notes |
|---|---|---|---|
| Primary training | Dr. SCI clean (numerical + expression + MCQ) | ~65k | After prose-gold filtering |
| Supplemental training | 6.8k curated corpus (same types) | ~4k | Higher difficulty, more diverse sources |
| SFT demonstrations | GPT-4o TIR traces (conditional) | 2–5k | Auto-exec filtered; only if Stage 0 probe fails |
| Dev | Stratified holdout, balanced by answer type | 1k | Fixed before any training; used for checkpoint eval |
| Test — in-domain | Held-out corpus problems | 500 | Fixed split |
| Test — OOD | MATH-500 hard subset | 500 | Generalization check |

Goldilocks filter (pass_rate ∈ [0.15, 0.85] from existing pass@8 data) applied to GRPO training set.

---

## 5. Baselines

| Baseline | Description |
|---|---|
| **CoT think-ON** | Qwen3.5-4B, thinking enabled, boxed LaTeX — existing zero-shot (48.9% non-truncated) |
| **CoT think-OFF** | Qwen3.5-4B, no thinking, boxed LaTeX — existing zero-shot (40.3% non-truncated) |
| **TIR zero-shot** | Qwen3.5-4B instruct prompted for single-block TIR, no training |
| **SFT TIR-only** | After SFT on TIR demonstrations, no GRPO — measures SFT ceiling |
| **CoT GRPO** | GRPO on boxed LaTeX answers (Dr. SCI-style) — primary direct comparison |
| **TIR GRPO (ours)** | Main system — GRPO on TIR trajectories with execution reward |

---

## 6. Evaluation

**Primary metrics:**
- Accuracy by answer type (numerical / expression / MCQ / equation)
- Accuracy by difficulty tier (SciBench-RL / UGPhysics / OlympiadBench / PHYBench)
- Code execution success rate and error type distribution
- Truncation rate vs. CoT baseline (target: reduce below 22%)
- Token count distribution per answer type

**Reward noise analysis (RQ2) — key novel contribution:**
- Pre-training: measure LaTeX verifier FN rate per answer type on the training corpus
- During training: evaluate per-type accuracy on the stratified dev split at checkpoints 500 / 1000 / 2000 gradient steps for both CoT-GRPO and TIR-GRPO
- Post-training: compute Pearson correlation between per-type FN rate and per-type TIR-GRPO accuracy gain at each checkpoint; test whether this relationship is consistent with the RLVεR prediction

**Behavioral analysis (RQ3) — 100–150 sampled outputs per condition:**
- Solution strategy categorization: direct compute, unit conversion scaffolding, error-free single-shot
- Failure mode breakdown: wrong physics setup vs. execution error vs. correct code + wrong interpretation
- Per-type accuracy gain: TIR-GRPO vs. CoT-GRPO

**Ablations (run in parallel in Week 5, GPU-abundant):**
- Cost penalty λ: effect on token efficiency vs. accuracy
- Curriculum vs. full training data from the start
- Scaling: 0.8B vs. 4B TIR-GRPO accuracy comparison (ablation, not an RQ)

**OOD generalization:**
- MATH-500 hard subset

---

## 7. Risks and Mitigations

| Risk | Severity | Mitigation |
|---|---|---|
| VeRL single-block injection implementation is brittle | High | Implement and test in Week 1 before any training; this is the gating item |
| SymPy timeout on complex expressions | High | Profile dev set timeout distribution before training; 30s default |
| Reward sparsity in early GRPO | High | Curriculum: numerical first; warm-start from SFT if needed; 0.8B smoke test before 4B |
| TIR zero-shot is already strong (>45%) | Medium | Reframe: analysis of what GRPO adds on top of a capable zero-shot TIR model |
| TIR-GRPO doesn't beat CoT-GRPO in aggregate | Medium | Fallback: reward noise analysis paper — still clean and publishable |
| SFT data quality (executes but wrong physics) | Medium | Auto-exec + cross-check against verifier on gold answers |
| Week 5 overloaded (ablations + scaling + OOD) | Low | All training jobs run in parallel with abundant GPUs; human time only needed for analysis |

---

## 8. Timeline

| Week | Goal |
|---|---|
| **1** | Stage 0 probe (100-problem TIR zero-shot); **implement and test single-block VeRL injection**; build execution sandbox (30s timeout); decide on SFT; measure LaTeX FN rate by answer type |
| **2** | SFT if needed (2–5k examples); validate execution success rate ≥ 30% on dev set; **instrument GRPO eval loop for per-type accuracy at checkpoints (500/1000/2000 steps)** |
| **3** | 0.8B GRPO smoke test on numerical curriculum; debug reward pipeline; first 4B GRPO run |
| **4** | Expand GRPO to expression + MCQ; compare TIR-GRPO vs. CoT-GRPO at checkpoints; start writing related work + methods |
| **5** | Core ablations and 0.8B scaling run **in parallel** (GPU-abundant); OOD eval on MATH-500; behavioral analysis on sampled outputs; continue writing |
| **6** | Freeze results; finalize figures; complete paper draft |

---

## 9. Related Work Positioning

**SimpleTIR (NeurIPS 2025)** and **GTPO (Nov 2025)** establish TIR-GRPO for math, addressing training instability from multi-turn rollouts. We build on this paradigm but study a structurally different verification domain. Physics expression and equation answers have high LaTeX FN rates (~68% before xVerify rescue) that do not exist in math benchmarks. We are the first to study how this verification noise shapes TIR-GRPO learning, connecting empirically to the **RLVεR (Jan 2026)** theoretical framework for verifier-noise effects on GRPO dynamics.

**Dr. SCI (Feb 2026)** trains on a 1M-problem scientific corpus with CoT-RLVR, demonstrating RLVR for physics broadly. We complement this by isolating the specific failure mode of symbolic verifier noise and showing TIR-based reward eliminates it.

---

## 10. Contribution Claims

1. **Reward noise analysis**: First empirical measurement of how LaTeX verifier FN rate by physics answer type predicts GRPO learning curve degradation at fixed compute checkpoints, with evidence that execution-based reward eliminates this noise on expression and MCQ types.
2. **TIR-GRPO for physics**: Single-block TIR training paradigm applied to a domain with structural symbolic verification noise, demonstrating accuracy gains over CoT-GRPO on the answer types where verification is hardest.
3. **Behavioral analysis**: Per-answer-type characterization of what TIR-GRPO learns vs. CoT-GRPO, with truncation reduction as a concrete efficiency metric.

---

## 11. Fallback Paper Path

If TIR-GRPO does not outperform CoT-GRPO in aggregate accuracy, the paper reframes as:

> *"We measure how symbolic verifier reward noise varies across physics answer types and show this predicts differential GRPO learning failure at fixed compute budgets. Execution-based reward eliminates the noise on expression and MCQ types but not equation derivations, revealing a principled boundary for when TIR is necessary vs. sufficient for reliable RL training."*

This is a clean empirical contribution independent of aggregate accuracy improvement.

---

## 12. Industry Framing

The transferable lesson:

> *"We designed an RLVR pipeline where deterministic execution feedback replaces a noisy symbolic verifier, and showed empirically that verifier reward quality directly determines GRPO learning curve behavior at fixed compute budgets. We trained a TIR agent on 107k physics problems and characterized where execution-based reward is necessary vs. sufficient — a finding that generalizes to any tool-use domain with symbolic output verification."*

This maps directly to reward design, verifier reliability, and execution-based RL for tool-using agents — active investment areas at all major AI labs.
