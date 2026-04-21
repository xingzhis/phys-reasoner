# Storyboard — Figures and Tables

One-liner per visual: **what it shows**, **why it's here**, **data source**, **status**.

---

## Figures

### Figure 1 — Hero: CoT vs TIR trajectory
- **What:** Side-by-side trajectory for one physics problem. Left: CoT-GRPO reasoning ending in a LaTeX expression. Right: TIR-GRPO reasoning, SymPy code, tool output, final boxed numerical answer.
- **Why:** Sets the paper's frame in one image — the reader sees what TIR does differently in under three seconds.
- **Data:** One hand-picked example from TIR-GRPO and CoT-GRPO rollouts on the same problem from in-domain test.
- **Status:** Placeholder until both models are trained. Candidate problems: pick a medium-difficulty expression-type problem where TIR visibly routes through code.
- **Placement:** Page 1, spans one column.

### Figure 2 — Rollout mechanism diagram
- **What:** Flow diagram of single-block TIR rollout: prompt → `<think>` reasoning → (think-interrupt if budget exceeded) → `<tool_call>` code → sandbox → `<tool_response>` output → `<think>` interpretation → `\boxed{}` answer.
- **Why:** Precise specification of what the agent does. Supports §3.2.
- **Data:** None — drawn with TikZ or a vector tool.
- **Status:** Static schematic; can build any time.
- **Placement:** Mid §3.

### Figure 3 — Per-answer-type learning curves
- **What:** 2×3 grid (or 2×2 if we cut equation/multi-part). Each subplot: accuracy on dev set over training steps, one line for CoT-GRPO and one for TIR-GRPO.
- **Why:** Central empirical finding: where the gain concentrates.
- **Data:** Dev-set eval at checkpoints every 200 steps. Per-type accuracy from xVerify-7B judge.
- **Status:** Waiting on training runs. Script: write today to pull from checkpoint eval logs; placeholder figure ready.
- **Placement:** §5.2.

### Figure 4 — Truncation and response length
- **What:** Two panels. (a) Fraction of rollouts truncated (response length ≥ cap and no boxed answer) vs training step, per condition. (b) Response length distribution (CDF) at final checkpoint, per condition.
- **Why:** TIR should produce shorter, more decisive answers on hard problems. Measurable and clean.
- **Data:** Rollout metadata from final training logs.
- **Status:** Placeholder until training.
- **Placement:** §5.4.

### Figure 5 (optional, appendix or §6) — Solution strategy breakdown
- **What:** Stacked-bar or Sankey diagram of qualitative strategy categories on 100–150 labeled outputs per condition.
- **Why:** Supports §6 behavioral analysis.
- **Status:** Only if we have labeling time. Otherwise, table form in §6.

---

## Tables

### Table 1 — Main benchmark results [HEADLINE]
- **What:** Accuracy across benchmarks.

| Model | In-domain | UGPhys | PHYBench | OlympiadBench | SciBench | (critpt) | Macro-avg |
|---|---|---|---|---|---|---|---|
| Qwen3.5-4B (zero-shot) | | | | | | | |
| Qwen3.5-4B (TIR zero-shot) | | | | | | | |
| CoT-GRPO (ours) | | | | | | | |
| **TIR-GRPO (ours)** | | | | | | | |

- **Why:** The claim-carrying table.
- **Data:** Final-checkpoint eval for each condition on each benchmark, xVerify-7B judge.
- **Status:** Structure set. Numbers arrive as training finishes.
- **Placement:** §5.1.

### Table 2 — Per-answer-type accuracy (in-domain test)
- **What:** Accuracy broken down by {numerical, expression, MCQ, equation, multi-part, interval}. Columns: {base, CoT-GRPO, TIR-GRPO, Δ(TIR−CoT)}.
- **Why:** Directly supports the "gains concentrate on expression-type" claim.
- **Data:** In-domain test set with answer-type labels (already stratified).
- **Status:** Waiting on final checkpoints.
- **Placement:** §5.2.

### Table 3 — Solution strategy and failure-mode frequencies
- **What:** Categorical breakdown of 100–150 hand-labeled outputs per condition.
- **Why:** Concrete evidence for §6 analysis.
- **Data:** Manual labeling after training; or LLM-assisted first pass + spot check.
- **Status:** Only if labeling gets done in time.

### Table 4 (appendix) — Hyperparameters
- **What:** Training config (steps, batch, LR, clip ratios, reward shaping), sampling config, sandbox limits.
- **Placement:** Appendix A.

### Table 5 (appendix, conditional) — Rule verifier coverage on TIR vs CoT outputs
- **What:** % of xVerify-correct outputs that rule verifier also accepts, per type, for each condition.
- **Why:** Motivates the reward-quality hypothesis in Discussion. Not in the body.
- **Data:** Final checkpoint rollouts, re-scored with rule verifier alongside xVerify.
- **Placement:** Appendix or §7 inline if we have room.

---

## Color / style conventions

- CoT-GRPO: neutral blue or grey.
- TIR-GRPO (ours): emphasis color — orange or red.
- Base / zero-shot references: dashed grey line.
- Consistent across all figures.

## Priority order if time runs out

Must-have: Table 1, Figure 3, Table 2.
Strong-to-have: Figure 4, Figure 1.
Nice-to-have: Figure 2 (can be replaced by a numbered list in §3.2 if needed).
Bonus: Figure 5, Table 3.

---

## TODO

- [ ] Sketch Figure 1 layout on paper; pick candidate example problems
- [ ] Write plotting script for Figure 3 (reads checkpoint eval logs, produces PDF)
- [ ] Write plotting script for Figure 4 (reads rollout metadata)
- [ ] Pre-register Table 1 layout in LaTeX template before porting
- [ ] Decide on critpt inclusion — cut if not ready by Apr 22
