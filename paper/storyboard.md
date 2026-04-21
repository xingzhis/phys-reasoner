# Storyboard — Figures and Tables

**Status:** v2 (Apr 21, post-Task-1b)

One-liner per visual: **what it shows**, **why it's here**, **data source**, **status**.

---

## Figures

### Figure 1 — Hero: two-failure-modes + recalibration (paired W−L)
- **What:** Two panels.
  - **Panel (a)** — per-benchmark paired W−L (zero-shot). x-axis: benchmarks; y-axis: W−L percentage points; each benchmark shows a single bar, colored by sign. Annotate "3 of 4 in-dist slices show TIR ≤ CoT."
  - **Panel (b)** — per-answer-type paired W−L (zero-shot). x-axis: answer types; each answer type shows one bar per benchmark (grouped or boxplot-styled). Expression bar highlighted. Annotate the physics slice's −10.7 pp.
  - (Post-RL version:) same two panels with post-RL bars overlaid or side-by-side.
- **Why:** Sets the paper's frame in one image. Two panels communicate aggregate (panel a) + mechanism (panel b).
- **Data:** `outputs/eval/paired_tir_vs_cot_by_benchmark.csv` and `paired_tir_vs_cot_by_type.csv` (zero-shot); same after RL (post-training).
- **Status:** Panel (a) and (b) zero-shot data in hand. Post-RL pending. Build a static first version now.
- **Placement:** Page 1, spans both columns.

### Figure 2 — Rollout mechanism diagram
- **What:** Flow diagram of single-block TIR rollout: prompt → `<think>` reasoning → (think-interrupt if budget exceeded) → `<tool_call>` code → sandbox → `<tool_response>` output → `<think>` interpretation → `\boxed{}` answer.
- **Why:** Precise specification of what the agent does. Supports §3.2.
- **Data:** None — drawn with TikZ or a vector tool.
- **Status:** Static schematic; can build any time.
- **Placement:** Mid §3.

### Figure 3 — Per-answer-type learning curves
- **What:** 2×3 grid of subplots (answer types: numerical, expression, equation, MCQ, interval, multi-part). Each subplot: accuracy on dev over training steps, one line for CoT-GRPO and one for TIR-GRPO.
- **Why:** Shows where RL gains concentrate; predicts the expression type gets the biggest TIR-GRPO lift (motivated by the zero-shot failure mode A).
- **Data:** Dev-set eval at checkpoints every 200 steps (HPC Task 2).
- **Status:** Pending training. Script hook present.
- **Placement:** §5.4.

### Figure 4 — Tool-use recalibration during training
- **What:** Two panels, TIR-GRPO only.
  - **(a)** call rate per answer type over training steps.
  - **(b)** paired W−L vs CoT-mode (on held-out dev) per answer type, checkpointed over training.
- **Why:** Shows the mechanism of recalibration — does call rate drop on expression? does call-quality rise on numerical?
- **Data:** HPC Task 2 (curves) + Task 1b on per-checkpoint rollouts.
- **Status:** Pending training.
- **Placement:** §5.5.

### Figure 5 (optional, may move to appendix) — Truncation + response length
- **What:** Truncation rate vs step; response-length CDF at final checkpoint.
- **Status:** Cut first if space is tight.
- **Placement:** §5.7 or appendix.

---

## Tables

### Table 1 — Main benchmark results [HEADLINE]
- **What:** Aggregate paired comparison across the four conditions.

| Model | pool_v2_drsci | pool_v2_physics | pool_v2_ugphysics | pool_v2_scibench | Olympiad | PHYBench (EED) | ABench-A | ABench-B | Macro |
|---|---|---|---|---|---|---|---|---|---|
| Qwen3-4B-Thinking (CoT zero-shot) | 64.4 | 43.5 | 41.0 | 56.2 | 5.1 | — | 19.5 | 63.2 | |
| Qwen3-4B-Thinking (TIR zero-shot) | 63.0 | 39.3 | 38.2 | 59.5 | 4.7 | — | 20.2 | 63.2 | |
| CoT-GRPO (ours) | | | | | | | | | |
| **TIR-GRPO (ours)** | | | | | | | | | |

- **Why:** Claim-carrying table.
- **Data:** zero-shot rows populated from existing eval; GRPO rows from HPC Task 4 post-training.
- **Placement:** §5.3.

### Table A — Zero-shot paired per-benchmark (sets up Table 1)
- **What:** W / L / TP / TF / W−L / TIR% / CoT% / TIR-call% per benchmark. 8 rows.
- **Data:** `paired_tir_vs_cot_by_benchmark.csv` — already in hand.
- **Placement:** §5.1.

### Table B — Zero-shot paired per-answer-type (mechanism)
- **What:** W−L / TIR% / CoT% / call% per (benchmark × answer_type). Filter to scalar types with n ≥ 10. Expression row highlighted.
- **Data:** `paired_tir_vs_cot_by_type.csv` — already in hand.
- **Placement:** §5.1.

### Table 2 — Per-answer-type in-distribution (post-RL)
- **What:** Accuracy by answer_type × {base, CoT-GRPO, TIR-GRPO}. Δ(TIR−CoT) column.
- **Data:** post-RL rollouts, HPC Task 4 + Task 1.
- **Placement:** §5.4.

### Table 3 — Solution-strategy / failure-mode frequencies (optional)
- **What:** Categorical breakdown of 100–150 hand-labeled outputs per condition.
- **Status:** Cut if labeling time runs out.

### Appendix tables
- Hyperparameters + training config.
- Rule-verifier coverage on TIR vs CoT (feeds Discussion paragraph).
- Per-benchmark × per-type full expansion of Table 2.

---

## Color / style conventions

- CoT-GRPO: neutral blue-grey.
- TIR-GRPO (ours): emphasis color (orange or red).
- Zero-shot references: dashed grey line.
- Expression-type highlight in Table B and Figure 1 panel (b): yellow-background row / thicker outline.
- Consistent across all figures.

## Priority order if time runs out

Must-have (claim-carrying): Table 1, Table A, Table B, Figure 1.
Strong-to-have: Figure 3 (per-type curves), Figure 4 (recalibration dynamics), Table 2.
Nice-to-have: Figure 2 (rollout diagram — can replace with numbered list).
Drop first: Figure 5 (truncation), Table 3 (qualitative).

---

## TODO

- [ ] Sketch Figure 1 panel layout with real W−L numbers from the CSVs.
- [ ] Draft caption text for Figure 1 (takeaway-first).
- [ ] Build Table A and Table B in LaTeX from the CSVs (zero-shot data is final).
- [ ] Write plotting script for Figure 3 when HPC Task 2 outputs land.
- [ ] Pre-register Table 1 layout in the Overleaf template.
