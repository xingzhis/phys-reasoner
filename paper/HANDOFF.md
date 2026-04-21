# HANDOFF — paper writing session context

**Created:** 2026-04-19.
**Purpose:** full, loss-free context dump of the planning discussion so a fresh session can resume paper writing without re-deriving decisions.
**Branch:** `paper-writing` (this branch). Not merged to main.
**Main only has:** `docs/tool_use_analysis_handoff.md` (the HPC-side runbook for tool-use analysis).

Read this first. Everything else in `paper/` follows from what's here.

---

## 1. Venue

- **Workshop:** ICML 2026 AI4Physics — https://ai4physics-workshop.github.io/
- **Deadline:** April 24, 2026 AOE (≈ Apr 25 evening local; aim to submit Apr 23 evening for safety).
- **Page limit:** 8 pages ICML 2-column, references excluded.
- **Topic fit:** "Physics-centric Scientific Reasoning with LLMs and Agents — tool-augmented agents" is a named direction in the CFP. The paper is squarely on-topic.

## 2. Base model (final)

`Qwen/Qwen3-4B-Thinking-2507` — this is the final base model for all training and eval.

This is a switch from Qwen3.5-4B that happened upstream. It matters because:

- Qwen3-Thinking is reasoning-tuned and already strong on physics (beats Qwen3.5-4B by 18–30pp across benchmarks per the archived sweep in `outputs/eval/README_QWEN35_SWEEP_IN_PROGRESS.md`).
- At zero-shot, in TIR mode, it **often skips the tool and pure-reasons**. The eval doc (`docs/eval_results.md`) notes this directly: "Qwen3-Thinking often pure-reasons inside TIR mode without invoking the tool."
- This shifts the paper's framing (see §8 below).

## 3. Training and comparison plan (what will produce the paper's numbers)

- **TIR-GRPO (ours):** Qwen3-4B-Thinking-2507 base + single-block TIR + Dr.GRPO + DAPO-lite.
- **CoT-GRPO (strict baseline):** same data, same steps, same reward stack, same VeRL infra, only difference is no code tool. This is the claim-carrying comparison.
- **Zero-shot references:** base-model CoT zero-shot and TIR zero-shot. These are already done (see §7).

All runs use Dr.GRPO + DAPO-lite configuration per `docs/training-decisions.md`:
- `algorithm.kl_ctrl.kl_coef=0.0`, `algorithm.norm_adv_by_std_in_grpo=False`
- `actor_rollout_ref.actor.use_kl_loss=False, kl_loss_coef=0.0`
- `actor_rollout_ref.actor.loss_agg_mode=token-mean`
- `actor_rollout_ref.actor.clip_ratio_low=0.2, clip_ratio_high=0.28`
- 4×A100-80G trainer + 5 A100 rollout nodes, `rollout.n=8`, `ppo_mini_batch_size=128`, 1500 steps (~1.8 epochs of the ~108k pool).
- Reward: binary `R_correct` on final `\boxed{}` via rule + xVerify-7B on a dedicated reward GPU.

## 4. Thesis — one sentence (locked)

> Training Qwen3-4B-Thinking-2507 with single-block tool-integrated RLVR on ~108k physics problems teaches the model *when* to reach for a symbolic-computation tool; against a strict CoT-GRPO baseline matched in data, training budget, and reward stack, accuracy improves across a suite of physics benchmarks with gains concentrated on expression-type answers and harder problems.

One claim. Falsifiable. Directly supported by the planned experiments.

## 5. Contributions (3, in priority order)

1. **Empirical.** First controlled TIR-GRPO vs CoT-GRPO comparison for physics at RL training level, across five physics benchmarks (four in-dist pool_v2 slices + four external: OlympiadBench, PHYBench, ABench-Phy A, ABench-Phy B). Matched data, steps, reward.
2. **Behavioral.** Per-answer-type decomposition showing where TIR gains concentrate (expression-type, hard problems) and where it does not (equation derivation remains open). Tool-use rate before vs after RL is the sharpest form of the story: "RL unlocks tool use on tool-suited problems."
3. **Recipe.** Single-block TIR with ScaleRL-style think-interrupt implemented inside VeRL for Qwen3-Thinking, plus the Dr.GRPO + DAPO-lite training configuration that made it stable. Reproducible artifact.

## 6. Framing decisions and rejected alternatives (read to avoid re-litigating)

### 6.1 What we rejected: the "reward-quality intervention" / FN-gap framing

The original proposal (`docs/physcode_proposal_v5.md`) led with RQ1 = aggregate accuracy, RQ2 = per-type LaTeX verifier FN rate correlates with per-type TIR gain. An earlier draft of this plan proposed a "hybrid" framing that made RQ2 the mechanistic explanation in §5.

**Why we rejected it:**

- **Scope creep.** Two stories stitched together, each weakened by sharing the 8-page budget.
- **No methodological contribution behind it.** We use published verifiers (rule + xVerify). To earn the reward-noise pitch we'd need a new verifier design or a noise-adaptive reward — we have neither.
- **Unfalsifiable with planned experiments.** The FN-vs-gap correlation is confounded by problem difficulty: hard-to-verify types may also be hard-to-solve types. Telling reward-quality effects apart from "tools help on hard problems" would require a rule-only-vs-rule+xVerify reward ablation we don't have time to run.
- **Dangerous timeline.** If the correlation comes out weak, we'd be rewriting the abstract 48h before deadline.
- **RLVεR already owns the theoretical point.** Citing it isn't the same as extending it.

**Where it lives now:** one honest paragraph in §7 Discussion, framed as a future-work hypothesis with the difficulty confound acknowledged. See `paper/sections/07-discussion.md` and the outline beats.

### 6.2 What we chose: "RL unlocks tool-use behavior" + aggregate accuracy

Zero-shot evaluation (§7) shows TIR ≈ CoT on the base model because Qwen3-Thinking ignores the tool in TIR mode. This makes the central empirical question operational:

> Does RL training teach the model *when* to reach for the tool, and does this shift produce accuracy gains on tool-suited problems?

This is sharper than "TIR wins" because:
- It's mechanistic (tool-use rate is observable, not inferred).
- It has a clean before/after (zero-shot vs post-RL tool-call rate).
- Small aggregate gains are still a strong story if the behavior shift is clean.
- Large aggregate gains are a strong story via the same mechanism.

Abstract language should reflect this: "TIR-GRPO teaches a reasoning-tuned model to use symbolic computation where it pays off; we show this shifts accuracy on tool-suited problem types while leaving pure-reasoning strengths intact."

## 7. Zero-shot eval results (already run, in main via `docs/eval_results.md`)

### 7.1 The numbers

**In-distribution (pool_v2 test):**

| benchmark (n) | tir/train | cot/train | tir/qwen | cot/qwen |
|---|---|---|---|---|
| pool_v2_scibench (153) | **0.599** | 0.566 | 0.572 | 0.579 |
| pool_v2_physics (191) | 0.393 | **0.435** | 0.367 | 0.398 |
| pool_v2_ugphysics (217) | 0.383 | **0.410** | 0.364 | 0.396 |
| pool_v2_drsci (503) | 0.630 | 0.644 | 0.612 | **0.654** |

**External (held out):**

| benchmark (n) | tir/train | cot/train | tir/qwen | cot/qwen |
|---|---|---|---|---|
| olympiad_oe_to_physics (236) pass@1 | 0.047 | 0.051 | 0.059 | **0.068** |
| phybench (1000) exact pass@1 | 0.015 | 0.010 | **0.022** | 0.017 |
| phybench mean EED (0-100) | 3.03 | 2.39 | **3.30** | 3.01 |
| abench_phy_a (400) | 0.203 | 0.195 | 0.203 | **0.215** |
| abench_phy_b (400) per-row | 0.633 | 0.633 | 0.640 | 0.620 |
| abench_phy_b (100) per-mid (all-4-subid) | **0.500** | 0.490 | 0.490 | 0.480 |

Sampling presets: `train` = `temp=1.0, top_p=1.0` (matches RL training); `qwen` = `temp=0.6, top_p=0.95, top_k=20` (Qwen team thinking-mode recommendation).

### 7.2 What these numbers say

- **TIR ≈ CoT on every in-dist cell (±5pp).** Expected, not worrying — see §8.
- **Qwen sampling slightly hurts TIR (~2–4pp) and slightly helps CoT (~1–2pp).** Lower entropy makes the thinking model skip the tool more confidently; CoT benefits marginally from the same determinism. We'll use the **train** preset in the paper for consistency with RL training.
- **OlympiadBench and PHYBench are punishing** — exact symbolic judges cap raw numbers at 1–7%. For PHYBench we use the continuous EED metric as primary; exact-match goes to the appendix.

### 7.3 Benchmarks — decided list

In-distribution (pool_v2 test splits — drawn from training pool):
- `pool_v2_drsci` (503), `pool_v2_physics` (191), `pool_v2_ugphysics` (217), `pool_v2_scibench` (153).

External (held out from training):
- `olympiad_oe_to_physics` (236) — exact-match.
- `phybench` (1000) — report EED (primary), exact-match in appendix.
- `abench_phy_a` (400) — 1% relative tolerance.
- `abench_phy_b` (100 per-mid; 400 per-row) — per-mid is the "all-4-parametric-variants correct" robustness metric.

**Dropped:** MATH-500 (distracts from physics focus). **Stretch (only if time):** critpt (research-style problems; mentioned by user as nice-to-have for signaling research capability, but not on the critical path).

## 8. Zero-shot TIR ≈ CoT is a feature, not a bug

Because Qwen3-Thinking often ignores the tool in TIR mode zero-shot, "TIR mode" and "CoT mode" are often the same behavior on the same problem. That is why the numbers are close. This is the exact gap that RL training fills: the tool-use distribution needs to shift from "rarely use" to "use where it pays off."

### Implications for the paper

1. **Don't oversell aggregate gains.** If post-RL gap is 2–5pp on aggregate, write "improves" not "substantially outperforms."
2. **Tool-use rate is a first-class finding, not a side analysis.** Add explicit tool-use figure/table (see §10).
3. **Per-type concentration becomes more important**, since small aggregate may coexist with sharp per-type shifts.
4. **Include zero-shot TIR and zero-shot CoT in Table 1** — the pre-RL parity is load-bearing for the "RL unlocks tool use" narrative.
5. **Watch for the collapse failure mode during training**: a reasoning-tuned base under RL might learn to skip the tool entirely (since tool calls cost tokens and sometimes fail). Log tool-call rate per training step. If it trends toward zero, consider (a) an auxiliary reward term encouraging tool use on hard problems, or (b) reporting the collapse as a finding.

## 9. Paper outline (page budget)

Full detail in `paper/outline.md`. Summary:

| § | Pages | Content |
|---|---|---|
| Abstract | — | ~180 words; "RL unlocks tool use + accuracy gains" frame. |
| 1. Introduction | 1.0 | Hook (reasoning-tuned models ignore tools zero-shot), what we do, contributions, Figure 1 hero. |
| 2. Related Work | 0.5 | RL for reasoning, TIR, physics benchmarks, verifiers. Deliberately short. |
| 3. Method | 1.75 | Trajectory, rollout mechanism (think-interrupt, sandbox), Dr.GRPO+DAPO-lite. Figure 2. |
| 4. Experimental Setup | 1.0 | Data, benchmarks, baselines, metrics, training config. |
| 5. Results | 2.25 | Main table + per-type + tool-use behavior + truncation. |
| 6. Analysis | 0.75 | Qualitative solution-strategy and failure-mode breakdown. |
| 7. Discussion | 0.5 | Scope, limitations, reward-noise hypothesis as future work. |
| 8. Conclusion | 0.25 | — |
| **total** | **8.0** | |
| Appendix | — | Hyperparams, ablations, extended examples. |

Pressure valve if tight: collapse §6 into §5; cut Related Work to 0.4 page.

## 10. Figures and tables (priority order)

Full detail in `paper/storyboard.md`. Priorities:

**Must-have:**
- **Table 1 — Main benchmark results.** Rows: {zero-shot CoT, zero-shot TIR, CoT-GRPO, TIR-GRPO}. Cols: {pool_v2 slices, UGPhysics, PHYBench (EED), OlympiadBench, ABench-Phy-A, ABench-Phy-B per-mid, macro-avg}.
- **Table 2 — Per-answer-type accuracy** on in-dist test. {numerical, expression, MCQ, equation, multi-part, interval} × {base, CoT-GRPO, TIR-GRPO, Δ}.
- **Figure 3 — Per-type learning curves.** CoT-GRPO vs TIR-GRPO over training steps, per answer type.

**Strong-to-have:**
- **Figure 4 — Tool-use rate pre-RL vs post-RL.** Per benchmark, per type. Promoted to first-class — NEW, added after base-model switch.
- **Figure 1 — Hero: CoT vs TIR trajectory** on a single problem.
- **Figure 5 — Truncation and response length.** (May downgrade to appendix if space tight.)

**Nice-to-have:**
- Figure 2 — rollout mechanism diagram (can be replaced by prose + numbered list if space is truly tight).
- Table 3 — solution-strategy and failure-mode frequencies from hand-labeled samples.

**Appendix only:**
- Table 4 — hyperparameters.
- Table 5 — rule verifier coverage on TIR vs CoT outputs (feeds the reward-noise future-work paragraph).

## 11. Novelty pitch + anticipated objections

Full detail in `paper/notes/novelty-pitch.md`. Top-level:

> We do the first controlled RL-level comparison of tool-integrated reasoning against chain-of-thought reasoning for physics problem solving, using matched data, steps, and reward stack. On a reasoning-tuned base that ignores tools zero-shot, TIR-GRPO teaches the model when to reach for symbolic computation; we show this shifts tool-use rate on tool-suited problems and improves accuracy across physics benchmarks.

Anticipated objections and responses:
- *"TIR on math has been done."* Physics answer types are more diverse; we run a strict apples-to-apples RL-level comparison that most prior TIR-RL work doesn't.
- *"Your verifier is published."* We don't claim verifier novelty. Contribution is the TIR rollout + think-interrupt + RL recipe plus empirical findings.
- *"Why only 4B?"* Scale ablation is appendix if time permits; main claim is stated conditional on 4B.
- *"TIR is obviously helpful."* The zero-shot parity and the per-type concentration *are* the surprises — gains are not uniform and a reasoning-tuned base doesn't reach for tools without training.
- *"Why not multi-block TIR?"* Scope decision; single-block is cleaner baseline for future multi-block extensions.

Things **not** to claim: mechanism (reward noise causes gain — only hypothesis), SOTA, cross-domain generalization, first-ever-tool-use-for-physics.

## 12. Timeline (from `paper/notes/timeline.md`)

| Date | What |
|---|---|
| Apr 19 (today) | Session handoff. Outline, storyboard, scaffolding in place. Tool-use handoff doc pushed to main. |
| Apr 20 | Related work + method skeleton. Draft §3 Method (1.75 pages) and §4 Setup (1.0 page) — mostly locked content. |
| Apr 21 | Draft §1 Introduction (1.0). Set up ICML 2026 LaTeX template in Overleaf. Port §2, §3, §4 drafts. Build Figure 2. |
| Apr 22 | First training checkpoints land. Write plotting scripts for Figures 3, 4. Pre-structure Table 1/Table 2 with placeholders. Start §5 around placeholders. |
| Apr 23 | Training ~complete. Run benchmark evals. Populate Table 1, Table 2, Figures 3, 4. Finalize §5. Sample outputs for §6 analysis. Draft §6, §7, §8. Write abstract. First read-through. Citation audit. |
| Apr 24 | Overflow check. Figure/table captions. Submit end-of-local-day. |

## 13. Risks and pressure-valve cuts

**Result-level risks:**
- Aggregate TIR gain is flat → reframe around per-type and tool-use-shift findings (already the frame; just strengthens §5.2/§5.3).
- Aggregate TIR gain is negative → hard pivot to "where tool use does not help" characterization. Decide by Apr 22.
- Tool-use collapses to zero under RL → treat as a finding ("RL on a reasoning-tuned base suppresses tool use"); add one figure on this.
- Training slips past Apr 22 → use partial-checkpoint eval for Table 1; note in paper.

**Space cuts (in order, if tight):**
1. Cut Figure 5 (truncation).
2. Collapse §6 into §5.
3. Cut Related Work to 0.4 page.
4. Move Table 3 to appendix.
5. Drop Figure 2; replace with numbered list in §3.2.

**Scope cuts (in order, if time tight):**
1. Cut critpt.
2. Skip manual solution-strategy labeling; rely on LLM-assisted first pass.
3. Skip 0.8B scaling ablation.
4. Ship without truncation analysis (Fig 5).

## 14. Citation policy (strict, no hallucinations)

- Every entry in `paper/bibliography/refs.bib` must be verified against a primary source (arXiv, publisher, proceedings).
- Prose uses `[CITE:tag]` markers; resolve against `paper/bibliography/citations-todo.md`.
- Known citations to verify (full list in `citations-todo.md`): DeepSeek-R1, GRPO (DeepSeekMath), DAPO, Dr. GRPO, Dr. SCI, RLVεR, ToRA, MathCoder, SimpleTIR, GTPO, math-verify, xVerify, UGPhysics, PHYBench, OlympiadBench, SciBench, Qwen3, VeRL, ScaleRL.
- Final manual audit by user before submission.
- When in doubt, do a WebSearch/WebFetch against the claimed source; only add to refs.bib if confirmed.

## 15. Repo state and branch topology

```
origin/main         — has docs/tool_use_analysis_handoff.md + latest training infra. No paper/ files.
origin/paper-writing — has main + paper/ folder (this branch).
```

Sync conventions from the previous session's message:

```bash
# on paper-writing: pick up main updates (e.g. new training results, tool-use output files)
git fetch origin
git merge origin/main
git push

# when merging back at the end
git checkout main
git merge --no-ff paper-writing
git push origin main
```

## 16. Where things live in paper/

```
paper/
  HANDOFF.md               # this file — read first
  README.md                # navigation + policies
  outline.md               # section beats + page budget
  storyboard.md            # figures/tables, one-liner each, priority order
  sections/
    00-abstract.md         # template + beats
    01-intro.md            # empty skeleton
    ...
    08-conclusion.md
  figures/                 # empty, subdirs per figure when we start building
  tables/                  # empty
  bibliography/
    refs.bib               # verified-only, currently empty
    citations-todo.md      # all [CITE:tag] markers + verification status
    sample-abstracts/      # empty, reference papers to drop here
  notes/
    novelty-pitch.md       # anticipated-reviewer-objections responses
    timeline.md            # day-by-day plan
```

## 17. Pending inputs from HPC session (user will provide)

From `docs/tool_use_analysis_handoff.md` (committed to main):

- `outputs/eval/tool_use_summary.txt` — aggregate per-cell tool-use rates.
- `outputs/eval/tool_use_by_type.csv` — per-answer-type tool-use rates on in-dist cells.
- `outputs/eval/tool_use_by_type_summary.txt` — human-readable version.

These will be committed to main and picked up via `git merge origin/main` into paper-writing. When they land:

- Use for Figure 4 zero-shot "before" data.
- Use for Table 1 "zero-shot TIR" row plausibility check (is tool-use > 0 at all? if not, the headline becomes "reasoning-tuned base ignores tools entirely").
- Use in §5.3 prose: "Before RL, TIR-mode called the tool on only X% of rollouts; CoT-mode by construction never calls the tool."

## 18. Immediate next steps for the new session

1. **Verify the scaffold** — read `paper/outline.md`, `paper/storyboard.md`, `paper/notes/novelty-pitch.md` and this file for any inconsistency; fix before drafting.
2. **Pull 2–3 sample workshop papers** (tool-use RL, physics reasoning; NeurIPS MATH-AI or ICML workshop 2024–25) via WebSearch/WebFetch. Drop abstracts + structure summaries into `paper/bibliography/sample-abstracts/`.
3. **Start citation verification** in parallel — resolve `[CITE:tag]` entries in `citations-todo.md` against primary sources; populate `refs.bib`.
4. **Begin drafting §3 Method and §4 Setup** — these are the most locked content; least dependent on results.
5. When tool-use outputs arrive on main, merge into paper-writing, update §5.3 draft + Figure 4.

## 19. Things decided that should not be re-opened (unless new evidence)

- Framing: "RL unlocks tool-use behavior" + aggregate accuracy. Not reward-noise mechanism.
- Base model: Qwen3-4B-Thinking-2507. Not Qwen3.5-4B.
- CoT-GRPO: strict apples-to-apples + base zero-shot reference. Both in Table 1.
- External benchmarks: OlympiadBench exact-match + PHYBench EED + ABench A/B all kept. MATH-500 dropped.
- Verifier noise analysis: one paragraph in §7 Discussion as future work. Not a headline section.
- Single-block TIR: design choice, mentioned in §3, not a pitched contribution.
- 8-page two-column ICML format.
- Work in markdown first, port to Overleaf around Apr 21.

## 20. Tone + style conventions for prose

- Honest. No overclaiming. "Improves" over "substantially outperforms."
- Mechanism claims require data that supports them; use "consistent with" or "suggests" for hypotheses.
- Short paragraphs in 2-column layout; avoid long blocks.
- Define terms once, use consistently: "TIR" (never "tool-integrated" after first mention), "CoT-GRPO" (never "chain-of-thought GRPO").
- Colors: TIR-GRPO = emphasis (orange/red); CoT-GRPO = neutral blue/grey; zero-shot references = dashed grey. Consistent across figures.
- Tables: use booktabs (`\toprule`, `\midrule`, `\bottomrule`). No vertical rules.
- Figure captions: first sentence is the takeaway, not just the legend.

---

End of handoff. If any decision above conflicts with something in the user's head, surface it before drafting prose. Good luck.
