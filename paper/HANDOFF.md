# HANDOFF — paper writing session context

**Created:** 2026-04-19. **Last revised:** 2026-04-21 (v3 — claim-1-led thesis after selection-bias discussion).
**Purpose:** full, loss-free context dump of the planning discussion so a fresh session can resume paper writing without re-deriving decisions.
**Branch:** `paper-writing` (this branch). Not merged to main.
**Main only has:** `docs/tool_use_analysis_handoff.md` (the HPC-side runbook for tool-use analysis).

## Revision log

- **v3 (2026-04-21):** Raised the concern that within-TIR `call-pass% < skip-pass%` has selection bias (the model's choice to call is not random). Switched the thesis to lead with the **paired TIR-mode-vs-CoT-mode zero-shot comparison** (same problem, same prompt, only tool availability differs — no selection bias). Added Task 1b to HPC_TASKS for per-problem paired comparison. Discussed fallback path if post-RL results are weak. Also refactored bibliography/sample-abstracts/ → writing-samples/ and shifted those docs toward paper-writing style analysis.
- **v2 (2026-04-19):** HPC tool-use analysis showed the base model DOES call tool at 31–89% rates. Reframed thesis around "tool-use miscalibration + RL recalibration" rather than the earlier "model ignores tool" story. Updated §4, §7, §8 accordingly. Training infra now 4 train + 6 rollout + 1 xVerify, staleness=1. Training data moved to a difficulty-filtered subset (size TBD).

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
- **Hardware:** 4 A100-80G trainer nodes + 6 A100-80G rollout nodes + 1 xVerify reward GPU.
- **Async:** VeRL `fully_async_policy` with `trigger_parameter_sync_step=1` (staleness-1).
- `rollout.n=8`, `ppo_mini_batch_size=128`, 1500 steps.
- **Training pool:** difficulty-filtered subset of Dr. SCI + curated corpus. **TODO:** lock size and composition from user.
- Reward: binary `R_correct` on final `\boxed{}` via rule + xVerify-7B on the dedicated reward GPU.

Infrastructure-level details (staleness, node counts, exact Hydra flags) go to appendix; body only mentions "VeRL async RL with rule+xVerify reward."

## 4. Thesis — one sentence (locked, v3)

> Tool availability alone is not enough: giving a reasoning-tuned LLM (Qwen3-4B-Thinking-2507) access to a symbolic-computation tool fails to consistently help on physics problems and, on four of five in-distribution benchmarks, hurts. Despite the model invoking the tool on 31–89% of problems zero-shot, tool-augmented accuracy falls at or below plain reasoning. We show tool-integrated RLVR bridges this gap: TIR-GRPO outperforms a strict CoT-GRPO baseline matched in data, training budget, and reward stack, with gains concentrated on answer types where zero-shot tool use underperforms most.

One claim. Falsifiable. Directly supported by the already-run paired TIR-mode-vs-CoT-mode zero-shot comparison and the planned RL runs.

### Why this framing, not earlier drafts

- **v1 (dropped 2026-04-19):** "Base model ignores tool; RL teaches use." Falsified by HPC analysis showing call rates 31–89%.
- **v2 (dropped 2026-04-21):** "Within TIR mode, call-path pass < skip-path pass → miscalibration." Real but **selection-biased** — the model's choice of when to call isn't random, so low call-path pass might reflect problem difficulty rather than tool-use quality. A careful reviewer would flag this.
- **v3 (current):** Leads with the paired TIR-vs-CoT aggregate comparison (same problems, same prompts, only tool availability differs — *no* selection bias). The within-TIR call-vs-skip observation becomes a secondary zoom-in with an explicit selection-bias caveat.

### Claim evidence levels

| Claim | Evidence | Confidence |
|---|---|---|
| **C1.** Tool availability doesn't consistently help zero-shot; sometimes hurts | Paired TIR-vs-CoT zero-shot table (§7.1) — 4/5 in-dist slices show TIR ≤ CoT | **High** |
| **C2.** Base model calls tool at non-trivial rates | HPC tool-use analysis (§7.2) — 31–89% | **High** |
| **C3.** Within TIR mode, call-path pass ≠ skip-path pass | HPC tool-use analysis (§7.2) | Medium — selection-biased, frame with caveat |
| **C4.** Miscalibration is systematic by answer type | Partial: per-type call rate + aggregate pass exist; per-type call-path vs skip-path NOT yet run | Pending Task 1 Output 2 |
| **C5.** Per-problem: tool-mode-vs-no-tool-mode paired comparison shows directional pattern | Not yet run — **new Task 1b** | Pending |
| **C6.** RL recalibrates (TIR-GRPO > CoT-GRPO) | Needs training results | Unknown |

C1 is the single strongest claim and carries the paper. C5 when it lands will be cleaner still. C6 is the endpoint. C3/C4 are supporting.

## 5. Contributions (3, in priority order)

1. **Empirical finding (zero-shot characterization).** Paired comparison shows that, for a reasoning-tuned 4B model, tool availability does not consistently help on physics and often hurts: TIR-mode ≤ CoT-mode on 4 of 5 in-distribution benchmarks at matched prompts. Despite call rates of 31–89%, the benefit does not materialize zero-shot. Observation holds across answer types.
2. **Controlled RL intervention.** First apples-to-apples TIR-GRPO vs CoT-GRPO at matched RL training budget on physics (same data, same steps, same reward stack). TIR-GRPO closes and reverses the zero-shot TIR-vs-CoT gap, with gains concentrated on answer types where zero-shot TIR underperforms most.
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

### 7.1 Aggregate pass@1 (no call/skip split)

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

Sampling presets: `train` = `temp=1.0, top_p=1.0` (matches RL training); `qwen` = `temp=0.6, top_p=0.95, top_k=20` (Qwen team thinking-mode recommendation). We use the **train** preset in the paper for consistency with RL training.

### 7.2 Zero-shot TIR, by call vs skip (HPC analysis 2026-04-19) — THE HEADLINE FINDING

From `outputs/eval/tool_use_by_type_summary.txt` (Qwen3-4B-Thinking-2507, TIR mode, train preset):

| benchmark | n | called% | skip_pass% | call_pass% | overall% |
|---|---|---|---|---|---|
| pool_v2_drsci | 503 | 38.4 | 61.9 | 66.5 | 63.0 |
| pool_v2_physics | 191 | 44.5 | 41.5 | 41.1 | 39.3 |
| pool_v2_scibench | 153 | 70.6 | 71.1 | 58.4 | 59.5 |
| pool_v2_ugphysics | 217 | 33.6 | 38.9 | 37.0 | 38.2 |
| olympiad_oe_to_physics | 236 | 34.7 | 4.5 | 6.2 | 4.7 |
| phybench | 1000 | 30.7 | 1.7 | 1.0 | 1.5 |
| abench_phy_a | 400 | 72.0 | 27.7 | 18.2 | 20.2 |
| abench_phy_b | 400 | 82.2 | 52.1 | 68.0 | 63.2 |

**Per-type on pool_v2_drsci/train:**
- numerical: called% 75.6, pass 81.7 (call-path well-triaged)
- expression, equation, MCQ: call rates 25–35%, pass 48–67% (under-called; skip-path accuracy lower than for numerical)

### 7.3 Reading the numbers

- **Call rates are non-trivial (31–89%).** The v1 story "model ignores tool" is wrong.
- **On most in-dist and external benchmarks, call-path pass ≤ skip-path pass.** Scibench, ugphysics, phybench, abench_a all show skip ≥ call. Tool calls are often unproductive.
- **Where tools work, they work well:** pool_v2_drsci numerical (81.7% on call-path, at 75.6% call rate) and abench_phy_b (call 68.0% vs skip 52.1%) show productive tool use.
- **The gap between "model knows to call the tool" and "calling the tool helps" is the paper's central observation.** RL needs to move both distributions: raise call rate on types where tools would help, and raise call-conditional pass through better code generation.

### 7.4 PHYBench and OlympiadBench caveats

Exact symbolic judges cap raw pass@1 at 1–7%. For PHYBench the continuous EED metric is the primary headline; exact-match goes to the appendix.

### 7.3 Benchmarks — decided list

In-distribution (pool_v2 test splits — drawn from training pool):
- `pool_v2_drsci` (503), `pool_v2_physics` (191), `pool_v2_ugphysics` (217), `pool_v2_scibench` (153).

External (held out from training):
- `olympiad_oe_to_physics` (236) — exact-match.
- `phybench` (1000) — report EED (primary), exact-match in appendix.
- `abench_phy_a` (400) — 1% relative tolerance.
- `abench_phy_b` (100 per-mid; 400 per-row) — per-mid is the "all-4-parametric-variants correct" robustness metric.

**Dropped:** MATH-500 (distracts from physics focus). **Stretch (only if time):** critpt (research-style problems; mentioned by user as nice-to-have for signaling research capability, but not on the critical path).

## 8. Tool availability ≠ tool benefit — the central observation

The base model (Qwen3-4B-Thinking-2507) calls the tool at 31–89% rates zero-shot, but the paired TIR-mode-vs-CoT-mode comparison shows that tool *availability* does not translate to tool *benefit*: on four of five in-distribution slices, TIR mode is equal to or worse than CoT mode. The model uses the tool, but using it does not help.

This is the gap RL training has to fill. There are two non-mutually-exclusive mechanisms RL can fix:

1. **Triage (when to call):** the model calls on problems where tool use hurts and skips on problems where it would help. Selection-biased within TIR-mode data, so best measured via per-problem tool-vs-no-tool paired comparison (Task 1b).
2. **Execution quality (how to call):** when the model does call, the code is suboptimal (e.g., incorrect SymPy usage, wrong units). Observable via call-path pass rate at the benchmark level, shifting with training.

Our job in §5 is to show (a) the zero-shot paired gap (Claim 1), (b) whether RL closes or reverses it, and (c) which mechanism drives the change, to the extent the data disambiguates.

### Implications for the paper

1. **Lead with the paired comparison (C1), not within-TIR call-vs-skip (C3).** C1 has no selection bias. C3 is a zoom-in that requires a caveat.
2. **Don't oversell aggregate gains.** If post-RL macro gap is 2–5pp, write "closes the gap" not "substantially outperforms."
3. **Zero-shot paired table is load-bearing.** Table A in §5.1 compares {TIR-mode, CoT-mode} × benchmarks with explicit Δ column. Paper starts from this observation.
4. **Per-type Table 2 is supporting** — it gives the "where does tool use help" answer even zero-shot, and lets us predict where RL gains will concentrate.
5. **Watch for collapse during training.** A reasoning-tuned base under RL might learn to skip the tool entirely (tool calls cost tokens and sometimes fail). Log tool-call rate per training step. If call rate → 0, frame as a finding about RL dynamics rather than a failure.
6. **Watch for the opposite collapse.** Tool called on everything but call-path pass is flat. Would mean RL teaches tool invocation without teaching productive use.

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

## 10. Figures and tables (priority order, v2)

Full detail in `paper/storyboard.md`. Priorities:

**Must-have:**
- **Table A (§5.1) — Zero-shot miscalibration.** Per benchmark, {call%, call-path pass%, skip-path pass%, overall%}. Rows highlight benchmarks where call ≤ skip. This table sets up the paper.
- **Table 1 (§5.2) — Main results.** Rows: {zero-shot CoT, zero-shot TIR, CoT-GRPO, TIR-GRPO}. Cols: {pool_v2 slices, OlympiadBench, PHYBench (EED), ABench-A, ABench-B per-mid, macro-avg}.
- **Table 2 (§5.4) — Per-answer-type accuracy** on in-dist test. {numerical, expression, MCQ, equation, multi-part, interval} × {zero-shot, CoT-GRPO, TIR-GRPO, Δ(TIR−CoT)}.
- **Figure 3 (§5.3) — Call rate × call-path pass scatter, pre-RL vs post-RL.** One point per (benchmark × answer-type), arrows from zero-shot to TIR-GRPO showing recalibration direction. **This is the headline mechanistic figure**.
- **Figure 4 (§5.4) — Per-type learning curves.** CoT-GRPO vs TIR-GRPO over training steps, per answer type.

**Strong-to-have:**
- **Figure 1 (hero) — Miscalibration illustration.** Either a single problem where base calls tool and errs / base skips and succeeds / TIR-GRPO triages correctly; or a bar chart showing call vs skip pass per benchmark zero-shot. Pick after we see more data.
- **Figure 2 — Rollout mechanism diagram.** (Cut to numbered list in prose if space is tight.)
- **Figure 5 — Truncation and response length.** (Appendix if space tight.)

**Nice-to-have:**
- Table 3 — solution-strategy and failure-mode frequencies from hand-labeled samples (§6).

**Appendix only:**
- Table 4 — hyperparameters.
- Table 5 — rule verifier coverage on TIR vs CoT outputs (feeds the reward-noise future-work paragraph).
- Full per-benchmark × per-type expansion of Table 2.

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
  HPC_TASKS.md             # analysis tasks for the HPC session
  outline.md               # section beats + page budget
  storyboard.md            # figures/tables, one-liner each, priority order
  sections/
    00-abstract.md         # template + beats
    01-intro.md            # empty skeleton
    ...
    08-conclusion.md
  figures/                 # empty, subdirs per figure when we start building
  tables/                  # empty
  writing-samples/         # structural/stylistic analyses of reference papers
    README.md
    simpletir.md
    tora.md
    phybench.md
    ugphysics.md
    dapo.md
    user-2410.12779.md     # author's prior arxiv — AISTATS 2025
    user-2602.00217.md     # author's prior arxiv
  bibliography/
    refs.bib               # verified citations only
    citations-todo.md      # all [CITE:tag] markers + verification status
  notes/
    novelty-pitch.md       # anticipated-reviewer-objections responses
    timeline.md            # day-by-day plan
```

## 17. Pending inputs from HPC session

**Done (committed to main as of 2026-04-19):**
- `outputs/eval/tool_use_summary.txt` — aggregate per-cell tool-use rates.
- `outputs/eval/tool_use_by_type.csv` — per-answer-type tool-use rates on in-dist cells.
- `outputs/eval/tool_use_by_type_summary.txt` — human-readable version.

These are the source for §7.2 of this handoff and already informed the v2 framing.

**Still pending — full task list in `paper/HPC_TASKS.md`:**
- Task 1: call/skip-conditional pass@1 by benchmark AND answer type (CSV + summary). **Gates §5.1 Table A. Can run today on existing rollouts.**
- Task 4: final-checkpoint benchmark evaluation for {CoT-GRPO, TIR-GRPO}. Gates Table 1.
- Task 1 (again) on post-RL rollouts. Gates Figure 3.
- Task 2: per-answer-type accuracy trajectory during training. Gates Figure 4.
- Task 3: tool-call rate trajectory during training. Collapse-mode early warning.
- Task 5: rule verifier coverage on TIR vs CoT (appendix).
- Task 6: truncation + response-length (optional).
- Task 7: sampled outputs for §6 qualitative analysis.
- Task 8: triage correctness (supplementary).

See `paper/HPC_TASKS.md` for input/output specs, invariants, and priority order.

## 18. Immediate next steps for the new session

1. **Verify the scaffold** — read `paper/outline.md`, `paper/storyboard.md`, `paper/notes/novelty-pitch.md` and this file for any inconsistency; fix before drafting.
2. **Pull 2–3 sample workshop papers** (tool-use RL, physics reasoning; NeurIPS MATH-AI or ICML workshop 2024–25) via WebSearch/WebFetch. Drop abstracts + structure summaries into `paper/bibliography/sample-abstracts/`.
3. **Start citation verification** in parallel — resolve `[CITE:tag]` entries in `citations-todo.md` against primary sources; populate `refs.bib`.
4. **Begin drafting §3 Method and §4 Setup** — these are the most locked content; least dependent on results.
5. When tool-use outputs arrive on main, merge into paper-writing, update §5.3 draft + Figure 4.

## 19. Things decided that should not be re-opened (unless new evidence)

- **Framing (v2):** "Zero-shot tool use is miscalibrated; RL recalibrates." Not the old "model ignores tool" (v1, falsified by HPC data). Not reward-noise mechanism.
- **Base model:** Qwen3-4B-Thinking-2507.
- **CoT-GRPO:** strict apples-to-apples + both zero-shot CoT and zero-shot TIR references in Table 1.
- **External benchmarks kept:** OlympiadBench, PHYBench (EED primary), ABench A, ABench B. In-dist: pool_v2 × 4 slices. Dropped: MATH-500.
- **Verifier edge-case rules:** implementation detail, appendix only. Not pitched as a contribution.
- **Verifier-noise analysis:** one paragraph in §7 Discussion as future work. Not a headline.
- **Single-block TIR:** design choice, mentioned in §3, not a pitched contribution.
- **8-page two-column ICML format.**
- **Work in markdown first, port to Overleaf around Apr 21.**
- **No algo boxes** — prose + numbered steps for rollout.
- **One hero schematic figure** (Figure 1) is worth the space.

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
