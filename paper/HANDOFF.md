# HANDOFF — paper writing session context

**Created:** 2026-04-19. **Last revised:** 2026-04-21 late (v4 — base model switched from Qwen3-4B-Thinking-2507 to Qwen3-4B base hybrid instruct/think; thesis provisional pending paired data on new rollouts).
**Purpose:** full, loss-free context dump of the planning discussion so a fresh session can resume paper writing without re-deriving decisions.
**Branch:** `paper-writing` (this branch). Not merged to main.
**Main only has:** `docs/tool_use_analysis_handoff.md` (the HPC-side runbook for tool-use analysis).

## Revision log

- **v4 (2026-04-21 late):** Base model switched from `Qwen3-4B-Thinking-2507` to `Qwen3-4B` (hybrid instruct/think — neither base-base nor RL'd thinking). Rationale: Thinking model was hard to train. All 32 zero-shot eval cells rerun on the new model. Aggregate picture is similar ("TIR ≠ consistent benefit") but the specific benchmark-level pattern is different: OlympiadBench now favors TIR (+5.1pp), ABench-B per-mid now penalizes TIR by 9pp, in-dist deltas are smaller. The v3.1 two-failure-modes thesis (expression hurts, numerical high-use low-benefit) was grounded in the Thinking model's paired per-type data. **Thesis is now provisional** pending Task 1 + Task 1b re-run on the Qwen3-4B rollouts.
- **v3.1 (2026-04-21 late):** Paired per-problem Task 1b and Task 1 outputs landed from HPC session (commit `713a737`). Numbers sharpened the story: the aggregate "4 of 5" claim in v3 was slightly overstated; the correct headline is **expression-type paired loss across 4 of 5 benchmarks (−1.7 to −10.7 pp)**. Numerical type: high call (60–82%) but near-zero paired effect. Updated §4 thesis and §7 numbers.
- **v3 (2026-04-21):** Raised the concern that within-TIR `call-pass% < skip-pass%` has selection bias (the model's choice to call is not random). Switched the thesis to lead with the **paired TIR-mode-vs-CoT-mode zero-shot comparison** (same problem, same prompt, only tool availability differs — no selection bias). Added Task 1b to HPC_TASKS for per-problem paired comparison. Discussed fallback path if post-RL results are weak. Also refactored bibliography/sample-abstracts/ → writing-samples/ and shifted those docs toward paper-writing style analysis.
- **v2 (2026-04-19):** HPC tool-use analysis showed the base model DOES call tool at 31–89% rates. Reframed thesis around "tool-use miscalibration + RL recalibration" rather than the earlier "model ignores tool" story. Updated §4, §7, §8 accordingly. Training infra now 4 train + 6 rollout + 1 xVerify, staleness=1. Training data moved to a difficulty-filtered subset (size TBD).

Read this first. Everything else in `paper/` follows from what's here.

---

## 1. Venue

- **Workshop:** ICML 2026 AI4Physics — https://ai4physics-workshop.github.io/
- **Deadline:** April 24, 2026 AOE (≈ Apr 25 evening local; aim to submit Apr 23 evening for safety).
- **Page limit:** 8 pages ICML 2-column, references excluded.
- **Topic fit:** "Physics-centric Scientific Reasoning with LLMs and Agents — tool-augmented agents" is a named direction in the CFP. The paper is squarely on-topic.

## 2. Base model (final — v4, 2026-04-21)

**`Qwen/Qwen3-4B`** — the hybrid instruct/think base. Not the pure base, not the thinking-RL'd variant (`Qwen3-4B-Thinking-2507`).

Rationale: the Thinking model was hard to train; the hybrid base is more stable. The hybrid base still supports `enable_thinking=True` — it just hasn't been further RL'd for chain-of-thought reasoning.

Cross-model zero-shot comparison (TIR/train on in-dist + olympiad):

| | scibench | physics | ugphysics | olympiad | drsci |
|---|---|---|---|---|---|
| Qwen3.5-4B (old target) | 0.30 | 0.29 | 0.20 | 0.10 | 0.41 |
| **Qwen3-4B (current target)** | **0.53** | **0.40** | **0.32** | **0.19** | **0.59** |
| Qwen3-4B-Thinking (prior v3 target) | 0.60 | 0.39 | 0.38 | 0.05 | 0.63 |

Qwen3-4B base is within ~5 pp of Thinking on most cells and beats it on OlympiadBench (Thinking's elaborate reasoning produces less-canonical symbolic forms that the strict AutoScoringJudge rejects). All 32 zero-shot cells are complete for Qwen3-4B; numbers are in §7.

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

## 4. Thesis — provisional (v4, pending Task 1b on Qwen3-4B)

### Safe fallback (v4.0, works today without paired data on the new model)

> Tool availability does not consistently translate to benefit on physics for a 4B hybrid instruct/think model: aggregate TIR-mode vs CoT-mode comparison is heterogeneous across benchmarks — TIR wins on OlympiadBench (+5.1 pp), loses on ABench-Physics-B per-mid (−9.0 pp) and SciBench (−4.6 pp), and is within ±2 pp on the remaining slices. We show tool-integrated RLVR converts this inconsistent zero-shot behavior into consistent benefit: TIR-GRPO outperforms a strict CoT-GRPO baseline at matched training budget, with per-type gains aligned with where zero-shot TIR underperforms most.

### Sharpened version (v4.1 — activate if paired per-type pattern survives)

If Task 1 + Task 1b on Qwen3-4B rollouts reveal the same expression-hurts / numerical-high-use-low-benefit structure we saw on Qwen3-Thinking, we can revert to the v3.1 two-failure-modes framing. That decision is made once the paired CSVs land for the new model.

### Why this is now provisional

The v3.1 thesis ("expression is where tool hurts; numerical is high-use low-benefit") was anchored in Qwen3-Thinking's per-type paired data (`paired_tir_vs_cot_by_type.csv`). Whether the same structural pattern holds for Qwen3-4B base is an open question — the aggregate landscape has shifted (OlympiadBench flipped, ABench-B per-mid shifted), and per-type may have shifted with it.

### Why this framing, not earlier drafts

- **v1 (dropped 2026-04-19):** "Base model ignores tool; RL teaches use." Falsified by HPC analysis showing call rates 31–89%.
- **v2 (dropped 2026-04-21):** "Within TIR mode, call-path pass < skip-path pass → miscalibration." Real but **selection-biased** — the model's choice of when to call isn't random, so low call-path pass might reflect problem difficulty rather than tool-use quality. A careful reviewer would flag this.
- **v3 (current):** Leads with the paired TIR-vs-CoT aggregate comparison (same problems, same prompts, only tool availability differs — *no* selection bias). The within-TIR call-vs-skip observation becomes a secondary zoom-in with an explicit selection-bias caveat.

### Claim evidence levels (v4, post-model-switch)

**These claims refer to the OLD model (Qwen3-4B-Thinking-2507) — kept for reference; reusable if per-type patterns survive on the new model.**

| Claim | Evidence (Thinking model) | Transfer to Qwen3-4B base |
|---|---|---|
| C1. Paired per-benchmark: TIR ≤ CoT on 3 of 4 in-dist pool_v2 | W−L drsci −1.4, physics −4.2, ugphysics −2.8, scibench +3.3 | **Confirmed aggregate-level** on new model (drsci −0.8, physics +1.1, ugphysics −1.4, scibench −4.6; see §7.1) |
| C2. Paired per-type: tool hurts on expression-type across 4 of 5 benchmarks | drsci −4.9, physics −10.7, ugphysics −1.7, olympiad −1.7, phybench +0.5 | **Unknown — Task 1b not yet rerun on Qwen3-4B rollouts** |
| C3. Paired per-type: numerical is high-use low-benefit | call% 60–82 with ±5.5 pp effect | **Unknown — needs Task 1b re-run** |
| C4. Base model calls tool at non-trivial rates overall | 31–89% aggregate | **Likely similar; needs Task 1 re-run for confirmation** |
| C5. Within-TIR call-path pass ≠ skip-path pass | Task 1 output, selection-biased | Needs Task 1 re-run |
| C6. RL recalibrates (TIR-GRPO > CoT-GRPO) | Needs training results | Needs training results |

**Action:** HPC session to rerun Task 1 + Task 1b on the new Qwen3-4B zero-shot rollouts (`outputs/eval/<bench>/<mode>__Qwen-Qwen3-4B__<tag>/`). Scripts are already parameterized via `--outputs-eval`.

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

## 7. Zero-shot eval results — Qwen3-4B base (current target, v4)

### 7.1 Aggregate pass@1, Qwen3-4B base

**In-distribution (pool_v2 test):**

| benchmark (n) | tir/train | cot/train | tir/qwen | cot/qwen | TIR−CoT (train) |
|---|---|---|---|---|---|
| pool_v2_scibench (153) | 0.533 | **0.579** | 0.579 | 0.579 | **−4.6** |
| pool_v2_physics (191) | 0.398 | 0.387 | 0.403 | 0.403 | +1.1 |
| pool_v2_ugphysics (217) | 0.318 | 0.332 | 0.309 | **0.336** | −1.4 |
| pool_v2_drsci (503) | 0.592 | **0.600** | 0.571 | 0.592 | −0.8 |

**External (held out):**

| benchmark (n) | tir/train | cot/train | tir/qwen | cot/qwen | TIR−CoT (train) |
|---|---|---|---|---|---|
| olympiad_oe_to_physics (236) | **0.191** | 0.140 | 0.157 | 0.153 | **+5.1** |
| phybench (1000) exact | 0.015 | 0.010 | 0.020 | 0.019 | +0.5 |
| phybench (1000) mean_EED | 2.93 | 2.53 | 3.21 | **3.29** | +0.40 |
| abench_phy_a (400) | 0.138 | 0.155 | 0.135 | **0.160** | −1.8 |
| abench_phy_b (400) per-row | 0.590 | **0.620** | 0.588 | 0.618 | −3.0 |
| abench_phy_b (100) per-mid | 0.390 | **0.480** | 0.410 | 0.470 | **−9.0** |

**Reading:** heterogeneous across benchmarks — TIR wins meaningfully on OlympiadBench (+5.1); ties or slight loss in-dist; loses on ABench-B per-mid (−9.0) and SciBench (−4.6). The "TIR ≈ CoT" statement holds in aggregate magnitude (|Δ| ≤ 5 pp for most cells) but directionality varies.

Sampling presets: `train` = `temp=1.0, top_p=1.0` (matches RL training); `qwen` = `temp=0.6, top_p=0.95, top_k=20`. We use **train** preset in the paper for consistency with RL.

### 7.1-archive. Qwen3-4B-Thinking zero-shot (prior v3 target)

Archived — see git history for v3.1 numbers. Kept only as reference for cross-model comparison in §2 above.

### 7.2 Paired TIR-mode vs CoT-mode (Task 1b, headline evidence) — STALE FOR QWEN3-4B-THINKING; NEEDS RE-RUN FOR QWEN3-4B BASE

**The tables below are for Qwen3-Thinking (prior target).** They remain useful for understanding the analysis structure. Once HPC reruns Task 1b on Qwen3-4B rollouts, this section will be replaced with the new numbers.


From `outputs/eval/paired_tir_vs_cot_by_benchmark.csv` (train preset, xverify-7b judge unless noted):

| benchmark | n | W | L | TP | TF | **W−L** | TIR% | CoT% | TIR_call% |
|---|---|---|---|---|---|---|---|---|---|
| pool_v2_drsci | 503 | 40 | 47 | 277 | 139 | **−1.4** | 63.0 | 64.4 | 38.4 |
| pool_v2_physics | 191 | 5 | 13 | 70 | 103 | **−4.2** | 39.3 | 43.5 | 44.5 |
| pool_v2_scibench | 153 | 12 | 7 | 79 | 55 | **+3.3** | 59.5 | 56.2 | 70.6 |
| pool_v2_ugphysics | 217 | 12 | 18 | 71 | 116 | **−2.8** | 38.2 | 41.0 | 33.6 |
| olympiad_oe_to_physics | 236 | 8 | 9 | 3 | 216 | **−0.4** | 4.7 | 5.1 | 34.7 |
| phybench (exact) | 1000 | 7 | 2 | 8 | 983 | **+0.5** | 1.5 | 1.0 | 30.7 |
| phybench (EED mean) | 1000 | — | — | — | — | **TIR−CoT = +0.64** | (cont.) | (cont.) | 30.7 |
| abench_phy_a | 400 | 19 | 16 | 62 | 303 | **+0.8** | 20.2 | 19.5 | 72.0 |
| abench_phy_b (per-row) | 400 | 9 | 9 | 244 | 138 | **0.0** | 63.2 | 63.2 | 82.2 |
| abench_phy_b (per-mid, n=100) | 100 | 3 | 2 | 47 | 48 | **+1.0** | 50.0 | 49.0 | — |

W = TIR correct + CoT wrong; L = TIR wrong + CoT correct; TP = both correct; TF = both wrong.

**3 of 4 in-dist pool_v2 slices show TIR ≤ CoT** (drsci, physics, ugphysics); scibench is the one exception. External benchmarks are near-tie.

### 7.3 Paired per-type (the SHARPEST finding)

From `paired_tir_vs_cot_by_type.csv` (train preset, n ≥ 10 rows, scalar answer types):

| benchmark | answer_type | n | **W−L** | TIR% | CoT% | call% |
|---|---|---|---|---|---|---|
| pool_v2_drsci | numerical | 82 | **+0.0** | 81.7 | 81.7 | 75.6 |
| pool_v2_drsci | expression | 82 | **−4.9** | 47.6 | 52.4 | 30.5 |
| pool_v2_drsci | equation | 108 | −0.9 | 51.9 | 52.8 | 25.0 |
| pool_v2_drsci | mcq | 231 | −0.9 | 67.1 | 68.0 | 34.2 |
| **pool_v2_physics** | **expression** | 56 | **−10.7** | **41.1** | **51.8** | **14.3** |
| pool_v2_physics | numerical | 71 | −2.8 | 50.7 | 53.5 | 67.6 |
| pool_v2_physics | mcq | 11 | 0.0 | 63.6 | 63.6 | 18.2 |
| pool_v2_scibench | numerical | 153 | +3.3 | 59.5 | 56.2 | 70.6 |
| pool_v2_ugphysics | expression | 60 | −1.7 | 36.7 | 38.3 | 11.7 |
| pool_v2_ugphysics | numerical | 73 | −5.5 | 43.8 | 49.3 | 60.3 |
| pool_v2_ugphysics | equation | 29 | 0.0 | 34.5 | 34.5 | 10.3 |
| pool_v2_ugphysics | true_false | 13 | −7.7 | 61.5 | 69.2 | 23.1 |
| olympiad_oe_to_physics | expression | 116 | −1.7 | 3.4 | 5.2 | 23.3 |
| olympiad_oe_to_physics | numerical | 113 | +0.9 | 6.2 | 5.3 | 45.1 |
| phybench | expression | 1000 | +0.5 | 1.5 | 1.0 | 30.7 |
| abench_phy_a | numerical | 400 | +0.8 | 20.2 | 19.5 | 72.0 |
| abench_phy_b | numerical | 400 | 0.0 | 63.2 | 63.2 | 82.2 |

**Two clean patterns:**

1. **Expression-type: tool hurts on 4 of 5 benchmarks.** W−L: drsci −4.9, physics −10.7, ugphysics −1.7, olympiad −1.7, phybench +0.5. The physics slice is the starkest: the model calls the tool on only 14% of expression problems, and when it does, it loses 10.7pp against CoT.
2. **Numerical-type: high call rate (60–82%) but near-zero aggregate effect.** The model attempts tool use most on numerical problems — yet TIR and CoT pass rates are within ±5.5 pp across all benchmarks, and zero on three of them. The model is using the tool but not extracting benefit.

### 7.4 Within-TIR call-vs-skip (Task 1, secondary — selection-biased)

From `call_vs_skip_by_benchmark.csv` (train preset, overall):

| benchmark | call% | call_pass% | skip_pass% | Δ |
|---|---|---|---|---|
| pool_v2_drsci | 38.4 | 64.8 | 61.9 | +2.8 |
| pool_v2_physics | 44.5 | 36.5 | 41.5 | **−5.0** |
| pool_v2_scibench | 70.6 | 54.6 | 71.1 | **−16.5** |
| pool_v2_ugphysics | 33.6 | 37.0 | 38.9 | −1.9 |
| abench_phy_a | 72.0 | 17.4 | 27.7 | **−10.3** |
| abench_phy_b | 82.2 | 65.7 | 52.1 | +13.5 |

Striking but selection-biased — the model's call decision is not random, so the gap reflects both "call quality" and "which problems the model chooses to call on." Use as a descriptive zoom-in; anchor causal claims on paired per-type data (7.3).

### 7.5 PHYBench and OlympiadBench caveats

Exact symbolic judges cap raw pass@1 at 1–7%. For PHYBench the continuous EED metric is the primary headline (0–100; higher = closer to exact); `tir−cot` EED mean difference is +0.64 with train preset. Exact-match goes to the appendix.

### 7.3 Benchmarks — decided list

In-distribution (pool_v2 test splits — drawn from training pool):
- `pool_v2_drsci` (503), `pool_v2_physics` (191), `pool_v2_ugphysics` (217), `pool_v2_scibench` (153).

External (held out from training):
- `olympiad_oe_to_physics` (236) — exact-match.
- `phybench` (1000) — report EED (primary), exact-match in appendix.
- `abench_phy_a` (400) — 1% relative tolerance.
- `abench_phy_b` (100 per-mid; 400 per-row) — per-mid is the "all-4-parametric-variants correct" robustness metric.

**Dropped:** MATH-500 (distracts from physics focus). **Stretch (only if time):** critpt (research-style problems; mentioned by user as nice-to-have for signaling research capability, but not on the critical path).

## 8. Two clean failure modes — where the paper gets its mechanism

The paired data reveals two structurally different failure modes for zero-shot tool use on a reasoning-tuned base:

**Failure mode A — "Tool hurts on expression-type."** On expression answers, the model calls the tool rarely (11–30%), and when it does, it loses 2–11 pp against plain reasoning. Across 4 of 5 benchmarks with adequate n (pool_v2_drsci, physics, ugphysics, olympiad), paired W − L on expression is negative. The starkest case is pool_v2_physics: 14.3% call rate, TIR 41.1% vs CoT 51.8% — **a 10.7pp loss with almost no tool use**. Reading: the small minority of tool calls are actively wrong.

**Failure mode B — "Tool is high-use, low-benefit on numerical."** On numerical answers, the model calls the tool 60–82% of the time — its default behavior — yet paired TIR vs CoT is within ±5.5 pp across every benchmark, and zero on three of them. Reading: the model is doing the work of making the call but not extracting accuracy from the result.

These two modes argue for different RL interventions:
- **Mode A (expression):** RL should suppress unproductive calls — either teach the model to skip, or teach it to produce calls whose output actually helps.
- **Mode B (numerical):** RL should improve call quality — better SymPy usage, better interpretation of tool output, better code.

A single reward signal (binary correctness on the final boxed answer) can in principle drive both. Whether it *does* drive both — or drives a collapse (call rate → 0 or → 1) — is what the TIR-GRPO training run is for.

### Implications for the paper

1. **Lead §5 with paired per-type data (C2), not within-TIR call-vs-skip (C5).** C2 is selection-bias-free and produces the sharpest mechanism claim. C5 is a zoom-in that requires a caveat.
2. **Don't oversell aggregate gains.** Post-RL aggregate macro-gap of 2–5 pp is plenty — the per-type story carries the paper.
3. **Two-plot hero figure:** top panel = paired TIR-vs-CoT W−L per benchmark (aggregate); bottom panel = paired W−L per answer-type across benchmarks (expression flagged).
4. **Watch for training-time collapse.** Log tool-call rate per step per type. If expression call rate → 0 under RL, that's actually *a productive resolution of failure mode A* — report it as such, not as a failure.
5. **Watch for opposite collapse.** Tool called on everything but call-path pass stays flat. Would mean RL teaches invocation without teaching productive use — failure mode B unresolved.

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

- **Framing (v4 provisional):** "Tool availability does not consistently translate to benefit; RL converts inconsistent zero-shot behavior into consistent benefit." Upgrade path to v3.1 two-failure-modes framing is available pending Task 1b on Qwen3-4B rollouts.
- **Base model (v4):** **Qwen/Qwen3-4B** (hybrid instruct/think). Not Qwen3-4B-Thinking-2507 (hard to train). Not Qwen3.5-4B (prior, dropped).
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
