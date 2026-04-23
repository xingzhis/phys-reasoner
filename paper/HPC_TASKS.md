# HPC analysis tasks — for the paper

**Audience:** the HPC-side session that runs jobs on Perlmutter / Yale cluster.
**Output destination:** `outputs/eval/` (commit to main; paper-writing branch picks up via `git merge origin/main`).

All tasks below specify **what** to compute and **what format** the output should be in. Implementation details (which script to extend, how to plumb args) are up to you — match the style of existing `eval/analyze_tool_use.py` and `eval/analyze_tool_use_by_type.py`.

**Rule of thumb:** same scripts should apply to (a) zero-shot rollouts in `outputs/eval/<cell>/rollouts.parquet` (already on disk) and (b) post-RL rollouts at each checkpoint, once training produces them. Parameterize by rollouts-parquet-path, not by hardcoded cell list.

---

## Task 1 — Call/skip-conditional pass@1, by benchmark AND by answer type

**Why:** Table A (§5.1) and Figure 3 (§5.3). This is the headline of the paper.

**Input:** a rollouts parquet with per-row fields:
- `problem_id`
- `answer_type` (numerical / expression / equation / MCQ / interval / multi-part)
- `called_tool` (bool — did this rollout emit a successful `<tool_call>`?)
- `pass` (bool — did this rollout's final `\boxed{}` verify correct?)
- `benchmark` (e.g., `pool_v2_drsci`, `olympiad_oe_to_physics`)
- `preset` (`train` or `qwen`)
- `condition` (`zero_shot_TIR`, `zero_shot_CoT`, `CoT-GRPO@step<k>`, `TIR-GRPO@step<k>`)

**Output 1:** `outputs/eval/call_vs_skip_by_benchmark.csv` with columns:
- `condition`, `preset`, `benchmark`, `n`, `call_pct`, `call_pass_pct`, `skip_pass_pct`, `overall_pass_pct`, `call_minus_skip_pass`

**Output 2:** `outputs/eval/call_vs_skip_by_benchmark_type.csv` with columns:
- `condition`, `preset`, `benchmark`, `answer_type`, `n`, `n_called`, `n_skipped`, `call_pct`, `call_pass_pct`, `skip_pass_pct`, `overall_pass_pct`, `call_minus_skip_pass`

**Output 3 (human-readable):** `outputs/eval/call_vs_skip_summary.txt` — pretty-printed version of Output 1 and the per-type summary for pool_v2_drsci.

**Test on:** the zero-shot rollouts you already have in `outputs/eval/` (8 in-dist TIR cells + external cells where applicable).
**Apply later to:** post-RL checkpoint rollouts when they land.

---

## Task 0 (2026-04-21 late) — URGENT: rerun Tasks 1 and 1b on Qwen3-4B zero-shot rollouts

**Context:** base model switched from Qwen3-4B-Thinking-2507 to Qwen3-4B (hybrid). All 32 zero-shot eval cells have been rerun and live under
`outputs/eval/<bench>/<mode>__Qwen-Qwen3-4B__<tag>/` (distinct from the older `Qwen-Qwen3-4B-Thinking-2507__` directories).

**Action:** rerun Task 1 and Task 1b on the new rollouts. Scripts are already parameterized — just point them at the Qwen3-4B cells.

**Outputs:** replace the existing `outputs/eval/call_vs_skip_*.csv` and `paired_tir_vs_cot_*.csv` with the new-model versions. Consider saving the old versions to `outputs/eval/archive_qwen3_thinking/` first, since they still have pedagogical value.

**Priority:** top priority, blocks §5.1 Table A and Table B in the paper. Everything else (Tasks 2–8) continues to be gated on training.

**Reason we care:** the v3.1 thesis ("expression hurts, numerical high-use low-benefit") was rooted in the Thinking model's per-type paired data. Whether the same pattern survives the model switch is open — aggregate patterns already differ (OlympiadBench flipped, ABench-B per-mid shifted). The new paired data decides whether we revert to a crisp two-failure-modes thesis (v4.1 sharpened) or stay on the safer aggregate-only v4.0 fallback.

---

## Task 1b — Per-problem paired TIR-mode vs CoT-mode comparison (the cleanest evidence)

**Why:** Table A in §5.1 and the strongest form of Claim 1 (no selection bias). For each problem evaluated in both TIR mode and CoT mode zero-shot, classify the pair as Win / Loss / Tie:
- **Win:** TIR correct AND CoT wrong
- **Loss:** TIR wrong AND CoT correct
- **Tie-pass:** both correct
- **Tie-fail:** both wrong

**Input:** pairs of rollouts parquets — one TIR-mode (`outputs/eval/<benchmark>/train/TIR/rollouts.parquet`) and one CoT-mode (`outputs/eval/<benchmark>/train/CoT/rollouts.parquet`) — same base model, same preset, same problems.

For each problem_id that appears in both, join on problem_id and compute the pairwise outcome.

**Output 1 — by benchmark:** `outputs/eval/paired_tir_vs_cot_by_benchmark.csv`:
- `benchmark`, `preset`, `n_paired`, `n_win`, `n_loss`, `n_tie_pass`, `n_tie_fail`, `win_rate`, `loss_rate`, `win_minus_loss`

**Output 2 — by benchmark × answer_type:** `outputs/eval/paired_tir_vs_cot_by_type.csv`:
- `benchmark`, `preset`, `answer_type`, `n_paired`, `n_win`, `n_loss`, `n_tie_pass`, `n_tie_fail`, `win_rate`, `loss_rate`, `win_minus_loss`

**Output 3 (human-readable):** `outputs/eval/paired_tir_vs_cot_summary.txt`.

**Notes:**
- If there are rollout counts >1 per problem (rollout.n > 1 in zero-shot), aggregate to problem-level pass@1 first (majority or mean), then classify.
- Keep only problems that were evaluated in **both** modes. Record the join yield.
- Use the same judge (xVerify-7B or benchmark-provided scorer) for both sides.

**Test on:** the existing `outputs/eval/` zero-shot rollouts. Both TIR and CoT cells exist for pool_v2 slices and most external benchmarks.
**Apply later to:** post-RL rollouts (TIR-GRPO rollouts vs CoT-GRPO rollouts on the same benchmarks). Same Win/Loss/Tie semantics.

**This is the strongest form of the zero-shot claim and the strongest form of the post-RL comparison.** Run before Task 2.

---

## Task 2 — Per-answer-type accuracy trajectory during training

**Why:** Figure 4 (§5.4) — learning curves per answer type for CoT-GRPO and TIR-GRPO.

**Input:** training-time dev-set evaluation rollouts. VeRL already runs dev eval every `test_freq` steps; ensure `answer_type` is preserved in the eval output parquet.

**Output:** `outputs/training/per_type_curve.csv` with columns:
- `run_id` (TIR-GRPO or CoT-GRPO)
- `step`
- `dev_slice` (dev set source)
- `answer_type`
- `n`
- `pass_pct`

**How to produce:** wrap VeRL's per-step dev-eval rollouts with a post-processing pass that groups by answer_type. Answer type must be in `extra_info` on the dev parquet (already there from data-pipeline Step 0).

**Test on:** first smoke-run checkpoints. Verify answer_type is non-null for every dev row.

---

## Task 3 — Tool-call rate trajectory during training (TIR-GRPO only)

**Why:** early-warning for collapse failure mode (call rate → 0 or → 1). Also input to Figure 3 pre/post arrows.

**Output:** `outputs/training/call_rate_curve.csv` with columns:
- `step`
- `dev_slice`
- `answer_type` (optional — include if available)
- `n`
- `call_pct`
- `call_pass_pct` (if easy; else pass_pct only)

**How to produce:** same dev-eval rollouts as Task 2, filtered to TIR-GRPO. A `called_tool` bool column must be present (trivial: detect `<tool_call>` in the generated text).

**Alerting:** flag if `call_pct` trends monotonically toward 0 or 1 over the first ~300 training steps. That's a signal to discuss.

---

## Task 4 — Final-checkpoint benchmark evaluation for Table 1

**Why:** Table 1 (§5.2).

**Input:** final checkpoint (or best-dev checkpoint) for each of {CoT-GRPO, TIR-GRPO}.

**Output:** `outputs/eval_final/<run_id>/<benchmark>/rollouts.parquet` — same format as the zero-shot `outputs/eval/` cells. Reuse the existing eval launcher.

**Cells to run:** every cell that was run zero-shot for the same model (pool_v2_drsci, pool_v2_physics, pool_v2_scibench, pool_v2_ugphysics, olympiad_oe_to_physics, phybench, abench_phy_a, abench_phy_b). Use **train preset** (temp=1.0, top_p=1.0) for consistency with training.

**For TIR-GRPO:** run in TIR mode (tool available). For CoT-GRPO: run in CoT mode (no tool). Do NOT cross-condition (i.e., don't run CoT-GRPO in TIR mode or vice versa — that's a different experiment we don't have space for).

**After rollouts:** run Task 1 on each resulting parquet to get the call/skip tables for post-RL.

---

## Task 5 — Rule verifier coverage on TIR vs CoT outputs (appendix feed)

**Why:** §7 Discussion future-work paragraph on reward-noise hypothesis; Table 5 in appendix.

**Input:** final-checkpoint rollouts from Task 4.

**Output:** `outputs/analysis/rule_coverage.csv` with columns:
- `condition` (TIR-GRPO or CoT-GRPO)
- `benchmark`
- `answer_type`
- `n_xverify_correct` (rollouts where xVerify-7B marked correct)
- `n_rule_correct_among_xverify_correct` (of those, how many the rule verifier also accepts)
- `rule_coverage` (the ratio)

**How to produce:** re-score each post-RL rollout with both rule verifier AND xVerify-7B. Then for each (condition × benchmark × answer_type), compute the conditional coverage.

**Expected pattern:** rule coverage should be higher for TIR-GRPO than CoT-GRPO, especially on expression types. If it isn't, that's important — tell us.

---

## Task 6 — Truncation + response-length analysis (optional, §5.5 / Figure 5)

**Why:** Figure 5 (truncation panel) and prose claims in §5.5.

**Output:** `outputs/analysis/length_truncation.csv` with columns:
- `condition`, `step` (if training; else `zero_shot`)
- `benchmark`
- `n`
- `truncated_pct` (fraction of rollouts hitting response cap without a `\boxed{}`)
- `response_len_p50`, `response_len_p95`, `response_len_mean`

**Cut trigger:** skip this if time is tight. Body figure is downgradable to appendix.

---

## Task 7 — Sampled outputs for §6 qualitative analysis

**Why:** Table 3 and representative example walkthroughs in §6.

**Output:** `outputs/analysis/qual_samples.jsonl` — 100–150 rollouts per condition (TIR-GRPO, CoT-GRPO) drawn uniformly at random, stratified by answer_type. Each record:
```json
{
  "condition": "TIR-GRPO",
  "benchmark": "pool_v2_drsci",
  "problem_id": "...",
  "answer_type": "expression",
  "question": "...",
  "gold": "...",
  "response": "...full rollout text...",
  "pass": true,
  "called_tool": true
}
```

**Labeling:** not strictly HPC work. The paper-writing session will do a LLM-assisted first pass (strategy category, failure mode) with spot checks.

**Trigger:** after Task 4 completes.

---

## Task 8 — Per-answer-type call-triage correctness (nice-to-have)

**Why:** strengthens §5.3 mechanism claim. "Did the model call when it should have, and skip when it should have?"

**Output:** `outputs/analysis/triage.csv`:
- `condition`, `benchmark`, `answer_type`
- `n_should_call` (where call-path pass > skip-path pass on this type, population-level)
- `call_pct_on_should_call` (fraction of should-call problems where this model actually called)
- `skip_pct_on_should_skip` — symmetric
- `triage_correctness` — weighted sum of "model did the right thing"

**How to produce:** determine the population-level "oracle" triage from the problem's per-type call/skip pass rate. Then score each rollout against that oracle.

**Caveat:** this is noisy for small n. Include it as a supplementary bar chart, not a headline.

**Trigger:** after Task 1 + Task 4 complete.

---

## Running order / priorities

| Order | Task | Blocking? | Notes |
|---|---|---|---|
| 1 | **Task 1b on existing zero-shot rollouts** | yes — gates §5.1 Table A | **Run today. Pure CPU. This is the headline evidence.** |
| 2 | Task 1 on existing zero-shot rollouts | yes — gates §5.2 Table 2a zoom-in | Run today. |
| 3 | Task 4 (post-RL rollouts per benchmark) | yes — gates §5.3 Table 1 | Needs training done. |
| 4 | Task 1b on post-RL rollouts | yes — strongest post-RL claim | After Task 4. |
| 5 | Task 1 on post-RL rollouts | yes — gates Fig 3 | After Task 4. |
| 6 | Task 2 and 3 (training curves) | yes — gates Fig 4 | Needs training logs. Should be streaming. |
| 7 | Task 5 (rule coverage) | no — appendix / discussion | After Task 4. |
| 8 | Task 7 (qualitative samples) | no — §6 nice-to-have | After Task 4. |
| 9 | Task 6 (truncation) | no — appendix fallback | Any time. |
| 10 | Task 8 (triage correctness) | no — supplementary | After Task 1 + 4. |

---

## Invariants to preserve across all outputs

1. **`answer_type` is a normalized string** from the set {`numerical`, `expression`, `equation`, `mcq`, `interval`, `multi-part`, `other`}. Do not emit raw Dr. SCI types — use the normalization already in `src/phys_reasoner/tir/` or `eval/`.
2. **`called_tool` detection is uniform**: a rollout called the tool if-and-only-if it emitted a `<tool_call>` block that parsed AND executed (timeout/error = still called, just failed; pass=False gets attributed to call-path). This matches the existing `analyze_tool_use_by_type.py` semantics.
3. **`preset`** is always logged — `train` for RL-training-consistent, `qwen` for the Qwen-recommended thinking-mode preset.
4. **Pass/fail semantics** use the same judge as the headline table: xVerify-7B for cells without a benchmark-provided scorer; the provided scorer (e.g. PHYBench EED, OlympiadBench exact-match) where available. Log which was used in a `judge` column.

---

## Things NOT to do

- Don't cross-condition rollouts (CoT-GRPO in TIR mode, or TIR-GRPO in CoT mode). Out of scope.
- Don't run MATH-500 — dropped from paper.
- Don't add new benchmarks unless user explicitly asks.
- Don't rescore zero-shot rollouts with anything new — the current `outputs/eval/*/rollouts.parquet` are the source of truth.
- Don't commit large analysis outputs to git if they're multi-GB. Use `outputs/` (gitignored); only the CSVs above go to git.

---

## When results land

Commit CSVs and summaries to main under `outputs/eval/` or `outputs/analysis/`. Paper-writing session merges main in and pulls numbers into Table A, Table 1, Table 2, Figures 3–4.
