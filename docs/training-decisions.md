# Training Decisions & Strategy Notes

Last updated: 2026-03-30

---

## SFT Stage: Conditional on Stage 0 Probe

**Decision:** Run a 100-problem zero-shot TIR probe (Stage 0) before committing to SFT.

**Threshold:** If verifier hit rate on the TIR zero-shot probe ≥ 15% → skip SFT entirely.
If < 15% → generate 2–5k TIR demonstrations via GPT-4o/Claude (auto-exec filtered) and fine-tune to stabilize format.

**Rationale:**
1. Qwen3.5-4B instruct (thinking OFF) may already produce valid single-block TIR format at useful rates, making SFT unnecessary overhead.
2. SFT is only needed to stabilize code format, SymPy usage, and post-execution reasoning — not to teach physics reasoning.
3. Auto-exec filter on SFT demos ensures only working code examples are used.
4. Papers like DAPO and Dr. GRPO show GRPO works from instruct model directly.

**Fallback:** If GRPO training is unstable (reward stuck near zero), warm-start from SFT checkpoint. Treat as last resort.

**Stage 0 probe details:**
- 100 problems stratified across numerical / expression / MCQ
- Measure: execution success rate, verifier hit rate, per-type accuracy
- Prompt for single-block TIR format: `[think]…[code]…[/code][output]…[think]…[answer]\boxed{}`

---

## Goldilocks Data Selection: Strategy B + D

**Context:** Pass@k estimates from a small sample tell us per-stratum Goldilocks rates but not per-problem pass rates across 113k rows. Strategy B uses those stratum-level rates to initialize weighted sampling; Strategy D updates per-problem pass rates iteratively using training checkpoints.

### Why not hard-filter to Goldilocks problems only?

- Corpus: hard filter leaves only ~1,200 problems (18% of 6,866) — too small for stable GRPO.
- Dr. SCI: 18% of 107k ≈ 19k problems would survive, but the filter requires per-problem pass rates we don't have for 99% of rows.
- Pass@k estimates from k=8 are noisy near the boundary.
- The Goldilocks zone shifts during training as the model improves — static pre-filtering discards future gradient signal. Strategy D handles this dynamically.

### Strategy B — Probe-based bucket weights (Stage 1 initialization)

Before Stage 1, run 8 TIR rollouts on a stratified probe subset (~2.5k rows) using the base model (Qwen3.5-4B instruct). Compute per-problem pass rate. Aggregate to per-stratum Goldilocks rates (pass_rate ∈ [0.05, 0.95]). Add a `_train_weight` column to the train parquets encoding each problem's stratum weight.

**Stratification dimensions:**
- Dr. SCI: `(inferred_answer_type × extra_info.from × difficulty_bin)` where difficulty_bin = low(0.0) / medium(0.125–0.375) / high(0.5–0.75)
- Corpus: `(primary_answer_type × source)`

Weights are proportional to stratum Goldilocks rate. Problems in strata with very low Goldilocks rate (mostly too hard) are downweighted, not excluded — they enter via the hard bank in later stages.

**Note on old weight table:** An earlier proxy table derived from 600-row pass@k samples has been superseded. The probe rollouts in Step 3 of `data-pipeline.md` produce the authoritative weights.

### Strategy D — Stage rescoring and dataset rebuild

After each training stage (~700 steps), run `rescore_goldilocks.py` with the stage checkpoint on a stratified subset (size TBD, baseline 5k rows; scale up if compute allows). Recompute per-problem pass rates. Update `_train_weight` and the hard bank:

- Problems that were too hard (pass_rate < 0.05 under base model) but now have pass_rate ∈ [0.05, 0.95] graduate from the hard bank into the active training set.
- Problems that have become too easy (pass_rate > 0.95) are downweighted.
- Updated `_train_weight` values take effect for the next stage.

This is how the model's improving capability is exploited: the Goldilocks zone expands as training progresses, and the dataset composition tracks it automatically.

### Why not an online per-batch filter (Strategy C)?

GRPO already handles the degenerate cases without extra engineering: if all k rollouts are correct, group-relative advantage = 1 − 1 = 0 (no gradient); if all wrong, advantage = 0 − 0 = 0. Explicit skip logic would only save the forward pass on those batches. With B+D keeping the dataset clean offline, the frequency of these degenerate batches is low. Strategy C is dropped.

### Hard problem bank

Problems with pass_rate < 0.05 in the initial probe are placed in `data/processed/hard_bank.parquet` rather than discarded. They are re-evaluated at each stage rescore (Strategy D). Once a problem's pass rate rises above 0.05, it enters the active training pool. This is the primary mechanism for the Stage 2+ curriculum expansion.

### Multi-stage training structure

| Stage | Data source | Curriculum | Steps |
|---|---|---|---|
| Stage 1 | Strategy B weights (probe-based) | Numerical-first | ~700 |
| Stage 2 | Strategy D rescore of Stage 1 ckpt | Numerical + expression + MCQ | ~700 |
| Stage 3 | Strategy D rescore of Stage 2 ckpt | Full mix + hard bank graduates | ~700 |

See `.claude/plans/data-pipeline.md` for the full pipeline spec.

---

## Dataset Sizes (as of 2026-04-10)

| Corpus | Training parquet | Rows | Split (train/dev/test) |
|--------|-----------------|------|------------------------|
| Corpus | `data/processed/corpus_train.parquet` | 6,866 | ~6,466 / 200 / 200 |
| Dr. SCI | `data/processed/drsci_train.parquet` | 105,729 | ~101,729 / 2,000 / 2,000 |

**Combined training pool: ~108,195 problems (after split).**

Dr. SCI is ~15× larger; without weighting it will dominate. Strategy B weights handle this — corpus sources (SciBench_RL, PHYSICS, UGPhysics, OlympiadBench, PHYBench) are oversampled relative to their raw share via per-stratum weights.

**Metadata enrichment (as of data pipeline Step 0):**
- Both train parquets will have enriched `extra_info` preserving: `from` (Dr. SCI), `domain_coarse` (corpus), `primary_answer_type` (corpus, normalizing 87 raw types to 7: numerical/expression/equation/mcq/true_false/interval/multi-part)
- Splits are stratified by these fields; see `data-pipeline.md` for details.

---

## Verifier Status (as of 2026-03-30)

All planned verifier work is complete:

| Component | Status |
|-----------|--------|
| Rule verifier (`math_verify_wrapper.py`) | ✅ Done — includes `_sympy_numerical_equiv` numerical substitution tier |
| xVerify integration | ✅ Done — xVerify-7B-I preferred; `local_files_only=True` in sbatch |
| Dr. SCI e-notation fix | ✅ Done |
| MCQ parenthesis normalization | ✅ Done |
| `_is_prose_gold` drop in cleaning | ✅ Done (step 6 in `drsci_clean.py`) |

The plan `floating-percolating-thunder.md` (numerical equiv checker) is **stale/complete**.

---

## TIR Training Decisions (PhysCode, 2026-03-31)

**Reward:** Binary R_correct on final `\boxed{}` answer (λ=0 initially). Token cost penalty added only in late ablation.

**Curriculum:** Numerical problems first (cleanest execution path, highest zero-shot accuracy). Expand to expression + MCQ once training is stable.

**TIR format:** Single code block per trajectory — `[think]…[code]…[/code][output]…[think]…[answer]\boxed{}`. Multi-block explicitly out of scope for MVP.

**Execution sandbox:** 30s timeout, subprocess + resource limits. Timeout distribution should be profiled on dev set before training.

### xVerify in reward function — open infrastructure problem

The rule-only verifier has ~68% FN rate on expression types (confirmed in experiments). This means many correct model answers get reward=0, directly hurting GRPO signal quality for expressions.

**Why xVerify can't be the default in `reward.py` today:**
VeRL calls `compute_score()` per-sample. xVerify-7B needs ~14GB GPU RAM and ~0.5s/call when warm. Loading it inside the reward function is not feasible (cold-load per call, or OOM with rollout model on same GPU).

**Options (in order of practicality):**

| Option | When | Notes |
|--------|------|-------|
| **Rule-only, numerical curriculum** | Smoke test + Phase 1 | Rule verifier FN rate on numerical is low (~5–10%). Start here. |
| **Dedicated reward GPU** | Production run | Run xVerify-7B on a separate A100; call via socket/HTTP from `compute_score()`. VeRL's custom reward function supports this pattern. |
| **Batch post-scoring** | Ablation only | Score rollout batch rule-first; queue expression-type unknowns for xVerify; update rewards before GRPO update step. Requires VeRL reward API change. |
| **xVerify co-located (careful)** | If reward GPU unavailable | Load xVerify-7B once as a module-level singleton in the reward worker process. Only works if VeRL uses a dedicated reward worker process (not same as rollout). |

**Current default in `reward.py`:** `xverify_judge=None` (rule-only). This is correct for the smoke test.

**Action required before full expression-type training:** set up a dedicated reward GPU running `xverify_judge` and wire it into `compute_score`. File an explicit task when starting the production run.

**CoT-GRPO baseline:** Must be run in parallel with TIR-GRPO for comparison (same data, same checkpoints). Required for RQ1 and RQ2.

---

## Immediate Next Steps (as of 2026-04-10)

1. **Think-interrupt patch** — implement in `verl/verl/experimental/agent_loop/tool_agent_loop.py` (spec in `physcode.md` Week 2). Gating item for all training.
2. **Data pipeline** — Steps 0–6 in `.claude/plans/data-pipeline.md`. Each step is a separate session. Steps 0 and 0b can start immediately; Steps 3+ require think-interrupt to be settled.
3. **Stage 1 GRPO run** — numerical curriculum, Strategy B weights, ~700 steps.

---

## Key Numbers for Paper Baselines

| Model | Condition | pass@1 (xV-7B) | Notes |
|-------|-----------|----------------|-------|
| Qwen3.5-4B | Think-ON, full corpus | 39.7% | `rescore_7b_v3.parquet` — authoritative |
| Qwen3.5-4B | No-Think, full corpus | 32.0% | `zero_shot_nothink_xverify_7b.parquet` |
| Qwen3.5-0.8B | No-Think, full corpus | 10.1% | `zero_shot_nothink_08b_xverify_7b.parquet` |
| Qwen3.5-4B | No-Think, corpus sample (pass@8) | 30.3% | `zero_shot_corpus_passk_xv7b.parquet` |
| Qwen3.5-4B | No-Think, Dr. SCI sample (pass@8) | 24.3% | `zero_shot_drsci_sample_xv7b.parquet` (prose-gold filtered) |

Non-truncated accuracy (think-ON): **49.0%** — the ceiling for fixed compute budget.

---

## Sampling Parameters for Thinking-Mode Rollouts (2026-04-13, final)

### The `!!!` Degeneration Problem — Root Cause: CUDA Graph Instability

Qwen3.5-4B with `enable_thinking=True` degenerates into repetitive single-token loops (`!!!...`) in ~96% of rollouts when using vLLM without `enforce_eager=True`. The degeneration appears **exclusively in phase 2** (after tool response injection), never in phase 1 thinking/tool-call generation. It is stochastic (mixed within every problem, not problem-dependent).

**Root cause: CUDA graph capture is numerically unstable with Qwen3.5's hybrid GDN (Gated Delta Network) linear attention layers.** With `enforce_eager=False` (vLLM's default), vLLM captures the forward pass into static CUDA graphs. These graphs produce subtly wrong logits on long-context phase 2 prompts, causing the model to enter absorbing repetition states. VeRL's `RolloutConfig` defaults to `enforce_eager=True`, which is why VeRL rollouts never exhibited this.

Diagnostic history (each row used `enforce_eager=False` unless noted):

| Run | `enforce_eager` | `top_p` | `temp` | `pp` | `!!!` degen |
|-----|-----------------|---------|--------|------|-------------|
| probe_calib_A/B | False | 0.9 | 1.0 | 0 | **96–97%** |
| probe_v2_A_fixed | False | 1.0 | 1.0 | 0 | **97%** |
| probe_fix_pp15 | False | 0.95 | 1.0 | 1.5 | 0% |
| probe_fix_t14 | False | 1.0 | 1.4 | 0 | 0% |
| **eager_test** | **True** | **1.0** | **1.0** | **0** | **0%** |

The `presence_penalty=1.5` and `top_p=0.95` "fixes" were masking the CUDA graph instability by constraining the distribution enough to avoid the degenerate states — not addressing the root cause.

### Decision: `enforce_eager=True`, standard RL sampling (`top_p=1.0`)

**Chosen config** (matching VeRL framework + RL theory):
```
enforce_eager=True
temperature=1.0, top_p=1.0, top_k=-1, presence_penalty=0.0, repetition_penalty=1.0
```

- `enforce_eager=True`: eliminates CUDA graph instability (matches VeRL `RolloutConfig` default)
- `top_p=1.0`: raw policy sampling for on-policy GRPO (no uncorrected IS mismatch)
- No `presence_penalty`: avoids context-dependent distribution modification during training
- Qwen model card params (`top_p=0.95, top_k=20, pp=1.5`) reserved for inference/eval only

### Bug Fixes in `dump_rollouts.py` vs VeRL (2026-04-13)

Four code-level discrepancies were found and fixed:

1. **Missing `enforce_eager=True`** (PRIMARY FIX) — `dump_rollouts.py` used vLLM's default `enforce_eager=False`, enabling CUDA graphs that are unstable with Qwen3.5's GDN attention. VeRL defaults to `enforce_eager=True`. Fixed: `LLM(enforce_eager=True)`.

2. **Missing `<|im_end|>` after `</tool_call>`** — `dump_rollouts.py` stops phase 1 generation at the `</tool_call>` stop string, so the model never emits the `<|im_end|>` token that closes the assistant turn. In VeRL there is no stop string; the model generates past `</tool_call>` and naturally emits `<|im_end|>`. Fixed: `_make_tool_injection` now prepends `<|im_end|>`.

3. **`enable_thinking=False` in tool injection** — `dump_rollouts.py` used `enable_thinking=False` when building the tool response injection, producing a pre-closed empty `<think>\n\n</think>\n\n` block. VeRL passes `enable_thinking=True` (from `apply_chat_template_kwargs`), producing an open `<think>\n` block that lets the model optionally reason before answering. Fixed: injection now uses the same `enable_thinking` flag as phase 1.

4. **`top_p` default** — Changed to 1.0 to match RL theory (raw policy sampling). VeRL framework default is also 1.0; the 0.9 in training scripts was a pre-fix artifact.

### TODO: Update VeRL training scripts before final run

The following VeRL training scripts still use `top_p=0.9` (a pre-fix artifact). Change to `top_p=1.0` for theoretical correctness (raw policy sampling in on-policy GRPO). **Do not edit yet** — preserve reproducibility of existing smoke test results. Apply before the final Stage 1 training run.

| Script | Line | Current | Target |
|--------|------|---------|--------|
| `scripts/smoke_tir_qwen35.sh` | ~207 | `top_p=0.9` | `top_p=1.0` |
| `scripts/smoke_tir.sh` | ~161 | `top_p=0.9` | `top_p=1.0` |
| `scripts/train_async.sh` | ~199 | `top_p=0.9` | `top_p=1.0` |

`enforce_eager=True` is already the VeRL `RolloutConfig` default — no change needed there.

**Open question:** The CUDA graph instability was confirmed on B200 (SM_100, Blackwell). VeRL defaults to `enforce_eager=True` regardless of GPU, suggesting this is a known cross-architecture issue with vLLM + hybrid attention models. Worth testing on H200/A100 if performance matters — CUDA graphs are 2-3x faster, so if they're stable on Ampere/Hopper the training scripts could conditionally enable them.

### Bug Fixes in Probe Pipeline (2026-04-12)

1. **`extract_answer()` now uses last `\boxed{}`** — standard practice (MATH, GSM8K eval). Fixes 0.35% of rollouts incorrectly scored 0.0 when model writes `\boxed{}` in both think block and final answer. Changed in `src/phys_reasoner/verifier/extract.py`.

2. **`dump_rollouts.py` early termination** — when model produces no tool call after phase 1, terminate immediately (no forced phase 2). Matches VeRL's `tool_agent_loop.py` line 361: `return AgentState.TERMINATED`. Previously the probe injected a fake `"(no code block)"` tool response and forced phase 2, causing confusion.

3. **`dump_rollouts.sbatch` env vars** — added missing `PYTHONPATH=/opt/phys-extras/`, `TRITON_CACHE_DIR`, `XDG_CACHE_HOME` (matching `dump_rollouts.sh`). Without these, Qwen3.5 model type and triton caching fail on compute nodes.

### Cluster Notes (2026-04-12)

- **`gpu_rtx6000` (rtx_pro_6000_blackwell) incompatible with this SIF** — flash-attn PTX compiled for older SM arch, Blackwell SM_120 not supported. `cudaErrorUnsupportedPtxVersion`. Avoid this partition.
- **CPU-only scoring jobs**: use `day_amd` partition (easier to get than `day`). `devel` has QOS limits if interactive session running. `gpu_b200` requires `--gres=gpu:1` minimum (QOSMinGRES).
