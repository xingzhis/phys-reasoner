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

## Goldilocks Data Selection: Strategy B + C

**Context:** Pass@k (k=8) was run on ~600-row stratified samples from each corpus — not on all 113k problems. Per-problem difficulty labels are unavailable for ~99% of the data.

### Why not hard-filter to Goldilocks problems only?

- Corpus: hard filter leaves only ~1,200 problems (18% of 6,866) — too small for stable GRPO.
- Dr. SCI: 18% of 107k ≈ 19k problems would survive, which is fine in isolation, but the filter requires per-problem pass rates we don't have.
- Pass@k estimates from k=8 are noisy near the boundary (a problem at 0.12 may truly be 0.18).
- The Goldilocks zone shifts during training as the model improves — static pre-filtering discards future gradient signal.

### Strategy B — Bucket weights at data loading time

Use measured Goldilocks rates per `(source × answer_type)` bucket as sampling weights. No additional inference needed — just a lookup table derived from the pass@k sample.

**Measured Goldilocks rates (xVerify-7B, pass@8):**

| Source | Answer type | Goldilocks rate | Suggested weight |
|--------|-------------|-----------------|------------------|
| SciBench_RL | any | ~40% | 1.0 |
| any | mcq | ~34% | 0.9 |
| MegaScience / WebInstruct | numerical | ~19–23% | 0.6 |
| UGPhysics | numerical | ~19% | 0.6 |
| any | expression | ~15% | 0.5 |
| OlympiadBench | any | ~14% | 0.4 |
| natural_reasoning | equation | ~10% | 0.3 |
| any | multi-part | ~4% | 0.1 |
| PHYBench | any | ~20% (high truncation) | 0.3 |

**Implementation:** Add a `_train_weight` column to `drsci_physics_clean.parquet` and `candidates_deduped.parquet` via a small preprocessing script. GRPO data loader reads this column for weighted sampling.

### Strategy C — Online filtering during GRPO

During training, GRPO generates k rollouts per problem. Before the gradient update, compute:
```
live_pass_rate = n_correct / k
```
If `live_pass_rate` falls outside `[0.05, 0.95]`, skip the gradient update for that step.
- Too easy (> 0.95): all rollouts correct, group-relative advantage ≈ 0 anyway.
- Too hard (< 0.05): all rollouts wrong, no signal, just noise.

This adapts dynamically — problems that were "too hard" early in training become learnable as the model improves. Implementation: a few lines in the verl reward wrapper.

### Why B + C together?

B ensures we preferentially sample from rich Goldilocks buckets at the data-loading level, reducing wasted forward passes. C discards the zero-gradient problems that slip through at runtime and adapts as the model's capability changes. Neither requires re-running inference.

---

## Dataset Sizes (as of 2026-03-30)

| Corpus | File | Rows | Notes |
|--------|------|------|-------|
| 6.8k corpus | `data/processed/candidates_deduped.parquet` | 6,866 | 5 sources; has `answer_type`, `difficulty`, `unit` |
| Dr. SCI | `data/processed/drsci_physics_clean.parquet` | 107,158 | 3 sources; has `inferred_answer_type`; prose-gold dropped |

**Combined training pool: ~114,024 problems.**

Dr. SCI is ~15× larger; without weighting it will dominate. Oversample the 6.8k corpus by a factor of ~3–5× relative to its raw size to maintain source diversity across all 5 corpus sources (SciBench_RL, PHYSICS, UGPhysics, OlympiadBench, PHYBench).

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

**CoT-GRPO baseline:** Must be run in parallel with TIR-GRPO for comparison (same data, same checkpoints). Required for RQ1 and RQ2.

---

## Immediate Next Steps (priority order)

1. **VeRL single-block injection** — implement and smoke-test mid-sequence `[output]` injection (gating item).
2. **Stage 0 probe** — 100-problem TIR zero-shot; decide SFT go/no-go.
3. **Execution sandbox** — subprocess isolation, 30s timeout.
4. **LaTeX FN rate measurement** — per-type FN rate on training corpus (gold-vs-gold round-trip with rule verifier).
5. **Add `_train_weight` column** to both clean parquets (Goldilocks B above).
6. **GRPO setup with verl** — binary reward, online filter (C above), numerical curriculum.

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
