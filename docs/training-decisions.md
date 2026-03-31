# Training Decisions & Strategy Notes

Last updated: 2026-03-30

---

## SFT Stage: Skipped

**Decision:** Do not run SFT before GRPO. Go directly from clean data → GRPO training.

**Rationale:**
1. GRPO with reward `R = R_correct − λ·C_action` discovers routing directly through reward signal — no oracle routing labels needed. The model explores all four actions and learns to prefer cheaper ones when they work.
2. SFT routing labels would be derived from noisy pass@k estimates on a small sample. Training on these could bias the model toward a fixed policy before GRPO can explore.
3. Qwen3.5-4B instruct follows action-selection instructions well enough for GRPO to start from. No warm-start required.
4. Consistent with DAPO, Dr. GRPO, and similar papers that go directly from instruct model to GRPO.

**Fallback:** If early GRPO training is unstable (reward stuck near zero for many steps), run a short warm-up SFT phase using only high-confidence **Answer** examples (problems where pass@8 > 0.85 — the model almost always gets these right with a direct response). This stabilises the policy without imposing a routing prior. Treat as a last resort, not a default step.

**Impact on checklist:** Week 2 SFT tasks (SFT data generation, Check demonstrations, SFT training) are removed. Week 2 is now focused entirely on the four prompting templates, parser, and tool wrappers — which are still needed for GRPO's action space.

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

## Immediate Next Steps (priority order)

1. **Add `_train_weight` column** to both clean parquets (Strategy B above).
2. **Write four prompting templates** — Answer / Check / Think-Deep / Tool-Check. These define the action space for GRPO. Required before any training.
3. **Implement parser** for Check and Tool-Check structured outputs (schema validation, required fields).
4. **Implement tool wrappers** — `check_equation`, `check_units`, `plug_values`, `check_root`, `compare_expr`.
5. **Set up GRPO with verl** — reward function, online filter (Strategy C), cost penalty λ.
6. **Dev run** on ~500 problems (Qwen3.5-0.6B or 4B) to validate reward signal before full training.

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
