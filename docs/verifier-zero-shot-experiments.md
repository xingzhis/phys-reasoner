# Verifier & Zero-Shot Baseline Experiments (2026-03-21/22)

## Overview

End-to-end experiments running Qwen3.5-4B zero-shot on a 10-sample diagnostic set
(OlympiadBench only, sampled via `run_zero_shot_diag.py --max_samples 10`).
Goal: validate the full verifier pipeline (rule + xVerify) before the full run.

---

## Pipeline Confirmed Working

```
Model output (raw)
  → strip_thinking()        # remove <think>...</think>
  → extract_answer()        # extract \boxed{} contents
  → split_by_comma()        # split multi-part answers
  → rule_verify()           # math-verify + sympy numeric tolerance
  → xVerify fallback        # LLM judge for unparseable cases
```

**Inference settings (confirmed):**
- Model: `Qwen/Qwen3.5-4B`, vLLM 0.17, bfloat16
- Thinking mode: `enable_thinking=True` via chat template
- Sampling: temperature=1.0, top_p=0.95, top_k=20, presence_penalty=1.5
- `max_new_tokens=32768` (avg natural EOS ~15k tokens, p90 ~16k, max observed ~33k)
- `max_model_len=36864` (32768 + 4096 prompt budget)

**Token distribution (from 10-sample diag with 61k cap, OlympiadBench):**
- min: ~3.7k tokens, median: ~8.6k, p90: ~16k, max: ~33k
- None truncated at 61k cap → 32k cap covers the vast majority

---

## Bugs Found and Fixed (2026-03-21/22)

### Bug 1: `normalize_answer_type` fails on JSON-encoded list strings
**File:** `src/phys_reasoner/data/normalize.py`

OlympiadBench multi-part answer types are stored as JSON strings like `'["expression", "numerical"]'`.
`normalize_answer_type` saw the comma and split on `","` → got `['["expression"', '"numerical"]']`
→ both mapped to `"unknown"` (quotes not stripped) → `primary_type = "unknown"` → returned -1.0
immediately, bypassing xVerify entirely.

**Fix:** Detect strings starting with `[` and JSON-parse them before comma-split.

**Impact:** Sample 0 (multi-part OlympiadBench) went from -1.0 to +1.0 (xVerify correctly verified).

---

### Bug 2: `split_by_comma` splits on `\,` (LaTeX thin space)
**File:** `src/phys_reasoner/verifier/extract.py`

`\,` is a LaTeX thin space command but contains a literal comma. `split_by_comma("5780 \, \mathrm{K}")`
split at the `\,` comma → pred_parts = `['5780 \\', '\\mathrm{K}']` (2 parts vs 1 gold) →
short-circuits to 0.0 before xVerify is called.

**Fix:** In `split_by_comma`, skip commas preceded by a backslash.

**Impact:** Sample 7 (`5780 \, \mathrm{K}` vs `5.76 × 10³`) went from 0.0 to 1.0
(numeric tolerance now fires correctly).

---

### Bug 3: `rule_verify` numeric tolerance doesn't fire for LaTeX fractions
**File:** `src/phys_reasoner/verifier/math_verify_wrapper.py`

The tolerance pre-check calls `float(pred_str)` which fails for `\frac{10}{7}`.
After math_verify parses it to sympy `Rational(10,7)`, `verify()` uses exact `float_rounding=6`
not percentage tolerance — so 1.4286 ≠ 1.43.

**Fix:** After both sides are parsed by math_verify, evaluate numerically with `sympy.N()`
and apply percentage tolerance before calling `verify()`.

**Impact:** `\frac{10}{7}` vs `1.43` (0.1% difference) now correctly returns True.
Also fixed `2.67 \mathrm{~m/s}` vs `2.667` (rule_verify now returns True directly,
no longer needs xVerify rescue).

---

## Verifier Accuracy on 10-Sample Diagnostic

| Sample | Type | Gold (abbrev) | Pred boxed | Rule | +xVerify-3B | +xVerify-7B | Notes |
|--------|------|---------------|------------|------|-------------|-------------|-------|
| 0 | [expr, num] | `\frac{1}{2}(m_e c^2/...)^2, Δ=3.63e-11` | `3.63e-11` | -1→ | **+1.0** | 0.0 | 3B correct; 7B too strict on partial |
| 1 | equation | bubble radius eq | rearranged form | -1 | 0.0 | 0.0 | **Suspected xVerify FN** (algebraic rearrangement ×R₀⁴) |
| 2 | equation | `v dρ/dx + ρ dv/dx = 0` | `d(ρv)/dx = 0` | -1 | 0.0 | **+1.0** | 7B rescues product-rule equivalence; 3B misses |
| 3 | equation | radiation ODE | different form | -1 | 0.0 | 0.0 | Genuinely wrong |
| 4 | expression | `3NkT/2 - a'N²/V` | `3Nk_BT/2` | 0 | 0.0 | 0.0 | Genuinely wrong (missing van der Waals term) |
| 5 | expression | `ω = ℏk²/2m` | `ω = ℏk²/2m` | -1→ | **+1.0** | **+1.0** | `\hbar` unparseable by rule; xVerify rescues |
| 6 | expression | `v₀²/2g, g/2v₀²` | `z_0=..., k=...` | -1→ | **+1.0** | **+1.0** | Multi-part with variable labels; xVerify correct |
| 7 | numerical | `5.76×10³` | `5780 \, \mathrm{K}` | +1* | +1.0 | +1.0 | *After Bug 2+3 fix; was 0.0 before |
| 8 | numerical | `2.667` | `2.67 \mathrm{~m/s}` | +1* | +1.0 | +1.0 | *After Bug 3 fix; was 0.0 before |
| 9 | numerical | `1.43` | `\frac{10}{7}` | +1* | +1.0 | +1.0 | *After Bug 3 fix; was 0.0 before |

**Final accuracy (after bug fixes):**
- Rule-only: 3/10 (30%)
- Rule + xVerify-3B-Ib: 6/10 (60%)
- Rule + xVerify-7B-I: 6/10 (60%, but different 6)

Note: all 10 samples are OlympiadBench (hardest tier). Expect higher accuracy on PHYSICS/UGPhysics.

---

## xVerify Model Comparison: 3B-Ib vs 7B-I

Tested on the 10-sample diagnostic using `scripts/compare_xverify.py`.

| Aspect | xVerify-3B-Ib | xVerify-7B-I |
|--------|--------------|--------------|
| Overall accuracy (this sample) | 6/10 | 6/10 |
| Algebraic rearrangement (Sample 1) | ✗ | ✗ |
| Calculus identity / product rule (Sample 2) | ✗ | ✓ |
| Multi-part with variable labels (Sample 6) | ✓ | ✓ |
| Partial answer for multi-alternative gold (Sample 0) | ✓ (lenient) | ✗ (strict) |
| Physics symbols (`\hbar`, `\omega`) | ✓ | ✓ |
| Speed | faster | ~2x slower |

**Decision deferred** pending larger sample. With only 10 samples the swap is a wash.
Key question: is 3B-Ib's leniency on Sample 0 a feature (partial credit) or a bug
(false positive on incomplete answer)?

---

## Remaining Known Limitations (from 10-sample diagnostic)

1. **xVerify false negatives on algebraic rearrangements** (multiply/divide both sides):
   Both 3B-Ib and 7B-I fail to verify `R₁⁴ - R₀³R₁ = q²/(32π²ε₀Pₐ)` ≡ gold × R₀⁴.
   **Status as of 2026-03-22: NOT fixed. Still an open known limitation.**
   Potential fix: try sympy simplify/expand on equation difference after parse.
   Difficulty: high (requires correct LaTeX→sympy parse of physics expressions).

2. **xVerify false negatives on calculus identities** (3B-Ib only):
   `d(ρv)/dx = 0` ≡ `v dρ/dx + ρ dv/dx = 0` by product rule.
   7B-I handles this correctly. → **Resolved by switching to 7B-I.**

3. **Diagnostic issue-classifier fires on cross-pair permutation calls**:
   In `diagnose_verifier.py`, the cross-pair xVerify calls (all pred×gold combinations)
   are logged as issues even when the matched permutation is correct (sample 6).
   Fix needed in the diagnostic script — does not affect real verifier.

---

## Full Zero-Shot Baseline: All 8 Chunks (2026-03-22)

**Dataset:** 6,866 samples (8 × ~859, stratified by source × answer_type, n_per_tier=10000)
**Model:** Qwen/Qwen3.5-4B, thinking enabled, max_new_tokens=32768
**Result files:**
- Inference: `data/results/zero_shot_chunk{0..7}.parquet` (columns: problem_id, source, answer_type, gold_answer, pred_text, raw_output, score, truncated)
- xVerify-3B rescore (chunks 0–3 only): `data/results/rescore_3b.parquet`
- xVerify-7B rescore (chunks 0–3 only): `data/results/rescore_7b.parquet`
- xVerify-3B rescore (all 8 chunks): `data/results/rescore_3b_all8.parquet` ✓ done 2026-03-22
- xVerify-7B rescore (all 8 chunks): `data/results/rescore_7b_all8.parquet` — pre-fix snapshot (pre-LaTeX-unit-fix + pre-MCQ-parens-fix)
- xVerify-7B rescore **corrected** (all 8 chunks): `data/results/rescore_7b_all8_fixed.parquet` ✓ done 2026-03-23 — **authoritative baseline**

### Truncation Stats

| | Count | % |
|---|---|---|
| Total samples | 6,866 | 100% |
| Truncated (hit 32k limit) | 1,511 | 22.0% |
| Not truncated | 5,355 | 78.0% |

Truncated samples have near-zero accuracy (rule: 1.3%, 7B-xVerify: 7.1%). This is the key motivation for future interrupted-thinking experiments — see note below.

### Accuracy Comparison — Corrected Final (all 8 chunks, n=6,866)

**Authoritative result file: `data/results/rescore_7b_all8_fixed.parquet`** (2026-03-23)
Incorporates all verifier fixes: LaTeX unit parsing, MCQ parenthesis normalization.

| Subset | n | Rule-only | +xVerify-7B |
|--------|---|-----------|-------------|
| **ALL** | **6,866** | **19.0%** | **39.7%** |
| Non-truncated | 5,355 | 24.0% | **48.9%** |
| Truncated | 1,511 | 1.3% | 7.1% |

*For reference, pre-fix snapshot (`rescore_7b_all8.parquet`): 38.5% overall, 47.3% non-truncated. The +1.2pp gain comes from 62 MCQ parenthesis fixes + 7 LaTeX unit fixes.*
*xVerify-3B comparison available in `rescore_3b_all8.parquet`: 31.6% overall, 39.6% non-truncated.*

### Accuracy by Source (corrected, rescore_7b_all8_fixed)

| Source | n | Rule-only | +xVerify-7B |
|--------|---|-----------|-------------|
| SciBench_RL | 280 | 65.7% | 82.5% |
| PHYSICS | 805 | 21.1% | 40.1% |
| OlympiadBench | 230 | 12.2% | 28.3% |
| UGPhysics | 5,451 | 16.9% | 38.4% |
| PHYBench | 100 | 0.0% | 12.0% |

### Accuracy by Answer Type (corrected, rescore_7b_all8_fixed)

| Type | n | Rule-only | +xVerify-7B |
|------|---|-----------|-------------|
| numerical | 2,698 | 38.3% | 54.5% |
| expression | 2,030 | 1.9% | 31.1% |
| equation | 792 | 2.7% | 32.7% |
| multi(2)-numerical | 296 | 10.5% | 21.0% |
| mcq | 269 | 40.9% | 63.9% |
| multi(2)-expression | 225 | 0.0% | 8.9% |
| true_false | 152 | 37.5% | 37.5% |
| interval | 65 | 12.3% | 26.2% |

Key takeaways:
- xVerify is essential for expression/equation types (+29–31pp vs rule-only)
- MCQ now shows large xVerify gain (+23pp) after parenthesis normalization fix rescues 62 cases
- true_false: xVerify adds nothing (exact-match only, correct)
- **48.9% non-truncated accuracy with 7B is the authoritative baseline for interrupted-thinking experiments**

### Note: Truncation and Interrupted-Thinking Experiments

22% of outputs were truncated at 32k tokens, and truncated samples have ~1–7% accuracy vs ~24–49% for non-truncated. This is the baseline for the upcoming **interrupted-thinking** experiment: if we interrupt `<think>` early to save tokens, we expect more truncation and lower accuracy. Compare against:
- `rescore_7b_all8_fixed.parquet` (full thinking, all fixes applied) — the authoritative baseline

---

## FN Spot-Check: Manual Inspection of Negatives (2026-03-22)

### FN Category Breakdown (based on rescore_7b.parquet, 3,436 rows)

Of 2,276 negatives (score_xverify=0):

| Category | Count | Notes |
|----------|-------|-------|
| Truncated (hit 32k limit) | 740 | Expected — model ran out of tokens before answering |
| Non-truncated, no `\boxed{}` | 52 | Model failed to produce a boxed answer |
| Non-truncated, has `\boxed{}`, score=0 | 1,484 | Mix of genuine wrong answers + verifier FNs |

Of the 1,484 "has boxed but scored 0" cases, manual inspection of a 15-sample random subset reveals:

**Dominant pattern (>90%): Genuinely wrong answers.** The model boxed an incorrect value.

**Confirmed verifier FN type: unit mismatch in SciBench_RL** (1 confirmed, likely ~5–10 cases):
- Example: `SciBench_RL_00147` (de Broglie wavelength)
  - Gold: `0.332` (implicit unit: nm)
  - Model: `3.32 × 10⁻¹⁰ m` (correct — 0.332 nm in SI)
  - Verifier: 0.0 — fails because SI vs implicit nm
  - Root cause: SciBench_RL answers are in non-SI units without stating the unit in gold

**Suspected verifier FN type: algebraic rearrangements** (low frequency, high difficulty to fix):
- Same category as smoke-test Sample 1: model rewrites equation by multiplying both sides
- xVerify-7B still fails on these (not fixed since 10-sample diagnostic)

**Script for further investigation:**
```bash
python scripts/spot_check_fn.py \
    --rescore data/results/rescore_7b_all8.parquet \
    --source SciBench_RL --n 30 --show_problem
```

---

---

## Verifier Update Rescore (2026-03-24)

### Motivation

`math_verify_wrapper.py` was updated with additional FN-reduction logic (symbolic numerical
equivalence via random substitution, coefficient-extraction FP guard, improved sci-notation
preprocessing). Re-scored all 8 chunks to confirm no regression and measure any FN rate improvement.

### What changed in the scorer

- Added `_sympy_numerical_equiv()`: random-substitution check for algebraically equivalent
  expressions/equations (catches `a*(b+c)` vs `a*b+a*c`, equation side-swaps, scalar multiples).
- Added `_is_suspicious_simplification()` guard: prevents math-verify from reporting True when
  it silently extracts a small integer from a complex LaTeX expression (e.g. `2 \cosh(...)` → 2).
- Added latex2sympy fallback numeric eval: rescues cases where `math_verify.parse()` drops symbolic
  terms (e.g. `\frac{\sqrt{\pi}}{2} \times 10^{14.5}` — parse() extracted only the `10^{14.5}` part).

### Regression check

Rule-only scores are **bit-for-bit identical** to the previous baseline (`rescore_7b_all8_fixed`).
No false positives introduced.

### Results: new v2/v3 vs corrected old baseline

| | rescore_7b_all8_fixed (old) | rescore_7b_v3 (new) | Δ |
|---|---|---|---|
| Rule-only | 19.01% | 19.01% | 0.00pp |
| +xVerify-7B | 39.69% | **39.70%** | +0.01pp |
| FN rescue rate (rule=0 → xv=1) | 25.55% | 25.57% | +0.01pp |

The scorer changes produced negligible measurable gain on this dataset. This is expected:
the new paths (symbolic substitution, coefficient guard) fire on patterns that are either
rare in this corpus or already rescued by xVerify.

**Conclusion:** no regression, no meaningful improvement on the current corpus.
`rescore_7b_v3.parquet` is now the canonical baseline (same numbers, cleaner pipeline).

### By source — 7B v3 final numbers

| Source | n | Rule-only | +xVerify-7B |
|--------|---|-----------|-------------|
| SciBench_RL | 280 | 65.7% | **85.0%** |
| PHYSICS | 805 | 21.1% | **40.2%** |
| UGPhysics | 5,451 | 16.9% | **38.3%** |
| OlympiadBench | 230 | 12.2% | **28.3%** |
| PHYBench | 100 | 0.0% | **12.0%** |

Differences from old baseline are within ±0.3pp (stochastic xVerify variation on borderline cases).

### By answer type — 7B v3 final numbers

| Type | n | Rule-only | +xVerify-7B | +xVerify-3B |
|------|---|-----------|-------------|-------------|
| numerical | 2,698 | 38.3% | **54.7%** | 50.7% |
| expression | 2,030 | 1.9% | **30.8%** | 20.1% |
| equation | 792 | 2.7% | **32.7%** | 22.0% |
| multi(2)-numerical | 296 | 10.5% | **21.3%** | 18.2% |
| mcq | 269 | 40.9% | **63.9%** | 63.9% |
| multi(2)-expression | 225 | 0.0% | **8.9%** | 4.9% |
| true_false | 152 | 37.5% | **37.5%** | 37.5% |
| multi(2)-equation | 130 | 0.0% | **14.6%** | 6.2% |
| interval | 65 | 12.3% | **26.2%** | 18.5% |

7B vs 3B gap is largest on symbolic types: +10.8pp on expression, +10.7pp on equation.

### 3B vs 7B summary

| | rescore_3b_v2 | rescore_7b_v3 | Δ (7B−3B) |
|---|---|---|---|
| Overall | 33.2% | **39.7%** | +6.5pp |
| SciBench_RL | 82.1% | **85.0%** | +2.9pp |
| PHYSICS | 33.7% | **40.2%** | +6.6pp |
| UGPhysics | 31.5% | **38.3%** | +6.8pp |
| OlympiadBench | 23.5% | **28.3%** | +4.8pp |
| PHYBench | 7.0% | **12.0%** | +5.0pp |

### Updated authoritative baseline (2026-03-24)

**`data/results/rescore_7b_v3.parquet`** — use this for all future comparisons.

Previous files (`rescore_7b_all8.parquet`, `rescore_7b_all8_fixed.parquet`, `rescore_3b_all8.parquet`)
are superseded. `rescore_3b_v2.parquet` is the companion 3B file for ablation comparisons.

### Infrastructure fixes made during this session

- **scipy missing from overlay** — `transformers/loss/loss_for_object_detection.py` imports
  `scipy.optimize.linear_sum_assignment` at model-load time; base SIF has broken scipy. Fixed by
  adding `scipy>=1.11` to `pyproject.toml` and installing into `phys-reasoner-overlay-017.img`.
- **huggingface-hub version** — base SIF has `0.36.2`; transformers requires `>=1.3.0`. Installed
  `1.7.2` into the 017 overlay.
- **FUSE2FS mount issue** — `phys-reasoner-overlay.img` was not cleanly unmounted on one node,
  causing silent mount failures on other nodes (overlay packages invisible). Switched sbatch to
  use `phys-reasoner-overlay-017.img` which mounts cleanly.
- **`local_files_only=True`** — added to `XVerifyJudge.__init__` to prevent HF API calls for
  `additional_chat_templates` (404 on compute nodes without outbound HTTPS).
- **`export PYTHONNOUSERSITE=1`** — added as a shell export in the sbatch script (not just as
  `--env` to apptainer) to reliably block `~/.local` from shadowing overlay packages.

---

## Extended Token Budget Rerun — Truncated Samples (2026-03-24/25)

### Motivation

22% of outputs (1,511/6,866) hit the 32768-token limit and had near-zero accuracy (rule: 1.3%, 7B-xV: 7.1%). To measure the accuracy ceiling for these samples, the rerun script gives them 81920 tokens (Qwen3.5 complex-problem recommendation).

### Setup

- **Script:** `scripts/run_zero_shot_rerun.py` + `scripts/zero_shot_rerun.sbatch`
- **Input:** truncated rows from `zero_shot_chunk{0..7}.parquet` (1,511 rows)
- **Split:** 32 chunks (47–48 samples each); submit as parallel sbatch array
- **max_new_tokens:** 81920; max_model_len: 86016

### Status (2026-03-25)

Only **chunk 0/32** completed (job 1418892). Chunks 1–31 stalled / hit time limit and need re-submission. Chunk 0 contains **48 samples, all OlympiadBench**.

Rescored locally with xVerify-3B-Ib → `data/results/rescore_rerun0_3b.parquet`
and xVerify-7B-I → `data/results/rescore_rerun0_7b.parquet`.

### Results: chunk 0, 46 completed samples (excl. 2 still truncated at 81920)

| Scorer | Before (32768 tok, truncated) | After (81920 tok) |
|--------|-------------------------------|-------------------|
| Rule-only | 0.0% (0/46) | **17.4%** (8/46) |
| Rule + xVerify-3B | ~2% | **28.3%** (13/46) |
| Rule + xVerify-7B | ~4% | **30.4%** (14/46) |

All 48 samples were previously truncated (score=0); 46 completed at 81920 tokens, 2 still hit the limit.

### By answer type (completed 46)

| Type | n | Rule | +xVerify-3B | +xVerify-7B |
|------|---|------|-------------|-------------|
| numerical | 23 | 34.8% | 43.5% | 43.5% |
| expression | 22 | 0.0% | 13.6% | 18.2% |
| equation | 1 | 0.0% | 0.0% | 0.0% |

Expression-type gains are the most informative: the model needs space to derive symbolic results, so the longer budget directly rescues these cases.

### TODO

- Re-submit chunks 1–31 (remaining 1,463 truncated samples) to complete the full rerun.
- Run `scripts/merge_rerun_chunks.py` after all chunks complete to produce `zero_shot_merged.parquet`.
- Rescore merged file with xVerify-7B to get the final combined baseline.

---

## No-Think Zero-Shot Baseline (2026-03-26)

### Motivation

Run the same zero-shot evaluation with Qwen3's **thinking mode disabled** (`enable_thinking=False`) to quantify the value of chain-of-thought for physics problem solving. Serves as a direct ablation: same model, same corpus, same verifier pipeline.

### Setup

- **Script:** `scripts/run_zero_shot_nothink.py` + `scripts/zero_shot_nothink_full.sbatch`
- **Model:** `Qwen/Qwen3.5-4B`, bfloat16, vLLM 0.17
- **Thinking mode:** `enable_thinking=False` in chat template
- **Sampling:** temperature=0.7, top_p=0.8, top_k=20 (Qwen3 official non-thinking params — no presence_penalty)
- **max_new_tokens:** 8192; max_model_len: 12288
- **Input:** `data/processed/candidates_deduped.parquet` (all 6,866 training candidates)
- **Output:** `data/results/zero_shot_nothink.parquet`
- **Rescored:** `data/results/zero_shot_nothink_xverify.parquet` (3B), `data/results/zero_shot_nothink_xverify_7b.parquet` (7B)
- **Job:** 1422176 (A100, ~1.5 hrs)

### Token distribution

- Mean: 3,576 tokens; min: 118; max: 8,192
- Truncated at 8,192: **1,568/6,866 (22.8%)** — similar truncation rate to thinking-on (22.0% at 32,768)

### Results vs Think-ON baseline

**Overall (n=6,866)**

| Scorer | Think-ON | No-Think | Gap |
|--------|----------|----------|-----|
| Rule-only | 19.0% | 17.7% | −1.3pp |
| +xVerify-3B | 33.2% | 27.0% | −6.2pp |
| **+xVerify-7B** | **39.7%** | **32.0%** | **−7.7pp** |

**By source (+xVerify-7B)**

| Source | n | Think-ON | No-Think | Gap |
|--------|---|----------|----------|-----|
| SciBench_RL | 280 | 85.0% | 73.2% | −11.8pp |
| PHYSICS | 805 | 40.2% | 33.2% | −7.1pp |
| UGPhysics | 5,451 | 38.3% | 30.7% | −7.6pp |
| OlympiadBench | 230 | 28.3% | 19.1% | −9.1pp |
| PHYBench | 100 | 12.0% | 8.0% | −4.0pp |

**By answer type (+xVerify-7B, no-think)**

| Type | n | Rule-only | +xVerify-3B | +xVerify-7B |
|------|---|-----------|-------------|-------------|
| numerical | 2,698 | 33.9% | 41.3% | **44.0%** |
| expression | 2,030 | 2.5% | 15.9% | **24.5%** |
| equation | 792 | 1.9% | 17.3% | **26.9%** |
| multi(2)-numerical | 296 | 9.1% | 14.9% | **16.9%** |
| mcq | 269 | 53.2% | 53.2% | **53.2%** |
| multi(2)-expression | 225 | 0.4% | 5.8% | **8.4%** |
| true_false | 152 | 34.9% | 34.9% | **34.9%** |
| multi(2)-equation | 130 | 0.0% | 6.2% | **10.0%** |
| interval | 65 | 6.2% | 9.2% | **12.3%** |

### Caveat: truncated samples included in all accuracy numbers

All accuracy figures above include truncated outputs. Truncated outputs score near-zero (rule: ~1%, xV-7B: ~4–7%) because the answer is cut off. This suppresses the overall accuracy and makes the think vs no-think comparison partially confounded:

- **Think-ON** truncated at 32,768 tokens — only very long, hard problems hit the limit
- **No-Think** truncated at 8,192 tokens — moderate-difficulty problems that happen to be verbose also get cut

The truncated sets are therefore *different problems at different difficulty levels*, so restricting to non-truncated samples is not a clean apples-to-apples comparison either (easier problems are less likely to truncate under either condition).

### Non-truncated accuracy (+xVerify-7B)

| | Think-ON | No-Think | Gap |
|---|---|---|---|
| Non-truncated n | 5,355 (78.0%) | 5,298 (77.2%) | — |
| Overall +xV-7B | **49.0%** | **40.3%** | −8.7pp |
| Truncated-only +xV-7B | 6.8% | 4.0% | — |

**By source, non-truncated only (+xVerify-7B)**

| Source | Think-ON n | Think-ON xV-7B | NoThink n | NoThink xV-7B | Gap |
|--------|-----------|----------------|-----------|---------------|-----|
| SciBench_RL | 262 | 89.7% | 257 | 79.4% | −10.3pp |
| PHYSICS | 675 | 46.7% | 675 | 38.7% | −8.0pp |
| UGPhysics | 4,241 | 47.2% | 4,207 | 38.6% | −8.6pp |
| OlympiadBench | 154 | 39.6% | 134 | 28.4% | −11.2pp |
| PHYBench | 23 | 47.8% | 25 | 28.0% | −19.8pp |

> **PHYBench caveat:** both conditions truncate ~75–77% of PHYBench samples, leaving only 23–25 non-truncated rows. The PHYBench non-truncated numbers are unreliable (high variance, likely selection bias toward easy problems).

The gap on non-truncated samples (~8–9pp) is consistent with the all-sample gap (~8pp), suggesting truncation does not strongly confound the think vs no-think comparison at the overall level. The fairest comparison would be a joint re-run where both conditions are given the same token budget — left for future work.

### Key observations

- Thinking mode provides a consistent **~7–8pp lift** across all sources with xVerify-7B scoring.
- The xVerify delta is **larger for no-think** (+14.3pp 7B) than for think-on (+20.7pp 7B) — thinking-on already resolves many symbolic answers in-context, while no-think leaves more correct but non-standard-format answers for xVerify to rescue.
- SciBench_RL shows the biggest thinking-mode gap (−11.8pp), consistent with it being multi-step derivation heavy.
- PHYBench is weakest under both conditions (8–12%), and least sensitive to thinking mode (−4.0pp).
- 7B xVerify adds ~5pp over 3B in both conditions, consistent with earlier findings.

### Authoritative no-think baseline

**`data/results/zero_shot_nothink_xverify_7b.parquet`** — use this for all think vs no-think comparisons.

---

## No-Think Zero-Shot Baseline — Qwen3.5-0.8B (2026-03-26)

### Setup

- **Script:** `scripts/run_zero_shot_nothink.py` + `scripts/zero_shot_nothink_08b.sbatch`
- **Model:** `Qwen/Qwen3.5-0.8B`, bfloat16, vLLM 0.17
- **Thinking mode:** `enable_thinking=False`
- **Sampling:** temperature=0.7, top_p=0.8, top_k=20
- **max_new_tokens:** 8192; max_model_len: 12288
- **Input:** `data/processed/candidates_deduped.parquet` (all 6,866 training candidates)
- **Output:** `data/results/zero_shot_nothink_08b.parquet`
- **Rescored:** `data/results/zero_shot_nothink_08b_xverify_7b.parquet` (7B)
- **Job:** 1428883 (A100, ~47 min)

### Token distribution

- Mean: 4,103 tokens; min: 163; max: 8,192
- Truncated at 8,192: **2,457/6,866 (35.8%)** — notably higher than 4B no-think (22.8%)

### Results — all samples (n=6,866)

| Source | Rule-only | +xVerify-7B | n |
|--------|-----------|-------------|---|
| SciBench_RL | 24.3% | 28.9% | 280 |
| PHYSICS | 6.3% | 9.6% | 805 |
| UGPhysics | 6.0% | 9.4% | 5,451 |
| OlympiadBench | 2.2% | 10.9% | 230 |
| PHYBench | 0.0% | 0.0% | 100 |
| **OVERALL** | **6.6%** | **10.1%** | **6,866** |

### Results — non-truncated only (n=4,409, 64.2%)

| Source | Rule-only | +xVerify-7B | n |
|--------|-----------|-------------|---|
| SciBench_RL | 29.3% | 34.5% | 232 |
| PHYSICS | 9.0% | 13.1% | 543 |
| UGPhysics | 9.1% | 13.3% | 3,482 |
| OlympiadBench | 2.4% | 14.3% | 126 |
| PHYBench | 0.0% | 0.0% | 26 |
| **OVERALL** | **9.9%** | **14.3%** | **4,409** |

### Comparison: 0.8B vs 4B, no-think (+xVerify-7B)

| Source | 0.8B (all) | 4B (all) | Δ | 0.8B (non-trunc) | 4B (non-trunc) | Δ |
|--------|-----------|---------|---|-----------------|----------------|---|
| SciBench_RL | 28.9% | 73.2% | −44.3pp | 34.5% | 79.4% | −44.9pp |
| PHYSICS | 9.6% | 33.2% | −23.6pp | 13.1% | 38.7% | −25.6pp |
| UGPhysics | 9.4% | 30.7% | −21.3pp | 13.3% | 38.6% | −25.3pp |
| OlympiadBench | 10.9% | 19.1% | −8.2pp | 14.3% | 28.4% | −14.1pp |
| PHYBench | 0.0% | 8.0% | −8.0pp | 0.0% | 28.0% | −28.0pp |
| **OVERALL** | **10.1%** | **32.0%** | **−21.9pp** | **14.3%** | **40.3%** | **−26.0pp** |

PHYBench non-truncated numbers are unreliable for 0.8B (only 26 samples survive; ~74% truncation rate).

### Key observations

- 0.8B is dramatically weaker than 4B under no-think: **−22pp overall, −26pp non-truncated**.
- Higher truncation rate (35.8% vs 22.8%) — 0.8B writes more tokens per step without making progress.
- xVerify-7B lift is modest (+3.5pp overall) vs 4B (+14.3pp), consistent with 0.8B generating fewer correct but non-standard-format answers and more genuinely wrong ones.
- OlympiadBench shows the best xVerify lift (+8.7pp all / +11.9pp non-trunc), suggesting rule matching fails on its answer formats even when answers are correct.

### Authoritative file

**`data/results/zero_shot_nothink_08b_xverify_7b.parquet`**

---

## Pass@k / Goldilocks Difficulty Profiling (2026-03-30)

### Motivation

To identify which problems are "learnable" for GRPO training (the Goldilocks zone), we need a per-problem difficulty estimate: too easy → always correct, no gradient; too hard → never correct, no gradient. We estimate difficulty via **empirical pass rate** over k independently sampled completions: `pass_rate = n_correct / k`. Problems where `pass_rate ∈ [0.15, 0.85]` are considered learnable.

### How pass@k is computed here (vs. standard definition)

The standard (unbiased) **pass@k** estimator from the Codex paper is:

```
pass@k = E[1 − C(n−c, k) / C(n, k)]
```

where n = number of samples drawn per problem, c = number of correct samples among them. When k = n, this reduces to: `pass@k = 1 if c ≥ 1 else 0` — a binary "did any sample pass?"

**We do not use the standard estimator.** Instead we compute:

```
pass_rate = c / n   (empirical mean accuracy, i.e. unbiased estimator of pass@1)
```

This is the **expected accuracy of a single randomly drawn completion**, estimated from n = 8 samples. It serves as a continuous difficulty score in [0, 1], which is what Goldilocks filtering requires. A binary pass@k would collapse all problems with ≥1 correct answer into the same bucket, losing all difficulty resolution.

Both metrics use the same n=8 samples. Our `pass_rate` is strictly more informative for Goldilocks selection; the standard pass@k is more relevant for estimating functional solve-rate (e.g. with best-of-n decoding).

### Truncation handling

A completion truncated at `max_new_tokens` typically has no `\boxed{}` and scores 0.

Two metrics are tracked per problem:
- **`pass_rate`** — `n_correct / k`: truncated completions count as incorrect. This is the conservative Goldilocks signal and is used as the primary metric.
- **`pass_rate_nontrunc`** — `n_correct / (k − n_truncated)`: truncated completions excluded from denominator. Separates "wrong answer" from "ran out of tokens." Useful when truncation rate is high.
- **`n_truncated`** — count of truncated completions per problem. Problems with `n_truncated ≥ k/2` have unreliable pass rates (most of their samples never produced an answer) and should be treated with caution in Goldilocks selection.

### Infrastructure note: why xVerify is separate

vLLM holds ~90% of GPU memory during inference. Loading xVerify on top would OOM. The correct pipeline is:

1. **Inference job** (`run_zero_shot_corpus_passk.py` / `run_zero_shot_drsci.py`): vLLM generates k completions, saves `all_texts: list[str]` (all k rollout texts) + `scores` (rule-only).
2. **Rescore job** (`rescore_passk_xverify.py`): loads xVerify only (no vLLM), iterates over `all_texts`, overwrites `scores` / `pass_rate` with xVerify results, preserves rule-only scores in `*_rule` columns.

All k rollout texts are saved so rescoring never requires re-inference.

---

### Corpus pass@8 (6.8k training corpus, stratified sample) (2026-03-30)

**Setup:**
- Script: `scripts/run_zero_shot_corpus_passk.py` + `scripts/zero_shot_corpus_passk.sbatch`
- Model: `Qwen/Qwen3.5-4B`, thinking OFF, temperature=0.7/top_p=0.8/top_k=20
- Sample: stratified by (source × simplified answer_type), 20 rows/stratum → **n=311 problems**
- k = 8 completions per problem (2,488 total inference calls)
- max_new_tokens: 8192; rule-only scoring during inference, xVerify-7B rescored separately
- Output: `data/results/zero_shot_corpus_passk.parquet` (inference + rule scores)
- Rescored: `data/results/zero_shot_corpus_passk_xv7b.parquet` (xVerify-7B scores, job 1445187)

**Truncation:**

| | Count | % |
|---|---|---|
| Truncated completions | 550 / 2,488 | 22.1% |
| Problems with ≥4/8 truncated | 71 / 311 | 22.8% |

**Overall accuracy (pass@1 estimated from k=8):**

| | All completions | Excl. truncated | Goldilocks [15%–85%] |
|---|---|---|---|
| Rule-only | 19.2% | 22.7% | 29/311 (9.3%) |
| **xVerify-7B** | **30.3%** | **36.3%** | **52/311 (16.7%)** |

**By source (xVerify-7B):**

| Source | n | rule (all) | xv7b (all) | xv7b (excl-trunc) | Goldilocks |
|--------|---|---|---|---|---|
| SciBench_RL | 20 | 60.6% | 70.6% | 73.8% | 8/20 (40.0%) |
| PHYSICS | 107 | 22.1% | 30.8% | 34.0% | 12/107 (11.2%) |
| UGPhysics | 120 | 16.8% | 30.3% | 35.8% | 22/120 (18.3%) |
| OlympiadBench | 44 | 8.5% | 18.8% | 25.4% | 6/44 (13.6%) |
| PHYBench | 20 | 0.0% | 12.5% | 34.8% | 4/20 (20.0%) |

PHYBench is the most truncation-distorted: 12.5% (all) vs 34.8% (excl-truncated) — its problems require long derivations that don't fit in 8192 tokens.

**By answer type (xVerify-7B):**

| Type | n | rule (all) | xv7b (all) | xv7b (excl-trunc) |
|------|---|---|---|---|
| numerical | 80 | 41.1% | 48.1% | 56.6% |
| expression | 80 | 2.7% | 25.2% | 32.1% |
| equation | 43 | 1.7% | 28.2% | 34.5% |
| mcq | 40 | 40.9% | 40.9% | 43.1% |
| true_false | 22 | 25.6% | 25.6% | 31.2% |
| multi | 41 | 4.6% | 3.7% | 4.2% |
| interval | 5 | 0.0% | 0.0% | 0.0% |

Key observations:
- Rule-only nearly fails on expression/equation (1–3%); xVerify recovers to 25–34%. xVerify is essential.
- Multi-part answers remain broken (~4%) under both rule and xVerify — a persistent verifier gap.
- Goldilocks yield is low overall (16.7%) but SciBench_RL is richest (40%).

---

### Dr. SCI gold quality: prose-style answer detection and drop (2026-03-30)

A spot-check of Dr. SCI pass@k false negatives revealed a class of gold answers that are computation
narratives or explanatory prose rather than clean mathematical expressions — e.g.
`t = -Tln(0.01) = 4.61T`, `x(t) = A*cos(ωt+φ), where ω = sqrt(k/m), A is amplitude...`,
`The final voltage across the 1μF capacitor is \frac{5000}{13}V.`
These score near-zero (7.2% pass@8 vs 24.3% for clean gold) and contribute no useful training
signal since neither rule nor xVerify can reliably judge correctness against prose golds.

**Detection heuristics** (added as step 6 in `drsci_clean.py`, tag: `prose_gold_dropped`):

| Pattern | Rule | Example |
|---|---|---|
| `var_equals_chain` | `^[A-Za-z_] = .{5,} = [0-9]` | `k = mg/x = 784 N/m` |
| `plain_prose_long` | no LaTeX markup + English words + len>60 | `x(t) = A*cos(ωt+φ), where ω = sqrt(k/m)...` |
| `prose_sentence` | starts Capital+lowercase word + len>40 | `The final voltage across the 1μF capacitor is...` |
| `label_colon_math` | `[Word]+: $math` or `[Word]+: \math` | `Geodesic equation: {D/Dt}{ds/dt} = 0` |
| `multiline_prose` | newline + alphabetic char | multi-sentence derivations |
| `approx_numeric` | `\approx [digit]` or `approximately [digit]` | `I = ... \approx -1.23 A` (pure `\approx` as math symbol is NOT flagged) |

**Numbers (from `drsci_physics_deduped.parquet`, 112,369 rows):**

| Drop reason | Count | % |
|---|---|---|
| `truncated_dropped` | 46 | 0.04% |
| `prose_dropped` (step 5, existing) | 3,633 | 3.2% |
| `prose_gold_dropped` (step 6, new) | 1,532 | 1.4% |
| **Final clean rows** | **107,158** | **95.3%** |

Note: 786 of the originally-flagged 2,318 rows were already removed by the earlier `prose_dropped`
step (overlapping patterns); only the 1,532 not caught earlier are newly dropped here.

**Distribution impact** (verified before dropping):

| | equation | expression | numerical | mcq |
|---|---|---|---|---|
| Before | 35.9% | 20.2% | 21.8% | 19.4% |
| After | 35.1% | 20.5% | 22.1% | 19.7% |
| Δ | −0.8pp | +0.3pp | +0.3pp | +0.3pp |

`mcq` and `numerical` types have zero flagged rows — those distributions are completely unaffected.
No source loses more than 1pp on any answer-type bucket.

**Authoritative clean file:** `data/processed/drsci_physics_clean.parquet` — **107,158 rows**, regenerated 2026-03-30.

---

### Dr. SCI pass@8 (Goldilocks sample) (2026-03-30)

**Setup:**
- Script: `scripts/run_zero_shot_drsci.py` + `scripts/zero_shot_drsci_sample.sbatch`
- Model: `Qwen/Qwen3.5-4B`, thinking OFF, temperature=0.7/top_p=0.8/top_k=20
- Input: `data/processed/drsci_goldilocks_sample.parquet` (599 rows, stratified from clean parquet)
- k = 8 completions per problem (4,792 total inference calls)
- max_new_tokens: 8192
- Output: `data/results/zero_shot_drsci_sample.parquet` (inference + rule scores)
- Rescored: `data/results/zero_shot_drsci_sample_xv7b.parquet` (xVerify-7B, job 1445224)

**Truncation:**

| | Count | % |
|---|---|---|
| Truncated completions | 723 / 4,792 | 15.1% |
| Problems with ≥4/8 truncated | 78 / 599 | 13.0% |

Lower truncation than corpus (15% vs 22%) — Dr. SCI answers tend to be shorter expressions.

**Overall accuracy (pass@1 estimated from k=8):**

| | All completions | Excl. truncated | Goldilocks [15%–85%] |
|---|---|---|---|
| Rule-only | 16.0% | 18.7% | 68/599 (11.4%) |
| **xVerify-7B** | **23.8%** | **27.5%** | **108/599 (18.0%)** |

xVerify recovers +7.8pp overall. Unverifiable rows drop from 20.9% (rule) to 3.2% (xVerify) — xVerify resolves most of the previously unscored equation/expression answers.

**By source (xVerify-7B):**

| Source | n | rule (all) | xv7b (all) | xv7b (excl-trunc) | Goldilocks |
|--------|---|---|---|---|---|
| MegaScience | 319 | 23.7% | 29.4% | 35.6% | 73/319 (22.9%) |
| WebInstruct-Verified | 52 | 10.6% | 28.1% | 30.6% | 12/52 (23.1%) |
| natural_reasoning | 228 | 6.5% | 14.9% | 15.9% | 23/228 (10.1%) |

`natural_reasoning` is the weakest source and lowest Goldilocks yield (10.1%) — its problems are harder or less well-matched to the model's capabilities. `MegaScience` and `WebInstruct-Verified` are comparable in Goldilocks yield (~23%).

**By answer type (xVerify-7B):**

| Type | n | rule (all) | xv7b (all) | xv7b (excl-trunc) | Goldilocks |
|------|---|---|---|---|---|
| numerical | 131 | 23.3% | 28.9% | 31.6% | 25/131 (19.1%) |
| expression | 118 | 8.2% | 21.8% | 24.9% | 18/118 (15.3%) |
| equation | 212 | 0.5% | 11.4% | 13.2% | 24/212 (11.3%) |
| mcq | 119 | 45.9% | 45.9% | 55.2% | 41/119 (34.5%) |
| unknown | 19 | 0.0% | 0.0% | 0.0% | 0/19 (0.0%) |

Key observations:
- `equation` goes from 0.5% (rule) to 11.4% (xVerify) — rule verifier nearly blind to equation-type answers; xVerify essential.
- `mcq` is unchanged by xVerify (rule already handles exact match) but shows the highest Goldilocks yield (34.5%).
- `unknown` (19 rows) scores 0% under both — these are unannotated answer types where neither rule nor xVerify can judge correctness.

**Comparison with 6.8k corpus sample:**

| | Corpus (n=311) | Dr. SCI (n=599) |
|---|---|---|
| xv7b pass@8 (all) | 30.3% | 23.8% |
| xv7b pass@8 (excl-trunc) | 36.3% | 27.5% |
| Goldilocks (xv7b) | 16.7% | 18.0% |
| Truncation rate | 22.1% | 15.1% |

Dr. SCI is harder overall (−6.5pp) but yields slightly more Goldilocks problems (18.0% vs 16.7%) — its difficulty distribution is better spread across the learnable zone once xVerify resolves the equation scoring.

---

## Scripts Reference

| Script | Purpose |
|--------|---------|
| `scripts/run_zero_shot.py` | Main zero-shot inference + scoring |
| `scripts/run_zero_shot_diag.py` | Diagnostic variant: no max_tokens cap, per-sample token stats, `--max_samples` arg |
| `scripts/rescore_xverify.py` | Re-score chunk parquets with xVerify (adds score_xverify column) |
| `scripts/rescore_all_chunks.sbatch` | SLURM template for full 8-chunk rescore |
| `scripts/analyze_zero_shot.py` | Full comparison analysis: rule vs 3B vs 7B, by source/type/truncation |
| `scripts/spot_check_fn.py` | Manual FN inspection: random sample of negatives with gold/pred display |
| `scripts/spot_check_xverify.py` | Re-score a parquet with xVerify enabled |
| `scripts/compare_xverify.py` | Side-by-side comparison of multiple xVerify model sizes |
| `scripts/diagnose_verifier.py` | Full pipeline trace: gold_parts, pred_parts, rule, xVerify per sample |
| `scripts/run_zero_shot_rerun.py` | Re-run truncated samples with extended token budget |
| `scripts/merge_rerun_chunks.py` | Merge rerun chunks back into full baseline parquet |
| `scripts/run_zero_shot_corpus_passk.py` | Pass@k inference on stratified corpus sample (saves all_texts for rescoring) |
| `scripts/run_zero_shot_drsci.py` | Pass@k inference on Dr. SCI Goldilocks sample (saves all_texts for rescoring) |
| `scripts/rescore_passk_xverify.py` | Re-score pass@k parquet (all_texts) with xVerify; preserves rule scores in *_rule columns |
| `scripts/zero_shot_corpus_passk.sbatch` | SLURM job: corpus pass@k inference |
| `scripts/zero_shot_drsci_sample.sbatch` | SLURM job: Dr. SCI pass@k inference |
| `scripts/rescore_passk_xverify.sbatch` | SLURM job: xVerify rescore of any pass@k parquet |
| `scripts/zero_shot_diag.sbatch` | SLURM: diagnostic run (30 min, --max_samples 10) |
| `scripts/zero_shot_preview.sbatch` | SLURM: small preview run (n_per_tier=5) |
| `scripts/zero_shot_chunk.sbatch` | SLURM: full chunked run template |
| `scripts/zero_shot_rerun.sbatch` | SLURM: rerun template (submit with CHUNK_ID + N_CHUNKS) |
| `scripts/rescore_rerun0_xverify.sbatch` | SLURM: one-off rescore of rerun chunk 0 with both xVerify models |
| `scripts/run_zero_shot_nothink.py` | Zero-shot inference with thinking OFF (`enable_thinking=False`), records `n_tokens` per sample |
| `scripts/zero_shot_nothink_smoke.sbatch` | SLURM: 1-sample smoke test for no-think pipeline |
| `scripts/zero_shot_nothink_full.sbatch` | SLURM: full no-think run (all 6,866 samples, single job) |
| `scripts/rescore_nothink_xverify.sbatch` | SLURM: xVerify rescore for no-think output (set XV_MODEL + XV_OUTPUT) |
| `scripts/zero_shot_nothink_08b.sbatch` | SLURM: full no-think run for Qwen3.5-0.8B (all 6,866 samples, single job) |