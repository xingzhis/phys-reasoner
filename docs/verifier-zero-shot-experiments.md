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
- xVerify-7B rescore (all 8 chunks): `data/results/rescore_7b_all8.parquet` ✓ done 2026-03-22 (pre-LaTeX-unit-fix; net impact +7 samples)

### Truncation Stats

| | Count | % |
|---|---|---|
| Total samples | 6,866 | 100% |
| Truncated (hit 32k limit) | 1,511 | 22.0% |
| Not truncated | 5,355 | 78.0% |

Truncated samples have near-zero accuracy (rule: 1.3%, 7B-xVerify: 7.3%). This is the key motivation for future interrupted-thinking experiments — see note below.

### Accuracy Comparison — Final (all 8 chunks, n=6,866)

Result files:
- `data/results/rescore_3b_all8.parquet` — rule + xVerify-3B scores
- `data/results/rescore_7b_all8.parquet` — rule + xVerify-7B scores

| Subset | n | Rule-only | +xVerify-3B | +xVerify-7B |
|--------|---|-----------|-------------|-------------|
| **ALL** | **6,866** | **19.0%** | **31.6%** | **38.5%** |
| Non-truncated | 5,355 | 24.0% | 39.6% | **47.3%** |
| Truncated | 1,511 | 1.3% | 3.2% | 7.3% |

### Accuracy by Source

| Source | n | Rule-only | +xVerify-3B | +xVerify-7B |
|--------|---|-----------|-------------|-------------|
| SciBench_RL | 280 | 65.7% | 74.3% | 76.8% |
| PHYSICS | 805 | 21.1% | 32.6% | 39.3% |
| OlympiadBench | 230 | 12.2% | 23.0% | 28.7% |
| UGPhysics | 5,451 | 16.9% | 30.1% | 37.3% |
| PHYBench | 100 | 0.0% | 7.0% | 12.0% |

### Accuracy by Answer Type

| Type | n | Rule-only | +xVerify-3B | +xVerify-7B |
|------|---|-----------|-------------|-------------|
| numerical | 2,698 | 38.3% | 48.3% | 52.5% |
| expression | 2,030 | 1.9% | 19.9% | 31.1% |
| equation | 792 | 2.7% | 22.6% | 32.8% |
| multi(2)-numerical | 296 | 10.5% | 19.9% | 24.7% |
| mcq | 269 | 40.9% | 40.9% | 40.9% |
| multi(2)-expression | 225 | 0.0% | 6.7% | 15.1% |
| interval | 65 | 12.3% | 18.5% | 27.7% |

Key takeaways:
- xVerify is essential for expression/equation types (+29–31pp vs rule-only)
- MCQ and true_false: xVerify adds nothing (rule handles exact-match correctly)
- 47.3% non-truncated accuracy with 7B is the **clean baseline** for interrupted-thinking experiments

### Impact of LaTeX unit fix (committed b19fef4, after rescores ran)

The `rescore_7b_all8.parquet` was produced **before** the LaTeX unit fix. To quantify the fix:

- Rule tier: 14 new TPs on SciBench_RL (all unit-conversion cases: pC, nm, mA, nm², kJ/mol, cm)
- Of those 14, **7 were already rescued by xVerify-7B** (it guessed units from problem context)
- **Net new gains: 7 samples** — the fix adds these definitively at the rule tier, before xVerify is even called
- Overall accuracy impact: +0.1pp (7/6866) — small but eliminates a class of systematic FNs
- All 7 are SciBench_RL answers where the model correctly gave SI units but gold was stored in non-SI

To get fully-corrected baseline numbers, re-run:
```bash
python scripts/rescore_xverify.py \
    --chunks data/results/zero_shot_chunk{0..7}.parquet \
    --model IAAR-Shanghai/xVerify-7B-I \
    --output data/results/rescore_7b_all8_fixed.parquet
```

### Note: Truncation and Interrupted-Thinking Experiments

22% of outputs were truncated at 32k tokens, and truncated samples have ~1–6% accuracy vs ~24–42% for non-truncated. This is the baseline for the upcoming **interrupted-thinking** experiment: if we interrupt `<think>` early to save tokens, we expect more truncation and lower accuracy. Compare against:
- `rescore_7b_all8.parquet` (this run, full thinking) for the authoritative baseline

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
| `scripts/zero_shot_diag.sbatch` | SLURM: diagnostic run (30 min, --max_samples 10) |
| `scripts/zero_shot_preview.sbatch` | SLURM: small preview run (n_per_tier=5) |
| `scripts/zero_shot_chunk.sbatch` | SLURM: full chunked run template |