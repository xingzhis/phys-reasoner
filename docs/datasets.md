# Dataset Inventory

Last updated: 2026-03-20. All row counts verified from downloaded data.

## Confirmed Downloads

| Dataset | Location | Rows | Split(s) |
|---|---|---|---|
| desimfj/PHYSICS | `data/hf_cache` | 2,000 | test only |
| UGPhysics/ugphysics (EN) | `data/hf_cache` | 5,520 | en, zh (zh = exact translations) |
| OlympiadBench OE_TO_physics_en_COMP | `data/hf_cache` | 236 | train |
| OlympiadBench OE_MM_physics_en_COMP | `data/hf_cache` | 456 | train |
| OlympiadBench physics_en_no_proof | `data/hf_cache` | 692 | test (= OE_TO + OE_MM) |
| PHYBench | `data/hf_cache` | 1,000 | train |
| SciBench-RL | `data/hf_cache` | 427 train / 153 test | train, test |
| CritPt | `data/hf_cache` | 70 | train |
| Yale NLP Physics | `data/raw/yale-physics/` | 1,297 root / 999 textonly | see tiers below |
| ABench Phy_A | `data/raw/abench/Phy_A_fixed_400.csv` | 400 | — |
| ABench Phy_B | `data/raw/abench/Phy_B_dynamic_100.csv` | 400 variants (100 base × 4) | — |

## Yale NLP Physics Tiers

| Tier | Rows | Notes |
|---|---|---|
| PHYSICS-textonly | 999 | No graphs, clean text+LaTeX |
| PHYSICS-hard | 523 | Harder subset of root |
| PHYSICS-eval | 297 | Eval split |
| PHYSICS-test | 1,000 | Test split |
| root | 1,297 | Full dataset (eval 297 + test 1000); graphs as base64 in `graphs` field |

## Answer Types by Dataset

### desimfj/PHYSICS (2,000 raw → 827 loaded)
```
Numerical: 1774,  Expression: 1158,  Equation: 302
MCQ: 132,  Open-end: 76,  T/F: 34,  Interval: 18
```
Note: answer field is `List(List(string))` — outer list is the set of parts, inner list is
alternatives per part (same quantity in different units, or paired equations, or distinct
sub-question answers). "T/F" is a variant spelling of "True/False".

**Loader drops applied (2026-03-20):**
- 953 Chinese-language rows: the dataset interleaves ZH/EN translations of the same problems.
  Keeping both would duplicate physics concepts and skew distribution; `drop_chinese=True`.
- 173 English rows where any answer part has >1 alternative: the inner list structure is
  semantically ambiguous (true unit-alternatives vs required pairs vs mispacked sub-questions)
  and cannot be reliably resolved programmatically. `drop_multi_alternative=True`.
- 47 Open-end rows: assigned to `eval_tier1` (not rule-verifiable), not `train_candidate`.
**Net loaded: 827 rows.**

### UGPhysics EN (5,520 rows, all loaded)
```
NV (Numerical Value): 2035    EX (Expression): 1605
EQ (Equation): 759            MC (Multiple Choice): 221
TF (True/False): 149          IN (Inequality): 64
multi-part combos: ~591       dirty labels cleaned (see below)
```
Note: `answers` field is a plain string (not a list), always with `\boxed{}`. Multi-part answers
use comma-separated types e.g. `'NV, NV'`; the answer string may contain multiple `\boxed{}`.

**Known dirty labels cleaned by `_clean_ugphysics_answer_type` (2026-03-20):**
- `'NV\n   \nThe final answer...'` → `'NV'`
- `'EX\n\`\`\`'` → `'EX'`
- `'\\\nMC'` → `'MC'` (2 rows in SemiconductorPhysics: spurious backslash on line 0,
  code on line 1 — fix: scan all lines not just the first)

### OlympiadBench OE_TO_physics_en_COMP (236 raw → 231 loaded)
```
Expression: 116,  Numerical: 113,  Equation: 3
Expression+Numerical: 3,  Equation+Numerical: 1
```
Note: `final_answer` is `List(string)`. `error` field gives numerical tolerance. `unit` separate field.

**Loader drops applied (2026-03-20):**
- 5 rows where answer-part count ≠ answer-type count (raw data inconsistency):
  - 3 rows pack multiple values into one answer string while `answer_type` says multi-part
  - 2 rows have 2 separate answer strings while `answer_type` is a single type
  No programmatic fix is safe for these; dropped.

### OlympiadBench OE_MM_physics_en_COMP (456 rows, multimodal)
```
Expression: 243,  Numerical: 189,  Equation: 14
Expression+Numerical: 8,  Equation+Numerical: 1,  Interval: 1, etc.
```
Note: Images in `image_1`..`image_5` fields as PIL Images.

### PHYBench (1,000 rows)
```
MECHANICS: 382,  ELECTRICITY: 284,  THERMODYNAMICS: 132
MODERN: 84,  OPTICS: 82,  ADVANCED: 36
```
Note: `answer` field is raw symbolic LaTeX, no `\boxed{}`, e.g. `$$F_{\max} = 5m\omega^2R$$`.
Verifiability: ~70% SymPy-verifiable (simple algebraic); ~30% prose/complex need LLM judge.

### SciBench-RL (427 train / 153 test)
```
Sources: atkins(105), stat(72), fund(71), thermo(66), calculus(42), chemmc(38), quan(33)
```
Note: `answer_number` + `unit` fields. `reward_model.style: 'rule'` already set.
Physics content: fund + thermo + quan + calculus ≈ 212 rows; rest is chemistry (atkins, chemmc).

### CritPt (70 rows)
All `problem_type: 'main'`. Answer is Python function in `answer_code` field.
Verified via code execution. Pure research-level theoretical physics.

### ABench
- Phy_A: 400 rows, columns: `mid`, `standard_question`, `standard_answer`. Numerical, 1% tolerance. Static.
- Phy_B: 400 rows (100 base × 4 numerical variants), columns: `mid`, `subid`, `standard_question`, `standard_answer`.
- Note: CSV has BOM character in `mid` column header — needs stripping.

## Verifiability

| Dataset | Verifiable? | Method |
|---|---|---|
| desimfj/PHYSICS (Numerical/Expression/Equation) | **Yes** | numerical tolerance + SymPy |
| desimfj/PHYSICS (MCQ, T/F) | **Yes** | exact match |
| desimfj/PHYSICS (Open-end) | **No** | LLM judge |
| UGPhysics (NV, EX, EQ, MC, TF) | **Yes** | numerical tolerance / SymPy / exact match |
| OlympiadBench (Numerical, Expression, Equation) | **Yes** | tolerance (via `error` field) / SymPy |
| OlympiadBench (Interval) | **Yes** | interval containment check |
| PHYBench (~70%) | **Partial** | SymPy for simple algebraic; LLM judge for rest |
| SciBench-RL | **Yes** | numerical + unit, rule already defined |
| CritPt | **Yes** | code execution |
| ABench Phy_A/B | **Yes** | numerical, 1% tolerance |
| Yale NLP PHYSICS-textonly | **Mostly No** | complex symbolic derivations + prose; ~20% SymPy |
| IPhO II-Vietnam | **No** | LLM judge only (explicitly `type: "llm_judge"`) |

## Corpus Assignment

### Training candidates (rule-verifiable)
| Dataset | Problems | Notes |
|---|---|---|
| UGPhysics EN (NV+EX+EQ+MC+TF types) | ~5,520 | after dirty label cleanup |
| desimfj/PHYSICS (80% of non-Open-end) | ~1,500 | reserve 20% for Tier 1 eval |
| OlympiadBench OE_TO_physics_en_COMP | 236 | text-only EN competition |
| SciBench-RL train | 427 | already RL-formatted |
| PHYBench (filtered simple subset) | ~700 | after filtering prose answers |
| **Total** | **~8,383** | before dedup |

### Evaluation
| Tier | Dataset | Problems |
|---|---|---|
| Tier 1 (in-domain) | desimfj/PHYSICS (held-out 20%) | ~400 |
| Tier 2 (dynamic) | ABench Phy_A + Phy_B | 800 |
| Tier 3 (frontier) | CritPt | 70 |
| Tier 3 (frontier) | Yale NLP PHYSICS-textonly | 999 |

## Known Data Quality Issues

Issues marked **FIXED** are handled in the loader; issues marked **OPEN** require future work.

| # | Dataset | Issue | Status |
|---|---------|-------|--------|
| 1 | UGPhysics | ~10 dirty `answer_type` labels with newlines/backticks/spurious chars | **FIXED** in `_clean_ugphysics_answer_type` |
| 2 | ABench | BOM character in `mid` column header | **FIXED** via `encoding='utf-8-sig'` |
| 3 | OlympiadBench | Multi-part `answer_type` comma-joined (`'Expression,Numerical'`) | **FIXED** via split in loader |
| 4 | OlympiadBench | 5 rows with answer/type count mismatch (raw data error) | **FIXED** — dropped |
| 5 | PHYBench | Answers are raw LaTeX without `\boxed{}` (8 rows do have it — source inconsistency) | **OPEN** — verifier must handle both |
| 6 | PHYSICS | Answer field doubly nested `List[List[str]]` | **FIXED** — unwrapped in loader |
| 7 | PHYSICS | 173 English rows with multi-alternative inner lists (semantically ambiguous) | **FIXED** — dropped (`drop_multi_alternative=True`) |
| 8 | PHYSICS | 953 Chinese rows (mislabeled as `language=en`, translations of EN problems) | **FIXED** — dropped (`drop_chinese=True`) |
| 9 | PHYSICS | `answer_type` length = total alternatives, not total parts (in multi-alt rows) | **FIXED** — resolved by dropping multi-alt rows |
| 10 | PHYSICS | `T/F` used as variant spelling of `True/False` | **OPEN** — benign; verifier should handle both |
| 11 | SciBench-RL | 143 chemistry rows (atkins, chemmc) in physics dataset | **FIXED** — filtered by default (`exclude_chemistry=True`) |
| 12 | UGPhysics ZH | Exact CN translations of EN problems | **FIXED** — EN split loaded only |
9. **Cross-dataset dedup**: PHYBench and OlympiadBench both draw from competition problems — may overlap; desimfj/PHYSICS may overlap with OlympiadBench
