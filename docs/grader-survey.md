# Grader/Verifier Survey: Existing Verification Systems Across Our Datasets

*Compiled 2026-03-20. Source repos cloned to `data/raw/grader-repos/`.*

---

## Executive Summary

We studied the graders from all 7 datasets in our corpus. Key takeaways:

1. **No single grader handles everything we need.** Each is tailored to its dataset's answer format.
2. **HuggingFace `math-verify`** (used by PHYSICS) is the most production-ready rule-based verifier — pip-installable, handles extraction + parsing + comparison with timeout protection. Strong candidate for our backbone.
3. **OlympiadBench's grader** is the most thorough hand-built system — SymPy equivalence, per-problem tolerances, interval/equation handling, multi-part matching.
4. **UGPhysics's grader** has the richest type dispatch (7 answer types) but has significant bugs (exponent comparison disabled, naive constant removal).
5. **PHYBench's EED** is unique — a continuous 0–100 partial-credit score via tree edit distance on symbolic expressions. Potentially useful as a dense reward signal.
6. **None of them do proper unit verification.** All either strip units or ignore them. This is our biggest opportunity for differentiation.
7. **Rule-only accuracy is ~39% on physics** (PHYSICS paper). Two-stage (rule + LLM) reaches ~96%. We need an LLM fallback.

### Common Patterns Across Graders

| Pattern | Used By | Adopt? |
|---------|---------|--------|
| `\boxed{}` extraction (last, brace-depth) | All | **Yes** — universal |
| Try-everything cascade (numeric → symbolic → equation) | OlympiadBench, UGPhysics | **Yes** — robust fallback |
| SymPy `simplify(a - b) == 0` | OlympiadBench, UGPhysics, PHYBench | **Yes** — core symbolic check |
| Equation equiv via ratio test (`simplify(eq1/eq2) ∈ ℤ`) | OlympiadBench, UGPhysics | **Yes** — handles proportional forms |
| `\pm` expansion into two answers | OlympiadBench, UGPhysics | **Yes** — common in physics |
| Bracket-aware comma splitting for multi-part | OlympiadBench, UGPhysics | **Yes** — handles `f(x,y), g(a,b)` |
| Timeout on SymPy (SIGALRM or decorator) | All except SciBench | **Yes** — essential |
| Strip `\left`, `\right`, `\mathrm{}`, `$` | All | **Yes** — standard preprocessing |
| `=`/`\approx` → take RHS | All | **Yes** — handles `x = 42` |
| LLM fallback for edge cases | PHYSICS, UGPhysics | **Yes** — catches false negatives |
| Unit stripping (regex, no comparison) | ABench, UGPhysics, SciBench | **No** — we should compare units |
| `exec()` for LaTeX→float | ABench | **No** — security risk |

---

## 1. PHYSICS (`desimfj/PHYSICS`)

**Source:** No evaluation code in the repo. Paper describes using `math-verify` + fine-tuned `physics-xVerify-8B-I`.
**Repo:** `data/raw/grader-repos/PHYSICS/` (README only), `data/raw/grader-repos/math-verify/`

### Architecture: Two-Stage Rule + Model

```
1. result ← math-verify(question, ground_truth, model_output)
2. if result == Correct → return Correct
3. else → return physics-xVerify(question, ground_truth, model_output)
```

Rule-only achieves **38.6%** accuracy on physics; Rule + physics-xVerify reaches **95.9%**.

### math-verify (Rule-Based Component)

Three-stage pipeline:
1. **Extraction** — priority-ranked regex patterns for `\boxed{}`, `$...$`, `\[...\]`, "final answer is" anchors
2. **Parsing** — `latex2sympy2_extended` with configurable normalization (units, malformed operators, etc.)
3. **Comparison** — multi-strategy:
   - String equality (fast path)
   - Numeric: `float()` with `float_rounding=6`, percentage detection (`X%` → `X/100`)
   - Symbolic: `simplify(a - b).is_zero`
   - Relational: equations/inequalities, solve + compare solution sets
   - Sets/intervals: element-wise with precision, open/close matching
   - Matrix: element-wise numeric comparison
   - Assignment unwrap: `x = 1` → compare `1`

**Key detail:** Cartesian product matching — `any(compare(g,t) for g,t in product(gold, target))`.

**Timeout:** 5s via `signal.SIGALRM`.

### physics-xVerify (Model Fallback)

- Fine-tuned from xVerify-8B-I on 918 samples (physics + math, correct + incorrect pairs)
- Prompt asks for step-by-step evaluation, output JSON with `reasoning` + `judgement`
- **Model weights not released** — we'd need to train our own or use a different LLM

### Reusable

- **`math-verify` is pip-installable** (`pip install math-verify[antlr4_13_2]`) — strongest candidate for our rule backbone
- `latex2sympy2_extended` (bundled with math-verify) handles many LaTeX edge cases
- The xVerify prompt template is well-designed for physics judge use

### Limitations

- Unit conversion rules are "pre-defined" but **not published** — documented failures with kHz/Hz, nm/m
- No per-problem tolerance (fixed `float_rounding=6`)
- Open-ended answers entirely delegated to model fallback
- `SIGALRM` breaks in threaded environments

---

## 2. OlympiadBench

**Source:** `eval/auto_scoring_judge.py` (class `AutoScoringJudge`)
**Repo:** `data/raw/grader-repos/OlympiadBench/`
**Test data:** `eval/scoring_examples.json` (24 test cases)

### Architecture: Fixed Cascade (No Type Dispatch)

Despite defining 4 answer types (Numerical, Expression, Equation, Interval), the grader uses a **fixed cascade** regardless of type:

```
1. Exact string match (after preprocessing)
2. Interval equality (if bracket chars detected)
3. Numerical equality (float() cast)
4. Expression equality (SymPy parse_latex + simplify)
5. Equation equality (if both contain '=', ratio test)
```

Each step wrapped in `try/except`; failures silently fall through.

### Preprocessing (`preprocess()`)

1. Extract `\boxed{}` via brace-counting stack (not regex). Multiple `\boxed{}` concatenated with commas.
2. Strip `\left`, `\right`, `$`, `\approx`→`=`, `^\circ`, `%`, fullwidth punctuation
3. `\in` removal: `c \in (-\infty, 1]` → `(-\infty, 1]`
4. Strip `\mathrm{}`, `\mathbf{}` wrappers
5. Remove Chinese characters

### Tolerance: Absolute, Per-Problem

```python
abs(item - prediction) <= self.precision * 1.01
```

- Uses the dataset's `error` field as **absolute** tolerance (not relative!)
- Default: `1e-8` (effectively exact match)
- Per-element precision for multi-part answers: `[1e34, 1e-3]`
- **Percentage trick:** Tests GT/100, GT, GT×100 — so `0.6` matches `60` (aggressive, potential false positives)

### SymPy Usage

- `parse_latex()` → `sympify()` → direct equality check
- Substitute `pi` with `math.pi` for numeric evaluation
- **Both numeric:** `abs(expr1.evalf() - expr2.evalf()) <= precision * 1.01`
- **Both symbolic:** `simplify(expr1 - expr2).evalf()`, hardcoded tolerance `1e-3` (ignores per-problem precision!)
- **Equation:** ratio test — `simplify(eq1/eq2)` is nonzero integer?
- Power overflow guard: rejects exponents > 1000

### Multi-Part Handling

- Bracket-aware `split_by_comma()`
- `\pm` expansion into two entries
- **Unordered set matching** — order doesn't matter, both lists must have same length

### 24 Test Cases

Covers: equation rearrangement, subscripts, physics formulas, intervals with union, multi-part with reordering, units-as-symbols (`0.64N`), percentage, scientific notation, per-element precision, power overflow, `\pm` expansion.

### Reusable

- Preprocessing pipeline (well-tested, thorough)
- Bracket-aware comma splitter
- `\pm` expansion
- Unordered set matching for multi-part
- Equation equivalence via ratio test
- Power overflow guard
- Per-element precision list
- **24 test cases** — directly usable for our test suite

### Limitations

- No unit handling (units treated as algebraic symbols by SymPy)
- Silent exception swallowing everywhere
- **No timeout on SymPy** — only guard is power overflow check
- Hardcoded `1e-3` tolerance for symbolic comparison
- Union intervals must be in same order
- Percentage matching overly aggressive (GT=1 would match prediction=100)

---

## 3. UGPhysics

**Source:** `codes/eval.py`, `codes/judge.py`, `codes/math_equivalence.py`, `codes/utils.py`
**Repo:** `data/raw/grader-repos/UGPhysics/`

### Architecture: Two-Stage (Deterministic + LLM)

```
1. correctness ← auto_judge(prediction, gold_answers, precision)
2. if not correctness:
3.     correctness ← aux_judge(prediction, gold, problem, solution)  # GPT-4o
```

### Answer Extraction (Priority Cascade)

1. Strip trailing punctuation
2. "Therefore" split — take after last "Therefore"/"therefore"
3. `\boxed{}` — last occurrence, brace-matching
4. Keyword fallback: "answer is", "answer:", etc.
5. (Disabled in production) Last `$...$` or last number

### Normalization (Two Layers)

**Layer 1 (`normalize_answer`):** Remove `\left`/`\right`/`$`, replace `\approx`→`=`, strip `\mathrm{}`/`\text{}` wrappers, fix bare `\frac`/`\sqrt`, then call Layer 2.

**Layer 2 (`_strip_string`, from MATH dataset):** Remove linebreaks, replace `tfrac`/`dfrac`→`frac`, strip units after `\text{ `, strip short LHS of equations (≤2 chars), remove all spaces, convert `0.5`→`\frac{1}{2}`, convert `a/b`→`\frac{a}{b}`.

### Core: `is_equal()` — Try All 7 Types

Unlike OlympiadBench, UGPhysics tries **all 7 answer type judges** for every answer pair (no type dispatch), each with 5-second SIGALRM timeout:

```python
for answer_type in ["MC", "TF", "NV", "IN", "EX", "EQ", "TUP"]:
    if judge(pred, gold) or judge(remove_const(pred), remove_const(gold)):
        return True
```

Also tries after stripping physical constants (second pass).

**Judge details:**

| Type | Logic |
|------|-------|
| **MC** | String match, handles `[A]` and `A: description` |
| **TF** | Normalizes TRUE/FALSE/T/F/YES/NO/Y/N (+ Chinese) |
| **NV** | `float()` → relative tolerance. **BUG:** scientific notation exponent comparison commented out! `4.39×10²` matches `4.39×10⁵` |
| **IN** | Parse `(a,b]`, split on `\cup`, compare bounds via EX judge |
| **EX** | `parse_latex` + `sympify`, `simplify(a-b)`, numeric eval for pure numbers, symbolic diff for variables |
| **EQ** | Both must contain `=`. Parse sides, check `simplify(eq1-eq2)==0`, then ratio test |
| **TUP** | Strip parens, split by comma, element-wise NV comparison |

### Physical Constants Handling

`PHY_CONST` list: `G, N_A, R, e, m_e, h, ℏ, α, ε₀, μ₀, ...` (32 entries)

`remove_const()` does naive string replacement of each constant with `""`. **Major bug:** removing `"e"` corrupts `"velocity"`, `"3e5"`; removing `"h"` corrupts `"theta"`.

### LLM Fallback (`aux_judge`)

- **LLM:** GPT-4o (`gpt-4o-2024-08-06`)
- **When:** Every time `auto_judge` returns False
- **Prompt:** 4-shot physics examples, asks for equivalence judgment + justification
- **Output:** `## Equivalence Judgement\nTRUE/FALSE`
- Retries indefinitely on error, `temperature=0.0`
- **Only called when answer was extracted** — if extraction fails, returns False directly

### Multi-Answer Handling

- `split_by_comma` with bracket depth tracking
- `\pm` expansion
- **Greedy bipartite matching** (not optimal — can fail to find valid pairing)

### Inline Test Cases (bottom of `judge.py`)

```python
pred = "\\boxed{\\pi}",  gold = "3.14159265358979"
pred = "\\boxed{4.39 \\times 10^{3}}",  gold = "\\boxed{439}"
pred = "\\boxed{1}",  gold = "\\boxed{\\hbar}"   # constant stripping
pred = "\\boxed{\\sigma(\\omega)=\\frac{ne^{2}\\tau}{m(1-i\\omega\\tau)}}"
gold = "\\boxed{\\sigma(\\omega)=\\frac{ne^2\\tau}{m}\\frac{1}{1-i\\omega\\tau}}"
```

### Reusable

- Normalization pipeline (two-layer, handles many LaTeX variants)
- `split_by_comma` with bracket depth
- `\pm` expansion
- SymPy expression comparison pattern (`simplify(a-b)`, then numeric eval)
- Equation comparison via ratio test
- Physical constants list (good starting point, needs safer removal)
- LLM fallback prompt template

### Bugs & Limitations

- **Scientific notation exponent comparison commented out** — `4.39×10²` matches `4.39×10²⁰`
- **Naive constant removal** corrupts variable names and numbers
- **Tries all types blindly** — numeric `"F"` could match as MC answer "F"
- **No unit comparison** — units stripped entirely
- **Greedy matching** — may fail to find valid multi-answer pairing
- **SIGALRM not portable**, conflicts with multiprocessing
- Default precision mismatch: constructor `1e-8` vs CLI `1e-2`

---

## 4. ABench

**Source:** `Physics/eval.py`, `Physics/utils.py`, `Physics/latex2python.py`, `Physics/number_modify.py`
**Repo:** `data/raw/grader-repos/ABench/`

### Architecture: Single-Stage Numeric

```
1. Extract last \boxed{} (brace-depth tracking)
2. Strip units (regex on \mathrm{} patterns)
3. Split on = / \approx, take RHS
4. Convert LaTeX → float (via latex2sympy2)
5. Check 1% relative tolerance: |answer - pred| ≤ 0.01 × |answer|
```

### Key Details

- **Tolerance:** 1% **relative** (not absolute like OlympiadBench)
- **LaTeX→float:** Uses `latex2sympy2.latex2sympy()` → generates Python code → `exec()` → `float()`
- **Fallback extraction:** If `latex2sympy` fails, uses regex for scientific notation (`3.5 \times 10^{-4}`)
- **Unit stripping:** Regex matches `\mathrm{...}` with superscripts/subscripts, compound units with `\cdot`/`/`

### MID-Level Scoring (Phy_B)

```python
# Static: only subid=0
static_acc = data[data['subid'] == 0][col].sum() / (total // 4)
# Dynamic: all 4 variants must pass
dynamic_acc = (data.groupby('mid')[col].sum() == 4).sum() / (total // 4)
```

### Reusable

- `\boxed{}` extraction with brace-depth tracking
- Unit stripping regex patterns (as reference, not for production)
- Scientific notation regex patterns (3 variants)
- Splitting on `=`/`\approx` and taking RHS
- "Last `\boxed{}`" heuristic

### Limitations

- No symbolic equivalence (everything → float)
- No unit-aware comparison (stripped entirely)
- `exec()` security risk
- No multi-part support
- Silent failure everywhere (`except: pass`)

---

## 5. PHYBench

**Source:** `EED/` directory — `EED.py`, `latex_pre_process.py`, `extended_zss.py`, `tree_node.py`
**Repo:** `data/raw/grader-repos/PHYBench/`

### Architecture: Three-Stage with Partial Credit (EED Score)

Unique among all graders — returns a **continuous 0–100 score**, not binary.

```
Stage 1: String match → score 100
Stage 2: SymPy equivalence (simplify, expand, equals) → score 100
Stage 3: Tree Edit Distance → score 0–60 (partial credit)
```

### LaTeX Preprocessing (`master_convert()`)

The most thorough preprocessing of any grader:
1. Extract from `\boxed{}`, take content after last `=`/`\approx`/`\geq`
2. Kill commands: `\begin`, `\end` (remove with content)
3. Unwrap commands: `\text`, `\mathbf`, `\mathrm`, `\hat`, `\overline`, `\boldsymbol` (keep content)
4. Remove tokens: `\,`, `$`, `\left`, `\right`, `\displaystyle`, `\infty`
5. Replace: `\dfrac`→`\frac`, `\times`→`\bar{times}`, `\partial`→`\bar{partial}`, `\pm`→`+`, `\mp`→`-`
6. Fix fraction notation: `\frac\alpha2` → `\frac{\alpha}{2}`
7. Vector normalization: `\vec x` → `\vec{x}`
8. Exponent-fraction fix: `^\frac{a}{b}` → `^{\frac{a}{b}}`
9. Convert via `latex2sympy2_extended`

### SymPy Equivalence (Stage 2)

- `posify()` — make symbols positive for simplification
- `simplify()` with 30s timeout
- `expand(answer - test) == 0`
- `expr1.equals(expr2)` with 10s timeout

### Tree Edit Distance (Stage 3)

Converts SymPy expressions to labeled trees:
- Labels: `number_<val>`, `symbol_<name>`, `operator_<Add|Mul|Pow>`, `function_<name>`
- Uses modified Zhang-Shasha algorithm with **subtree insertion/deletion**
- Cluster discount for large subtrees: `min(s, 0.6*(s-5)+5)` for s > 5

**Scoring:**
```python
score = max(0, 100 * 0.6 - 100 * tree_dist / tree_size)
# Exact match: 100
# Partial credit: 0–60
# Zero: when tree_dist ≥ 0.6 * tree_size
```

### Reusable

- **LaTeX preprocessing pipeline** — most thorough of any grader, handles many real-world edge cases
- Three-stage approach (exact → symbolic → partial credit) is a good architecture template
- Timeout protection pattern for SymPy
- **EED partial credit** could be valuable as dense RLVR reward signal (reported 204% higher sample efficiency vs binary)

### Limitations

- No integrals or sums (returns 0)
- Length guard: test >3× longer than answer → 0
- No unit handling
- `posify()` assumes all symbols positive (problematic for negative physics quantities)
- `\infty` removed in preprocessing (breaks expressions with infinity)

---

## 6. SciBench-RL

**Source:** `eval/eval_zero.py`, `eval/post_process.py`
**Repo:** `data/raw/grader-repos/SciBench-RL/`

### Architecture: Simple Numeric

```python
def equiv(model_output, answer, unit):
    model_output = model_output.replace(',', '')
    ans = float(answer.strip())
    # Try full string, then first token
    return math.isclose(float(model_output), ans, rel_tol=0.05) or \
           math.isclose(float(model_output.split()[0]), ans, rel_tol=0.05)
```

### Key Details

- **5% relative tolerance** via `math.isclose(rel_tol=0.05)`
- **Unit field is accepted but NEVER checked** — `unit` parameter ignored
- Comma stripping for formatted numbers
- Fallback: tries first whitespace-delimited token
- `\boxed{}` extraction via brace-depth counting
- Scientific notation handling for unit strings

### Reusable

- `rel_tol=0.05` matches our planned 5% tolerance
- Simple, correct baseline for numeric comparison

### Limitations

- No unit verification (stored but ignored!)
- No symbolic equivalence
- No LaTeX parsing
- Tolerance inconsistency: `eval_zero.py` uses `rel_tol=0.05`, `ana_error.py` uses `abs_tol=0.1`

---

## 7. CritPt

**Source:** Client-side code only; grading logic is server-side (private)
**Repo:** `data/raw/grader-repos/CritPt/`

### Architecture: Remote Server + Code Execution

- Answers are **Python functions**, not raw values
- Model fills in a code template with typed signature
- Submitted to `https://artificialanalysis.ai/api/v2/critpt/evaluate`
- Server runs code in sandboxed environment (30s timeout, bounded memory)
- Rate limit: 10 submissions/24h

### Grading Modes (Server-Side, Undocumented)

| Mode | Description |
|------|-------------|
| Numeric | Expert-provided per-problem tolerance, 12 significant digits default |
| Symbolic | Hierarchical SymPy + custom routines per expression type |
| Parametric | Execute function against curated test cases |
| Composite | Element-wise grading, all parts must match |

### Reusable

- Two-step generation/parsing separation (reasoning vs answer formatting)
- Code-template answer format for structured extraction
- Per-problem expert tolerances (philosophy, not code)
- AST-based code parsing

### Limitations

- **Cannot replicate locally** — grading server is private
- Must use remote API for Tier 3 evaluation
- No test cases included in public data

---

## Synthesis: Design Implications for Our Verifier

### 1. Architecture Decision: Use `math-verify` as Backbone?

**Pros:**
- Production-ready, pip-installable
- Handles extraction + parsing + comparison
- Already used by the PHYSICS dataset (our largest source)
- Maintained by HuggingFace
- Has `latex2sympy2_extended` bundled (handles more LaTeX than raw `parse_latex`)
- Timeout protection built in

**Cons:**
- No unit-aware comparison (we'd need to add this on top)
- Fixed `float_rounding=6` (no per-problem tolerance)
- SIGALRM-based timeout (not thread-safe)
- May be opinionated in ways that conflict with other datasets' grading

**Recommendation:** Use `math-verify` for extraction + parsing, but build our own comparison layer on top that adds:
- Unit-aware comparison via `pint`
- Per-problem tolerance (absolute or relative)
- Physics-specific normalization from UGPhysics/OlympiadBench

### 2. Preprocessing: Merge the Best

Combine normalization from multiple graders:
- **Base:** math-verify's `NormalizationConfig` (most comprehensive)
- **Add:** OlympiadBench's `\in` removal, Chinese character stripping
- **Add:** UGPhysics's `_strip_string` special cases (`0.5`→`½`, short LHS stripping)
- **Add:** PHYBench's vector/fraction notation fixes
- **Skip:** UGPhysics's naive constant removal (too dangerous)

### 3. Comparison Pipeline: Layered Dispatch

```
1. Extract answer (last \boxed{}, keyword fallback)
2. Normalize (merged preprocessing)
3. Route by answer_type:
   a. MCQ/TF    → exact string match
   b. Numerical  → float parse + relative tolerance (default 5%)
   c. Unit-aware → pint parse + dimensional comparison
   d. Symbolic   → SymPy simplify(a-b), with timeout
   e. Equation   → SymPy + ratio test
   f. Interval   → bracket type + bound comparison
   g. Multi-part → split + element-wise + unordered matching
4. If all fail → LLM fallback (optional, for eval only)
```

### 4. What We're Adding That No Grader Has

| Feature | Status Across Graders | Our Plan |
|---------|----------------------|----------|
| **Unit-aware comparison** | All strip/ignore units | `pint` dimensional analysis |
| **Explicit type dispatch** | Only UGPhysics (buggy) | Clean dispatch by normalized `answer_type` |
| **Relative + absolute tolerance** | Mixed/inconsistent | Per-problem, configurable |
| **Safe constant handling** | UGPhysics naive removal | None initially; add later if needed |
| **Thread-safe timeout** | None (all use SIGALRM) | `multiprocessing.Process` with timeout |
| **Structured logging** | None (all swallow errors) | Log every comparison attempt + result |

### 5. Test Cases: Harvest from Existing Graders

Sources for our `test_verifier.py`:
- OlympiadBench `scoring_examples.json` — 24 cases
- UGPhysics `judge.py` inline tests — 7 cases
- Our own gold data round-trip — ~7,000 cases
- Synthetic perturbation tests (±1%, ±10%, sign flip, unit swap)

### 6. Open Questions

1. **Should we use EED partial credit as RLVR reward?** PHYBench reports 204% higher sample efficiency. But continuous rewards may destabilize GRPO training. Worth experimenting.
2. **How to handle the PHYSICS dataset's unpublished unit conversion rules?** We'll need to reverse-engineer what they consider equivalent from their test data.
3. **LLM fallback for training vs eval?** Using GPT-4o as reward during RLVR training is expensive and non-reproducible. Maybe: rule-only for training reward, rule+LLM for evaluation.
4. **Thread safety:** Our GRPO training will call the verifier from multiple workers. Must avoid SIGALRM. Use `multiprocessing.Process` with timeout or `concurrent.futures.ProcessPoolExecutor`.
