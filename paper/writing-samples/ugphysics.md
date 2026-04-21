# UGPhysics — writing reference (physics voice + verifier framing)

**arXiv:** https://arxiv.org/abs/2502.00334 • **Venue:** ICML 2025

---

## Why read this

Same "physics-benchmark voice" as PHYBench, but with a **verifier design** section (MARJ: Model-Assistant Rule-based Judgment). The verifier-design framing is useful for us as a comparison point — MARJ is the published hybrid rule-plus-LLM physics scorer, closest public analog to our rule+xVerify stack.

## Structural observations

- ICML 2025 paper, standard 9-page format + appendix.
- Notable: the paper separates the **benchmark contribution** (5,520 problems, stratified metadata) from the **evaluation methodology contribution** (MARJ) — two clean chapters that stand on their own.
- Multi-language corpus (English + Chinese) — they address the translation/normalization issue head-on in the curation section.

## What to steal

- **The "math ability ≠ physics ability" motivation sentence.** One well-placed citation can do a lot of work in our §1 hook; UGPhysics is the authoritative source for this claim.
- **Stratified taxonomy presentation.** They describe 13 subjects × 7 answer types × 4 reasoning skills in a compact table. We need a similar compact table for our own stratification dimensions in §4.
- **Verifier design framing:** they cast MARJ as an explicit design choice with tradeoffs. We should cast our rule+xVerify as a similar design choice ("we use xVerify because…") rather than implementation plumbing.

## What NOT to steal

- Don't dedicate a section to verifier design — it's *not* our contribution. Treat it as one paragraph in §3.3 and one appendix table.
- Don't use their MARJ pipeline directly as the eval judge (we use benchmark-provided scorers where available; xVerify elsewhere). Different strategy.

## Position in our paper

- §1: cite as authoritative source for "physics reasoning is distinct from math reasoning."
- §2: one sentence in benchmark paragraph + MARJ mentioned in verification paragraph.
- §4: our answer-type taxonomy should mirror theirs (numerical / expression / equation / MCQ / interval / multi-part / ...) — reference their taxonomy explicitly if we align with it.
