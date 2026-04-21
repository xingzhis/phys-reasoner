# PHYBench — writing reference (physics benchmark voice)

**arXiv:** https://arxiv.org/abs/2504.16074

---

## Why read this

Physics-benchmark papers establish a distinct voice — they have to *motivate physics specifically* and distinguish from math. Useful for our §1 motivation and §2 benchmark paragraph.

## Structural observations (from abstract + public info)

- Main contribution is a benchmark; paper structure tends to: problem → curation methodology → evaluation pipeline → results → analysis.
- Introduces a novel metric (EED, tree-edit-distance over expressions) — pitched as overcoming the "exact-match is too strict" problem.
- Contrast framing vs. math benchmarks: they explicitly point out that physics problems require symbolic reasoning under units/constraints, not just algebraic manipulation.

## What to steal

- **The physics-specificity motivation.** Sentences like "Physics problems require dimensional analysis, symbolic reasoning with units, and multi-step derivation under physical constraints — properties that math benchmarks do not stress." This framing is workshop-appropriate and we should adapt it for our §1.
- **The 'why a new metric' argument.** Applies directly to our discussion of rule-verifier vs. xVerify: when a continuous metric (EED) or a fallback judge gives more signal than exact-match, motivate the choice.

## What NOT to steal

- Benchmark-paper structure is different from ours. Don't try to spend 2 pages on data curation — we're training a model, not releasing a dataset.

## Position in our paper

- §2 Related Work: one sentence as a benchmark citation, grouped with UGPhysics / OlympiadBench / SciBench.
- §4 Setup: one paragraph introducing benchmarks with EED noted for PHYBench.
- §5 Results: PHYBench EED appears as a column in Table 1.

## Voice note

Physics-benchmark papers tend to be more expository than ML method papers — they spend time explaining physics domain properties to an ML audience. Our workshop audience overlaps with theirs; we can afford one or two sentences of domain motivation in §1 without feeling indulgent.
