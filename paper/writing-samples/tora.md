# ToRA — writing reference (SFT-era TIR)

**arXiv:** https://arxiv.org/abs/2309.17452 • **Venue:** ICLR 2024

---

## Why read this

Not structurally closest (SFT era, pre-RL), but it's the paper everyone cites for TIR. Useful to see how the field *first* pitched tool-integrated reasoning — what felt new then, and how that framing has evolved.

## Structural observations

- ICLR-length (9 pages + appendix). Structured along the SFT-era template: Introduction → Approach → Experiments → Analysis → Related Work → Conclusion. **Related Work at the end, after results** — an older convention than SimpleTIR's upfront related-work placement.
- Heavy table-driven results — ~6-8 tables comparing ToRA variants against dozens of baselines on multiple math benchmarks.
- Prose is descriptive rather than mechanism-driven; they describe what they built rather than pitching a single-mechanism intervention.

## What to steal

- **Tables-over-figures** when results are primarily comparative across many model×benchmark combinations.
- Subsection structure in method: "Format → Trajectory Curation → Training" is a clean three-part for our own §3 to mirror.

## What NOT to steal

- Related-work-at-end placement: outdated for 2026 workshops.
- Long appendix with many variants: we don't have the ablation count to justify.
- "We built X" framing: feels tech-report-y. Our miscalibration observation is a stronger hook than "we built TIR."

## Position in our related work (§2)

One sentence credit as TIR's canonical reference: "Tool-integrated reasoning was formalized by ToRA [cite], which trained models via SFT on curated interleaved reasoning-and-tool-call trajectories."

Don't give it more than one sentence. It's context, not a target of comparison — we're in the RL era.
