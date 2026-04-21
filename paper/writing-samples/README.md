# Writing samples — style and structure references

**Purpose:** learn paper-writing craft from strong work in adjacent areas. Each file analyzes a specific paper for:
- Section arrangement and length allocation
- Intro architecture (hook, motivation, contributions)
- Related-work organization
- Method-section structure
- Results presentation: tables, figures, captions
- Paragraph-level flow
- Specific rhetorical techniques worth stealing

**Not the same as related work.** Papers analyzed here may or may not be cited in our own §2. For citation management see `paper/bibliography/`.

## Current samples

| File | Why picked |
|---|---|
| `simpletir.md` | Structurally closest: TIR + RL for reasoning. Template for our "surprising failure mode → clean intervention" arc. |
| `tora.md` | Foundational TIR (SFT era). Useful to see how *tool-use* gets pitched without the RL angle. |
| `phybench.md` | Physics benchmark paper voice — how physics-specific framing is done. |
| `ugphysics.md` | Physics voice + verifier discussion (MARJ). Useful for our §2 and §4 prose. |
| `dapo.md` | Training recipe voice — "crisp technique list" style. |
| `user-2602.00217.md` | Author's prior paper — voice calibration. |
| `user-2410.12779.md` | Author's prior paper — voice calibration. (Analysis pending — full text not yet extractable.) |

## Reading order for a new writer

1. `simpletir.md` first — closest structural template for our paper.
2. `user-*.md` — match the established voice.
3. `phybench.md` / `ugphysics.md` — calibrate physics-specific tone.
4. `dapo.md` — steal the "crisp technique list" style for §3.3.

## What to extract, not what to copy

- Steal: paragraph structures, transition phrases, caption formats, table layouts.
- Don't steal: thesis shape, specific claims, voice.

## Update notes

- **2026-04-19:** folder was `bibliography/sample-abstracts/`, held citation-positioning notes.
- **2026-04-21:** renamed to `writing-samples/` to make the style-learning purpose explicit. Per-file docs rewritten with structural/stylistic focus. Citation positioning moved to `bibliography/citations-todo.md`.

## Still to add

- A short workshop paper (4–8 pages, AI4Science/MATH-AI workshop) for length calibration.
- xVerify paper structural analysis if we can get the PDF.
- GTPO structural analysis if it exists in published form.
