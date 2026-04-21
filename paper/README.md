# Paper workspace — PhysCode, ICML 2026 AI4Physics workshop

> **Start here:** [`HANDOFF.md`](./HANDOFF.md) — full, loss-free context dump of the planning session. Read it first if you're picking up this workspace in a fresh session.

**Deadline:** April 24, 2026 AOE
**Page limit:** 8 pages (ICML 2-column), references excluded
**Workshop:** https://ai4physics-workshop.github.io/
**Target topic (from CFP):** Physics-centric Scientific Reasoning with LLMs and Agents — tool-augmented agents

## Single-claim thesis

Training Qwen3.5-4B with single-block tool-integrated RLVR on ~108k physics problems yields higher accuracy across a suite of physics benchmarks than a strict CoT-GRPO baseline trained with matched data, steps, and reward stack, with gains concentrated on expression-type answers and harder problems.

One experiment carries one claim. No reward-noise mechanism pitch in the body — held for Discussion as a future-work hypothesis.

## Where things live

| Path | Purpose |
|---|---|
| `HANDOFF.md` | **Read first.** Full session context: thesis, framing decisions, rejected alternatives, zero-shot eval numbers, figure/table priorities, timeline, risks. |
| `HPC_TASKS.md` | Complete, unambiguous list of analysis tasks to hand to the HPC session. Input/output specs + priority order. |
| `outline.md` | Section-by-section story beats. Source of truth for structure. |
| `storyboard.md` | One-liner per figure and table. What each shows and why. |
| `sections/` | Per-section prose drafts (markdown). Stitched into `main.md` at the end. |
| `figures/` | Figure source data, plotting scripts, exported PDFs. One subdir per figure. |
| `tables/` | Table source CSVs and rendered tex. |
| `bibliography/refs.bib` | Verified citations only. Every entry checked against primary source. |
| `bibliography/citations-todo.md` | Placeholder citation markers and verification checklist. |
| `bibliography/sample-abstracts/` | Abstracts of strong related papers for reference. |
| `notes/novelty-pitch.md` | Crystallized thesis + reviewer-anticipation notes. |
| `notes/timeline.md` | Day-by-day plan through Apr 24. |
| `main.md` | Stitched draft for reading / sharing. Built from `sections/` when ready. |

## Workflow

1. Iterate on `outline.md` and `storyboard.md` until the story is locked.
2. Draft prose into `sections/0X-*.md`.
3. As training results arrive, populate `figures/` and `tables/`.
4. Port to Overleaf when section drafts stabilize (~Apr 20).
5. Final audit of citations in `refs.bib` before submit.

## Citations — no hallucinations policy

- Every entry in `refs.bib` must be verified against a primary source (arXiv, venue proceedings, publisher site).
- Placeholder citations in prose use markers like `[CITE:topic,year]`.
- `citations-todo.md` tracks which markers remain to be resolved.
- Final manual audit before submission.
