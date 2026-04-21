# Sample abstracts — reference papers for writing

Five initial samples pulled on 2026-04-19. Each file has a short summary, position in our related work, and a BibTeX stub to verify.

| File | Role |
|---|---|
| `simpletir.md` | Structurally closest prior work (TIR + RL, math). arXiv 2509.02479. |
| `tora.md` | Foundational TIR (SFT era). ICLR 2024. |
| `phybench.md` | Primary external physics benchmark. arXiv 2504.16074. |
| `ugphysics.md` | External physics benchmark + verifier (MARJ) reference. ICML 2025. |
| `dapo.md` | RL training recipe reference. arXiv 2503.14476. |

## What's missing (for user to add or next session to pull)

- A recent **physics RL** paper beyond Dr. SCI — if any exists in the Mar 2026 window.
- **GTPO** (mentioned in proposal as TIR-RL, Nov 2025).
- **xVerify** paper (IAAR-Shanghai).
- An **ICML workshop or NeurIPS MATH-AI short paper** for length/style calibration (4–8 pages, similar scope).
- User's own prior paper style references (user mentioned arxiv 2410.12779 and 2602.00217 earlier).

## Style calibration notes (from what we've read)

- SimpleTIR: ~8-pages-ish ICLR paper. Clean "mechanism → fix → result" arc. Plots training-dynamics not just final accuracy.
- ToRA: longer ICLR paper. Lots of tables. SFT era, so shape is different.
- PHYBench / UGPhysics: benchmark papers. Pay attention to how they motivate physics specifically.
- DAPO: tech-report-flavored but with a clean "four techniques" pitch. We're doing the opposite — one thesis, no laundry list.

## For the new writer

Read these summaries before drafting §2 Related Work. Structure of our §2:
1. RL for reasoning w/ verifiable rewards (Deepseek-R1, GRPO, DAPO, Dr. GRPO, Dr. SCI)
2. Tool-integrated reasoning (ToRA, MathCoder, SimpleTIR, GTPO)
3. Physics reasoning benchmarks (UGPhysics, PHYBench, OlympiadBench, SciBench, Dr. SCI corpus)
4. Verification for open-ended answers (math-verify, xVerify, MARJ)
