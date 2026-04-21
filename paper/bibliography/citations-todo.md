# Citations to verify

Every entry below must be confirmed against a primary source before it goes into `refs.bib`.

Format: `[CITE:tag]` — topic — proposed primary source — status.

## RL for reasoning with verifiable rewards

- [CITE:deepseek-r1] — R1's GRPO-from-base pipeline — arXiv 2501.12948 — UNVERIFIED
- [CITE:grpo-original] — GRPO algorithm — DeepSeekMath paper (Shao et al.) — UNVERIFIED
- [CITE:dapo] — Asymmetric PPO clip + token-level loss + dynamic sampling — ByteDance Seed team, arXiv 2503.14476 — UNVERIFIED
- [CITE:dr-grpo] — Length/std normalization removal — UNVERIFIED
- [CITE:dr-sci] — Dr. SCI corpus + training — UNVERIFIED (need arXiv)
- [CITE:rlver] — RLVεR theoretical framework — UNVERIFIED (one-line nod in Discussion only)

## Tool-integrated reasoning

- [CITE:tora] — ToRA, tool-integrated math reasoning (Gou et al., ICLR 2024) — UNVERIFIED
- [CITE:mathcoder] — MathCoder (Wang et al.) — UNVERIFIED
- [CITE:simpletir] — SimpleTIR (NeurIPS 2025 per proposal) — UNVERIFIED
- [CITE:gtpo] — GTPO (Nov 2025) — UNVERIFIED
- [CITE:nemotron-tir] — NeMo tool-integrated — UNVERIFIED (optional)
- [CITE:pal] — Program-Aided Language Models (Gao et al.) — UNVERIFIED

## Verification

- [CITE:math-verify] — HuggingFace math-verify — UNVERIFIED
- [CITE:xverify] — IAAR-Shanghai xVerify — UNVERIFIED

## Physics reasoning benchmarks

- [CITE:ugphysics] — UGPhysics — UNVERIFIED
- [CITE:phybench] — PHYBench — UNVERIFIED
- [CITE:olympiadbench] — OlympiadBench — UNVERIFIED
- [CITE:scibench] — SciBench (Wang et al., ICML 2024) — UNVERIFIED
- [CITE:physreason] — PhysReason — UNVERIFIED (optional)
- [CITE:phy-physics] — PHYSICS benchmark — UNVERIFIED

## Base model + training infra

- [CITE:qwen3] — Qwen3 technical report — UNVERIFIED
- [CITE:verl] — VeRL training framework — UNVERIFIED

## Think-interrupt / budget control

- [CITE:scalerl] — ScaleRL-style thinking truncation — UNVERIFIED (proposal references it)

## Other

- [CITE:gsm8k] — Reference for math RL comparison, if used — UNVERIFIED (skip if not cited)
- [CITE:math500] — MATH benchmark — UNVERIFIED (skip since we dropped it)

## Verification workflow

1. For each tag, run a WebSearch or WebFetch to confirm title, authors, venue, year.
2. If confirmed, add a proper entry to `refs.bib`.
3. If not confirmed (hallucination risk), either find an alternative source or drop the citation.
4. Final audit before submission: diff `refs.bib` against this list; every prose `\cite{}` must match a verified entry.
