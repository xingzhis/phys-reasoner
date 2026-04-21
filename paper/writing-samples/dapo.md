# DAPO — writing reference (crisp technique-list style)

**arXiv:** https://arxiv.org/abs/2503.14476

---

## Why read this

DAPO is the cleanest example of the "crisp technique list" paper genre. Each technique is named, justified in 2–3 sentences, ablated, and combined. Useful as a style reference for our §3.3 where we describe Dr.GRPO + DAPO-lite.

## Structural observations

- Four techniques, each with:
  - A **name** (Clip-Higher, Dynamic Sampling, Token-Level Loss, Overlong Shaping).
  - A **motivation** (one sentence: what goes wrong without it).
  - A **description** (one sentence: what it does).
  - An **ablation** (later in Results: this technique alone adds X pp).
- Strong visual language: curves showing entropy-over-steps for each technique on/off.
- Ablation table has the four techniques as rows, not as runs — reads bottom-to-top as "ablating each piece reduces performance by…"

## What to steal

- **Named-component structure.** In our §3.3, name our choices explicitly: "Dr. GRPO (no length/std normalization)…", "DAPO-lite (clip-higher, token-mean loss)…". Readers remember names.
- **One-sentence motivation pattern:** "[Problem]. [Technique] fixes this by [Mechanism]." Compact and information-dense.
- **Ablation-as-rows table format** for the appendix ablations, if we get to run any.

## What NOT to steal

- Don't try to pitch four techniques. We have exactly *one* intervention (single-block TIR with think-interrupt). The crisp-list format is for papers whose contribution is a bundle of techniques; ours isn't.
- Don't mimic DAPO's tech-report feel. Our paper has a thesis; DAPO has a system.

## Position in our paper

- §2: one sentence crediting DAPO for clip-higher + token-level loss + overlong shaping, which we adopt.
- §3.3: "Our training follows Dr. GRPO [cite] with the async-compatible subset of DAPO [cite] — clip-higher (0.2/0.28), token-mean loss aggregation, overlong-response shaping. We omit DAPO's dynamic sampling, which is sync-only; zero-advantage groups contribute zero loss under the Dr.GRPO + KL-off setting." Two sentences, done.
