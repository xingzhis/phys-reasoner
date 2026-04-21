# SimpleTIR — primary structural template

**Full title:** SimpleTIR: End-to-End Reinforcement Learning for Multi-Turn Tool-Integrated Reasoning
**Venue:** arXiv:2509.02479 (Sep 2025); NeurIPS 2025 DL4C workshop (poster); ICLR 2026.
**arXiv:** https://arxiv.org/abs/2509.02479

---

## Why it's our primary template

Same direction as us: TIR + RL from base model, no SFT. Math instead of physics; multi-turn instead of single-block. Paper shape is "observation → mechanism → intervention → result" — *exactly* our shape.

## Structure (from PDF analysis)

Total ~22 pages (arXiv version). For our 8-page workshop submission we compress proportionally — but the beat-level structure still maps cleanly.

### Section allocation (arXiv version)

| Section | Role |
|---|---|
| Introduction | Problem → motivation → contributions + links |
| Related Work | Short, integrated-citation style |
| Method | Problem formulation → approach → technical details |
| Experiments | Dominant section; 4–6 pages. Multiple tables, figures |
| Discussion / Conclusions | Brief, forward-looking |

### Intro architecture (steal this)

Paragraph 1: positions tool use in LLMs as the current frontier; cites contemporary work (RAGen, ReTool, etc.).
Paragraph 2–3: identifies the specific failure — multi-turn RL with tool feedback is unstable; describes the symptom (catastrophic gradient explosions) concretely.
Paragraph 4: introduces SimpleTIR as the remedy, one-sentence description of the intervention.
Paragraph 5: headline numeric result (AIME24: 22.1 → 50.5).
Paragraph 6: contributions bullet list + project links.

**What to steal:** the "problem is specific, symptom is concrete, fix is one sentence, result is one number" structure. This is what makes an intro memorable on a skim.

### Related work (steal this)

- ~0.5 page equivalent.
- Cites integrated into prose, not siloed under a "Related Work" heading with lots of sub-headers.
- Theme-clustered: (a) RL for reasoning, (b) tool-integrated approaches, (c) multi-turn stability.
- Dense citation in the first paragraph of each theme; thinning in subsequent sentences.

### Method (steal the "mechanism box" style)

- Prose-heavy but with explicit formulae where needed.
- Has algorithm/pseudocode for the key step (void-turn filtering).
- Separates "problem formulation" from "our intervention" into distinct subsections — makes it easy to skim past the recap.

### Experiments — what they do *right*

- **Training dynamics plots, not just final-accuracy tables.** Loss curves, gradient-norm curves. This is the single most useful thing to copy: *show the reader the dynamics that motivate the claim*, not just the endpoint.
- Multiple benchmarks in one main table; per-subset breakdowns in companion tables.
- Figures captioned takeaway-first: legend-style label + short explanatory phrase.

### Rhetorical techniques worth stealing

- "We show…" — agency-forward, confident voice.
- Contrast framing: "In contrast to prior work which [does X], we [do Y]." Used once in each section to re-anchor the contribution.
- Minimal hedging. Where they hedge, they do it explicitly: "We conjecture…" "Our experiments suggest…"
- Headlines in table captions: `Table 3: SimpleTIR improves AIME24 from 22.1 → 50.5 with no SFT.`

---

## How this maps to our 8-page paper

- **Our intro** should have exactly the same 6-paragraph structure but compressed: ~1 page.
- **Our related work** should emulate the theme-clustered integrated-citation style in ~0.5 page.
- **Our method** should have one "our intervention" subsection (§3.2 rollout mechanism + think-interrupt) that reads like their §X on void-turn filtering — concrete, specific, one idea per subsection.
- **Our results** section must include training-dynamics figures (tool-call rate over steps, call-path pass over steps) — not just a final-checkpoint accuracy table. This is the SimpleTIR lesson.

## Differences to preserve

- They pitch *RL stability*; we pitch *tool-use benefit*. Different axis.
- They have one clear instability mechanism; we have a less-crisp observation ("tool availability doesn't help"). Our writing needs to work harder to make the observation feel surprising, not just reported.

## Citation position in our §2

Cite in the "Tool-integrated reasoning" paragraph. One sentence of credit, one sentence of difference: "SimpleTIR [cite] demonstrates multi-turn TIR-RL can be stabilized via void-turn filtering. Our work differs in domain (physics vs. math), scope (single-block vs. multi-turn), and framing (calibration of tool *benefit* vs. training stability)."
