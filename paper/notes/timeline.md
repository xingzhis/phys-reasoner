# Timeline — 8 days to Apr 24 AOE

**Today:** Apr 16 2026. **Deadline:** Apr 24 2026 AOE (≈ Apr 25 evening local).

## Day-by-day plan

### Apr 16 (today) — Scaffold + plan
- [x] Agree on single-claim framing
- [x] Create `paper/` folder, outline, storyboard, novelty notes
- [ ] Pull 2–3 sample workshop papers (tool-use RL, physics reasoning) for style → `bibliography/sample-abstracts/`
- [ ] Start `bibliography/citations-todo.md`
- [ ] Decide if/how CoT-GRPO baseline compute fits alongside TIR-GRPO

### Apr 17 — Related work + method skeleton
- [ ] Draft §2 Related Work (0.5 page) with `[CITE:…]` markers
- [ ] Draft §3 Method (1.75 pages) — this is mostly locked content
- [ ] Draft §4 Experimental Setup (1.0 page) — also mostly locked
- [ ] Sketch Figure 2 rollout diagram

### Apr 18 — Intro + Overleaf port
- [ ] Draft §1 Introduction (1.0 page), leaving results bullets as placeholders
- [ ] Set up ICML 2026 LaTeX template in Overleaf
- [ ] Port §2, §3, §4 drafts into Overleaf — check line lengths, table fits, first overflow check
- [ ] Build Figure 2 final

### Apr 19 — Results scaffolding
- [ ] First training checkpoints should be landing (CoT-GRPO and TIR-GRPO)
- [ ] Write plotting scripts for Figures 3 and 4 against partial data
- [ ] Pre-structure Table 1 and Table 2 in LaTeX with placeholder values
- [ ] Start drafting §5 prose around placeholder numbers

### Apr 20 — Fill in results
- [ ] Training likely complete or near-complete
- [ ] Run benchmark evals (in-domain + UGPhysics + PHYBench + OlympiadBench + SciBench)
- [ ] Populate Table 1, Table 2, Figures 3, 4 with real numbers
- [ ] Finalize §5 prose

### Apr 21 — Analysis
- [ ] Sample 100–150 outputs per condition; label strategies and failure modes
- [ ] Draft §6 Analysis (0.75 page) with Table 3
- [ ] Hand-pick Figure 1 example from rollouts; render Figure 1
- [ ] Decide on critpt inclusion (cut if not ready)

### Apr 22 — Polish pass 1
- [ ] Draft §7 Discussion (0.5 page) and §8 Conclusion (0.25 page)
- [ ] Write abstract from finished body
- [ ] Full read-through for flow and clarity
- [ ] Check every `[CITE:…]` marker is resolved with verified bib entry

### Apr 23 — Polish pass 2 + buffer
- [ ] Overflow check — is it 8 pages?
- [ ] Figure captions, table captions
- [ ] Appendix cleanup
- [ ] Second full read-through; author pass
- [ ] Prepare submission materials

### Apr 24 — Submit (with local-day buffer)
- [ ] Submit by end of your local day (AoE gives you until Apr 25 evening, but don't cut it close)

## Risks and triggers

- **Training slips past Apr 21:** use partial-checkpoint eval for Table 1; note as "X checkpoint" in the paper. If results are still missing Apr 23, submit with best-available numbers.
- **Aggregate TIR gain is flat:** reframe abstract and intro to emphasize per-type findings and behavioral characterization. §5.2 becomes the headline, not §5.1.
- **Aggregate TIR gain is negative:** hard pivot required. Candidate: "where tool use does NOT help" characterization paper. Decide by Apr 22.
- **xVerify reward GPU setup slips:** fall back to rule-only reward for smoke; reward noise is higher but signal still present for numerical. Flag in discussion.

## Pressure-valve cuts (in order, if space tight)

1. Cut Figure 5 if present.
2. Collapse §6 into §5.
3. Cut Related Work to 0.4 page.
4. Move Table 3 to appendix.
5. Make Figure 4 one panel instead of two.

## Pressure-valve cuts (if time tight)

1. Cut critpt.
2. Skip manual solution-strategy labeling; rely on LLM-assisted first pass only.
3. Skip 0.8B scaling ablation.
4. Ship without truncation analysis (Fig 4).
