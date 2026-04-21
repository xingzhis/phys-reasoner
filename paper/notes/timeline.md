# Timeline — 3 days to Apr 24 AOE

**Today:** Apr 21 2026. **Deadline:** Apr 24 2026 AOE (≈ Apr 25 evening local time).

**Remaining calendar days:** Apr 22, Apr 23, Apr 24. That's it.

## Day-by-day plan (revised for 3-day window)

### Apr 21 (today) — Framing + scaffold
- [x] Claim-1-led thesis locked (TIR-mode vs CoT-mode paired comparison as headline)
- [x] HPC_TASKS updated with Task 1b (per-problem paired comparison)
- [x] Outline updated
- [ ] Writing samples refactored (in progress)
- [ ] Start drafting §3 Method and §4 Setup — these are ~90% locked content
- [ ] HPC runs Task 1b on existing zero-shot rollouts (blocks §5.1)

### Apr 22 — Prose draft 1 + Overleaf port
- [ ] Complete §3 Method and §4 Setup drafts (md)
- [ ] Draft §1 Introduction around locked contributions; leave results-sentence as placeholder
- [ ] Draft §2 Related Work with `[CITE:tag]` markers
- [ ] Set up Overleaf template (user provides preference or fresh ICML template)
- [ ] Port §2, §3, §4 to Overleaf; first overflow check
- [ ] When Task 1b lands: pull §5.1 Table A numbers in
- [ ] Training should be well underway; monitor call-rate collapse

### Apr 23 — Results pull + §5 and §6 draft
- [ ] Training results landing — run Task 4 evaluations on final/best checkpoints
- [ ] Task 1b on post-RL rollouts → Table 1 numbers
- [ ] Draft §5 Results around real numbers; build Figures 1, 3, 4
- [ ] Draft §6 Analysis using sampled outputs (LLM-assisted labeling)
- [ ] Draft §7 Discussion and §8 Conclusion
- [ ] Write Abstract from finished body
- [ ] Full read-through; citation audit
- [ ] If training is incomplete or results are weak: evaluate fallback framing (characterization-only paper)

### Apr 24 — Polish + submit
- [ ] Overflow check — is it 8 pages?
- [ ] Figure/table captions (takeaway-first)
- [ ] Abstract polish
- [ ] Verify every `[CITE:tag]` is resolved in refs.bib
- [ ] Second read-through
- [ ] Submit by end of local day (AoE buffers to Apr 25 evening)

## Risks and triggers (compressed)

**Result-level risks:**
- Training slips past Apr 23 → use whatever checkpoints exist; report honestly.
- Aggregate TIR-GRPO gain is negative → hard pivot Apr 23 to characterization framing. Don't wait until Apr 24.
- Call rate collapses to 0 → frame as a finding; paper becomes "RL dynamics under tool availability on reasoning-tuned models" rather than "TIR beats CoT."

**Time-level risks:**
- Overleaf porting drags → port §3 and §4 first (most locked); leave §1 and §5 in markdown until last.
- Citation verification drags → skip the Apr 23 full audit; do only `\cite` vs `refs.bib` name-match check.

## Pressure-valve cuts (in order, if tight)

### Space cuts (if exceeds 8 pages)
1. Cut Figure 5 (truncation).
2. Collapse §6 into §5.
3. Cut Related Work to 0.4 page.
4. Move Table 3 to appendix.
5. Drop Figure 2 (rollout diagram); replace with numbered list in §3.2.

### Scope cuts (if time tight)
1. Cut critpt benchmark.
2. Skip manual solution-strategy labeling; LLM-assisted only.
3. Skip 0.8B scaling ablation.
4. Ship without Figure 5.
5. Skip §6 qualitative analysis entirely — present results and discussion only.

## What the paper-writing session works on each day

### Apr 21 evening:
- Sections most ready: §3 Method, §4 Setup (just need data size placeholder).
- Start drafting in parallel with HPC Task 1b.

### Apr 22:
- §1, §2 drafts.
- Overleaf setup.
- Integrate Task 1b numbers into §5.1.

### Apr 23:
- §5 real numbers.
- §6, §7, §8.
- Abstract.

### Apr 24:
- Polish only. No new content.
