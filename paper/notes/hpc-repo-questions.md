# Repo questions for the HPC session

Questions the paper-writing session needs answered from the repo. HPC session has write access to the repo and can inspect files directly; writing session only sees what's in paper-writing branch.

**Format:** each question is self-contained. HPC session answers by editing this file in place (or replying via commit) and pushing.

---

## Q1. Training data — filtered subset size and composition

**Context:** §4.2 needs a concrete number for the training pool size.

**What I need:**
- Total row count of the filtered training parquet.
- Breakdown by source (Dr. SCI vs corpus vs UGPhysics vs …).
- Breakdown by answer type.
- Filter rule used (e.g., "probe pass rate ∈ [0.05, 0.95]" or equivalent).

**Where to look:**
- `data/processed_tir/data/train.parquet` or wherever the production train file lives.
- Any README under `data/` documenting the filter.
- Session notes and recent commits mentioning "filter" or "goldilocks."

**How to answer:** fill in below, then push.

```
total rows: [N]
by source:
  dr_sci:     [n]
  scibench:   [n]
  ugphysics:  [n]
  olympiad:   [n]
  phybench:   [n]
  other:      [n]
by answer type:
  numerical:  [n]
  expression: [n]
  equation:   [n]
  mcq:        [n]
  interval:   [n]
  multi-part: [n]
  other:      [n]
filter rule: [describe]
```

---

## Q2. Final hardware layout

**Context:** §4.5 says "4 trainer nodes + 6 rollout nodes + 1 xVerify" per user's recent clarification. But `scripts/perlmutter/prod_tir_het.sbatch` shows 1 trainer node + 5 rollout + 1 xVerify, which is the older config.

**What I need:** confirm which layout is used for the production run.

**How to answer:**
- "4+6+1" (user's stated config, sbatch needs update).
- "1+5+1" (current sbatch file).
- Something else — specify.
- Number of GPUs per node (assumed 4 × A100-80G).

---

## Q3. Total training steps for the production run

**Context:** §4.5 states `TOTAL_STEPS=3000`, taken from the sbatch default. HANDOFF.md earlier discussed 1500. outline.md says 1500.

**What I need:** the actual `TOTAL_STEPS` the prod run is set to (or will be set to). Also: is there a separate value for CoT-GRPO vs TIR-GRPO?

---

## Q4. Paper title candidate — any preference?

**Context:** working title placeholders currently read "PhysCode" in legacy docs; HANDOFF says naming is deferred.

**What I need (preference only — not blocking):**
- Working title if one exists.
- Any candidates you've considered.
- Or "undecided — propose later."

---

## Q5. Author list + affiliations

**Context:** needed for Overleaf title page; not blocking prose drafting.

**What I need:**
- Author list in intended order.
- Affiliations (one per author).
- Corresponding author + email.

---

## Q6. Think-interrupt phrase — match between spec and code?

**Context:** spec in `src/phys_reasoner/tir/prompts.py` has
`THINK_INTERRUPT_PHRASE = "\nOkay, I've thought enough. Time to write my response.\n</think>\n"`.
CLAUDE.md earlier mentioned a different phrase ("time is up, formulate a final answer").

**What I need:** the exact phrase used in production training (from VeRL patch or prod sbatch). This goes in a figure caption and the appendix.

---

## Q7. Reward verifier configuration

**Context:** §3.3 says "rule-plus-xVerify". The prod sbatch uses `PHYS_REQUIRE_XVERIFY=1`.

**What I need:** Brief confirmation:
- Rule verifier file(s) used.
- xVerify model ID (xVerify-7B-I or -3B-Ib?).
- Threshold / boolean semantics of the xVerify fallback.
- Any preset differences between training-time reward and eval-time judge.

---

## Q8. PHYBench EED metric implementation

**Context:** §4.3 says PHYBench reports EED (primary) and exact-match (appendix).

**What I need:** confirm:
- EED is computed via the PHYBench-provided scorer (not a reimplementation).
- Which commit / version of the scorer is used.
- Score range (0–100, lower is better per our storyboard).

---

## Q9. ABench-Physics A vs B evaluation

**Context:** §4.3 lists ABench A (n=400) and B (n=100 per-mid).

**What I need:** confirm:
- A is 1% relative-tolerance numerical pass@1.
- B "per-mid" = problem is correct iff all 4 parametric sub-variants pass.
- Both use benchmark-provided scoring (not our rule+xVerify).

---

## Q10. Any missing numbers for zero-shot Table 1

**Context:** HANDOFF §7 quotes numbers from `docs/eval_results.md`. I want to confirm those are the **authoritative** numbers for Table 1 zero-shot rows.

**What I need:**
- Confirm the numbers in HANDOFF §7.1 and 7.2 are final for the paper.
- Flag any that have been rerun since (e.g. with a different preset).
- Are we missing any benchmark × {train, qwen} × {TIR, CoT} cells?

---

## Answering protocol

Edit this file in place. Each section: either fill in the requested info, or answer "N/A / TBD / not yet". Then `git add paper/notes/hpc-repo-questions.md && git commit -m "answer paper repo questions" && git push`. Writing session will `git pull` and integrate.

Urgency order: Q1 (blocks §4.2 prose), Q6 (blocks §3.2 caption), Q2/Q3/Q7 (block §4.5 prose), Q10 (blocks Table 1 placeholders), rest are non-blocking.
