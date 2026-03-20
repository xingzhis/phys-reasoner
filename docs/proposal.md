## **Do Verifiable Rewards Teach Physics?**

## **RLVR Training with a Robust Physics Verifier for Transfer to Dynamic and Research-Level Reasoning**

---

**One-line thesis:** We build the first rigorous physics answer verifier, train an open LLM with RLVR on a multi-level physics corpus, and study whether the resulting gains transfer from textbook problems to dynamic variants and research-level tasks — or whether they collapse outside narrow answer formats.

---

## **Core Research Question**

Does RLVR on verifiable physics tasks produce *transferable* reasoning gains beyond the training distribution?

Sub-questions:

- Does RLVR improve over SFT on in-domain textbook physics?
- Do those gains survive on dynamic variants that break template matching (ABench-Physics Phy_B)?
- Do any gains transfer to frontier benchmarks (CritPt)?
- Which answer types benefit most: numeric, symbolic, unit-sensitive, or multi-step?
- Can equivalent gains be obtained by inference-time scaling alone (Best-of-N)?

---

## **Central Hypothesis**

RLVR will improve performance substantially on verifiable textbook and numeric physics tasks, but transfer to dynamic and frontier physics will be smaller unless the reward pipeline explicitly handles physics-specific equivalences — units, symbolic forms, and multi-part structured answers.

---

## **The Three Contributions**

1. **A physics answer verifier that actually works.** No existing math verifier handles physics answers reliably. We build a layered verifier handling: numerical answers with units (via `pint`), symbolic expressions (via `SymPy`), multi-part answers, and order-of-magnitude estimates — with an LLM fallback for edge cases. We benchmark the verifier against human annotations (target: ≥95% precision).
2. **The first broad RLVR training run on a multi-level physics corpus.** Using the verifier, we train with GRPO (via VeRL) on PHYSICS + UGPhysics + OlympiadBench (~25–27k problems after dedup), spanning high school to graduate/Olympiad level.
3. **A diagnostic analysis of what RLVR actually learns in physics.** Does the model learn physical invariants (dimensional analysis, conservation laws) or procedural templates (plug-and-chug)? This is the finding that gets cited.

---

## **Method in Brief**

| **Stage** | **Details** |
| --- | --- |
| Base model | Qwen3.5-4B (primary); Qwen3.5-9B (scaling check) |
| Training | Base → SFT warm-up → GRPO with physics-aware verifier reward |
| Training data | PHYSICS + UGPhysics + OlympiadBench ≈ 25–27k problems (tentative, can expand if needed) |
| Verifier reward | Binary (correct/incorrect); partial credit for multi-part problems |
| Baselines | Base (zero-shot), SFT-only, SFT+RLVR, Best-of-N inference scaling |

---

## **Three-Tier Evaluation**

- **Tier 1 — Textbook (in-domain):** Held-out split from PHYSICS. Confirms the method works at all.
- **Tier 2 — Dynamic robustness:** ABench-Physics Phy_B. Tests whether models learned physical relations or memorized answer templates.
- **Tier 3 — Frontier/research-level:** CritPt (~70 PhD-level problems, <10% for frontier models). Tests genuine scientific reasoning transfer. (can include other ones if needed, such as openai frontiers benchmark)

---

## **What Makes This Publishable**

Any one of these outcomes yields a strong paper:

- RLVR gives textbook gains + moderate dynamic transfer → some real physical reasoning is acquired
- RLVR gains collapse at Tier 2–3 → current verifiable-reward recipes are insufficient for real science
- Physics-aware verifier materially changes conclusions vs. naive exact-match
- RLVR beats test-time scaling at matched compute, or vice versa

---

## **Open-Source Artifacts**

- Physics verifier library (Python, with benchmarking results)
- Preprocessed annotated training dataset (PHYSICS + UGPhysics + OlympiadBench)
- GRPO-trained Qwen3.5-4B checkpoint on Hugging Face

**Target venue:** NeurIPS 2026 (primary), ICLR 2027 (fallback)