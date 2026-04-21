# 4. Experimental Setup

**Target:** 1.0 page. Status: empty skeleton.

## 4.1 Training data
- Dr. SCI clean + curated corpus. ~108k training pool. Stratified splits.

## 4.2 Benchmarks
- In-domain held-out (Dr. SCI test + corpus test).
- External physics: UGPhysics, PHYBench, OlympiadBench, SciBench.
- (Stretch) critpt.

## 4.3 Baselines and models
- Base Qwen3.5-4B (zero-shot).
- TIR zero-shot.
- CoT-GRPO strict.
- TIR-GRPO (ours).

## 4.4 Metrics
- Pass@1 accuracy (xVerify-7B judge).
- Per-answer-type accuracy on in-domain.
- Truncation rate, tool-execution success rate.

## 4.5 Training config
- Hardware, steps, batch, key hyperparameters. Full table in appendix.

---

<!-- prose goes here -->
