# 2. Related Work

**Target:** 0.5 page. Status: first draft (Apr 21).

---

## RL with verifiable rewards for reasoning

Reinforcement learning from verifiable rewards (RLVR) has driven
substantial progress on mathematical reasoning, beginning with
DeepSeek-R1 [CITE:deepseekr1]'s demonstration that GRPO [CITE:grpo]
suffices to elicit strong chain-of-thought behavior from a base model
without supervised warm-up. Subsequent recipes have sharpened the
practical setup: DAPO [CITE:dapo] proposed clip-higher, token-mean
loss aggregation, dynamic sampling, and overlong reward shaping;
Dr. GRPO [CITE:drgrpo] removed the length- and standard-deviation-
normalization terms that bias the original GRPO estimator; and
Dr. SCI [CITE:drsci] applied these techniques to a physics training
pool. Our training configuration combines Dr. GRPO with the
async-compatible subset of DAPO.

## Tool-integrated reasoning

Tool-integrated reasoning (TIR) interleaves natural-language reasoning
with calls to external programs. Early systems relied on
supervised fine-tuning on curated tool-use trajectories
(ToRA [CITE:tora], MathCoder [CITE:mathcoder], PAL [CITE:pal]). Recent
work trains TIR directly with RL: SimpleTIR [CITE:simpletir]
identified a distributional-drift failure mode in multi-turn TIR-RL
and introduced trajectory filtering to stabilize training, raising
AIME24 from 22.1 to 50.5 on Qwen2.5-7B. Our setting differs along three
axes: (i) **domain** — physics rather than competition math;
(ii) **scope** — a single-block trajectory with at most one tool call;
and (iii) **framing** — we ask whether tool availability alone helps a
reasoning-tuned base, not whether RL can stabilize tool use.

## Physics reasoning benchmarks

Recent benchmarks have characterized LLM physics-reasoning capability
at multiple levels: PHYBench [CITE:phybench] for Olympiad-level
problems with a continuous expression-edit-distance metric;
UGPhysics [CITE:ugphysics] for undergraduate physics with a rule-plus-
LLM scoring pipeline (MARJ); OlympiadBench [CITE:olympiadbench] for
competition problems; SciBench [CITE:scibench] for university-level
STEM. Results on these benchmarks consistently show that physics
ability does not reduce to math ability — a motivation that holds
equally for training-time interventions.

## Verification of open-ended answers

Rule-based verifiers such as `math-verify` [CITE:mathverify] handle
canonical numerical and algebraic forms but miss semantically
equivalent paraphrases. LLM-based judges such as xVerify
[CITE:xverify] close this gap at the cost of a separate inference
pass. Our reward stack combines both in a rule-first cascade, matching
the pattern used in UGPhysics's MARJ pipeline; we do not pitch the
verifier design as a contribution.
