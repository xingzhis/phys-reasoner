# 4. Experimental Setup

**Target:** 1.0 page. Status: first draft (Apr 21).

---

## 4.1 Base model and baselines

We train from **Qwen3-4B (hybrid instruct/think)** [CITE:qwen3], a 4B-parameter
reasoning-tuned base released by the Qwen team. Four conditions are
compared throughout the paper:

1. **Zero-shot CoT** — the base model in plain chain-of-thought mode, no
   tool access.
2. **Zero-shot TIR** — the base model with tool access (sandbox,
   hermes-format tool calls), single-block trajectory as in §3.
3. **CoT-GRPO** — the base model trained with Dr. GRPO + DAPO-lite
   in chain-of-thought mode, same data pool, same training budget.
4. **TIR-GRPO (ours)** — the base model trained with Dr. GRPO + DAPO-lite
   in tool-integrated mode, same data pool, same training budget.

The TIR-GRPO vs CoT-GRPO comparison is strict apples-to-apples: identical
hyperparameters, data, reward verifier stack, and training steps. The
only difference is whether the rollout environment includes the Python
tool.

## 4.2 Training data

Our training pool combines **Dr. SCI** [CITE:drsci] (Chinese-origin
undergraduate physics, translated to English) with a curated corpus
drawn from SciBench [CITE:scibench], UGPhysics [CITE:ugphysics],
OlympiadBench [CITE:olympiadbench], and PHYBench [CITE:phybench].
Problems are stratified by answer type (numerical, expression, equation,
multiple choice, interval, multi-part) and filtered by difficulty using
a zero-shot probe on the base model, retaining problems in a Goldilocks
pass-rate band where the reward signal is neither saturated nor absent.
The resulting pool contains approximately **[DATA_SIZE]** problems.
Stratified held-out dev and test splits (≈ 2 k each) are drawn from the
same sources before training; external benchmarks listed in §4.3 are
held out from the training pool entirely. Data pipeline details appear
in Appendix [TODO].

## 4.3 Benchmarks

We evaluate on eight physics benchmarks spanning two tiers.

**In-distribution held-out** — drawn from the training pool but disjoint
from the training split:

- `pool_v2_drsci` (n=503), `pool_v2_physics` (n=191),
  `pool_v2_ugphysics` (n=217), `pool_v2_scibench` (n=153).

**External** — held out from the training pool:

- `OlympiadBench` (OE_TO_physics subset, n=236) [CITE:olympiadbench] —
  exact-match on symbolic or numerical answers.
- `PHYBench` (n=1,000) [CITE:phybench] — we report the
  continuous Expression Edit Distance (EED) metric as primary and
  exact-match in the appendix.
- `ABench-Physics` A (n=400) and B (n=100 per-mid) [CITE:abench] —
  high-difficulty benchmark; B evaluates robustness across four
  parametric variants of each problem.

Where a benchmark ships with its own scorer (PHYBench EED,
OlympiadBench exact-match), we use it. For the remaining benchmarks
without a dedicated judge, we use the same rule-plus-xVerify stack
used for the training reward.

## 4.4 Metrics

- **Pass@1** on each benchmark, using the benchmark-native scorer
  where available and xVerify-7B otherwise. For PHYBench the primary
  metric is EED (lower is better); exact-match is reported in the
  appendix.
- **Tool-call rate** — fraction of TIR rollouts in which the model
  emits a well-formed `<tool_call>`.
- **Call-conditional and skip-conditional pass@1** — pass rates within
  TIR mode conditioned on whether the tool was invoked. Reported with
  the selection-bias caveat described in §5.2.
- **Paired TIR-vs-CoT outcomes** (Win/Loss/Tie per problem) — the
  strongest form of the zero-shot observation, since the same problem
  is scored under both conditions (§5.1).
- **Per-answer-type accuracy** on the in-distribution test split.
- **Truncation rate** — fraction of rollouts reaching the response cap
  without emitting a `\boxed{}` answer.

## 4.5 Training configuration

All RL runs share the following settings: rollout group size
$G = 8$, PPO mini-batch 128 problems per step, Dr. GRPO + DAPO-lite
(clip-higher 0.2/0.28, token-mean loss, overlong shaping, KL off), no
length penalty, total training [TOTAL_STEPS=3000] steps. Response
budget decomposes as thinking 12,288 + tool-call 2,048 + answer 4,096
tokens (CoT collapses the last two into a single 6,144 answer budget).
Rollouts sample at temperature 1.0, top-$p$ 1.0, matching the
on-policy GRPO specification.

Hardware: 4 trainer nodes (4 × A100-80G each), 6 rollout nodes
(4 × A100-80G each), and 1 A100 dedicated to the xVerify-7B reward
server. Training uses VeRL's `fully_async_policy` with parameter sync
every step (staleness 1). Full Hydra configuration is reproduced in
Appendix [TODO].
