# Novelty pitch + reviewer-anticipation notes

## The pitch in one breath

> We do the first controlled RL-level comparison of tool-integrated reasoning against chain-of-thought reasoning for physics problem solving, using matched data, steps, and reward stack. TIR-GRPO improves accuracy across five physics benchmarks with gains concentrated on expression-type answers and harder problems. We release a reproducible recipe: single-block TIR with think-interrupt on Qwen3.5 hybrid attention, trained with Dr.GRPO + DAPO-lite.

## Why this is interesting for a physics-reasoning workshop

- Tool-augmented agents is an explicit direction in the CFP.
- Existing TIR-RL work (SimpleTIR, GTPO) is math-only. Physics has a richer answer-type structure where the TIR vs CoT comparison is more informative.
- The per-answer-type analysis is a finding in its own right: tool use has structurally different value depending on what's being answered.

## Anticipated reviewer objections and our responses

### "TIR on math has been done. What's new for physics?"
- Physics answer types are more diverse than math (equations with units, expressions with named constants, mixed prose). This makes the per-type analysis richer.
- We run a *strict apples-to-apples RL-level comparison* that most prior work doesn't. SimpleTIR/GTPO compare TIR-RL to external baselines, not to their own CoT-RL trained with matched compute.

### "Your verifier is published (xVerify). Where is the method novelty?"
- We do not claim verifier novelty. The method contribution is the TIR rollout + think-interrupt + RL training recipe. The verifier is tooling.
- A workshop paper earns its place with an empirical contribution on a well-defined question.

### "Why only 4B? Would this hold at scale?"
- Scale ablation is appendix (0.8B) — if time permits we include. Main claim is conditional on 4B and stated as such.
- 4B was chosen for training cost and because Qwen3.5-4B instruct has non-trivial tool-call capability out of the box (22% zero-shot TIR hit rate on our probe).

### "TIR is obviously helpful because tools are useful. What's surprising?"
- The per-type decomposition *is* the surprise: gains are not uniform. They concentrate on expression-type and harder problems, and are near-zero on some categories. This is useful for deciding when tool integration is worth the rollout complexity.

### "The reward-noise hypothesis in the Discussion — why not test it?"
- We flag it as a testable future-work hypothesis, with the specific ablation described (rule-only vs rule+xVerify reward). Acknowledging the confound (problem difficulty) is better than oversmoking the causal claim.

### "Why not multi-block TIR?"
- Scope decision. Single-block is simpler, has tighter token-budget discipline, and is a clean baseline for future multi-block extensions. Multi-block would confound rollout mechanism with number of tool calls.

## Things NOT to claim

- Do not claim mechanism (reward noise causes gain). It's a hypothesis, not a result.
- Do not claim state-of-the-art on any benchmark. Context, not leaderboard.
- Do not claim generalization to other scientific domains without testing.
- Do not claim this is the first tool-integrated work for physics — there are prior SFT / inference works.

## What we DO claim

- First controlled **RL-level** TIR vs CoT comparison for physics.
- A reproducible recipe that works with a 4B open model.
- A per-type behavioral finding about where TIR helps.
