# DAPO — training recipe reference

**Full title:** DAPO: An Open-Source LLM Reinforcement Learning System at Scale
**Authors:** ByteDance Seed team
**Venue:** arXiv:2503.14476 (Mar 2025); open-source release built on VeRL.
**arXiv:** https://arxiv.org/abs/2503.14476
**Site:** https://dapo-sia.github.io/

## One-paragraph summary

Open-source RL system reaching 50 points on AIME 2024 with Qwen2.5-32B. Four techniques: (1) **Clip-Higher** — asymmetric PPO clip, upper 0.28, lower 0.2, to prevent entropy collapse; (2) **Dynamic Sampling** — filter groups where all rollouts are correct or all wrong (sync-only); (3) **Token-Level Policy Gradient Loss** — token-mean aggregation, critical for long CoT; (4) **Overlong Reward Shaping** — soft penalty for rollouts exceeding length cap. Beats DeepSeek-R1-Zero-Qwen-32B at 50% fewer training steps.

## How we use it

Our training config is "Dr. GRPO + DAPO-lite":
- Clip-higher (0.2 / 0.28) — adopted directly.
- Token-mean loss aggregation — adopted.
- Overlong shaping — soft form adopted.
- Dynamic sampling — **dropped** (sync-only; not compatible with our `fully_async_policy`). Compensated by running more steps so that zero-advantage groups contribute zero loss (KL off).

## Position in our related-work

Cite in §2 paragraph on "RL for reasoning with verifiable rewards." One sentence on the four techniques, one sentence on which ones we adopt.

## Citation BibTeX stub

```bibtex
@article{yu2025dapo,
  title={DAPO: An Open-Source LLM Reinforcement Learning System at Scale},
  author={Yu, Qiying and others},
  journal={arXiv preprint arXiv:2503.14476},
  year={2025}
}
```

Resolve authors from PDF before committing.
