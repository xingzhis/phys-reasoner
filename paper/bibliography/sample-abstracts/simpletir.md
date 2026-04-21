# SimpleTIR — structurally closest prior work

**Full title:** SimpleTIR: End-to-End Reinforcement Learning for Multi-Turn Tool-Integrated Reasoning
**Authors:** Xue, Zheng, et al.
**Venue:** arXiv:2509.02479 (Sep 2025); appears as poster at NeurIPS 2025 DL4C workshop; ICLR 2026 main.
**arXiv:** https://arxiv.org/abs/2509.02479
**GitHub:** https://github.com/ltzheng/SimpleTIR

## One-paragraph summary

SimpleTIR addresses instability in multi-turn TIR-RL: tool feedback causes distributional drift → low-probability tokens → catastrophic gradient explosions. Their fix is trajectory filtering: identify "void turns" (no code block AND no final answer) and drop them from the policy update. Starting from Qwen2.5-7B, they raise AIME24 from 22.1 (text-only baseline) to 50.5 with multi-turn TIR-RL. They explicitly avoid SFT, letting RL discover diverse patterns (self-correction, cross-validation).

## Why it's our structurally closest reference

- TIR + RL from base model (no SFT), same direction as our work.
- Math focus (AIME), not physics — leaves our domain open.
- Multi-turn; we're single-block — this is a scope difference we should acknowledge.
- Their stability fix (void-turn filtering) is orthogonal to our work; we could cite it as a different instability mitigation.

## Style notes for us

- They frame the paper around an *instability mechanism* (distributional drift) with a clean fix. This is the template: observation → mechanism → intervention → result. Our miscalibration framing has the same shape.
- They back every claim with loss-curve or gradient-norm plots, not just accuracy tables. We should do this too for call-rate and call-pass trajectories during training.
- Short related-work (~0.5 page) focused on TIR and RL stability.

## Citation BibTeX stub (to verify)

```bibtex
@article{xue2025simpletir,
  title={SimpleTIR: End-to-End Reinforcement Learning for Multi-Turn Tool-Integrated Reasoning},
  author={Xue, Zheng and others},
  journal={arXiv preprint arXiv:2509.02479},
  year={2025}
}
```

Resolve authors and venue details from arXiv PDF before committing to refs.bib.
