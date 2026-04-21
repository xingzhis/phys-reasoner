# ToRA — foundational TIR (SFT era)

**Full title:** ToRA: A Tool-Integrated Reasoning Agent for Mathematical Problem Solving
**Authors:** Zhibin Gou, Zhihong Shao, Yeyun Gong, Yelong Shen, Yujiu Yang, Minlie Huang, Nan Duan, Weizhu Chen
**Venue:** ICLR 2024
**arXiv:** https://arxiv.org/abs/2309.17452
**GitHub:** https://github.com/microsoft/ToRA

## One-paragraph summary

ToRA introduces the classic TIR format: natural-language reasoning interleaved with programmatic tool calls (symbolic solvers, compute libraries). Training is **SFT on curated interactive tool-use trajectories**, plus "output space shaping" to further refine reasoning behavior. ToRA-7B reaches 44.6% on MATH (vs. WizardMath-70B's lower score). ToRA-Code-34B is the first open-source model >50% on MATH.

## Why it's a reference for us

- Foundational TIR-for-math paper. Everyone cites it.
- SFT-only — no RL. Our paper replaces SFT with GRPO, which is a recent-literature trend (SimpleTIR, GTPO).
- Multi-turn trajectories — a scope difference from our single-block setup.
- Math only; no physics.

## Position in our related-work

Cite in §2 paragraph on tool-integrated reasoning: "ToRA [cite] pioneered interleaved reasoning-and-tool-call format trained via SFT on distilled trajectories."

## Citation BibTeX stub

```bibtex
@inproceedings{gou2024tora,
  title={ToRA: A Tool-Integrated Reasoning Agent for Mathematical Problem Solving},
  author={Gou, Zhibin and Shao, Zhihong and Gong, Yeyun and Shen, Yelong and Yang, Yujiu and Huang, Minlie and Duan, Nan and Chen, Weizhu},
  booktitle={International Conference on Learning Representations (ICLR)},
  year={2024}
}
```
