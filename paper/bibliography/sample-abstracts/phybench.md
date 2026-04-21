# PHYBench — primary external physics benchmark

**Full title:** PHYBench: Holistic Evaluation of Physical Perception and Reasoning in Large Language Models
**Venue:** arXiv:2504.16074 (Apr 2025); OpenReview record exists.
**arXiv:** https://arxiv.org/abs/2504.16074
**Site:** https://www.phybench.cn/

## One-paragraph summary

500 original physics problems (high-school through Olympiad), covering mechanics, electromagnetism, thermodynamics, optics, modern physics, and advanced topics. Two eval metrics: Expression Edit Distance (EED) Score (tree edit distance between generated and reference expressions) and exact-match accuracy. Best model (Gemini 2.5 Pro): 36.9% accuracy vs human experts' 61.9%. PHYBench cleanly separates reasoning vs non-reasoning models — DeepSeek-R1 vs V3 gap is larger here than on math benchmarks.

## How we use it

- PHYBench is in our external-benchmark list. We report EED as primary (continuous) and exact-match in appendix.
- Note: exact-match is punishing (our zero-shot Qwen3-4B-Thinking gets 1–2%); EED gives a more informative signal.

## Citation BibTeX stub

```bibtex
@article{qiu2025phybench,
  title={PHYBench: Holistic Evaluation of Physical Perception and Reasoning in Large Language Models},
  author={Qiu, Shi and others},
  journal={arXiv preprint arXiv:2504.16074},
  year={2025}
}
```

Resolve full author list from PDF before committing.
