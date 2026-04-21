# UGPhysics — primary external benchmark (in-dist slice source)

**Full title:** UGPhysics: A Comprehensive Benchmark for Undergraduate Physics Reasoning with Large Language Models
**Venue:** ICML 2025 poster
**arXiv:** https://arxiv.org/abs/2502.00334 (Feb 2025)
**GitHub:** https://github.com/YangLabHKUST/UGPhysics

## One-paragraph summary

5,520 undergraduate-level physics problems, English + Chinese, 13 subjects, 7 answer types, 4 reasoning skills. Rigorously screened for data leakage. Top score: OpenAI-o1-mini at 49.8%. They introduce MARJ (Model-Assistant Rule-based Judgment) — a hybrid rule + LLM scoring pipeline tailored to physics answer-type heterogeneity. The pipeline is directly relevant to our verifier discussion.

## How we use it

- UGPhysics problems are in our training pool (pool_v2_ugphysics slice).
- We also report held-out evaluation on UGPhysics as one of the external-physics benchmarks in Table 1.
- Cite MARJ in §2 verification paragraph as prior work on hybrid physics scoring.

## Notes

- The "seven answer types" they describe maps closely to our {numerical, expression, equation, MCQ, interval, multi-part, ...}. Our stratification dimension aligns with theirs — useful for the setup section.
- Their observation that math ability ≠ physics ability is a useful citation to motivate physics-specific training.

## Citation BibTeX stub

```bibtex
@inproceedings{ugphysics2025,
  title={UGPhysics: A Comprehensive Benchmark for Undergraduate Physics Reasoning with Large Language Models},
  author={Yang, Ming and others},
  booktitle={International Conference on Machine Learning (ICML)},
  year={2025}
}
```
