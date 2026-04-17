# Evaluation Plan — PhysCode (TIR vs CoT GRPO)

**Status:** Planned. Eval harness not yet implemented as of 2026-04-16.
**Venue target:** ICML AI4Physics workshop.
**Purpose:** This doc is a handoff for the next session — it captures the benchmark list and rationale so an eval harness can be built without re-deriving the decisions.

---

## Context

We're evaluating two trained checkpoints against one base model:
- **PhysCode (TIR-GRPO)** — Qwen3.5-4B trained with tool-integrated RL (main method)
- **CoT-GRPO baseline** — same model, same data, same hparams, no tool (ablation)
- **Qwen3.5-4B** — zero-shot for context

Both trained runs are launched via `scripts/perlmutter/prod_tir_het.sbatch` and `prod_cot_het.sbatch`. See `scripts/perlmutter/README.md` for the training flow.

**Training data after the corpus rebuild (2026-04-16):** Dr. SCI (~102k) + UGPhysics (~5.4k). Everything else was dropped from training specifically so it could serve as contamination-free eval. See `docs/training-decisions.md` + `.claude/plans/declarative-churning-ladybug.md` for that history.

---

## Eval Suite

Two tiers: **in-distribution** (our own held-out splits) and **external** (public benchmarks). Drop external benchmarks with known contamination; for benchmarks that started in our corpus and got removed, use the full benchmark as eval because we now see none of it during training.

### In-distribution

| Dataset | Size | Source | Scorer | Path |
|---|---|---|---|---|
| Dr. SCI test | 2,000 | Our stratified split (seed=42) | Our verifier | `data/processed/drsci_test.parquet` |
| Corpus test (UGPhysics) | 200 | Our stratified split (answer_type × domain_coarse) | Our verifier | `data/processed/corpus_test.parquet` |

Both are mirrored to HF at `xingzhi0/phys-tir` (split=test) and `xingzhi0/phys-cot`.

Our verifier is `src/phys_reasoner/verifier/router.py` — rule-first, xVerify-7B fallback on expression types. xVerify is launched as a separate server in eval (same pattern as training: see `scripts/perlmutter/serve_xverify_perlmutter.sbatch`).

### External (planned — held out from training)

| Benchmark | Size | Type | HF / source | Official scorer | Our path | Notes |
|---|---|---|---|---|---|---|
| **PHYSICS** (desimfj) | 793 usable | Physics, text+MCQ | [`desimfj/PHYSICS`](https://huggingface.co/datasets/desimfj/PHYSICS) (2k total, HF split=test) | **None — use our verifier** | `data/hf_cache/` via loader | 793 = after loader filters (Chinese / multi-alt / Open-end dropped). Not a public leaderboard — our numbers aren't externally comparable, but fills the "textbook physics" eval tier. |
| **OlympiadBench OE_TO physics** (EN, text-only competition) | 236 | Physics competition | [`Hothan/OlympiadBench`](https://huggingface.co/datasets/Hothan/OlympiadBench), config `OE_TO_physics_en_COMP` | [`eval/auto_scoring_judge.py`](https://github.com/OpenBMB/OlympiadBench) | `data/hf_cache/` via loader | Handles numerical (uses `error` field for tolerance) + expression. Multimodal subset (`OE_MM_*`) excluded — text-only model. |
| **SciBench-RL test** | 153 | Textbook STEM | [`Sihangli/scibench-rl`](https://huggingface.co/datasets/Sihangli/scibench-rl) | Original SciBench has LLM-judge grader (needs OpenAI API) — our verifier is fine for numerical+unit | `data/hf_cache/` via loader | Only the 153-row official test split. 427-row train split was in our corpus → now dropped. |
| **PHYBench** | 1,000 | Physics competition | [`Eureka-Lab/PHYBench`](https://huggingface.co/datasets/Eureka-Lab/PHYBench) | [EED scorer](https://github.com/phybench-official/phybench) — Expression Edit Distance on SymPy trees, gives 0-100 continuous score | `data/hf_cache/` via loader | 100 rows were in our corpus (filtered simple subset) → now dropped. Full 1k for eval is clean. Answers are raw LaTeX without `\boxed{}` — need wrapper. |
| **ABench Phy_A + Phy_B** | 400 + 400 | Physics grad/Olympiad; Phy_B is dynamic (parametric variants) | [`inclusionAI/ABench`](https://github.com/inclusionAI/ABench) — GitHub only, not on HF | Official `src/eval.py` — numerical, 1% tolerance. Phy_B needs ALL 4 variants of a base problem correct for credit | `data/raw/abench/Phy_A_fixed_400.csv`, `Phy_B_dynamic_100.csv` | Never in training. Phy_B is the dynamic-robustness check. |
| **CritPt** | 70 | Frontier research-level physics | [`CritPt-Benchmark/CritPt`](https://huggingface.co/datasets/CritPt-Benchmark/CritPt) | Server-graded via `artificialanalysis.ai/api/v2/critpt/evaluate` (Python functions executed server-side) | `data/hf_cache/` via loader | **Rate limit: 10 full-benchmark submissions / 24h.** Each submission = all 70 problems. Answer format: `def solution(): return <value>` Python functions. Not `\boxed{}`. Requires format-wrapper work. **Stretch goal** — do it if time permits after the main eval is solid. |

### Explicitly NOT included (and why)

| Benchmark | Why skipped |
|---|---|
| MATH-500 | Math, not physics. Distraction for AI4Physics workshop. Every paper reports it, but it doesn't support our core physics claims. Revisit only if we need a math-domain comparison. |
| GPQA / GPQA Diamond | MCQ-only, mixed domain (physics + chem + bio). Limited signal. No clean physics-only split. |
| SuperGPQA | Too broad (285 disciplines). Signal-to-noise too low for a physics paper. |
| FrontierMath | Private dataset (only 12 public samples). Can't use. |
| Humanity's Last Exam (HLE) | Multimodal + mixed domain + partly private. Out of scope. |
| Yale NLP Physics | ~80% of answers are not verifiable (prose / complex symbolic derivations). Our verifier stack can't score it cleanly. |
| TheoremQA | Rare in RL papers — no strong comparability story. |
| JEEBench | Rare in RL papers. Similar reason. |
| MinerVa Math, AIME, AMC | Math competition benchmarks — same reason as MATH-500. |

---

## Metrics

- **pass@1 everywhere** — this is what every RL reasoning paper headlines
- **avg@8** only for small benchmarks (<200 problems), to tame variance:
  - CritPt (70) — though rate limit makes avg@k expensive (each `k` = 1 full submission)
  - Maybe corpus_test (200) as a hedge

Per-type accuracy breakdown (numerical / expression / MCQ / equation) is the core of **RQ2** (reward noise analysis). This is done post-hoc on the rollouts we already produce — no extra inference needed.

Training-time monitoring: VeRL logs `critic/score/mean` on the val set every `test_freq` steps (already configured to 100 in `prod_tir_het.sbatch`). Since the reward is binary, that metric **is** val-set pass@1.

---

## Eval Harness — Not Yet Built

The eval harness is the next major piece of work. Suggested structure (from earlier discussion):

```
eval/                            # NEW folder, separate from messy scripts/
  run_eval.py                    # main harness: (checkpoint, benchmark) → results
  benchmarks/                    # one loader per benchmark
    __init__.py
    drsci.py                     # our own test splits
    corpus.py
    physics_desimfj.py
    olympiad_bench.py
    scibench_rl.py
    phybench.py
    abench.py
    critpt.py                    # separate because of server API + Python-function format
  scoring/                       # one scorer per benchmark (wraps their official)
    __init__.py
    our_verifier.py              # re-exports src/phys_reasoner/verifier for in-dist
    olympiad_official.py         # wraps OpenBMB/OlympiadBench eval/auto_scoring_judge.py
    phybench_eed.py              # wraps phybench-official/phybench EED scorer
    abench_official.py           # wraps inclusionAI/ABench src/eval.py
    critpt_server.py             # posts to artificialanalysis.ai API
  inference/
    tir_rollout.py               # TIR generation — reuse logic from scripts/dump_rollouts.py
    cot_rollout.py               # single-turn CoT generation
  configs/                       # per-benchmark eval config (batch size, max_tokens, etc.)
  results/                       # output parquets go here (gitignored)
```

**Why a new folder:** `scripts/` is currently messy (preprocessing + training + testing + outdated files mixed). Starting eval in a separate `eval/` folder keeps it clean and makes a future archive-and-cleanup easier.

**Logic reuse:**
- TIR inference should mirror `scripts/dump_rollouts.py` — the faithful VeRL-matching 2-phase rollout with think-interrupt
- CoT inference is simpler — single-turn, just sample until EOS or max_tokens
- Our scorer already has the rule + xVerify router at `src/phys_reasoner/verifier/router.py`

**First eval to implement (easiest first):**
1. Our test splits (drsci_test, corpus_test) — zero new code, just inference + our verifier
2. OlympiadBench OE_TO — small (236 rows), official scorer is a local script, no server needed
3. ABench Phy_A — 400 static numerical problems with 1% tolerance
4. Then PHYBench (needs EED wrapper), SciBench-RL test, PHYSICS
5. CritPt last — needs Python-function formatter and server API client

---

## Paper Framing Notes

**Main claim:** TIR-GRPO beats CoT-GRPO on physics problems because execution-based reward eliminates the LaTeX verifier FN noise (~68% on expression types) that degrades CoT-GRPO.

**Primary comparison:** TIR vs CoT per-type accuracy across all 7 eval datasets. This is the core figure.

**Fallback (if TIR doesn't beat CoT in aggregate):** reframe as a reward-noise analysis — show that FN-rate per answer-type predicts differential learning failure. This stands as a clean empirical contribution regardless of aggregate accuracy.

**Ablations (if time — dropped from scope 2026-04-14):**
- Cost-penalty λ > 0: NOT in current training, skip
- Curriculum vs full-data: NOT in plan (decided against), skip
- 0.8B vs 4B scaling: probably too expensive for the deadline, skip

---

## Stretch / Future Work

- **CritPt** — 10 submissions/24h means only ~3-4 evals per model per week. Get base model baseline submitted ASAP to save budget.
- **Official OlympiadBench LLM-judge path** — their grader supports LLM-judge for proof problems. We're skipping those (text-only answer types only) but could add later.
- **Per-source analysis on Dr. SCI test** — break down by `extra_info.from` to see which source benchmarks within Dr. SCI are hardest. Not essential but interesting.

---

## Handoff Checklist for Next Session

- [ ] Read `scripts/perlmutter/README.md` to understand the training flow (TIR + CoT sbatch setup)
- [ ] Read `.claude/plans/declarative-churning-ladybug.md` for the corpus-rebuild rationale
- [ ] Read `docs/training-decisions.md` §"Dataset Sizes" and §"Verifier Status" for context
- [ ] Check if any training checkpoints exist yet (`ls outputs/physcode_tir/`) — if not, the eval harness can be built + unit-tested against the base model zero-shot first
- [ ] If building the eval harness: start with the `eval/` folder structure above, and pick the "first eval" list order (drsci_test → olympiad → abench → phybench → scibench → physics → critpt)
- [ ] For CritPt specifically: submit the base model first (costs 1 of the 10/24h submissions) to lock in the baseline number before training runs complete
