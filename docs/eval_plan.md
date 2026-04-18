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

**Training data after pool v2 rebuild (2026-04-18):** ~23.6k training problems from Dr.SCI (difficulty ≥ 0.5, 17,827) + UGPhysics (4,980) + PHYSICS (502) + SciBench-RL native train (280). Under the compute-constrained subsample regime, PHYSICS and SciBench-RL are now partially included in training — OlympiadBench / PHYBench / ABench / CritPt remain fully held out. See `docs/training-decisions.md` + the pool rebuild script `scripts/build_pool_v2.py`.

---

## Eval Suite

Two tiers: **in-distribution** (our own held-out splits of training-pool sources) and **external** (public benchmarks never in training). Drop external benchmarks with known contamination; use our own held-out test splits on sources that are partially in training.

### In-distribution (held-out splits from pool v2)

| Dataset | Size | Source | Scorer | Path |
|---|---|---|---|---|
| Dr.SCI test (≥0.5) | 503 | Our stratified split (seed=42) | Our verifier | `data/processed/pool_v2/tir/test.parquet` (filter `pool == 'drsci'`) |
| UGPhysics test | 217 | Our stratified split by (domain × primary_answer_type) | Our verifier | `data/processed/pool_v2/tir/test.parquet` (filter `data_source == 'UGPhysics'`) |
| PHYSICS test | 191 | Our stratified 80/20 self-split by primary_answer_type | Our verifier | `data/processed/pool_v2/tir/test.parquet` (filter `data_source == 'PHYSICS'`) |
| SciBench-RL test | 153 | **Official native test split** (textbooks class/diff/matter, disjoint from train) | Our verifier | `data/processed/pool_v2/tir/test.parquet` (filter `data_source == 'SciBench_RL'`) |

Combined pool v2 test = 1,064 rows. Mirrored to HF at `xingzhi0/phys-tir` (split=test) and `xingzhi0/phys-cot`.

Our verifier is `src/phys_reasoner/verifier/router.py` — rule-first, xVerify-7B fallback on expression types. xVerify is launched as a separate server in eval (same pattern as training: see `scripts/perlmutter/serve_xverify_perlmutter.sbatch`).

### External (held out entirely from training)

| Benchmark | Size | Type | HF / source | Official scorer | Our path | Notes |
|---|---|---|---|---|---|---|
| **OlympiadBench OE_TO physics** (EN, text-only competition) | 236 | Physics competition | [`Hothan/OlympiadBench`](https://huggingface.co/datasets/Hothan/OlympiadBench), config `OE_TO_physics_en_COMP` | [`eval/auto_scoring_judge.py`](https://github.com/OpenBMB/OlympiadBench) | `data/hf_cache/` via loader | Whole config held out (no native train/test divide). Handles numerical + expression. Multimodal subset (`OE_MM_*`) excluded — text-only model. |
| **PHYBench** | 1,000 | Physics competition | [`Eureka-Lab/PHYBench`](https://huggingface.co/datasets/Eureka-Lab/PHYBench) | [EED scorer](https://github.com/phybench-official/phybench) — Expression Edit Distance on SymPy trees, gives 0-100 continuous score | `data/hf_cache/` via loader | Whole dataset held out. Answers are raw LaTeX without `\boxed{}` — need wrapper. |
| **ABench Phy_A + Phy_B** | 400 + 400 | Physics grad/Olympiad; Phy_B is dynamic (parametric variants) | [`inclusionAI/ABench`](https://github.com/inclusionAI/ABench) — GitHub only, not on HF | Official `src/eval.py` — numerical, 1% tolerance. Phy_B needs ALL 4 variants of a base problem correct for credit | `data/raw/abench/Phy_A_fixed_400.csv`, `Phy_B_dynamic_100.csv` | Never in training. Phy_B is the dynamic-robustness check. |
| **CritPt** | 70 | Frontier research-level physics | [`CritPt-Benchmark/CritPt`](https://huggingface.co/datasets/CritPt-Benchmark/CritPt) | Server-graded via `artificialanalysis.ai/api/v2/critpt/evaluate` (Python functions executed server-side) | `data/hf_cache/` via loader | **Rate limit: 10 full-benchmark submissions / 24h.** Each submission = all 70 problems. Answer format: `def solution(): return <value>` Python functions. Not `\boxed{}`. Requires format-wrapper work. **Stretch goal** — do it if time permits after the main eval is solid. |

**PHYSICS and SciBench-RL move to in-distribution (2026-04-18):** Under compute-constrained subsampling, we accepted that these two benchmarks would partially enter training and report results only on our held-out splits (`pool_v2` test). Trade-off: lose cross-paper comparability on SciBench-RL and any would-be "textbook physics" eval on PHYSICS, gain ~780 curated training rows. OlympiadBench / PHYBench / ABench / CritPt remain clean external benchmarks.

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
