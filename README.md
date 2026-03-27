# Adaptive Compute Routing for Physics Reasoning

Research project training a small open model (Qwen3.5-4B) to route each physics problem to the right compute mode — direct answer, structured internal verification, deeper reasoning, or restricted tool-based verification — using SFT + RL with a correctness-minus-cost reward.

Core claim: a learned routing policy achieves a better accuracy-cost frontier than fixed strategies or heuristic routers, with interpretable structure across problem types.

Target venue: NeurIPS 2026.

---

## Environment

All commands run inside an Apptainer container. The base SIF is read-only; a writable overlay holds all installed packages.

```bash
ROOT=$(pwd)   # repo root — derived automatically in sbatch scripts
SIF=$ROOT/verl_vllm017.latest.sif
OVERLAY=$ROOT/phys-reasoner-overlay-017.img

CMD="PYTHONNOUSERSITE=1 apptainer exec --overlay $OVERLAY --bind /etc/pki:/etc/pki $SIF"
CMD_GPU="PYTHONNOUSERSITE=1 apptainer exec --nv --overlay $OVERLAY --bind /etc/pki:/etc/pki $SIF"

# Run tests
$CMD python -m pytest tests/ -v

# GPU inference/training
$CMD_GPU python scripts/run_zero_shot.py ...
```

**`PYTHONNOUSERSITE=1` is required** — prevents `~/.local` packages from shadowing the overlay.

HuggingFace models are loaded from the standard cache (`~/.cache/huggingface`). Models download automatically on first use if internet is available.

### Machine-local configuration

Copy `.env.example` to `.env` (gitignored) to override machine-specific settings:

```bash
cp .env.example .env
# Edit .env as needed (HF_HOME, Slurm partition/QOS, etc.)
```

Sbatch scripts automatically source `.env` from the project root if it exists. For interactive use, source it in your shell first:

```bash
. .env   # or: source .env
```

### Installing packages into the overlay

After cloning or when `pyproject.toml` changes:

```bash
PYTHONNOUSERSITE=1 apptainer exec --overlay "$OVERLAY" --bind /etc/pki:/etc/pki "$SIF" \
    pip install -e "$ROOT[dev]"
```

---

## Data Pipeline

Run once after cloning (or after any loader/dedup logic changes):

```bash
. .env                          # load machine-local overrides if any
bash scripts/prepare_data.sh    # downloads everything, validates, builds parquet
```

This single script runs four steps:
1. **Download** — HuggingFace datasets + Yale NLP Physics JSONL files + ABench CSVs
2. **Validate** — checks all loaders produce sane output (`validate_loaders.py`)
3. **Build raw parquet** — loads all sources into `data/processed/candidates_raw.parquet`
4. **Dedup** — exact + fuzzy (MinHash LSH) + eval-contamination check → `data/processed/candidates_deduped.parquet` (~6,866 rows)

### Dataset sources

| Source | Type | Location after download | Role |
|---|---|---|---|
| desimfj/PHYSICS | HuggingFace | `data/hf_cache` | Training (Tier 1 eval held-out) |
| UGPhysics/ugphysics | HuggingFace | `data/hf_cache` | Training |
| lscpku/OlympiadBench-official | HuggingFace | `data/hf_cache` | Training |
| Eureka-Lab/PHYBench | HuggingFace | `data/hf_cache` | Training |
| Sihangli/scibench-rl | HuggingFace | `data/hf_cache` | Training |
| CritPt-Benchmark/CritPt | HuggingFace | `data/hf_cache` | Eval Tier 3 |
| yale-nlp/Physics (GitHub) | 6 JSONL files | `data/raw/yale-physics/` | Eval Tier 3 |
| inclusionAI/ABench (GitHub) | 2 CSV files | `data/raw/abench/` | Eval Tier 2 |

Yale and ABench are **eval-only** benchmarks. They are never in the training set — the dedup pipeline explicitly removes any training candidates that overlap with them.

### Zero-shot baseline

```bash
sbatch scripts/zero_shot_preview.sbatch   # quick smoke test (n_per_tier=5)
sbatch --export=ALL,CHUNK_ID=0,N_CHUNKS=4 scripts/zero_shot_chunk.sbatch  # full run
```

---

## Project Structure

```
phys-reasoner/
├── src/phys_reasoner/
│   ├── data/
│   │   ├── loaders.py          # One loader per source dataset
│   │   ├── normalize.py        # Raw answer_type → canonical label
│   │   └── schema.py           # PhysicsProblem dataclass
│   ├── verifier/
│   │   ├── router.py           # Main entry: verify_answer()
│   │   ├── extract.py          # \boxed{} extraction, split_by_comma, expand_pm
│   │   ├── math_verify_wrapper.py  # math-verify rule tier
│   │   ├── unit_check.py       # pint unit-aware comparison
│   │   └── xverify_judge.py    # xVerify-3B-Ib LLM fallback
│   └── training/
│       └── reward.py           # VeRL-compatible compute_score()
├── scripts/
│   ├── run_dedup.py            # Build candidates_deduped.parquet
│   ├── validate_parquet.py     # Post-dedup correctness check
│   ├── run_zero_shot.py        # Zero-shot baseline (vLLM)
│   ├── benchmark_xverify.py    # xVerify model size selection
│   └── spot_check_eval.py      # Verifier spot-check on eval sets
├── tests/
│   └── test_verifier.py        # Verifier test suite (D1–D8)
├── data/
│   ├── raw/                    # Source data (git submodules + HF cache)
│   └── processed/
│       ├── candidates_raw.parquet      # Pre-dedup (from loaders)
│       └── candidates_deduped.parquet  # Final training corpus (~6,866 rows)
└── docs/
```

---

## Datasets

### Training corpus (~6,866 problems after dedup)

| Dataset | Count | Answer types |
|---|---|---|
| UGPhysics | 5,451 | numerical, expression, equation, interval |
| PHYSICS (Yale) | 805 | numerical, expression |
| SciBench_RL | 280 | numerical |
| OlympiadBench | 230 | mixed multi-part |
| PHYBench | 100 | numerical |

### Evaluation tiers

| Dataset | Tier | Purpose |
|---|---|---|
| PHYSICS held-out | Tier 1 | In-domain textbook (same distribution as training) |
| ABench-Physics Phy_B (dynamic) | Tier 2 | Robustness: same problem, varied numerical parameters |
| CritPt (~70 problems) | Tier 3 | PhD-level frontier problems |

---

## Verifier

Layered pipeline: unit check (pint) → rule verify (math-verify) → LLM fallback (xVerify-7B-I).

- Rule tier accuracy on zero-shot corpus: **19.0%** (1,305/6,866)
- +xVerify-7B-I accuracy: **39.7%** overall, **48.9%** on non-truncated samples
- xVerify model: `IAAR-Shanghai/xVerify-7B-I` (+6.5pp over 3B-Ib, especially on expression/equation types)
- Authoritative baseline file: `data/results/rescore_7b_v3.parquet`
- Full experiment log: `docs/verifier-zero-shot-experiments.md`

```python
from phys_reasoner.verifier.router import verify_answer

score = verify_answer(
    pred_text="\\boxed{9.8}",
    gold_answer="9.8",
    answer_type="numerical",
    gold_unit="m/s^2",
)
# Returns: 1.0 (correct), 0.0 (wrong), -1.0 (unverifiable)
```

---

## Models

- **Primary**: `Qwen/Qwen3.5-4B` (base vs instruct TBD; see `docs/standalone_proposal_v2_5_2.md` §5)
- **Debug**: `Qwen/Qwen3.5-0.6B`

## Training Pipeline (planned)

1. Fixed-action baseline profiling (Answer / Check / Think-Deep / Tool-Check)
2. Heuristic + classifier router baseline
3. SFT warm-up for action-conditioned formatting
4. GRPO with correctness-minus-cost reward (via VeRL)
5. Routing analysis vs falsifiable prediction
