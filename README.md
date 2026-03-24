# Do Verifiable Rewards Teach Physics?

Research project studying whether RLVR (Reinforcement Learning with Verifiable Rewards) improves physics reasoning and whether gains transfer beyond textbook problems to dynamic and research-level tasks.

Target venue: NeurIPS 2026.

---

## Environment

All commands run inside an Apptainer container. The base SIF is read-only; a writable overlay holds all installed packages.

```bash
ROOT=/gpfs/radev/scratch/krishnaswamy_smita/xs272/phys-reasoner
SIF=$ROOT/verl_vllm017.latest.sif
OVERLAY=$ROOT/phys-reasoner-overlay-017.img   # primary overlay (use this one)

CMD="PYTHONNOUSERSITE=1 apptainer exec --overlay $OVERLAY --bind /etc/pki:/etc/pki $SIF"
CMD_GPU="PYTHONNOUSERSITE=1 apptainer exec --nv --overlay $OVERLAY --bind /etc/pki:/etc/pki $SIF"

# Run tests
$CMD python -m pytest tests/ -v -m "not slow"

# GPU inference/training
$CMD_GPU python scripts/run_zero_shot.py ...
```

**`PYTHONNOUSERSITE=1` is required** — prevents `~/.local` packages from shadowing the overlay (the base SIF has `huggingface-hub==0.36.2`; the overlay has the correct `>=1.3.0`).

HuggingFace model cache: `HF_HOME=$ROOT/hf_cache` (xVerify models pre-cached; use `local_files_only=True` in `from_pretrained` calls).

### Installing packages into the overlay

After cloning or when `pyproject.toml` changes:

```bash
PYTHONNOUSERSITE=1 apptainer exec --overlay "$OVERLAY" --bind /etc/pki:/etc/pki "$SIF" \
    pip install -e "$ROOT[dev]"
```

---

## Data Pipeline

The training corpus is assembled from multiple sources, deduplicated, and saved as a single parquet file. **Re-run `run_dedup.py` whenever loaders or normalization logic changes.**

### Step 1 — Raw data

Data is stored in `data/raw/`:

| Source | Location | Notes |
|---|---|---|
| UGPhysics | HuggingFace (`luffycodes/ugphysics`) | loaded via `loaders.py` |
| PHYSICS (Yale) | `data/raw/yale-physics/` | git submodule |
| OlympiadBench | `data/raw/grader-repos/OlympiadBench/` | git submodule |
| PHYBench | HuggingFace | loaded via `loaders.py` |
| SciBench_RL | HuggingFace | loaded via `loaders.py` |
| ABench-Physics | `data/raw/abench/` | eval only (Phy_A, Phy_B CSV) |

### Step 2 — Build training parquet

```bash
$CMD python -u scripts/run_dedup.py
# Output: data/processed/candidates_deduped.parquet (~6,866 rows)
```

This runs three passes:
1. **Exact dedup** — SHA-256 on normalized question text
2. **Fuzzy dedup** — MinHash LSH at Jaccard ≥ 0.8
3. **Contamination check** — removes training rows that overlap with any eval set

Priority when deduplicating: `OlympiadBench > PHYSICS > UGPhysics > PHYBench > SciBench_RL`

Answer types are normalized to canonical labels (`numerical`, `expression`, `equation`, `interval`, `mcq`, `true_false`, `open_end`).

### Step 3 — Validate parquet

```bash
$CMD python -u scripts/validate_parquet.py
# Checks: canonical types, row count, 200-row smoke verify
```

### Step 4 — Zero-shot baseline

```bash
$CMD_GPU python -u scripts/run_zero_shot.py \
    --n_per_tier 200 \
    --output data/results/zero_shot_scores.parquet
```

Uses vLLM, thinking enabled, `max_new_tokens=4096`.

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

- **Primary**: `Qwen/Qwen3.5-4B` (Base → SFT warm-up → GRPO)
- **Scaling check**: `Qwen/Qwen3.5-9B`

## Training Pipeline (planned)

1. Base zero-shot baseline (`scripts/run_zero_shot.py`)
2. SFT warm-up on training corpus
3. GRPO with physics verifier reward (via VeRL)
4. Evaluation on all three tiers
