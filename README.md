# Do Verifiable Rewards Teach Physics?

Research project studying whether RLVR (Reinforcement Learning with Verifiable Rewards) improves physics reasoning and whether gains transfer beyond textbook problems to dynamic and research-level tasks.

Target venue: NeurIPS 2026.

---

## Environment Setup (HPC with Singularity)

This project runs on a SLURM HPC cluster using a Singularity container with an overlay for additional packages.

### Prerequisites

- Singularity installed on HPC
- Base SIF with PyTorch + VeRL (see `scripts/env_setup.sh`)

### Install project in overlay

```bash
# Activate your overlay and SIF
singularity exec --overlay /path/to/overlay.img /path/to/base.sif bash

# Inside container: install project in editable mode
pip install -e ".[dev]"
```

### Running scripts

All scripts should be run inside the container:

```bash
singularity exec --overlay /path/to/overlay.img /path/to/base.sif \
    python scripts/prepare_data.py
```

Or via SLURM batch jobs (see `scripts/slurm/`).

---

## Project Structure

```
phys-reasoner/
├── pyproject.toml
├── src/phys_reasoner/
│   ├── verifier/        # Physics answer verifier (numeric, unit, symbolic, ...)
│   ├── data/            # Dataset loaders + preprocessing
│   ├── training/        # VeRL reward wrapper
│   └── eval/            # Evaluation scripts + analysis
├── tests/
├── scripts/
│   ├── prepare_data.py  # Download + normalize all datasets
│   ├── train_grpo.sh    # GRPO training entry point
│   └── run_eval.py      # Evaluation entry point
└── docs/
    └── proposal.md
```

## Datasets

| Dataset | Split | Purpose |
|---|---|---|
| PHYSICS | train | Textbook problems (Tier 1 training + held-out eval) |
| UGPhysics | train | Undergraduate physics |
| OlympiadBench (physics) | train | Competition problems |
| ABench-Physics Phy_B | eval | Dynamic robustness (Tier 2) |
| CritPt | eval | PhD-level frontier problems (Tier 3) |

## Models

- **Primary**: Qwen3.5-4B (Base → SFT → GRPO)
- **Scaling check**: Qwen3.5-9B

## Baselines

1. Base zero-shot
2. SFT-only
3. SFT + RLVR (GRPO with physics verifier reward)
4. Best-of-N (SFT model, scored by verifier)
