# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Active Plan

**Read first:** `.claude/plans/clever-launching-moonbeam.md` — the 2-month implementation plan. Currently at end of Phase 1 / start of Phase 2.

**Verifier MVP plan:** `.claude/plans/scalable-snacking-feigenbaum.md` — Phases A–E complete. Phase F (zero-shot baseline) in progress. See plan Status section for exact state and pending items before GRPO training.

## Project Overview

Research project: "Do Verifiable Rewards Teach Physics?" — building a physics answer verifier and training LLMs with RLVR (Reinforcement Learning with Verifiable Rewards) to study whether reasoning gains transfer beyond textbook problems to dynamic and research-level physics tasks. Target venue: NeurIPS 2026.

## Planned Architecture

Three main components:

1. **Physics Answer Verifier** (Python library): Layered verification pipeline for physics answers:
   - Numerical answers with units (via `pint`)
   - Symbolic expressions (via `SymPy`)
   - Multi-part answers
   - Order-of-magnitude estimates
   - LLM fallback for edge cases
   - Target: ≥95% precision vs human annotations

2. **RLVR Training Pipeline**: GRPO training (via VeRL) on physics corpus (~25-27k problems from PHYSICS + UGPhysics + OlympiadBench):
   - Base models: Qwen3.5-4B (primary), Qwen3.5-9B (scaling check)
   - Pipeline: Base → SFT warm-up → GRPO with physics-aware verifier reward
   - Baselines: Base zero-shot, SFT-only, SFT+RLVR, Best-of-N

3. **Three-Tier Evaluation**:
   - Tier 1: Textbook (in-domain held-out split from PHYSICS)
   - Tier 2: Dynamic robustness (ABench-Physics Phy_B)
   - Tier 3: Frontier/research-level (CritPt ~70 PhD-level problems)

## Environment

Project lives at `/gpfs/radev/scratch/krishnaswamy_smita/xs272/phys-reasoner`.

All Python commands run inside an Apptainer container:
```bash
SIF=/gpfs/radev/scratch/krishnaswamy_smita/xs272/phys-reasoner/verl_vllm017.latest.sif
OVERLAY=/gpfs/radev/scratch/krishnaswamy_smita/xs272/phys-reasoner/phys-reasoner-overlay-017.img
PYTHONNOUSERSITE=1 apptainer exec --overlay "$OVERLAY" --bind /etc/pki:/etc/pki "$SIF" <command>
```

**GPU commands require `--nv`** (passes through host NVIDIA driver):
```bash
PYTHONNOUSERSITE=1 apptainer exec --nv --overlay "$OVERLAY" --bind /etc/pki:/etc/pki "$SIF" python scripts/run_zero_shot.py
```

**Always set `PYTHONNOUSERSITE=1`**: prevents `~/.local/lib/python3.12/site-packages` from leaking into the container. A broken `boto3` (missing `botocore`) in `~/.local` causes `accelerate → transformers` import failure.

Run tests: `PYTHONNOUSERSITE=1 apptainer exec --overlay "$OVERLAY" --bind /etc/pki:/etc/pki "$SIF" python3 -m pytest tests/ -v`

HF model cache: `HF_HOME=/gpfs/radev/scratch/krishnaswamy_smita/xs272/phys-reasoner/hf_cache`

## Key Dependencies (Planned)

- `pint` — unit handling for physics answers
- `sympy` — symbolic math equivalence checking
- `verl` — GRPO/RLVR training framework
- Qwen3.5 model family via HuggingFace
