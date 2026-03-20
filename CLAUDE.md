# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Active Plan

**Read first:** `.claude/plans/clever-launching-moonbeam.md` — the 2-month implementation plan. Currently executing Phase 1 (data acquisition + project scaffolding).

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
SIF=/gpfs/radev/scratch/krishnaswamy_smita/xs272/phys-reasoner/verl_vllm011.latest.sif
OVERLAY=/gpfs/radev/scratch/krishnaswamy_smita/xs272/phys-reasoner/phys-reasoner-overlay.img
apptainer exec --overlay "$OVERLAY" --bind /etc/pki:/etc/pki "$SIF" <command>
```

Run tests: `apptainer exec --overlay "$OVERLAY" --bind /etc/pki:/etc/pki "$SIF" python -m pytest tests/ -v`

## Key Dependencies (Planned)

- `pint` — unit handling for physics answers
- `sympy` — symbolic math equivalence checking
- `verl` — GRPO/RLVR training framework
- Qwen3.5 model family via HuggingFace
