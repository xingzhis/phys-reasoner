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

### Container images

Two files are relevant:

| File | Purpose |
|---|---|
| `verl_vllm017.latest.sif` | Base Apptainer SIF (read-only; never modify) |
| `phys-reasoner-overlay-017.img` | **Primary overlay** — use for all sbatch jobs and interactive work |
| `phys-reasoner-overlay.img` | Legacy overlay — has FUSE2FS mount issues on some nodes; avoid for sbatch |

The overlay holds all pip-installed packages (phys-reasoner, scipy, huggingface-hub, etc.) layered on top of the base SIF.

### Running commands

```bash
ROOT=/gpfs/radev/scratch/krishnaswamy_smita/xs272/phys-reasoner
SIF=$ROOT/verl_vllm017.latest.sif
OVERLAY=$ROOT/phys-reasoner-overlay-017.img

# CPU command
PYTHONNOUSERSITE=1 apptainer exec --overlay "$OVERLAY" --bind /etc/pki:/etc/pki "$SIF" <command>

# GPU command (--nv passes through host NVIDIA driver)
PYTHONNOUSERSITE=1 apptainer exec --nv --overlay "$OVERLAY" --bind /etc/pki:/etc/pki "$SIF" <command>
```

**Always set `PYTHONNOUSERSITE=1`**: prevents `~/.local/lib/python3.12/site-packages` from leaking into the container and shadowing overlay packages (e.g. old `huggingface-hub==0.36.2` in `~/.local` would shadow `1.7.2` in the overlay, breaking transformers).

Run tests:
```bash
PYTHONNOUSERSITE=1 apptainer exec --overlay "$OVERLAY" --bind /etc/pki:/etc/pki "$SIF" python3 -m pytest tests/ -v
```

HF model cache: `HF_HOME=$ROOT/hf_cache` — xVerify models are pre-cached; always use `local_files_only=True` when loading them to avoid network calls to the HF API.

### Installing / updating packages into the overlay

When `pyproject.toml` dependencies change, reinstall into the overlay:

```bash
PYTHONNOUSERSITE=1 apptainer exec --overlay "$OVERLAY" --bind /etc/pki:/etc/pki "$SIF" \
    pip install -e "/gpfs/radev/scratch/krishnaswamy_smita/xs272/phys-reasoner[dev]"
```

To upgrade a specific package (e.g. after a base-SIF version conflict):
```bash
PYTHONNOUSERSITE=1 apptainer exec --overlay "$OVERLAY" --bind /etc/pki:/etc/pki "$SIF" \
    pip install "<package>==<version>"
```

**Do NOT upgrade `huggingface-hub`** — keep the base SIF's `0.36.2`. See "Known overlay pitfalls" below.

### sbatch jobs

See `scripts/rescore_xverify.sbatch` for the template. Key points:
- Uses `phys-reasoner-overlay-017.img` (`:ro` mount — safe for concurrent jobs)
- Exports `PYTHONNOUSERSITE=1` in the shell **before** calling apptainer (not just via `--env`)
- Uses `--no-home` to prevent `$HOME` from being mounted, further isolating from `~/.local`
- GPU partition: `gpu`, gres `a100:1`, qos `qos_nmi`

Submit pattern:
```bash
sbatch --export=ALL,XV_MODEL=IAAR-Shanghai/xVerify-3B-Ib,XV_OUTPUT=data/results/rescore_3b_v2.parquet scripts/rescore_xverify.sbatch
sbatch --export=ALL,XV_MODEL=IAAR-Shanghai/xVerify-7B-I,XV_OUTPUT=data/results/rescore_7b_v3.parquet scripts/rescore_xverify.sbatch
```

### Known overlay pitfalls

- **FUSE2FS "unchecked fs" warning**: if the overlay was not cleanly unmounted (e.g. node crash during a writable session), other nodes may fail to mount it. Symptom: packages installed in the overlay are invisible and the base SIF's old versions are used instead. Fix: run `e2fsck -fp <overlay.img>` while the overlay is not mounted.
- **`~/.local` shadowing**: always use `PYTHONNOUSERSITE=1`. The base SIF has `huggingface-hub==0.36.2`; `~/.local` may also have stale packages. The overlay has the correct versions.
- **HF API calls in sbatch**: compute nodes may not have outbound HTTPS. Use `local_files_only=True` in any `from_pretrained` call when the model is already in `hf_cache`.
- **Do NOT upgrade `huggingface-hub`**: keep the base SIF's `0.36.2`. With hub `>=1.3.0`, `list_repo_tree` exists and transformers `list_repo_templates` makes a live HTTP call for `additional_chat_templates` — this raises a fatal 404 for models that don't have that directory (e.g. `Qwen3.5-4B`). With `0.36.2`, `list_repo_tree` is absent so the call fails silently and inference works fine. The transformers `dependency_versions_check` warning about hub `<1.0` is harmless — ignore it.

## Key Dependencies (Planned)

- `pint` — unit handling for physics answers
- `sympy` — symbolic math equivalence checking
- `scipy>=1.11` — required by transformers (qwen2 object detection loss module loads it at model-load time)
- `huggingface-hub` — use base SIF's `0.36.2`; do NOT upgrade (see "Known overlay pitfalls")
- `verl` — GRPO/RLVR training framework
- Qwen3.5 model family via HuggingFace
