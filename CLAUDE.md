# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Active Plan

**Read first:** `.claude/plans/physcode.md` — the 6-week implementation plan. Started 2026-03-31, Week 1 in progress.

**Session notes (start here for new sessions):** `.claude/session-notes.md` — current state, completed work, and next steps in priority order.

**Key decisions:** `docs/training-decisions.md` — Goldilocks B+C strategy, dataset sizes, baseline numbers, TIR-specific decisions.

## Project Overview

Research project: "PhysCode: Tool-Integrated Reasoning for Physics Problem Solving via RLVR" — training Qwen3.5-4B with RLVR to solve physics problems using single-block TIR: model reasons, executes one Python/SymPy code block, receives output, reasons to final answer. Core claim: execution-based reward eliminates the symbolic verification noise (~68% LaTeX FN rate on expression types) that degrades CoT-GRPO, connecting empirically to the RLVεR theoretical framework. Target venue: NeurIPS 2026.

Proposal: `docs/physcode_proposal_v5.md`

## Architecture

Three main components:

1. **TIR Trajectory Format** (single code block per rollout):
   ```
   [think] reasoning... [code] import sympy... print(...) [/code] [output] result [think] interpretation... [answer] \boxed{X}
   ```
   - Model generates until `[/code]` stop token
   - Sandbox executes code (30s timeout, subprocess isolation)
   - `[output] {result}\n` injected as fixed continuation
   - Model resumes to `[answer]` stop token

2. **GRPO Training Pipeline**:
   - Model: Qwen3.5-4B instruct (thinking OFF); debug: Qwen3.5-0.8B
   - Reward: binary R_correct on final \boxed{} (λ=0 initially; token cost penalty in late ablation)
   - Primary data: Dr. SCI clean (~65k numerical + expression + MCQ) + 6.8k curated corpus
   - Curriculum: numerical first → expression + MCQ
   - SFT: conditional on Stage 0 probe (skip if zero-shot TIR hit rate ≥15%)
   - Secondary benchmark: MATH-500 hard subset

3. **Verification Stack** (unchanged from existing pipeline):
   - `src/phys_reasoner/verifier/` — rule verifier + xVerify-7B fallback
   - Execution reward: final \boxed{} verified against gold answer
   - LaTeX FN rate measurement: per-type, used for RQ2 correlation analysis

## Environment

Project lives at `/gpfs/radev/scratch/krishnaswamy_smita/xs272/phys-reasoner`.

### Container images

Two files are relevant:

| File | Purpose |
|---|---|
| `verl_vllm017.latest.sif` | Base Apptainer SIF (read-only; never modify) |
| `phys-reasoner-overlay-017.img` | **Primary overlay** — use for all sbatch jobs and interactive work after post-install upgrades |

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

**Always set `PYTHONNOUSERSITE=1`**: prevents `~/.local/lib/python3.12/site-packages` from leaking into the container and shadowing overlay packages (e.g. old `huggingface-hub==0.36.2` in `~/.local` would shadow the upgraded version in the overlay, breaking transformers).

Run tests (interactive — use `-017` overlay):
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

**Upgrade `huggingface-hub` and `transformers`** after initial overlay install (see README § "Installing packages into the overlay").

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
- **Upgrade `huggingface-hub` and `transformers`**: the base SIF's `0.36.2` is outdated. After `pip install -e .[dev]`, run `pip install --upgrade huggingface-hub transformers --no-deps` to get verl-compatible versions (1.8.0+ and 5.4.0+). This resolves the 404 issue with newer transformers and enables proper model downloading.

## Key Dependencies (Planned)

- `pint` — unit handling for physics answers
- `sympy` — symbolic math equivalence checking
- `scipy>=1.11` — required by transformers (qwen2 object detection loss module loads it at model-load time)
- `huggingface-hub` — upgrade to 1.8.0+ after initial install (see README § "Installing packages into the overlay")
- `verl` — GRPO/RLVR training framework
- Qwen3.5 model family via HuggingFace
