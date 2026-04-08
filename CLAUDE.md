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
   reasoning... [code] import sympy... print(...) [/code] [output] result interpretation... [answer] \boxed{X}
   ```
   - **`enable_thinking=True` for phase 1, `False` for phase 2** — In Qwen3.5's tool-call format the assistant turn has no plain-content slot; with thinking suppressed, reasoning leaks into Python code comments. With thinking ON, the model reasons in `<think>...</think>` then calls the tool with clean code. `</think>` closes before `<tool_call>` cleanly (confirmed 2026-04-02). Phase 2 injection uses `enable_thinking=False` (final answer needs no thinking).
   - **Token length caveat**: thinking traces can be long on hard problems. Planned mitigations: token-budget checkpoint, stop-thinking injection, RL length penalty. See `prompts.py` THINKING MODE section.
   - Model generates until `[/code]` stop token
   - Sandbox executes code (30s timeout, subprocess isolation)
   - `[output] {result}\n` injected as fixed continuation
   - Model resumes to EOS/max_tokens; `[answer]` is a format marker only, not a stop token

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
| `verl_vllm017.latest.sif` | Base Apptainer SIF — vllm 0.17.0, transformers 4.57.6 (read-only; never modify) |
| `phys-reasoner-overlay-017b.img` | **Active overlay** — use for all sbatch jobs and interactive work |

The overlay holds project packages plus `/opt/phys-extras/` which contains upgraded versions of transformers, hub, and flash-linear-attention layered above the SIF base.

### Running commands

```bash
ROOT=/gpfs/radev/scratch/krishnaswamy_smita/xs272/phys-reasoner
SIF=$ROOT/verl_vllm017.latest.sif
OVERLAY=$ROOT/phys-reasoner-overlay-017b.img

# CPU command — PYTHONPATH is required for /opt/phys-extras/ packages
PYTHONNOUSERSITE=1 apptainer exec \
  --overlay "$OVERLAY" --bind /etc/pki:/etc/pki \
  --env "PYTHONPATH=/opt/phys-extras/" "$SIF" <command>

# GPU command (--nv passes through host NVIDIA driver)
PYTHONNOUSERSITE=1 apptainer exec --nv \
  --overlay "$OVERLAY" --bind /etc/pki:/etc/pki \
  --env "PYTHONPATH=/opt/phys-extras/" "$SIF" <command>
```

**Always set `PYTHONNOUSERSITE=1`**: prevents `~/.local/lib/python3.12/site-packages` from leaking into the container and shadowing overlay packages.

**Always set `PYTHONPATH=/opt/phys-extras/`**: the SIF ships transformers 4.57.6 (no qwen3_5 support). The upgraded packages (transformers 5.3.0, hub 1.8.0, flash-linear-attention) live in `/opt/phys-extras/` inside the overlay and must be prepended to sys.path. See "Known overlay pitfalls" for why we use `--target` instead of pip upgrade.

Run tests:
```bash
PYTHONNOUSERSITE=1 apptainer exec \
  --overlay "$OVERLAY" --bind /etc/pki:/etc/pki \
  --env "PYTHONPATH=/opt/phys-extras/" "$SIF" \
  python3 -m pytest tests/ -v
```

HF model cache: `HF_HOME=$ROOT/hf_cache` — xVerify models are pre-cached; always use `local_files_only=True` when loading them to avoid network calls to the HF API.

### Installing / updating packages into the overlay

When `pyproject.toml` dependencies change, reinstall into the overlay:

```bash
PYTHONNOUSERSITE=1 apptainer exec \
  --overlay "$OVERLAY" --no-home --bind /etc/pki:/etc/pki \
  --env "PYTHONNOUSERSITE=1" --env "PYTHONPATH=/opt/phys-extras/" "$SIF" \
  pip install -e "/gpfs/radev/scratch/krishnaswamy_smita/xs272/phys-reasoner[dev]"
```

To add NEW packages needed above the SIF baseline (e.g. for a newer transformers feature):
```bash
# Install to /opt/phys-extras/ — does NOT require whiteout, works on all nodes
PYTHONNOUSERSITE=1 apptainer exec \
  --overlay "$OVERLAY" --no-home --bind /etc/pki:/etc/pki \
  --env "PYTHONNOUSERSITE=1" "$SIF" \
  pip install --no-deps --target /opt/phys-extras/ "<package>==<version>"
# Then run e2fsck -fp $OVERLAY to mark filesystem clean
```

**Do NOT use plain `pip install <package>` to replace SIF packages** — upgrading SIF-installed packages via pip in an overlay creates OverlayFS whiteout entries that are silently ignored on RHEL 8 compute nodes when the overlay is mounted `:ro`. The compute node falls back to the SIF's old version. Use `--target /opt/phys-extras/` instead.

### Async training extras (`$ROOT/.async-extras/`)

The `fully_async_policy` path (scripts/train_async.sh, scripts/train_smoke_async.sh) needs
numpy 2.x because `ray.util.collective` and the NCCL checkpoint engine both pull in cupy,
and the SIF-shipped `cupy-cuda12x 14.0.1` wheel is built against numpy 2.x ABI. The SIF
itself ships numpy 1.26, so `import cupy` crashes with
`numpy.core.multiarray failed to import` unless numpy 2.x is on sys.path first.

We do NOT put numpy 2.x into `/opt/phys-extras/` (the overlay), because the non-async path
is fine with numpy 1.26 and mixing numpy versions in the overlay is fragile. Instead it goes
into a host directory under the repo (`$ROOT/.async-extras/`) that is auto-bound by
apptainer, and is only added to `PYTHONPATH` by the async launchers:

```bash
# One-time install (host directory, NOT the overlay)
mkdir -p "$ROOT/.async-extras"
PYTHONNOUSERSITE=1 apptainer exec \
  --overlay "$OVERLAY:ro" --no-home --bind /etc/pki:/etc/pki \
  --env "PYTHONNOUSERSITE=1" "$SIF" \
  pip install --no-deps --target "$ROOT/.async-extras" "numpy>=2.1,<2.3"
```

Only `scripts/train_async.sh` prepends this path:
```
--env "PYTHONPATH=$ROOT/.async-extras:/opt/phys-extras/"
```
The regular `scripts/train.sh` keeps the old `PYTHONPATH=/opt/phys-extras/` and continues
to use the SIF's numpy 1.26. Rollback is a plain `rm -rf $ROOT/.async-extras` — the overlay
is never touched.

When moving to a new cluster, re-run the `pip install --target` line once inside the same
SIF+overlay. The host dir travels with the repo via any `$ROOT` that apptainer auto-binds.

### sbatch jobs

Key points for all sbatch scripts:
- Uses `phys-reasoner-overlay-017b.img` (`:ro` mount — safe for concurrent jobs)
- Exports `PYTHONNOUSERSITE=1` **and** `PYTHONPATH=/opt/phys-extras/` via `--env`
- Uses `--no-home` to prevent `$HOME` from being mounted, further isolating from `~/.local`
- Unsets `SIF` and `OVERLAY` before sourcing `env.sh` to prevent SLURM-inherited stale values
- GPU partition: `gpu`, gres `h200:1`, qos `qos_nmi`

Submit pattern:
```bash
sbatch --export=ALL,XV_MODEL=IAAR-Shanghai/xVerify-3B-Ib,XV_OUTPUT=data/results/rescore_3b_v2.parquet scripts/rescore_xverify.sbatch
sbatch --export=ALL,XV_MODEL=IAAR-Shanghai/xVerify-7B-I,XV_OUTPUT=data/results/rescore_7b_v3.parquet scripts/rescore_xverify.sbatch
```

### Known overlay pitfalls

- **OverlayFS whiteout ignored on RHEL 8 compute nodes** (critical): when `pip install` upgrades a package that already exists in the SIF, it uninstalls the old version by creating OverlayFS whiteout entries. On RHEL 8 compute nodes with the overlay mounted `:ro`, these whiteouts are silently ignored and the SIF's old package becomes visible again. Symptom: compute node shows old package version despite successful login-node install. Fix: use `pip install --no-deps --target /opt/phys-extras/` for packages that need to be newer than the SIF baseline, then set `PYTHONPATH=/opt/phys-extras/` so Python finds them first. This is why our overlay uses `/opt/phys-extras/` for hub/transformers/flash-linear-attention.
- **FUSE2FS "unchecked fs" warning**: if the overlay was not cleanly unmounted (e.g. session crash during a writable install), compute nodes may refuse to mount it. Symptom: overlay-installed packages are invisible, base SIF versions used instead. Fix: run `e2fsck -fp <overlay.img>` while the overlay is NOT mounted. `setup_overlay.sh` runs this automatically at the end.
- **SLURM environment inheritance**: SLURM passes all shell environment variables to batch jobs by default (`--export=ALL`). If `OVERLAY` or `SIF` are set in your shell from a previous session, they override `env.sh`. Fix: `unset SIF OVERLAY` before sourcing `env.sh` in batch scripts (already done in `smoke_tir_qwen35.sh`).
- **`~/.local` shadowing**: always use `PYTHONNOUSERSITE=1`. Without it, packages in `~/.local/lib/python3.12/site-packages/` (from outside the container) leak in and can shadow overlay packages.
- **HF API calls in sbatch**: compute nodes may not have outbound HTTPS. Use `local_files_only=True` in any `from_pretrained` call when the model is already in `hf_cache`.

## Key Dependencies

- `pint` — unit handling for physics answers
- `sympy` — symbolic math equivalence checking
- `scipy>=1.11` — required by transformers at model-load time
- `transformers==5.3.0` — installed in `/opt/phys-extras/`; required for Qwen3.5 (qwen3_5 model type added in 5.2.0)
- `huggingface-hub==1.8.0` — installed in `/opt/phys-extras/`; required by transformers 5.3.0
- `flash-linear-attention==0.4.2` + `fla-core==0.4.2` — installed in `/opt/phys-extras/`; required by Qwen3.5's GDN linear attention layers
- `verl` — GRPO/RLVR training framework (installed from `verl/` submodule with `--no-deps`)
- Qwen3.5 model family via HuggingFace

## Required VeRL Overrides for Qwen3.5-4B (GRPO)

These Hydra overrides must be set in every GRPO / smoke-test run:

```bash
# SDPA attention — VeRL's Ulysses flash-attention monkey-patch is incompatible with
# Qwen3.5 hybrid attention (GDN layers) in transformers 5.3.0; causes CUDA illegal
# memory access. SDPA bypasses the monkey-patch entirely.
'+actor_rollout_ref.model.override_config={attn_implementation:sdpa}'

# Offload AdamW optimizer to CPU — 4.54B params × float32 × 2 moments ≈ 36 GB.
# Combined with vLLM KV cache this exceeds H200 (80 GB) without offloading.
actor_rollout_ref.actor.fsdp_config.optimizer_offload=True

# Explicit FSDP wrap — Qwen3_5ForCausalLM._no_split_modules incorrectly lists
# Qwen3_5VisionBlock (a vision class absent from the text-only model).
# Without this override VeRL crashes with "Could not find transformer layer class".
'+actor_rollout_ref.actor.fsdp_config.wrap_policy.transformer_layer_cls_to_wrap=[Qwen3_5DecoderLayer]'
'+actor_rollout_ref.ref.fsdp_config.wrap_policy.transformer_layer_cls_to_wrap=[Qwen3_5DecoderLayer]'

# Multi-turn TIR with qwen3_coder tool-call format
actor_rollout_ref.rollout.multi_turn.format=qwen3_coder
actor_rollout_ref.rollout.multi_turn.enable=true
actor_rollout_ref.rollout.agent.default_agent_loop=tool_agent
```

Reference: `scripts/smoke_tir_qwen35.sh` (validated configuration, 2026-04-02).
