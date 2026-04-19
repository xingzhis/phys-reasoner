# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Active Plan

**Read first:** `.claude/plans/physcode.md` — the 6-week implementation plan. Started 2026-03-31, Week 2 in progress.

**Data pipeline:** `.claude/plans/data-pipeline.md` — step-by-step plan for Goldilocks data filtering (Steps 0–6 + Strategy D rescoring). Each step is a separate session.

**Session notes (start here for new sessions):** `.claude/session-notes.md` — current state, completed work, and next steps in priority order.

**Key decisions:** `docs/training-decisions.md` — Goldilocks B+C strategy, dataset sizes, baseline numbers, TIR-specific decisions.

## Project Overview

Research project: "PhysCode: Tool-Integrated Reasoning for Physics Problem Solving via RLVR" — training Qwen3.5-4B with RLVR to solve physics problems using single-block TIR: model reasons, executes one Python/SymPy code block, receives output, reasons to final answer. Core claim: execution-based reward eliminates the symbolic verification noise (~68% LaTeX FN rate on expression types) that degrades CoT-GRPO, connecting empirically to the RLVεR theoretical framework. Target venue: NeurIPS 2026.

Proposal: `docs/physcode_proposal_v5.md`

## Architecture

Three main components:

1. **TIR Trajectory Format** (Qwen3.5 native `qwen3_coder` tool-call format):
   ```
   <think>reasoning...</think>
   <tool_call><function=python><parameter=code>
   import sympy as sp
   ...
   print(result)
   </parameter></function></tool_call>
   <tool_response>
   stdout
   </tool_response>
   interpretation... \boxed{answer}
   ```
   - **`enable_thinking=True` for phase 1, `False` for phase 2** (confirmed 2026-04-02):
     Qwen3.5's tool-call format has no plain-content slot in the assistant turn. With thinking OFF, reasoning leaks into Python code comments. With thinking ON, the model reasons in `<think>...</think>` then calls the tool with clean code. `</think>` closes before `<tool_call>` cleanly. Phase 2 (after tool response injection) uses `enable_thinking=False`.
   - **Format constants** (`src/phys_reasoner/tir/prompts.py`):
     `TIR_SYSTEM_PROMPT`, `COT_SYSTEM_PROMPT`, `PYTHON_TOOL_SCHEMA` (OpenAI function-schema passed via `tools=[...]` kwarg of `apply_chat_template`), `TOOL_CALL_STOP = "</tool_call>"`, `THINK_INTERRUPT_PHRASE`, `extract_tool_call_code()`.
   - **Rollout flow** (reference impl: `eval/inference/rollout.py`; matches `verl/experimental/agent_loop/tool_agent_loop.py`):
     1. Phase 1: `enable_thinking=True`, `stop=["</tool_call>"]`, `max_tokens=thinking_budget`
     2. No `<tool_call>` found → terminate; phase 1 text is the final answer
     3. Else: sandbox executes code (`src/phys_reasoner/tir/sandbox.py` — 30s timeout, subprocess isolation)
     4. Tool response injected as `{"role": "tool", "content": stdout}` via `apply_chat_template(..., enable_thinking=False)`
     5. Phase 2: resume to EOS or `answer_budget`; final `\boxed{X}` extracted by scorer
   - **Think-interrupt** (budget escape for runaway thinking):
     Fires when `len(phase1_tokens) >= thinking_budget AND </think> NOT in tokens`.
     Phrase: `"\nOkay, I've thought enough. Time to write my response.\n</think>\n"`.
     Followed by phase 1b: `max_tokens=tool_call_budget`, `stop=["</tool_call>"]`.
     Mask=0 for interrupt tokens in training loss.
     Budget identity: `response_length = thinking + interrupt + tool_call + tool_response + answer`.

2. **GRPO Training Pipeline**:
   - Model: Qwen3-4B-Thinking-2507 (current, switched 2026-04-18); debug: Qwen3-0.6B. Historical: Qwen3.5-4B (see "If reverting" footnote below).
   - Reward: binary R_correct on final `\boxed{}` via `src/phys_reasoner/verifier/router.py` (λ=0 initially; token cost penalty in late ablation).
   - Primary data: Dr. SCI clean (~65k numerical + expression + MCQ) + 6.8k curated corpus. External benchmarks (PHYSICS, OlympiadBench OE_TO, SciBench-RL, PHYBench) held out from training.
   - Curriculum: numerical first → expression + MCQ.
   - SFT: skipped (2026-04-14 Stage-0 probe: 22.18% hit rate, 49.68% pass@1 → above 15% threshold).
   - Two runs, same hparams: TIR-GRPO (main) via `scripts/perlmutter/prod_tir_het.sbatch`; CoT-GRPO (ablation) via `prod_cot_het.sbatch`. CoT uses VeRL's `single_turn_agent`; TIR uses `tool_agent_loop`. Tool-call format differs per model: Qwen3-Thinking emits hermes (JSON) natively → `multi_turn.format=hermes`; Qwen3.5-Coder emits qwen3_coder (XML) → `multi_turn.format=qwen3_coder`. Both runs share identical total `response_length` — in CoT, the `tool_call + tool_response + answer` share collapses into the post-interrupt answer budget (see `verl/experimental/agent_loop/single_turn_agent_loop.py`).
   - Secondary benchmark: MATH-500 hard subset.

3. **Verification Stack**:
   - `src/phys_reasoner/verifier/router.py` — rule verifier + xVerify-7B fallback
   - xVerify served via `scripts/serve_xverify.py` (HTTP) during training — binary correct/incorrect output, no threshold semantics (the server's `>= 0.5` check is a no-op since score is always 0.0 or 1.0)
   - For eval, `XVerifyJudge` is called in-process directly (same class, same behavior, no HTTP overhead)
   - LaTeX FN rate (~68% on expression types) is the core RQ2 quantity — measured post-hoc from training rollouts

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

## Active model: Qwen3-4B-Thinking-2507 (GRPO)

The Qwen3.5-4B GDN-specific overrides (SDPA attention, Qwen3_5DecoderLayer
wrap policy) are **no longer needed** — Qwen3-4B-Thinking is a vanilla
transformer. The current required overrides are minimal:

```bash
# Optimizer offload — 4B params × fp32 × 2 moments ≈ 32 GB; combined with
# vLLM KV cache exceeds 80 GB without offloading on a single trainer node.
# Drop this once moving to 2-trainer-node FSDP-8.
actor_rollout_ref.actor.fsdp_config.optimizer_offload=True

# Ulysses sequence parallelism — 4 on 4-GPU trainer (FSDP=1, DP=1) or
# 2 on 4-GPU trainer (DP=2 for FSDP shard). Set TRAIN_SP=1 to disable on
# single-GPU smokes.
actor_rollout_ref.actor.ulysses_sequence_parallel_size=2
actor_rollout_ref.ref.ulysses_sequence_parallel_size=2

# Multi-turn TIR with hermes tool-call format (Qwen3-4B-Thinking-2507 chat
# template emits hermes-style <tool_call>{json}</tool_call>, NOT the
# qwen3_coder XML format Qwen3.5 used).
actor_rollout_ref.rollout.multi_turn.format=hermes
actor_rollout_ref.rollout.multi_turn.enable=true
actor_rollout_ref.rollout.agent.default_agent_loop=tool_agent
```

Reference: `scripts/train_async.sh` (canonical), `scripts/perlmutter/smoke_tir_het.sbatch` (Perlmutter wrapper). Migration plan: `.claude/plans/qwen3-switch.md`.

### If reverting to Qwen3.5-4B (historical)

Re-add: `+actor_rollout_ref.model.override_config={attn_implementation:sdpa}`
(GDN+flash-attn monkey-patch crashes), and
`+actor_rollout_ref.actor.fsdp_config.wrap_policy.transformer_layer_cls_to_wrap=[Qwen3_5DecoderLayer]`
(plus the `ref.` twin) because `Qwen3_5ForCausalLM._no_split_modules`
incorrectly lists `Qwen3_5VisionBlock`. Switch `multi_turn.format` back
to `qwen3_coder`, set `INTERRUPT_LEN=17` (vs 16 for Qwen3), and remove
the Ulysses SP lines entirely. See `scripts/smoke_tir_qwen35.sh`
(validated 2026-04-02) for the full set.
