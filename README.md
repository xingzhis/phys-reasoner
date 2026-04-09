# PhysCode: Tool-Integrated Reasoning for Physics via RLVR

Research project training Qwen3.5-4B with RLVR to solve physics problems using single-block tool-integrated reasoning (TIR): the model reasons, executes one Python/SymPy code block, receives the output, and reasons to a final answer. Execution-based reward eliminates the symbolic verification noise (~68% LaTeX FN rate on expression types) that degrades CoT-GRPO, connecting empirically to the RLVεR theoretical framework.

Target venue: NeurIPS 2026. Proposal: `docs/physcode_proposal_v5.md`.

---

## Environment

All commands run inside an Apptainer container. The base SIF is read-only; a writable overlay image holds additional installed packages.

**Container images:**

| File | Purpose |
|---|---|
| `verl_vllm017.latest.sif` | Base Apptainer SIF — vllm 0.17.0, transformers 4.57.6 |
| `phys-reasoner-overlay-017b.img` | **Active overlay** — holds project packages + `/opt/phys-extras/` |

**One-time setup per session:**

```bash
source env.sh   # sets ROOT, SIF, OVERLAY, HF_HOME, PYTHONNOUSERSITE
```

After sourcing, use these aliases for interactive commands:

```bash
# CPU (add PYTHONPATH for upgraded packages in /opt/phys-extras/)
PYTHONNOUSERSITE=1 apptainer exec \
  --overlay "$OVERLAY" --bind /etc/pki:/etc/pki \
  --env "PYTHONPATH=/opt/phys-extras/" "$SIF" <command>

# GPU
PYTHONNOUSERSITE=1 apptainer exec --nv \
  --overlay "$OVERLAY" --bind /etc/pki:/etc/pki \
  --env "PYTHONPATH=/opt/phys-extras/" "$SIF" <command>

# Run tests
PYTHONNOUSERSITE=1 apptainer exec \
  --overlay "$OVERLAY" --bind /etc/pki:/etc/pki \
  --env "PYTHONPATH=/opt/phys-extras/" "$SIF" \
  python3 -m pytest tests/ -v
```

All sbatch scripts source `env.sh` and pass `PYTHONPATH=/opt/phys-extras/` automatically.

**`PYTHONNOUSERSITE=1` is required** — prevents `~/.local` packages from shadowing the overlay. `env.sh` exports it automatically.

### Machine-local configuration

```bash
cp .env.example .env   # override HF_HOME, SIF, OVERLAY, Slurm partition/QOS, etc.
```

### Fresh environment setup (one-time, from scratch)

See **`docs/setup.md`** for the full step-by-step tutorial (clone → configure → install → datasets → tests → training).

```bash
# 1. Pull the SIF and create the overlay (sbatch — takes ~10 min)
sbatch pull_docker.sbatch

# 2. Install all packages into the overlay (run on login node, ~5 min)
bash scripts/setup_overlay.sh
```

`setup_overlay.sh` installs:
- `phys-reasoner[dev]` (project + all declared deps)
- `verl` from `verl_repo/` with `--no-deps`
- `transformers==5.3.0`, `huggingface_hub==1.8.0`, `flash-linear-attention` into `/opt/phys-extras/`
- Runs `e2fsck` to mark the overlay clean for compute nodes

**Why `/opt/phys-extras/`?** The 017 SIF ships transformers 4.57.6 (no qwen3_5 support). We need 5.3.0. Upgrading in-place via pip fails on RHEL 8 compute nodes — OverlayFS whiteout entries for SIF packages are not respected when the overlay is mounted `:ro`. The `--target /opt/phys-extras/` approach writes fresh files to a new path, and `PYTHONPATH` makes Python find them first. No whiteout needed.

---

## TIR Trajectory Format

Each solution follows a fixed single-execution structure using Qwen3.5's native XML tool-call tokens:

```
<think>
...reasoning...
</think>
<tool_call>
<function=python>
<parameter=code>
import sympy as sp
# code here
print(result)
</parameter>
</function>
</tool_call>
<tool_response>
{"output": "31622.776"}
</tool_response>
So ω ≈ 3.16 × 10⁴ rad/s. \boxed{C}
```

The model generates exactly one code block per trajectory. The sandbox executes it, injects the tool response, and the model reasons to a final `\boxed{}` answer. The verifier checks the final answer.

**`enable_thinking` per phase:**
- **Phase 1 (generation)**: `enable_thinking=True` — reasoning goes into `<think>...</think>`, keeping Python code clean
- **Phase 2 (after tool response)**: `enable_thinking=False` — model writes final answer directly

With thinking suppressed in phase 1, reasoning leaks into Python code comments (no plain-content slot in Qwen3.5's tool-call assistant turn). Always use thinking=True for phase 1. See `src/phys_reasoner/tir/prompts.py` for full explanation.

**Token length warning**: thinking traces can be very long on hard problems (`response_length/clip_ratio ≈ 0.94` at 1536 tokens). Planned mitigations: token-budget injection, stop-thinking at threshold, RL length penalty.

**Why native tool-call tokens**: Qwen3.5's `</tool_call>` is a special token the model is pre-trained on — no format confusion, no looping issues. Custom bracket formats (`[code]...[/code]`) are fragile.

**Qwen3 vs Qwen3.5 tool-call format difference:**

| Model | Format | VeRL `multi_turn.format` |
|---|---|---|
| Qwen3 (0.6B, 1.7B, 4B) | hermes: `{"name": "python", "arguments": {...}}` | `hermes` |
| Qwen3.5 (0.8B, 4B) | qwen3_coder: `<function=python><parameter=code>...</parameter></function>` | `qwen3_coder` |

---

## Training Pipeline

**Stage 0 — Zero-shot probe:**
Run 100-problem zero-shot check (Qwen3.5-4B instruct, thinking on). Measure execution success and verifier hit rate. If hit rate ≥ 15% on numerical problems → skip SFT.

```bash
sbatch scripts/stage0_probe.sbatch
```

**Stage 1 — SFT (conditional):**
If zero-shot format compliance < 15%, fine-tune on 2–5k TIR demonstrations.

**Stage 2 — GRPO:**
Binary reward on final `\boxed{}` answer via VeRL. Curriculum: numerical first → expression + MCQ.

```bash
sbatch scripts/train.sbatch
```

### Required VeRL overrides for Qwen3.5-4B

These must be in every GRPO/smoke run for Qwen3.5-4B to work correctly:

```bash
# Use SDPA (not flash_attention_2) for FSDP actor/ref.
# VeRL's Ulysses monkey-patch conflicts with Qwen3.5's hybrid attention
# (GDN linear layers + full attention layers) in transformers 5.3.0.
'+actor_rollout_ref.model.override_config={attn_implementation:sdpa}'

# Offload optimizer state to CPU — AdamW for 4.54B params needs ~36 GB GPU
# (2 moments × float32). With vLLM KV cache this exceeds H200's 80 GB.
actor_rollout_ref.actor.fsdp_config.optimizer_offload=True

# Explicit FSDP wrap class — Qwen3_5ForCausalLM incorrectly lists
# Qwen3_5VisionBlock in _no_split_modules (it's a text-only model).
# Without this, VeRL crashes looking for the non-existent vision block.
'+actor_rollout_ref.actor.fsdp_config.wrap_policy.transformer_layer_cls_to_wrap=[Qwen3_5DecoderLayer]'
'+actor_rollout_ref.ref.fsdp_config.wrap_policy.transformer_layer_cls_to_wrap=[Qwen3_5DecoderLayer]'

# TIR multi-turn format for Qwen3.5
actor_rollout_ref.rollout.multi_turn.format=qwen3_coder
actor_rollout_ref.rollout.multi_turn.enable=true
actor_rollout_ref.rollout.agent.default_agent_loop=tool_agent
```

See `scripts/smoke_tir_qwen35.sh` for the complete validated configuration.

---

## Data Pipeline

### Dr. SCI (primary, ~107k problems)

```bash
bash scripts/prepare_drsci.sh   # download → dedup → clean
```

Final file: `data/processed/drsci_physics_clean.parquet`

Cleaning steps: prose extraction, dollar-stripping, double-unescaping, truncated drops, prose-gold drops. 1.3% of rows filtered for figure references.

### Curated 6.8k corpus (hard-problem supplement)

```bash
bash scripts/prepare_data.sh   # download → validate → dedup
```

Final file: `data/processed/candidates_deduped.parquet`

| Source | Count |
|---|---|
| UGPhysics | 5,451 |
| PHYSICS (Yale) | 805 |
| SciBench_RL | 280 |
| OlympiadBench | 230 |
| PHYBench | 100 |

### Evaluation tiers

| Dataset | Tier | Purpose |
|---|---|---|
| PHYSICS held-out | Tier 1 | In-domain textbook |
| ABench-Physics (dynamic) | Tier 2 | Robustness across numerical variations |
| CritPt (~70 problems) | Tier 3 | PhD-level frontier |

---

## Verifier

Layered pipeline: unit check (pint) → rule verify (math-verify) → LLM fallback (xVerify-7B-I).

- Authoritative baseline: `data/results/rescore_7b_v3.parquet` (6,866 rows, 39.7% accuracy)
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

## Project Structure

```
phys-reasoner/
├── src/phys_reasoner/
│   ├── tir/
│   │   ├── prompts.py          # TIR system prompt, stop tokens, extract_code()
│   │   ├── sandbox.py          # Subprocess code execution (30s timeout, whitelist)
│   │   └── tir_agent_loop.py   # VeRL AgentLoopBase subclass (@register "physcode_tir")
│   ├── eval/
│   │   └── stage0_probe.py     # Zero-shot TIR probe (vLLM, no Ray)
│   ├── verifier/
│   │   ├── router.py           # Main entry: verify_answer()
│   │   ├── extract.py          # \boxed{} extraction
│   │   ├── math_verify_wrapper.py
│   │   ├── unit_check.py       # pint unit-aware comparison
│   │   └── xverify_judge.py    # xVerify-7B-I LLM fallback
│   ├── data/
│   │   ├── loaders.py
│   │   ├── normalize.py
│   │   └── schema.py
│   └── training/
│       └── reward.py           # VeRL-compatible compute_score()
├── scripts/
│   ├── smoke_tir_qwen35.sh     # Validated Qwen3.5-4B smoke test (H200)
│   ├── smoke_tir_qwen35.sbatch # sbatch wrapper for smoke test
│   ├── grpo_train.sh           # GRPO training launcher
│   ├── setup_overlay.sh        # Install packages into overlay (run after pull_docker)
│   ├── dump_rollouts.py        # Offline rollout inspector (vLLM, no Ray)
│   ├── stage0_probe.sbatch     # Stage 0 probe job
│   ├── prepare_drsci.sh        # Dr. SCI end-to-end pipeline
│   ├── prepare_data.sh         # 6.8k corpus pipeline
│   └── drsci_clean.py          # Dr. SCI cleaning steps
├── tests/
│   ├── test_tir.py             # TIR pipeline tests (43 tests)
│   └── test_verifier.py        # Verifier test suite
└── docs/
    ├── physcode_proposal_v5.md
    ├── training-decisions.md   # Goldilocks strategy, dataset sizes, baselines
    └── verifier-zero-shot-experiments.md
```

---

## Models

- **Primary**: `Qwen/Qwen3.5-4B` (instruct, thinking ON for phase 1)
- **Debug**: `Qwen/Qwen3.5-0.8B`
