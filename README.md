# PhysCode: Tool-Integrated Reasoning for Physics via RLVR

Research project training Qwen3.5-4B with RLVR to solve physics problems using single-block tool-integrated reasoning (TIR): the model reasons, executes one Python/SymPy code block, receives the output, and reasons to a final answer. Execution-based reward eliminates the symbolic verification noise (~68% LaTeX FN rate on expression types) that degrades CoT-GRPO, connecting empirically to the RLVεR theoretical framework.

Target venue: NeurIPS 2026. Proposal: `docs/physcode_proposal_v5.md`.

---

## Environment

All commands run inside an Apptainer container. The base SIF is read-only; a writable overlay holds all installed packages.

**One-time setup per session:**

```bash
source env.sh   # sets ROOT, SIF, OVERLAY, HF_HOME, PYTHONNOUSERSITE
```

After sourcing, use these aliases for interactive commands:

```bash
# CPU
apptainer exec --overlay "$OVERLAY" --bind /etc/pki:/etc/pki "$SIF" <command>

# GPU
apptainer exec --nv --overlay "$OVERLAY" --bind /etc/pki:/etc/pki "$SIF" <command>

# Run tests
apptainer exec --overlay "$OVERLAY" --bind /etc/pki:/etc/pki "$SIF" python -m pytest tests/ -v
```

All sbatch scripts source `env.sh` automatically.

**`PYTHONNOUSERSITE=1` is required** — prevents `~/.local` packages from shadowing the overlay. `env.sh` exports it automatically.

### Machine-local configuration

```bash
cp .env.example .env   # override HF_HOME, SIF, OVERLAY, Slurm partition/QOS, etc.
```

### Installing packages into the overlay

```bash
PYTHONNOUSERSITE=1 apptainer exec --overlay "$OVERLAY" --bind /etc/pki:/etc/pki "$SIF" \
    pip install -e "$ROOT[dev]"
```

**⚠️ DO NOT upgrade `huggingface-hub` and `transformers`** — the overlay defaults are stable and fully tested.

---

## TIR Trajectory Format

Each solution follows a fixed single-execution structure (using Qwen's native tool-call tokens):

```
<tool_call>
{"name": "python", "arguments": {"code": "import sympy as sp\n..."}}
</tool_call>
<tool_response>
{"output": "31622.776"}
</tool_response>
So ω ≈ 3.16 × 10⁴ rad/s. [answer] \boxed{C}
```

The model generates exactly one code block per trajectory. The sandbox executes it, injects the output, and the model reasons to a final `\boxed{}` answer. The verifier checks the final answer — not the code output directly.

**Why native tool-call tokens**: Qwen3.5's `</tool_call>` is a special token the model is pre-trained on — no format confusion, no looping issues. Custom bracket formats (`[code]...[/code]`) are fragile with this model.

**`enable_thinking=False` is required everywhere** — Qwen3.5 defaults to thinking mode ON (`<think>...</think>`), which breaks the TIR format. Two flags needed in training: `data.apply_chat_template_kwargs.enable_thinking=False` and `actor_rollout_ref.model.enable_thinking=False`. See `src/phys_reasoner/tir/prompts.py` for full explanation.

---

## Training Pipeline

**Stage 0 — Zero-shot probe:**
Run 100-problem zero-shot check (Qwen3.5-4B instruct, thinking off). Measure execution success and verifier hit rate. If hit rate ≥ 15% on numerical problems → skip SFT.

```bash
sbatch scripts/stage0_probe.sbatch
```

**Stage 1 — SFT (conditional):**
If zero-shot format compliance < 15%, fine-tune on 2–5k TIR demonstrations. Likely skippable with the instruct model.

**Stage 2 — GRPO:**
Binary reward on final `\boxed{}` answer via VeRL. Curriculum: numerical first → expression + MCQ.

```bash
sbatch scripts/grpo_train.sh
```

---

## Data Pipeline

### Dr. SCI (primary, ~107k problems)

```bash
bash scripts/prepare_drsci.sh   # download → dedup → clean
```

Final file: `data/processed/drsci_physics_clean.parquet`

Cleaning steps: prose extraction, dollar-stripping, double-unescaping, truncated drops, prose-gold drops. Known issue: ~4.2% of problems reference figures/diagrams not present in text — filter pending.

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
│   ├── stage0_probe.sbatch     # Stage 0 probe job
│   ├── grpo_train.sh           # GRPO training launcher
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

- **Primary**: `Qwen/Qwen3.5-4B` (instruct, thinking OFF)
- **Debug**: `Qwen/Qwen3.5-0.8B`
