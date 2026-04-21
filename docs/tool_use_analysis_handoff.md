# Handoff — zero-shot tool-use analysis for the paper

**Created:** 2026-04-19. **Target venue:** ICML 2026 AI4Physics workshop (Apr 24 AOE).

## Goal

Produce the zero-shot "before RL" tool-use numbers for the paper. Specifically:

1. **Per-cell tool-use rate** — fraction of TIR-mode rollouts that actually called the sandbox vs pure-reasoned. This is the headline "pre-RL tool-call rate" on Qwen3-4B-Thinking-2507.
2. **Per-cell tool-use × correctness breakdown** — pass@1 among rollouts that called the tool (sandbox OK / sandbox error) vs rollouts that skipped the tool.
3. **Per-answer-type tool-use rate** on in-distribution benchmarks — supports the per-type story in §5.

These numbers underpin the paper's "RL unlocks tool-use behavior" framing.

## Context (what already exists)

- All zero-shot rollouts are already generated and scored under `outputs/eval/<benchmark>/<mode>__Qwen-Qwen3-4B-Thinking-2507__<tag>/` — see `docs/eval_results.md` for the full layout.
- The analyzer `eval/analyze_tool_use.py` is already implemented. Reads `rollouts.parquet` + `scored.parquet` for every cell, emits a per-cell summary table.
- 32 cells total: 8 benchmarks × (TIR/CoT × train-preset/qwen-preset).
- Only TIR cells have meaningful tool-use data (CoT cells have `code=None` by construction — they're included for sanity).

## Environment setup (recap from CLAUDE.md)

```bash
ROOT=/gpfs/radev/scratch/krishnaswamy_smita/xs272/phys-reasoner  # adjust if on Perlmutter
SIF=$ROOT/verl_vllm017.latest.sif
OVERLAY=$ROOT/phys-reasoner-overlay-017b.img                     # or -017.img on some hosts

cd $ROOT
```

## Step 1 — aggregate tool-use rate (run the analyzer)

This is CPU-only; no GPU needed. Takes a few minutes.

```bash
PYTHONNOUSERSITE=1 apptainer exec \
  --overlay "$OVERLAY" --bind /etc/pki:/etc/pki \
  --env "PYTHONPATH=/opt/phys-extras/" "$SIF" \
  python3 eval/analyze_tool_use.py \
  | tee outputs/eval/tool_use_summary.txt
```

Expected output: table with columns `benchmark | cell | n | called% | call_OK | call_err | skipped% | call_OK_pass% | call_err_pass% | skip_pass% | overall%`.

The **`called%` column for TIR cells** is the headline zero-shot tool-call rate.

## Step 2 — per-answer-type tool-use rate on in-dist benchmarks

Run this as a one-shot Python script inside apptainer. Focus on in-dist cells only (pool_v2_*); external benchmarks have fewer answer types and less useful breakdown.

Save the following as `eval/analyze_tool_use_by_type.py` before running:

```python
"""Per-answer-type tool-use rate on in-dist pool_v2 TIR cells."""
from pathlib import Path
import pandas as pd

ROOT = Path(".")
MODEL_SLUG = "Qwen-Qwen3-4B-Thinking-2507"
BENCHMARKS = ["pool_v2_drsci", "pool_v2_physics",
              "pool_v2_ugphysics", "pool_v2_scibench"]
TAGS = ["train", "qwen"]

rows = []
for bench in BENCHMARKS:
    for tag in TAGS:
        cell = ROOT / f"outputs/eval/{bench}/tir__{MODEL_SLUG}__{tag}"
        if not (cell / "rollouts.parquet").exists():
            continue
        r = pd.read_parquet(cell / "rollouts.parquet")
        s = pd.read_parquet(cell / "scored.parquet")
        r = r.set_index(["problem_idx", "rollout_idx"])
        s = s.set_index(["problem_idx", "rollout_idx"])
        common = r.index.intersection(s.index)
        r, s = r.loc[common], s.loc[common]
        r["called"] = r["code"].apply(lambda c: c is not None and not isinstance(c, float))
        r["correct"] = s["correct"].astype(bool)
        # answer_type lives in extra_info struct
        if "answer_type" in r.columns:
            atype = r["answer_type"]
        else:
            atype = r["extra_info"].apply(lambda x: (x or {}).get("answer_type") if isinstance(x, dict) else None)
        r["answer_type"] = atype
        agg = r.groupby("answer_type").agg(
            n=("called", "size"),
            called_pct=("called", lambda c: 100 * c.mean()),
            overall_pass=("correct", lambda c: 100 * c.mean()),
        )
        agg["bench"] = bench
        agg["tag"] = tag
        rows.append(agg.reset_index())

df = pd.concat(rows, ignore_index=True)
# Pretty print
for (b, t), g in df.groupby(["bench", "tag"]):
    print(f"\n=== {b} / tir/{t} ===")
    print(g[["answer_type", "n", "called_pct", "overall_pass"]]
            .to_string(index=False, float_format=lambda x: f"{x:6.1f}"))

# Also save flat CSV for the paper
df.to_csv("outputs/eval/tool_use_by_type.csv", index=False)
print("\nWrote outputs/eval/tool_use_by_type.csv")
```

Then run:

```bash
PYTHONNOUSERSITE=1 apptainer exec \
  --overlay "$OVERLAY" --bind /etc/pki:/etc/pki \
  --env "PYTHONPATH=/opt/phys-extras/" "$SIF" \
  python3 eval/analyze_tool_use_by_type.py \
  | tee outputs/eval/tool_use_by_type_summary.txt
```

## Step 3 — report back

Commit (or paste in next session) these three files:

- `outputs/eval/tool_use_summary.txt` — aggregate tool-use rates per cell
- `outputs/eval/tool_use_by_type.csv` — per-answer-type tool-use rates (in-dist)
- `outputs/eval/tool_use_by_type_summary.txt` — human-readable version of the above

If the outputs are small (they will be — <100 KB total), the cleanest thing is to **commit them directly to this branch** so the next planning session can consume them without needing HPC access:

```bash
git add outputs/eval/tool_use_summary.txt \
        outputs/eval/tool_use_by_type.csv \
        outputs/eval/tool_use_by_type_summary.txt \
        eval/analyze_tool_use_by_type.py
git commit -m "add zero-shot tool-use analysis outputs for paper §5"
git push
```

Note: `outputs/eval/**/rollouts.parquet` and `scored.parquet` are typically gitignored — do NOT try to commit those.

## What to do if something breaks

- **Analyzer errors on a cell**: the script prints `[bench/cell] ERROR: ...` and continues. Report which cells failed; we'll skip them in the paper.
- **`code` column missing**: some CoT cells may not have the column; `analyze_tool_use.py` tolerates this. The per-type script above uses `r["code"]` with a fallback via `.apply()`; if it errors, add `if "code" not in r.columns: continue` after the `r, s = r.loc[common], s.loc[common]` line.
- **`answer_type` missing in `extra_info`**: the fallback in the script returns `None`. You'll see a `None` answer-type row in the output — that's fine, it just means the field wasn't populated for that row. Report if >10% of rows are None.

## Expected runtime

~5 minutes total, CPU only.

## After this: what the numbers enable

- Table row in paper §4.3: "Zero-shot tool-call rate on Qwen3-4B-Thinking-2507 (TIR mode)" per benchmark.
- Figure in §5: "Tool-call rate pre-RL vs post-RL" (post-RL comes after training finishes).
- Per-answer-type tool-call shift: the sharpest form of "RL unlocks tool use on tool-suited problems."

Thanks. Questions? Update `docs/eval_results.md` with any findings while you're at it.
