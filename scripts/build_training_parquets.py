"""Build VeRL-ready training parquets from cleaned source data.

This script is the single reproducible step between cleaned data and training.
It does two things for both datasets:

  1. Figure filter (Dr. SCI only): drop rows whose question text references an
     external figure, diagram, or table that is not present in the problem text.
     ~1.3% of Dr. SCI rows are dropped (~1,400 / 107,158).
     The corpus (candidates_deduped) is manually curated and has no image rows.

  2. Prompt rebuild: overwrite the 'prompt' column with the canonical TIR messages
     format imported directly from src/phys_reasoner/tir/prompts.py:

         [{"role": "system", "content": TIR_SYSTEM_PROMPT},
          {"role": "user",   "content": <raw question>}]

     For Dr. SCI:  raw question = extra_info.question
     For corpus:   raw question = problem

     VeRL's RLHFDataset reads doc["prompt"] and passes it through
     tokenizer.apply_chat_template(..., **apply_chat_template_kwargs).
     The grpo_train.sh already passes enable_thinking=False via
     data.apply_chat_template_kwargs.enable_thinking=False.

     ⚠️  KNOWN GAP: grpo_train.sh does NOT pass tools=[PYTHON_TOOL_SCHEMA] via
         apply_chat_template_kwargs, so the Qwen tool XML schema is NOT injected
         during training. Fix: add tools kwarg to grpo_train.sh (deferred until
         VeRL smoke test completes). The probe (stage0_probe.py) already passes
         tools= correctly.

SOURCE OF TRUTH:
    If TIR_SYSTEM_PROMPT changes in prompts.py, re-run this script to regenerate
    both parquets before the next training run. stage0_probe.py uses prompts.py
    directly (no regeneration needed for probe runs).

Inputs (already cleaned, do not modify these):
    data/processed/drsci_physics_clean.parquet   -- output of drsci_clean.py
    data/processed/candidates_deduped.parquet    -- output of run_dedup.py

Outputs:
    data/processed/drsci_train.parquet
    data/processed/corpus_train.parquet

Usage:
    python scripts/build_training_parquets.py
    python scripts/build_training_parquets.py --report    # stats only, no save
    python scripts/build_training_parquets.py --drsci-only
    python scripts/build_training_parquets.py --corpus-only
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from phys_reasoner.tir.prompts import TIR_SYSTEM_PROMPT

# ---------------------------------------------------------------------------
# Figure filter — high-confidence unanswerable rows only.
# Targets phrases that unambiguously reference an external figure not in text.
# Does NOT filter "sketch the graph" (model task) or ambiguous "table" mentions.
# Estimated precision ~90%+, recall covers the bulk of truly unanswerable rows.
# ---------------------------------------------------------------------------

_FIG_REF_PAT = re.compile(
    r"shown in (the |a )?(figure|diagram|graph|image|table|circuit|plot)"
    r"|(?:see|refer to|from|in) (the |a )?(figure|diagram|image|circuit|table)"
    r"\s*(above|below|\d+)?"
    r"|(figure|diagram|image|circuit) (above|below|\d+)"
    r"|as (shown|depicted|illustrated|given) (in|by|below|above)"
    r"|the (circuit|diagram|figure|image|graph) (shows|represents|depicts)"
    r"|\bfig\.\s*\d+",
    re.IGNORECASE,
)


def _has_fig_ref(text: str) -> bool:
    return bool(_FIG_REF_PAT.search(text))


# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------

def _build_prompt(question: str) -> list[dict]:
    """Return the canonical TIR messages list for VeRL's prompt column."""
    return [
        {"role": "system", "content": TIR_SYSTEM_PROMPT},
        {"role": "user",   "content": question},
    ]


# ---------------------------------------------------------------------------
# Per-dataset processing
# ---------------------------------------------------------------------------

def _build_reward_model(ground_truth: str) -> dict:
    """VeRL's naive reward manager accesses data['reward_model']['ground_truth'].
    Must be a dict column (parquet struct) — not a flat 'reward_model.ground_truth' column.
    """
    return {"ground_truth": str(ground_truth), "style": "rule"}


def _build_extra_info(answer_type: str, unit: str = "", tolerance: float = 0.05,
                      problem: str = "", **passthrough) -> dict:
    """Fields that compute_score reads from extra_info.
    VeRL passes extra_info to the reward fn; answer_type/unit/tolerance live here.
    """
    d = {"answer_type": str(answer_type), "unit": str(unit or ""),
         "tolerance": float(tolerance), "problem": str(problem or "")}
    d.update(passthrough)
    return d


def process_drsci(input_path: str, output_path: str, report: bool) -> None:
    print(f"\n=== Dr. SCI ===")
    print(f"Loading {input_path}...")
    df = pd.read_parquet(input_path)
    print(f"  {len(df):,} rows")

    # --- Figure filter ---
    q_col = "extra_info.question"
    fig_mask = df[q_col].astype(str).apply(_has_fig_ref)
    n_fig = fig_mask.sum()
    df_filtered = df[~fig_mask].copy()
    print(f"  Figure filter: dropped {n_fig} rows ({n_fig/len(df):.1%}) → {len(df_filtered):,} remain")

    # --- Prompt rebuild ---
    df_filtered["prompt"] = df_filtered[q_col].astype(str).apply(_build_prompt)
    print(f"  Prompt column rebuilt from TIR_SYSTEM_PROMPT + {q_col}")

    # --- VeRL reward_model struct (dict column, not flat dot-notation) ---
    df_filtered["reward_model"] = df_filtered["reward_model.ground_truth"].apply(_build_reward_model)

    # --- extra_info struct: answer_type, unit, tolerance for compute_score ---
    df_filtered["extra_info"] = df_filtered.apply(lambda r: _build_extra_info(
        answer_type=r.get("inferred_answer_type", "numerical"),
        unit="",
        tolerance=0.05,
        problem=str(r[q_col]),
        subject=str(r.get("extra_info.subject", "")),
        difficulty=str(r.get("extra_info.difficulty", "")),
    ), axis=1)

    # Keep only the columns VeRL needs (drop flat reward_model.* and extra_info.* columns)
    keep_cols = ["data_source", "prompt", "reward_model", "extra_info"]
    df_out = df_filtered[keep_cols]
    print(f"  Output columns: {keep_cols}")

    if not report:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        df_out.to_parquet(output_path, index=False)
        print(f"  Saved → {output_path}")


def process_corpus(input_path: str, output_path: str, report: bool) -> None:
    print(f"\n=== Corpus ===")
    print(f"Loading {input_path}...")
    df = pd.read_parquet(input_path)
    print(f"  {len(df):,} rows")

    # No figure filter — corpus is manually curated, no image-dependent rows.

    # --- Prompt rebuild ---
    df = df.copy()
    df["prompt"] = df["problem"].astype(str).apply(_build_prompt)
    print(f"  Prompt column built from TIR_SYSTEM_PROMPT + problem")

    # --- VeRL reward_model struct ---
    df["reward_model"] = df["answer"].apply(_build_reward_model)

    # --- extra_info struct ---
    def _safe_tol(v, default=0.05):
        try:
            return float(v)
        except (ValueError, TypeError):
            return default

    df["extra_info"] = df.apply(lambda r: _build_extra_info(
        answer_type=r.get("answer_type", "numerical"),
        unit=str(r.get("unit", "")),
        tolerance=_safe_tol(r.get("tolerance", 0.05)),
        problem=str(r.get("problem", "")),
        source=str(r.get("source", "")),
        difficulty=str(r.get("difficulty", "")),
    ), axis=1)

    keep_cols = ["data_source", "prompt", "reward_model", "extra_info"]
    # corpus may not have data_source column; add it
    if "data_source" not in df.columns:
        df["data_source"] = df.get("source", "corpus")
    df_out = df[keep_cols]
    print(f"  Output columns: {keep_cols}")

    if not report:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        df_out.to_parquet(output_path, index=False)
        print(f"  Saved → {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--report", action="store_true",
                        help="Print stats only — do not save")
    parser.add_argument("--drsci-only", action="store_true")
    parser.add_argument("--corpus-only", action="store_true")
    parser.add_argument("--drsci-input",
                        default="data/processed/drsci_physics_clean.parquet")
    parser.add_argument("--drsci-output",
                        default="data/processed/drsci_train.parquet")
    parser.add_argument("--corpus-input",
                        default="data/processed/candidates_deduped.parquet")
    parser.add_argument("--corpus-output",
                        default="data/processed/corpus_train.parquet")
    args = parser.parse_args()

    run_drsci  = not args.corpus_only
    run_corpus = not args.drsci_only

    if run_drsci:
        process_drsci(args.drsci_input, args.drsci_output, args.report)
    if run_corpus:
        process_corpus(args.corpus_input, args.corpus_output, args.report)

    print("\nDone.")
    if args.report:
        print("(--report mode: no files written)")
    else:
        print("Re-run this script whenever TIR_SYSTEM_PROMPT changes in prompts.py.")


if __name__ == "__main__":
    main()
