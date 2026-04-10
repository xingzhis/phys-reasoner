"""Build VeRL-ready training parquets from cleaned source data.

This script is the single reproducible step between cleaned data and training.
It does three things for both datasets:

  1. Row filtering:
     - Dr. SCI: drop rows whose question text references an external figure/diagram
       (~1.3% dropped). Drop rows with NaN `from` or `unknown` answer_type.
     - Corpus: no row-level filtering (quality filter is a prior step).

  2. Metadata enrichment (extra_info):
     - Dr. SCI: answer_type, difficulty (float), from, subject, problem, unit, tolerance
     - Corpus: answer_type, primary_answer_type, domain, domain_coarse, source,
               difficulty, problem, unit, tolerance

  3. Prompt rebuild: overwrite the 'prompt' column with the canonical TIR messages
     format imported directly from src/phys_reasoner/tir/prompts.py:

         [{"role": "system", "content": TIR_SYSTEM_PROMPT},
          {"role": "user",   "content": <raw question>}]

     For Dr. SCI:  raw question = extra_info.question
     For corpus:   raw question = problem

     VeRL's RLHFDataset reads doc["prompt"] and passes it through
     tokenizer.apply_chat_template(..., **apply_chat_template_kwargs).
     The grpo_train.sh passes enable_thinking=True via
     data.apply_chat_template_kwargs.enable_thinking=True.

     NOTE on tool schema: grpo_train.sh does NOT pass tools=[PYTHON_TOOL_SCHEMA]
     via apply_chat_template_kwargs, but this is NOT a gap. VeRL's ToolAgentLoop
     calls apply_chat_template(messages, tools=self.tool_schemas) fresh during
     each rollout (_handle_pending_state), so the Qwen tool XML schema IS injected
     into the actual training tokens. The data-loading tokenization (which uses
     apply_chat_template_kwargs) is only used for prompt length filtering and is
     never used as training tokens.

SOURCE OF TRUTH:
    If TIR_SYSTEM_PROMPT changes in prompts.py, re-run this script to regenerate
    both parquets before the next training run. stage0_probe.py uses prompts.py
    directly (no regeneration needed for probe runs).

Inputs (already cleaned, do not modify these):
    data/processed/drsci_physics_clean.parquet   -- output of drsci_clean.py
    data/processed/candidates_deduped.parquet    -- output of run_dedup.py
      (or candidates_filtered.parquet if filter_data_quality.py was run first)

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

# ---------------------------------------------------------------------------
# Corpus metadata: domain coarse mapping and primary_answer_type normalization
# ---------------------------------------------------------------------------

DOMAIN_COARSE_MAP: dict[str, str] = {
    # mechanics
    "ClassicalMechanics": "mechanics",
    "Mechanics": "mechanics",
    "MECHANICS": "mechanics",
    "TheoreticalMechanics": "mechanics",
    # em_electro
    "ClassicalElectromagnetism": "em_electro",
    "Electrodynamics": "em_electro",
    "Electromagnetism": "em_electro",
    "ELECTRICITY": "em_electro",
    # quantum_modern
    "QuantumMechanics": "quantum_modern",
    "AtomicPhysics": "quantum_modern",
    "Modern Physics": "quantum_modern",
    "MODERN": "quantum_modern",
    "ADVANCED": "quantum_modern",
    "quan": "quantum_modern",
    # thermo_stat
    "StatisticalMechanics": "thermo_stat",
    "Thermodynamics": "thermo_stat",
    "thermo": "thermo_stat",
    "stat": "thermo_stat",
    "THERMODYNAMICS": "thermo_stat",
    # optics
    "WaveOptics": "optics",
    "Optics": "optics",
    "GeometricalOptics": "optics",
    "OPTICS": "optics",
    # other
    "SemiconductorPhysics": "other",
    "Solid-StatePhysics": "other",
    "Relativity": "other",
    "OE_TO_physics_en_COMP": "other",
    "fund": "other",
    "calculus": "other",
}


def normalize_primary_answer_type(raw: str) -> str:
    """Map 87 raw answer_type values to 7 clean categories.

    Single-type values pass through as-is (numerical, expression, equation,
    mcq, true_false, interval). JSON list values → "multi-part".
    """
    raw = str(raw).strip()
    if raw.startswith("["):
        return "multi-part"
    return raw


def map_domain_coarse(raw_domain: str) -> str:
    """Map 29 raw domain values to 6 coarse buckets."""
    return DOMAIN_COARSE_MAP.get(str(raw_domain).strip(), "other")


# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------

_ANSWER_TYPE_HINTS: dict[str, str] = {
    "mcq":        "This is a multiple-choice question. Put only the letter (A/B/C/D/...) inside \\boxed{}.",
    "true_false": "This is a true/false question. Put only True or False inside \\boxed{}.",
}


def _build_prompt(question: str, answer_type: str = "") -> list[dict]:
    """Return the canonical TIR messages list for VeRL's prompt column.

    For MCQ and true/false questions a one-line hint is appended to the user
    message (not the system prompt) to discourage the model from writing the
    option value or using yes/no instead of True/False.
    """
    hint = _ANSWER_TYPE_HINTS.get(str(answer_type).lower().strip(), "")
    user_content = f"{question}\n\n{hint}".strip() if hint else question
    return [
        {"role": "system", "content": TIR_SYSTEM_PROMPT},
        {"role": "user",   "content": user_content},
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
    print(f"  {len(df):,} rows loaded")

    q_col = "extra_info.question"

    # --- Drop rows with NaN `from` ---
    nan_from_mask = df["extra_info.from"].isna()
    n_nan_from = nan_from_mask.sum()
    df = df[~nan_from_mask]
    print(f"  Dropped {n_nan_from} rows with NaN from → {len(df):,} remain")

    # --- Drop rows with unknown answer_type ---
    unknown_mask = df["inferred_answer_type"] == "unknown"
    n_unknown = unknown_mask.sum()
    df = df[~unknown_mask]
    print(f"  Dropped {n_unknown} rows with unknown answer_type → {len(df):,} remain")

    # --- Figure filter ---
    fig_mask = df[q_col].astype(str).apply(_has_fig_ref)
    n_fig = fig_mask.sum()
    df_filtered = df[~fig_mask].copy()
    print(f"  Figure filter: dropped {n_fig} rows ({n_fig/len(df):.1%}) → {len(df_filtered):,} remain")

    # --- Prompt rebuild (with answer_type hint for MCQ / true_false) ---
    df_filtered["prompt"] = df_filtered.apply(
        lambda r: _build_prompt(str(r[q_col]),
                                answer_type=r.get("inferred_answer_type", "")),
        axis=1,
    )
    print(f"  Prompt column rebuilt from TIR_SYSTEM_PROMPT + {q_col} (+ type hints)")

    # --- VeRL reward_model struct (dict column, not flat dot-notation) ---
    df_filtered["reward_model"] = df_filtered["reward_model.ground_truth"].apply(_build_reward_model)

    # --- extra_info struct: answer_type, unit, tolerance, problem, subject,
    #     difficulty (float), from ---
    # Track difficulty conversion fallbacks: a fallback to 0.0 would silently
    # mix with genuine difficulty=0.0 rows, so we count and warn.
    _difficulty_fallbacks = []

    def _drsci_difficulty(val, idx) -> float:
        try:
            return float(val)
        except (ValueError, TypeError):
            _difficulty_fallbacks.append((idx, val))
            return 0.0

    df_filtered["extra_info"] = df_filtered.apply(lambda r: _build_extra_info(
        answer_type=r.get("inferred_answer_type", "numerical"),
        unit="",
        tolerance=0.05,
        problem=str(r[q_col]),
        subject=str(r.get("extra_info.subject", "")),
        difficulty=_drsci_difficulty(r.get("extra_info.difficulty", 0.0), r.name),
        **{"from": str(r.get("extra_info.from", ""))},
    ), axis=1)

    if _difficulty_fallbacks:
        print(f"  WARNING: {len(_difficulty_fallbacks)} difficulty values fell back to 0.0!")
        for idx, val in _difficulty_fallbacks[:10]:
            print(f"    row {idx}: {val!r} (type={type(val).__name__})")
    else:
        print(f"  Difficulty: all {len(df_filtered):,} values converted to float, 0 fallbacks")

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

    # --- Prompt rebuild (with answer_type hint for MCQ / true_false) ---
    df = df.copy()
    # For prompt hints, use the primary_answer_type (not the raw JSON-list value)
    df["_primary_answer_type"] = df["answer_type"].apply(normalize_primary_answer_type)
    df["prompt"] = df.apply(
        lambda r: _build_prompt(str(r["problem"]),
                                answer_type=r["_primary_answer_type"]),
        axis=1,
    )
    print(f"  Prompt column built from TIR_SYSTEM_PROMPT + problem (+ type hints)")

    # --- VeRL reward_model struct ---
    df["reward_model"] = df["answer"].apply(_build_reward_model)

    # --- extra_info struct: answer_type (raw), primary_answer_type, domain,
    #     domain_coarse, source, difficulty, problem, unit, tolerance ---
    def _safe_tol(v, default=0.05):
        try:
            return float(v)
        except (ValueError, TypeError):
            return default

    df["extra_info"] = df.apply(lambda r: _build_extra_info(
        answer_type=r.get("answer_type", "numerical"),
        primary_answer_type=normalize_primary_answer_type(r.get("answer_type", "numerical")),
        unit=str(r.get("unit", "")),
        tolerance=_safe_tol(r.get("tolerance", 0.05)),
        problem=str(r.get("problem", "")),
        source=str(r.get("source", "")),
        difficulty=str(r.get("difficulty", "")),
        domain=str(r.get("domain", "")),
        domain_coarse=map_domain_coarse(r.get("domain", "")),
    ), axis=1)

    keep_cols = ["data_source", "prompt", "reward_model", "extra_info"]
    # data_source = actual source (e.g. UGPhysics, OlympiadBench, etc.)
    if "data_source" not in df.columns:
        df["data_source"] = df["source"]
    df_out = df[keep_cols]
    print(f"  Output columns: {keep_cols}")

    # Print metadata summary
    pat = df["_primary_answer_type"].value_counts()
    print(f"  primary_answer_type: {pat.to_dict()}")
    dc = df["domain"].apply(map_domain_coarse).value_counts()
    print(f"  domain_coarse: {dc.to_dict()}")

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
                        default="data/processed/candidates_filtered.parquet")
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
