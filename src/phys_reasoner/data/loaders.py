"""Dataset loaders — one function per source, each returning list[PhysicsProblem].

Field names verified against actual downloaded data; see docs/datasets.md.
Do NOT port from sketches/data/loaders.py — field names differ.

All HuggingFace datasets are loaded from data/hf_cache (relative to project root).
Pass cache_dir as an absolute path when running from a different working directory.
"""

from __future__ import annotations

import ast
import csv
import json
import re
from pathlib import Path
from typing import Sequence

from phys_reasoner.data.schema import PhysicsProblem


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_PROJECT_ROOT = Path(__file__).parent.parent.parent.parent  # phys-reasoner/
_DEFAULT_CACHE = str(_PROJECT_ROOT / "data" / "hf_cache")
_DEFAULT_RAW = _PROJECT_ROOT / "data" / "raw"

_ALL_UGPHYSICS_SUBJECTS = [
    "AtomicPhysics",
    "ClassicalElectromagnetism",
    "ClassicalMechanics",
    "Electrodynamics",
    "GeometricalOptics",
    "QuantumMechanics",
    "Relativity",
    "SemiconductorPhysics",
    "Solid-StatePhysics",
    "StatisticalMechanics",
    "TheoreticalMechanics",
    "Thermodynamics",
    "WaveOptics",
]


def _unwrap_single(lst: list) -> str | list:
    """Return lst[0] if it has exactly one element, else lst."""
    if isinstance(lst, list) and len(lst) == 1:
        return lst[0]
    return lst


# ---------------------------------------------------------------------------
# load_physics
# ---------------------------------------------------------------------------

def load_physics(cache_dir: str = _DEFAULT_CACHE) -> list[PhysicsProblem]:
    """Load desimfj/PHYSICS (test split, 2000 rows).

    Known QC applied at load time:
    - answer is List[List[str]] (doubly nested) — unwrap outer list.
    - Open-end rows assigned to eval_tier1 (not rule-verifiable).
    """
    from datasets import load_dataset

    ds = load_dataset("desimfj/PHYSICS", cache_dir=cache_dir, split="test")
    problems = []
    for i, row in enumerate(ds):
        raw_answer = row["answer"]
        raw_at = row["answer_type"]  # List[str]

        # Known fix: answer is List[List[str]] — unwrap outer list
        if (
            isinstance(raw_answer, list)
            and len(raw_answer) >= 1
            and isinstance(raw_answer[0], list)
        ):
            raw_answer = raw_answer[0]

        answer = _unwrap_single(raw_answer) if isinstance(raw_answer, list) else raw_answer
        answer_type = _unwrap_single(raw_at) if isinstance(raw_at, list) else raw_at

        # Open-end rows go to eval pool (not rule-verifiable)
        at_list = raw_at if isinstance(raw_at, list) else [raw_at]
        split = "eval_tier1" if "Open-end" in at_list else "train_candidate"

        problems.append(
            PhysicsProblem(
                problem_id=f"PHYSICS_{i:05d}",
                problem=row["question"],
                answer=answer,
                answer_type=answer_type,
                source="PHYSICS",
                domain=str(row.get("domain") or ""),
                difficulty=str(row.get("difficulty") or ""),
                split=split,
            )
        )
    return problems


# ---------------------------------------------------------------------------
# load_ugphysics
# ---------------------------------------------------------------------------

def load_ugphysics(
    subjects: Sequence[str] = _ALL_UGPHYSICS_SUBJECTS,
    cache_dir: str = _DEFAULT_CACHE,
    language: str = "en",
) -> list[PhysicsProblem]:
    """Load UGPhysics/ugphysics for given subjects (default: all 13 EN).

    Only loads the language split matching `language` ("en" or "zh").
    ZH is excluded by default (exact translations of EN).

    Known QC applied at load time:
    - Chemistry/non-physics subjects are all physics here — no filtering needed.
    - Dirty answer_type labels (newlines, backticks) are stripped and mapped.
    """
    from datasets import load_dataset

    problems = []
    for subject in subjects:
        ds_all = load_dataset("UGPhysics/ugphysics", subject, cache_dir=cache_dir)
        # UGPhysics has splits named "en" and "zh"
        if language not in ds_all:
            continue
        split_ds = ds_all[language]
        for i, row in enumerate(split_ds):
            raw_at = row.get("answer_type", "")
            # Known fix: strip dirty labels (newlines, backticks, prose)
            clean_at = _clean_ugphysics_answer_type(raw_at)

            problems.append(
                PhysicsProblem(
                    problem_id=f"UGPhysics_{subject}_{i:05d}",
                    problem=row["problem"],
                    answer=row["answers"],       # str, may contain \boxed{}
                    answer_type=clean_at,
                    source="UGPhysics",
                    unit=str(row.get("unit") or ""),
                    domain=subject,
                    language=language,
                    split="train_candidate",
                    metadata={
                        "is_multiple_answer": row.get("is_multiple_answer", False),
                    },
                )
            )
    return problems


_UGPHYSICS_AT_MAP = {
    "NV": "NV", "EX": "EX", "EQ": "EQ", "MC": "MC", "TF": "TF", "IN": "IN",
}


def _clean_ugphysics_answer_type(raw: str) -> str:
    """Strip dirty UGPhysics answer_type labels.

    Known dirty patterns:
      'NV\\n   \\nThe final answer...' → 'NV'
      'EX\\n```'                       → 'EX'
    For comma-joined multi-part (e.g. 'NV, NV'), kept as-is.
    """
    if not isinstance(raw, str):
        return str(raw)
    # Take only the first line (strips trailing prose/backticks)
    first_line = raw.split("\n")[0].strip()
    # Remove backticks and surrounding whitespace
    first_line = re.sub(r"`+", "", first_line).strip()
    # If it looks like a valid code (possibly multi-part), return it
    if re.match(r"^[A-Z]{2}(,\s*[A-Z]{2})*$", first_line):
        return first_line
    # Fallback: try to extract first valid 2-letter code
    m = re.match(r"([A-Z]{2})", first_line)
    if m:
        return m.group(1)
    return first_line


# ---------------------------------------------------------------------------
# load_olympiadbench
# ---------------------------------------------------------------------------

def load_olympiadbench(
    config: str = "OE_TO_physics_en_COMP",
    cache_dir: str = _DEFAULT_CACHE,
) -> list[PhysicsProblem]:
    """Load lscpku/OlympiadBench-official for one config.

    Recommended configs:
      'OE_TO_physics_en_COMP'  — text-only EN competition (236 rows)
      'OE_MM_physics_en_COMP'  — multimodal EN competition (456 rows, has images)

    answer_type may be comma-joined for multi-part problems (e.g. 'Expression,Numerical').
    """
    from datasets import load_dataset

    ds_all = load_dataset(
        "lscpku/OlympiadBench-official", config, cache_dir=cache_dir
    )
    # OlympiadBench has a single 'train' split
    split_name = list(ds_all.keys())[0]
    split_ds = ds_all[split_name]

    problems = []
    for i, row in enumerate(split_ds):
        raw_at = row.get("answer_type") or ""
        raw_answer = row.get("final_answer", [])  # List[str]

        answer = _unwrap_single(raw_answer) if isinstance(raw_answer, list) else raw_answer
        answer_type = _unwrap_single(raw_at.split(",")) if "," in str(raw_at) else raw_at

        problems.append(
            PhysicsProblem(
                problem_id=f"OlympiadBench_{config}_{i:05d}",
                problem=row["question"],
                answer=answer,
                answer_type=answer_type,
                source="OlympiadBench",
                unit=str(row.get("unit") or ""),
                tolerance=str(row.get("error") or ""),
                domain=config,
                split="train_candidate",
            )
        )
    return problems


# ---------------------------------------------------------------------------
# load_phybench
# ---------------------------------------------------------------------------

def load_phybench(cache_dir: str = _DEFAULT_CACHE) -> list[PhysicsProblem]:
    """Load Eureka-Lab/PHYBench (1000 rows).

    answer is raw symbolic LaTeX (no \\boxed{}).
    tag field holds the physics domain (e.g. 'MECHANICS').
    """
    from datasets import load_dataset

    ds = load_dataset("Eureka-Lab/PHYBench", cache_dir=cache_dir, split="train")
    problems = []
    for i, row in enumerate(ds):
        # Drop rows with empty problem or answer (800/1000 rows have empty answers)
        if not row.get("content", "").strip() or not row.get("answer", "").strip():
            continue
        problems.append(
            PhysicsProblem(
                problem_id=f"PHYBench_{i:05d}",
                problem=row["content"],
                answer=row["answer"],   # raw LaTeX, no \boxed{}
                answer_type="symbolic",
                source="PHYBench",
                domain=str(row.get("tag") or ""),
                split="train_candidate",
            )
        )
    return problems


# ---------------------------------------------------------------------------
# load_scibench_rl
# ---------------------------------------------------------------------------

_SCIBENCH_CHEMISTRY_SOURCES = {"atkins", "chemmc"}


def load_scibench_rl(
    cache_dir: str = _DEFAULT_CACHE,
    exclude_chemistry: bool = True,
) -> list[PhysicsProblem]:
    """Load Sihangli/scibench-rl (train split, 427 rows).

    Drops chemistry sources (atkins, chemmc) by default — 143 rows removed.
    Physics sources retained: fund, thermo, quan, calculus, stat (~284 rows).
    """
    from datasets import load_dataset

    ds = load_dataset("Sihangli/scibench-rl", cache_dir=cache_dir, split="train")
    problems = []
    for i, row in enumerate(ds):
        source_tag = str(row.get("source") or "")
        if exclude_chemistry and source_tag in _SCIBENCH_CHEMISTRY_SOURCES:
            continue
        problems.append(
            PhysicsProblem(
                problem_id=f"SciBench_RL_{i:05d}",
                problem=row["problem_text"],
                answer=str(row.get("answer_number") or ""),
                answer_type="numerical",
                source="SciBench_RL",
                unit=str(row.get("unit") or ""),
                domain=source_tag,
                split="train_candidate",
            )
        )
    return problems


# ---------------------------------------------------------------------------
# load_critpt
# ---------------------------------------------------------------------------

def load_critpt(cache_dir: str = _DEFAULT_CACHE) -> list[PhysicsProblem]:
    """Load CritPt-Benchmark/CritPt (70 rows, PhD-level physics).

    Answer is Python function source code in answer_code field.
    Evaluation requires remote API (10 submissions/day limit).
    Assigned to eval_tier3 (frontier evaluation).
    """
    from datasets import load_dataset

    ds = load_dataset("CritPt-Benchmark/CritPt", cache_dir=cache_dir, split="train")
    problems = []
    for i, row in enumerate(ds):
        pid = str(row.get("problem_id") or f"CritPt_{i:05d}")
        problems.append(
            PhysicsProblem(
                problem_id=f"CritPt_{pid}",
                problem=row["problem_description"],
                answer=str(row.get("answer_code") or ""),
                answer_type="code",
                source="CritPt",
                split="eval_tier3",
                metadata={
                    "code_template": row.get("code_template", ""),
                },
            )
        )
    return problems


# ---------------------------------------------------------------------------
# load_yale_physics
# ---------------------------------------------------------------------------

_YALE_TIER_TO_SUBDIR = {
    "textonly": "PHYSICS-textonly",
    "hard":     "PHYSICS-hard",
    "eval":     "PHYSICS-eval",
    "test":     "PHYSICS-test",
}


def load_yale_physics(
    tier: str = "textonly",
    raw_dir: Path = _DEFAULT_RAW,
) -> list[PhysicsProblem]:
    """Load Yale NLP Physics dataset from local JSONL files.

    tier options: 'textonly' (999 rows), 'hard' (523), 'eval' (297), 'test' (1000).

    Each JSONL has fields: id, questions, solutions, final_answers (list[str]), graphs.
    Assigned to eval_tier3.
    """
    subdir = _YALE_TIER_TO_SUBDIR.get(tier, tier)
    tier_dir = Path(raw_dir) / "yale-physics" / "PHYSICS" / subdir
    if not tier_dir.exists():
        raise FileNotFoundError(f"Yale NLP tier directory not found: {tier_dir}")

    problems = []
    for jsonl_path in sorted(tier_dir.glob("*.jsonl")):
        domain = jsonl_path.stem.replace("_dataset_textonly", "").replace("_dataset", "")
        with open(jsonl_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                pid = str(row.get("id", ""))
                raw_answers = row.get("final_answers", [])

                # final_answers is already a list (parsed from JSON)
                answer = _unwrap_single(raw_answers) if isinstance(raw_answers, list) else raw_answers

                problems.append(
                    PhysicsProblem(
                        problem_id=f"Yale_NLP_{pid}",
                        problem=row["questions"],
                        answer=answer,
                        answer_type="unknown",
                        source="Yale_NLP",
                        domain=domain,
                        split="eval_tier3",
                        metadata={
                            "has_graph": bool(row.get("graphs")),
                        },
                    )
                )
    return problems


# ---------------------------------------------------------------------------
# load_abench
# ---------------------------------------------------------------------------

def load_abench(
    subset: str = "Phy_B",
    raw_dir: Path = _DEFAULT_RAW,
) -> list[PhysicsProblem]:
    """Load ABench Physics from local CSV.

    subset: 'Phy_A' (400 static) or 'Phy_B' (400 dynamic, 100 base × 4 variants).

    Uses encoding='utf-8-sig' to strip BOM from column headers.
    Assigned to eval_tier2.
    """
    if subset == "Phy_A":
        csv_path = Path(raw_dir) / "abench" / "Phy_A_fixed_400.csv"
    elif subset == "Phy_B":
        csv_path = Path(raw_dir) / "abench" / "Phy_B_dynamic_100.csv"
    else:
        raise ValueError(f"Unknown ABench subset: {subset!r}. Use 'Phy_A' or 'Phy_B'.")

    if not csv_path.exists():
        raise FileNotFoundError(f"ABench CSV not found: {csv_path}")

    source = "ABench_Phy_A" if subset == "Phy_A" else "ABench_Phy_B"
    problems = []
    with open(csv_path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            mid = str(row.get("mid", f"{subset}_{i:04d}"))
            problems.append(
                PhysicsProblem(
                    problem_id=f"ABench_{subset}_{mid}_{i:04d}",
                    problem=row["standard_question"],
                    answer=str(row["standard_answer"]),
                    answer_type="numerical",
                    source=source,
                    split="eval_tier2",
                    metadata={"mid": mid},
                )
            )
    return problems
