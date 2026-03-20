"""Dataset loaders — normalize each source to list[PhysicsProblem].

Usage
-----
from phys_reasoner.data.loaders import load_physics, load_ugphysics, \
    load_olympiadbench, load_abench_phy_b, load_critpt
"""

from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import Iterator

try:
    from datasets import load_dataset
except ImportError:
    load_dataset = None  # type: ignore[assignment]

from .schema import AnswerType, PhysicsProblem

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _guess_answer_type(answer_type_raw: str) -> AnswerType:
    """Map source-specific answer type label to our canonical AnswerType."""
    raw = answer_type_raw.strip().lower()
    if raw in ("numerical", "nv", "float", "numeric"):
        return "numerical"
    if raw in ("expression", "expr", "formula"):
        return "expression"
    if raw in ("mc", "mcq", "multiple choice", "multiple_choice"):
        return "mcq"
    if raw in ("equation", "eq"):
        return "equation"
    if raw in ("interval",):
        return "expression"
    if raw in ("true/false", "tf"):
        return "mcq"
    return "unknown"


# ---------------------------------------------------------------------------
# PHYSICS dataset  (HF: desimfj/PHYSICS)
# ---------------------------------------------------------------------------

def load_physics(
    split: str = "test",
    language: str | None = None,
    cache_dir: str | None = None,
) -> list[PhysicsProblem]:
    """Load the PHYSICS benchmark (NeurIPS 2025).

    HF ID: desimfj/PHYSICS
    Splits: "test" (2,000 rows). Training data must be downloaded from
    https://github.com/Zhengsh123/PHYSICS and placed at data/raw/PHYSICS_train.jsonl

    Parameters
    ----------
    split : "test" | "train"
        Which split to load.  "train" requires the GitHub JSONL file.
    language : "en" | "zh" | None
        Filter by language.  None returns all.
    cache_dir : optional HuggingFace cache directory override.
    """
    if load_dataset is None:
        raise ImportError("Install `datasets` to use load_physics()")

    if split == "test":
        logger.info("Loading PHYSICS test split from HuggingFace (desimfj/PHYSICS)...")
        ds = load_dataset("desimfj/PHYSICS", split="test", cache_dir=cache_dir)
        rows = list(ds)
    elif split == "train":
        train_path = Path(__file__).parent.parent.parent.parent / "data" / "raw" / "PHYSICS_train.jsonl"
        if not train_path.exists():
            raise FileNotFoundError(
                f"PHYSICS train split not found at {train_path}.\n"
                "Download from https://github.com/Zhengsh123/PHYSICS and save as "
                "data/raw/PHYSICS_train.jsonl"
            )
        import json
        rows = [json.loads(line) for line in train_path.read_text().splitlines() if line.strip()]
    else:
        raise ValueError(f"Unknown split: {split!r}. Expected 'train' or 'test'.")

    problems: list[PhysicsProblem] = []
    for i, row in enumerate(rows):
        if language and row.get("language") != language:
            continue

        # answer and answer_type are parallel lists
        raw_answers = row.get("answer", [])
        raw_types = row.get("answer_type", [])
        if not isinstance(raw_answers, list):
            raw_answers = [raw_answers]
        if not isinstance(raw_types, list):
            raw_types = [raw_types] * len(raw_answers)

        answer_types: list[AnswerType] = [_guess_answer_type(t) for t in raw_types]

        if len(raw_answers) == 1:
            answer: str | list[str] = raw_answers[0]
            atype: AnswerType | list[AnswerType] = answer_types[0]
        else:
            answer = raw_answers
            atype = answer_types

        problems.append(PhysicsProblem(
            problem_id=f"PHYSICS_{split}_{i}",
            problem=row.get("question", ""),
            answer=answer,
            answer_type=atype,
            source="PHYSICS",
            difficulty=row.get("difficulty", ""),
            split=split,
            domain=row.get("domain", ""),
            language=row.get("language", "en"),
            metadata={
                "solution": row.get("solution", ""),
                "translate": row.get("translate", False),
            },
        ))

    logger.info("Loaded %d PHYSICS problems (split=%s)", len(problems), split)
    return problems


# ---------------------------------------------------------------------------
# UGPhysics  (HF: UGPhysics/ugphysics)
# ---------------------------------------------------------------------------

_UGPHYSICS_SUBJECTS = [
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


def load_ugphysics(
    subjects: list[str] | None = None,
    language: str | None = None,
    cache_dir: str | None = None,
) -> list[PhysicsProblem]:
    """Load UGPhysics benchmark (ICML 2025).

    HF ID: UGPhysics/ugphysics
    5,520 undergraduate-level physics problems across 13 subjects.

    Parameters
    ----------
    subjects : list of subject names to load, or None for all 13.
    language : "en" | "zh" | None (all).
    cache_dir : optional HuggingFace cache directory override.
    """
    if load_dataset is None:
        raise ImportError("Install `datasets` to use load_ugphysics()")

    subjects = subjects or _UGPHYSICS_SUBJECTS
    problems: list[PhysicsProblem] = []

    for subject in subjects:
        logger.info("Loading UGPhysics subject: %s ...", subject)
        try:
            ds = load_dataset("UGPhysics/ugphysics", subject, cache_dir=cache_dir)
        except Exception as exc:
            logger.warning("Failed to load UGPhysics/%s: %s", subject, exc)
            continue

        for split_name, split_ds in ds.items():
            for row in split_ds:
                if language and row.get("language") != language:
                    continue

                atype = _guess_answer_type(row.get("answer_type", ""))
                # If is_multiple_answer flag set, treat as multi-part if comma-separated
                answer_str: str = str(row.get("answers", ""))
                answer: str | list[str] = answer_str
                if row.get("is_multiple_answer") and "," in answer_str:
                    parts = [p.strip() for p in answer_str.split(",") if p.strip()]
                    answer = parts if len(parts) > 1 else answer_str

                problems.append(PhysicsProblem(
                    problem_id=f"UGPhysics_{subject}_{row.get('index', len(problems))}",
                    problem=row.get("problem", ""),
                    answer=answer,
                    answer_type=atype,
                    source="UGPhysics",
                    difficulty=str(row.get("level", "")),
                    split="test",  # UGPhysics is eval-only
                    unit=str(row.get("unit", "") or ""),
                    domain=subject,
                    language=row.get("language", "en"),
                    metadata={
                        "subject": row.get("subject", subject),
                        "topic": row.get("topic", ""),
                        "solution": row.get("solution", ""),
                    },
                ))

    logger.info("Loaded %d UGPhysics problems", len(problems))
    return problems


# ---------------------------------------------------------------------------
# OlympiadBench (physics subset)  (HF: lscpku/OlympiadBench-official)
# ---------------------------------------------------------------------------

_OLYMPIAD_PHYSICS_CONFIGS = [
    "OE_TO_physics_en_COMP",
    "OE_TO_physics_zh_CEE",
    # Multimodal configs omitted from text-only training set
    # "OE_MM_physics_en_COMP",
    # "OE_MM_physics_zh_CEE",
]


def load_olympiadbench(
    configs: list[str] | None = None,
    language: str | None = None,
    text_only: bool = True,
    cache_dir: str | None = None,
) -> list[PhysicsProblem]:
    """Load OlympiadBench physics subset (ACL 2024).

    HF ID: lscpku/OlympiadBench-official
    Text-only physics configs by default (multimodal configs require images).

    Parameters
    ----------
    configs : list of config names, or None to load defaults.
    language : "en" | "zh" | None (all).
    text_only : if True (default), skip multimodal configs.
    cache_dir : optional HuggingFace cache directory override.
    """
    if load_dataset is None:
        raise ImportError("Install `datasets` to use load_olympiadbench()")

    if configs is None:
        configs = _OLYMPIAD_PHYSICS_CONFIGS
    if text_only:
        configs = [c for c in configs if "_TO_" in c]

    problems: list[PhysicsProblem] = []

    for config in configs:
        lang = "zh" if "_zh_" in config else "en"
        if language and lang != language:
            continue

        logger.info("Loading OlympiadBench config: %s ...", config)
        try:
            ds = load_dataset("lscpku/OlympiadBench-official", config, cache_dir=cache_dir)
        except Exception as exc:
            logger.warning("Failed to load OlympiadBench/%s: %s", config, exc)
            continue

        for split_name, split_ds in ds.items():
            for row in split_ds:
                raw_answers = row.get("final_answer", [])
                if not isinstance(raw_answers, list):
                    raw_answers = [raw_answers]
                raw_answers = [str(a) for a in raw_answers if a is not None]

                atype = _guess_answer_type(row.get("answer_type", ""))
                if len(raw_answers) == 1:
                    answer: str | list[str] = raw_answers[0]
                    answer_type_out: AnswerType | list[AnswerType] = atype
                else:
                    answer = raw_answers
                    answer_type_out = [atype] * len(raw_answers)

                problems.append(PhysicsProblem(
                    problem_id=f"OlympiadBench_{config}_{row.get('id', len(problems))}",
                    problem=row.get("question", ""),
                    answer=answer,
                    answer_type=answer_type_out,
                    source="OlympiadBench",
                    difficulty=str(row.get("difficulty", "")),
                    split="test",
                    unit=str(row.get("unit", "") or ""),
                    tolerance=str(row.get("error", "") or ""),
                    domain=str(row.get("subfield", "") or ""),
                    language=lang,
                    metadata={
                        "is_multiple_answer": row.get("is_multiple_answer", False),
                        "question_type": row.get("question_type", ""),
                        "context": row.get("context", ""),
                        "config": config,
                    },
                ))

    logger.info("Loaded %d OlympiadBench problems", len(problems))
    return problems


# ---------------------------------------------------------------------------
# ABench-Physics Phy_B  (GitHub CSV only)
# ---------------------------------------------------------------------------

def load_abench_phy_b(
    csv_path: str | Path | None = None,
    cache_dir: str | None = None,
) -> list[PhysicsProblem]:
    """Load ABench-Physics Phy_B dynamic evaluation set.

    No HuggingFace release — download from GitHub:
    https://github.com/inclusionAI/ABench/tree/main/Physics

    Save the file as data/raw/Phy_B_dynamic_100.csv

    Parameters
    ----------
    csv_path : explicit path to CSV file.  If None, looks in data/raw/.
    """
    if csv_path is None:
        default = Path(__file__).parent.parent.parent.parent / "data" / "raw" / "Phy_B_dynamic_100.csv"
        csv_path = default

    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(
            f"ABench Phy_B CSV not found at {csv_path}.\n"
            "Download from https://github.com/inclusionAI/ABench/tree/main/Physics\n"
            "and save as data/raw/Phy_B_dynamic_100.csv"
        )

    problems: list[PhysicsProblem] = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            mid = row.get("mid", "")
            subid = row.get("subid", "")
            problems.append(PhysicsProblem(
                problem_id=f"ABench_Phy_B_{mid}_{subid}",
                problem=row.get("standard_question", ""),
                answer=row.get("standard_answer", ""),
                answer_type="numerical",  # Phy_B is strictly numerical
                source="ABench_Phy_B",
                difficulty="dynamic",
                split="test",
                tolerance="0.01",  # 1% relative error per ABench spec
                language="en",
                metadata={"mid": mid, "subid": subid},
            ))

    logger.info("Loaded %d ABench Phy_B problems", len(problems))
    return problems


# ---------------------------------------------------------------------------
# CritPt  (HF: CritPt-Benchmark/CritPt)
# ---------------------------------------------------------------------------

def load_critpt(
    cache_dir: str | None = None,
) -> list[PhysicsProblem]:
    """Load CritPt benchmark (~70 PhD-level physics problems).

    HF ID: CritPt-Benchmark/CritPt

    Note: CritPt answers are Python ``answer()`` functions.  The answer field
    stores the ``answer_code`` source string.  Evaluation requires submission
    to the authors' remote grading server (rate-limited to 10 submissions/day).
    """
    if load_dataset is None:
        raise ImportError("Install `datasets` to use load_critpt()")

    logger.info("Loading CritPt from HuggingFace (CritPt-Benchmark/CritPt)...")
    ds = load_dataset("CritPt-Benchmark/CritPt", cache_dir=cache_dir)

    problems: list[PhysicsProblem] = []
    for split_name, split_ds in ds.items():
        for row in split_ds:
            problems.append(PhysicsProblem(
                problem_id=f"CritPt_{row.get('problem_id', len(problems))}",
                problem=row.get("problem_description", ""),
                answer=row.get("answer_code", ""),
                answer_type="code",
                source="CritPt",
                difficulty="phd",
                split="test",
                domain="frontier_physics",
                language="en",
                metadata={
                    "problem_type": row.get("problem_type", ""),
                    "code_template": row.get("code_template", ""),
                    "answer_only_code": row.get("answer_only_code", ""),
                    "metadata_tag": row.get("metadata_tag", ""),
                    "metadata_problem_setup": row.get("metadata_problem_setup", ""),
                },
            ))

    logger.info("Loaded %d CritPt problems", len(problems))
    return problems


# ---------------------------------------------------------------------------
# Convenience: load all datasets
# ---------------------------------------------------------------------------

def load_all_training(
    language: str | None = "en",
    cache_dir: str | None = None,
) -> list[PhysicsProblem]:
    """Load all training sources: PHYSICS (train) + UGPhysics + OlympiadBench.

    Returns combined list.  Each problem has `split` set to "train" or "test"
    as appropriate.  For RLVR training, you'll want to re-split via prepare.py.
    """
    problems: list[PhysicsProblem] = []

    # PHYSICS train (requires local JSONL file from GitHub)
    try:
        problems += load_physics(split="train", language=language, cache_dir=cache_dir)
    except FileNotFoundError as e:
        logger.warning("Skipping PHYSICS train: %s", e)

    # PHYSICS test (available on HF) — we hold this out for Tier 1 eval
    try:
        test_problems = load_physics(split="test", language=language, cache_dir=cache_dir)
        for p in test_problems:
            p.split = "test"
        problems += test_problems
    except Exception as e:
        logger.warning("Skipping PHYSICS test: %s", e)

    # UGPhysics
    try:
        problems += load_ugphysics(language=language, cache_dir=cache_dir)
    except Exception as e:
        logger.warning("Skipping UGPhysics: %s", e)

    # OlympiadBench (text-only physics)
    try:
        problems += load_olympiadbench(language=language, cache_dir=cache_dir)
    except Exception as e:
        logger.warning("Skipping OlympiadBench: %s", e)

    return problems
