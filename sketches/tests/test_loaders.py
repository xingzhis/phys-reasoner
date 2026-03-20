"""Smoke tests for dataset loaders.

These tests mock HuggingFace `load_dataset` to avoid network calls.
They verify schema normalization logic, not dataset content.
"""

from __future__ import annotations
from unittest.mock import MagicMock, patch

from phys_reasoner.data.loaders import (
    load_physics,
    load_ugphysics,
    load_olympiadbench,
    load_abench_phy_b,
    load_critpt,
)
from phys_reasoner.data.schema import PhysicsProblem


# ---------------------------------------------------------------------------
# PHYSICS loader
# ---------------------------------------------------------------------------

_PHYSICS_ROWS = [
    {
        "question": "Find the velocity of a projectile at t=2s.",
        "answer": ["19.6 m/s"],
        "answer_type": ["Numerical"],
        "solution": "v = g*t",
        "difficulty": "High School",
        "domain": "Mechanics",
        "language": "en",
        "translate": False,
    },
    {
        "question": "What is the energy? (a) KE (b) PE",
        "answer": ["0.5mv^2", "mgh"],
        "answer_type": ["Expression", "Expression"],
        "solution": "",
        "difficulty": "Undergraduate Physics Major",
        "domain": "Mechanics",
        "language": "en",
        "translate": False,
    },
]


def test_load_physics_single_answer():
    mock_ds = MagicMock()
    mock_ds.__iter__ = lambda self: iter(_PHYSICS_ROWS[:1])

    with patch("phys_reasoner.data.loaders.load_dataset", return_value=mock_ds):
        problems = load_physics(split="test")

    assert len(problems) == 1
    p = problems[0]
    assert isinstance(p, PhysicsProblem)
    assert p.source == "PHYSICS"
    assert not p.is_multi_part()
    assert p.answer_type == "numerical"


def test_load_physics_multi_part():
    mock_ds = MagicMock()
    mock_ds.__iter__ = lambda self: iter(_PHYSICS_ROWS[1:])

    with patch("phys_reasoner.data.loaders.load_dataset", return_value=mock_ds):
        problems = load_physics(split="test")

    assert len(problems) == 1
    p = problems[0]
    assert p.is_multi_part()
    assert len(p.answer) == 2
    assert p.answer_type == ["expression", "expression"]


# ---------------------------------------------------------------------------
# UGPhysics loader
# ---------------------------------------------------------------------------

_UGPHYSICS_ROWS = [
    {
        "index": 0,
        "problem": "Calculate angular momentum.",
        "answers": "1.26 × 10^{6}",
        "answer_type": "NV",
        "unit": "kg m^2/s",
        "is_multiple_answer": False,
        "level": "undergraduate",
        "language": "en",
        "subject": "ClassicalMechanics",
        "topic": "Angular momentum",
        "solution": "L = I omega",
    }
]


def test_load_ugphysics():
    mock_split = MagicMock()
    mock_split.__iter__ = lambda self: iter(_UGPHYSICS_ROWS)
    mock_ds = {"en": mock_split}

    with patch("phys_reasoner.data.loaders.load_dataset", return_value=mock_ds):
        problems = load_ugphysics(subjects=["ClassicalMechanics"])

    assert len(problems) == 1
    p = problems[0]
    assert p.source == "UGPhysics"
    assert p.unit == "kg m^2/s"
    assert p.answer_type == "numerical"


# ---------------------------------------------------------------------------
# OlympiadBench loader
# ---------------------------------------------------------------------------

_OLYMPIAD_ROWS = [
    {
        "id": 1,
        "question": "A ball is thrown upward. Find max height.",
        "final_answer": ["$\\frac{v_0^2}{2g}$"],
        "answer_type": "Expression",
        "unit": "m",
        "error": "1e-4",
        "difficulty": "competition",
        "subfield": "Mechanics",
        "is_multiple_answer": False,
        "question_type": "open-ended",
        "context": "",
    }
]


def test_load_olympiadbench():
    mock_split = MagicMock()
    mock_split.__iter__ = lambda self: iter(_OLYMPIAD_ROWS)
    mock_ds = {"train": mock_split}

    with patch("phys_reasoner.data.loaders.load_dataset", return_value=mock_ds):
        problems = load_olympiadbench(configs=["OE_TO_physics_en_COMP"])

    assert len(problems) == 1
    p = problems[0]
    assert p.source == "OlympiadBench"
    assert p.tolerance == "1e-4"
    assert p.answer_type == "expression"


# ---------------------------------------------------------------------------
# ABench Phy_B loader
# ---------------------------------------------------------------------------

def test_load_abench_missing_file(tmp_path):
    import pytest
    with pytest.raises(FileNotFoundError, match="ABench"):
        load_abench_phy_b(csv_path=tmp_path / "nonexistent.csv")


def test_load_abench_phy_b(tmp_path):
    csv_content = "mid,subid,standard_question,standard_answer\n1,1,A block slides down.,3.14\n1,2,Same block faster.,4.20\n"
    csv_file = tmp_path / "Phy_B_dynamic_100.csv"
    csv_file.write_text(csv_content)

    problems = load_abench_phy_b(csv_path=csv_file)
    assert len(problems) == 2
    assert problems[0].source == "ABench_Phy_B"
    assert problems[0].answer_type == "numerical"
    assert problems[0].tolerance == "0.01"
    assert problems[0].metadata["mid"] == "1"


# ---------------------------------------------------------------------------
# CritPt loader
# ---------------------------------------------------------------------------

_CRITPT_ROWS = [
    {
        "problem_id": "CP001",
        "problem_type": "ising",
        "problem_description": "Compute the critical temperature of a 2D Ising model.",
        "code_template": "def answer():\n    Tc = ...\n    return Tc",
        "answer_code": "def answer():\n    Tc = 2.269\n    return Tc",
        "answer_only_code": "Tc = 2.269\nreturn Tc",
        "metadata_tag": "stat_mech",
        "metadata_problem_setup": "2D square lattice",
    }
]


def test_load_critpt():
    mock_split = MagicMock()
    mock_split.__iter__ = lambda self: iter(_CRITPT_ROWS)
    mock_ds = {"train": mock_split}

    with patch("phys_reasoner.data.loaders.load_dataset", return_value=mock_ds):
        problems = load_critpt()

    assert len(problems) == 1
    p = problems[0]
    assert p.source == "CritPt"
    assert p.answer_type == "code"
    assert "answer()" in p.answer
