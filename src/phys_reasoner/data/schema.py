"""Common schema for all physics datasets.

Every loader normalizes to a list of PhysicsProblem instances.

AnswerType values (from inspecting all datasets):
  PHYSICS:       'Numerical', 'Expression', 'Equation', 'MCQ', 'Open-end', 'True/False', 'Interval'
  UGPhysics:     'NV', 'EX', 'EQ', 'MC', 'TF', 'IN' (+ comma-joined multi-part combos)
  OlympiadBench: 'Numerical', 'Expression', 'Equation', 'Interval' (comma-joined multi-part)
  PHYBench:      no answer_type field — inferred 'symbolic'
  SciBench-RL:   no answer_type field — inferred 'numerical'
  CritPt:        no answer_type field — 'code' (Python function)
  ABench:        no answer_type field — 'numerical'
  Yale NLP:      no answer_type field — mixed
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Literal


AnswerType = Literal[
    "numerical",    # numeric value (possibly with unit)
    "expression",   # LaTeX expression / formula
    "equation",     # equation (LHS = RHS)
    "interval",     # interval notation
    "mcq",          # multiple choice
    "true_false",   # True/False
    "inequality",   # inequality expression
    "symbolic",     # symbolic/algebraic (PHYBench)
    "code",         # Python function source (CritPt)
    "open_end",     # open-ended text (not rule-verifiable)
    "unknown",      # fallback
]

Source = Literal[
    "PHYSICS",
    "UGPhysics",
    "OlympiadBench",
    "PHYBench",
    "SciBench_RL",
    "CritPt",
    "Yale_NLP",
    "ABench_Phy_A",
    "ABench_Phy_B",
]


@dataclass
class PhysicsProblem:
    """Normalized physics problem.

    Fields
    ------
    problem_id : str
        Unique identifier within source.
    problem : str
        Problem statement (may contain LaTeX).
    answer : str | list[str]
        Gold answer. String for single answers; list for multi-part.
        CritPt uses Python source code strings.
    answer_type : str | list[str]
        Raw answer type tag from the source dataset (not yet normalized).
    source : Source
        Which dataset this came from.
    difficulty : str
        Free-form difficulty label from the source dataset.
    split : str
        "train_candidate", "eval_tier1", "eval_tier2", or "eval_tier3".
        Final train/val split happens after Phase D (zero-shot filter).
    unit : str
        Physical unit string (may be empty). Populated when source provides it.
    tolerance : str
        Allowed numerical tolerance from source (e.g. "1e-4"), or "" if unset.
    domain : str
        Physics domain / subject (e.g. "Mechanics", "QuantumMechanics").
    language : str
        "en" or "zh".
    metadata : dict
        Any extra source-specific fields.
    """

    problem_id: str
    problem: str
    answer: str | list[str]
    answer_type: str | list[str]
    source: Source
    difficulty: str = ""
    split: str = "train_candidate"
    unit: str = ""
    tolerance: str = ""
    domain: str = ""
    language: str = "en"
    metadata: dict = field(default_factory=dict)

    def is_multi_part(self) -> bool:
        return isinstance(self.answer, list)

    def to_dict(self) -> dict:
        d = {
            "problem_id": self.problem_id,
            "problem": self.problem,
            "answer": self.answer,
            "answer_type": self.answer_type,
            "source": self.source,
            "difficulty": self.difficulty,
            "split": self.split,
            "unit": self.unit,
            "tolerance": self.tolerance,
            "domain": self.domain,
            "language": self.language,
        }
        d.update(self.metadata)
        return d
