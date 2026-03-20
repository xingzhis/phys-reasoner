"""Tests for the common data schema and dataset loaders."""

import pytest
from phys_reasoner.data.schema import PhysicsProblem


# ---------------------------------------------------------------------------
# Unit tests (fast, no HF/file access)
# ---------------------------------------------------------------------------

def test_single_answer():
    p = PhysicsProblem(
        problem_id="test_1",
        problem="What is the speed of light?",
        answer="3e8",
        answer_type="numerical",
        source="PHYSICS",
        unit="m/s",
    )
    assert not p.is_multi_part()
    d = p.to_dict()
    assert d["answer"] == "3e8"
    assert d["unit"] == "m/s"


def test_multi_part_answer():
    p = PhysicsProblem(
        problem_id="test_2",
        problem="(a) Find x. (b) Find y.",
        answer=["3 m", "4 m"],
        answer_type=["numerical", "numerical"],
        source="PHYSICS",
    )
    assert p.is_multi_part()
    d = p.to_dict()
    assert d["answer"] == ["3 m", "4 m"]


def test_to_dict_has_required_keys():
    p = PhysicsProblem(
        problem_id="x",
        problem="Q",
        answer="A",
        answer_type="numerical",
        source="UGPhysics",
    )
    d = p.to_dict()
    for key in ("problem_id", "problem", "answer", "answer_type", "source", "split"):
        assert key in d, f"Missing key: {key}"


# ---------------------------------------------------------------------------
# Integration tests (slow — require HF cache + local files)
# Mark with pytest.mark.slow; run with: pytest -m slow
# ---------------------------------------------------------------------------

pytestmark_slow = pytest.mark.slow


def _check_rows(rows: list, n: int = 3) -> None:
    """Shared assertions for all loader integration tests."""
    assert len(rows) >= n, f"Expected at least {n} rows, got {len(rows)}"
    for row in rows[:n]:
        assert isinstance(row, PhysicsProblem)
        assert isinstance(row.problem_id, str) and row.problem_id
        assert isinstance(row.problem, str) and row.problem
        assert row.answer is not None
        if isinstance(row.answer, list):
            assert all(isinstance(a, str) for a in row.answer), (
                f"answer list contains non-str: {row.answer}"
            )
        else:
            assert isinstance(row.answer, str)
        assert row.answer_type is not None
        assert row.source is not None
        assert row.split in (
            "train_candidate", "eval_tier1", "eval_tier2", "eval_tier3"
        ), f"Unexpected split: {row.split!r}"


@pytest.mark.slow
def test_load_physics():
    from phys_reasoner.data.loaders import load_physics
    rows = load_physics()
    _check_rows(rows)
    # answer should NOT be doubly nested after loading
    for r in rows[:20]:
        if isinstance(r.answer, list):
            assert not any(isinstance(a, list) for a in r.answer), (
                f"Doubly-nested answer not unwrapped: {r.answer}"
            )
    # Open-end rows should be eval_tier1
    oe_rows = [r for r in rows if "Open-end" in (
        r.answer_type if isinstance(r.answer_type, list) else [r.answer_type]
    )]
    assert all(r.split == "eval_tier1" for r in oe_rows)


@pytest.mark.slow
def test_load_ugphysics():
    from phys_reasoner.data.loaders import load_ugphysics
    rows = load_ugphysics(subjects=["ClassicalMechanics"])
    _check_rows(rows)
    # No dirty answer_type labels (newlines or backticks)
    for r in rows[:50]:
        at = r.answer_type if isinstance(r.answer_type, str) else str(r.answer_type)
        assert "\n" not in at, f"Newline in answer_type: {at!r}"
        assert "`" not in at, f"Backtick in answer_type: {at!r}"


@pytest.mark.slow
def test_load_olympiadbench():
    from phys_reasoner.data.loaders import load_olympiadbench
    rows = load_olympiadbench(config="OE_TO_physics_en_COMP")
    _check_rows(rows)


@pytest.mark.slow
def test_load_phybench():
    from phys_reasoner.data.loaders import load_phybench
    rows = load_phybench()
    _check_rows(rows)
    assert all(r.answer_type == "symbolic" for r in rows[:10])


@pytest.mark.slow
def test_load_scibench_rl():
    from phys_reasoner.data.loaders import load_scibench_rl
    rows = load_scibench_rl(exclude_chemistry=True)
    _check_rows(rows)
    # Chemistry sources should be excluded
    from phys_reasoner.data.loaders import _SCIBENCH_CHEMISTRY_SOURCES
    for r in rows:
        assert r.domain not in _SCIBENCH_CHEMISTRY_SOURCES, (
            f"Chemistry source not excluded: {r.domain}"
        )


@pytest.mark.slow
def test_load_critpt():
    from phys_reasoner.data.loaders import load_critpt
    rows = load_critpt()
    _check_rows(rows)
    assert all(r.split == "eval_tier3" for r in rows)
    assert all(r.answer_type == "code" for r in rows)


@pytest.mark.slow
def test_load_yale_physics():
    from phys_reasoner.data.loaders import load_yale_physics
    rows = load_yale_physics(tier="textonly")
    _check_rows(rows)
    assert all(r.split == "eval_tier3" for r in rows)


@pytest.mark.slow
def test_load_abench_phy_a():
    from phys_reasoner.data.loaders import load_abench
    rows = load_abench(subset="Phy_A")
    _check_rows(rows)
    assert all(r.split == "eval_tier2" for r in rows)
    assert all(r.source == "ABench_Phy_A" for r in rows)


@pytest.mark.slow
def test_load_abench_phy_b():
    from phys_reasoner.data.loaders import load_abench
    rows = load_abench(subset="Phy_B")
    _check_rows(rows)
    assert all(r.split == "eval_tier2" for r in rows)
    assert all(r.source == "ABench_Phy_B" for r in rows)
