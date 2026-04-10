"""Tests for the data pipeline: metadata enrichment, splits, and probe sampling.

Tests the logic in:
  - scripts/build_training_parquets.py  (Step 0: metadata enrichment)
  - scripts/split_train_dev_test.py     (Step 0b: stratified split)
  - scripts/subsample_probe.py          (Step 1: probe subsample)

Unit tests only (no file I/O or container required). Run with:
    python -m pytest tests/test_data_pipeline.py -v
"""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Import scripts as modules (they're not packages)
# ---------------------------------------------------------------------------

_SCRIPTS = Path(__file__).parent.parent / "scripts"

def _import_script(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    # Need src on path for prompts import
    sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
    spec.loader.exec_module(mod)
    return mod

btp = _import_script("build_training_parquets", _SCRIPTS / "build_training_parquets.py")
spl = _import_script("split_train_dev_test", _SCRIPTS / "split_train_dev_test.py")
sub = _import_script("subsample_probe", _SCRIPTS / "subsample_probe.py")


# ===========================================================================
# Step 0: Metadata enrichment
# ===========================================================================

class TestNormalizePrimaryAnswerType:
    def test_single_types_passthrough(self):
        for t in ["numerical", "expression", "equation", "mcq", "true_false", "interval"]:
            assert btp.normalize_primary_answer_type(t) == t

    def test_json_list_becomes_multi_part(self):
        assert btp.normalize_primary_answer_type('["numerical", "numerical"]') == "multi-part"
        assert btp.normalize_primary_answer_type('["equation"]') == "multi-part"

    def test_empty_string(self):
        assert btp.normalize_primary_answer_type("") == ""

    def test_whitespace_handling(self):
        assert btp.normalize_primary_answer_type("  numerical  ") == "numerical"


class TestMapDomainCoarse:
    def test_all_29_domains_mapped(self):
        assert len(btp.DOMAIN_COARSE_MAP) == 29

    def test_mechanics_bucket(self):
        for d in ["ClassicalMechanics", "Mechanics", "MECHANICS", "TheoreticalMechanics"]:
            assert btp.map_domain_coarse(d) == "mechanics", f"{d} should map to mechanics"

    def test_em_electro_bucket(self):
        for d in ["ClassicalElectromagnetism", "Electrodynamics", "Electromagnetism", "ELECTRICITY"]:
            assert btp.map_domain_coarse(d) == "em_electro", f"{d} should map to em_electro"

    def test_quantum_modern_bucket(self):
        for d in ["QuantumMechanics", "AtomicPhysics", "Modern Physics", "MODERN", "ADVANCED", "quan"]:
            assert btp.map_domain_coarse(d) == "quantum_modern", f"{d} should map to quantum_modern"

    def test_thermo_stat_bucket(self):
        for d in ["StatisticalMechanics", "Thermodynamics", "thermo", "stat", "THERMODYNAMICS"]:
            assert btp.map_domain_coarse(d) == "thermo_stat", f"{d} should map to thermo_stat"

    def test_optics_bucket(self):
        for d in ["WaveOptics", "Optics", "GeometricalOptics", "OPTICS"]:
            assert btp.map_domain_coarse(d) == "optics", f"{d} should map to optics"

    def test_other_bucket(self):
        for d in ["SemiconductorPhysics", "Solid-StatePhysics", "Relativity",
                   "OE_TO_physics_en_COMP", "fund", "calculus"]:
            assert btp.map_domain_coarse(d) == "other", f"{d} should map to other"

    def test_unknown_falls_to_other(self):
        assert btp.map_domain_coarse("UnknownDomain") == "other"

    def test_coarse_values_are_exhaustive(self):
        expected = {"mechanics", "em_electro", "quantum_modern", "thermo_stat", "optics", "other"}
        actual = set(btp.DOMAIN_COARSE_MAP.values())
        assert actual == expected


class TestBuildExtraInfo:
    def test_difficulty_stays_float(self):
        ei = btp._build_extra_info(answer_type="numerical", difficulty=0.25)
        assert isinstance(ei["difficulty"], float)
        assert ei["difficulty"] == 0.25

    def test_passthrough_kwargs(self):
        ei = btp._build_extra_info(answer_type="numerical", **{"from": "MegaScience"})
        assert ei["from"] == "MegaScience"

    def test_corpus_fields(self):
        ei = btp._build_extra_info(
            answer_type="numerical",
            primary_answer_type="numerical",
            domain="ClassicalMechanics",
            domain_coarse="mechanics",
            source="UGPhysics",
        )
        assert ei["primary_answer_type"] == "numerical"
        assert ei["domain_coarse"] == "mechanics"
        assert ei["source"] == "UGPhysics"


# ===========================================================================
# Step 0b: Stratified split
# ===========================================================================

class TestDifficultyBin:
    def test_low(self):
        assert spl.difficulty_bin(0.0) == "low"

    def test_medium(self):
        assert spl.difficulty_bin(0.125) == "medium"
        assert spl.difficulty_bin(0.25) == "medium"
        assert spl.difficulty_bin(0.375) == "medium"

    def test_high(self):
        assert spl.difficulty_bin(0.5) == "high"
        assert spl.difficulty_bin(0.625) == "high"
        assert spl.difficulty_bin(0.75) == "high"


class TestStratifiedSplit:
    @pytest.fixture
    def sample_df(self):
        """Create a small df with 3 strata of known sizes."""
        rng = np.random.RandomState(42)
        rows = []
        for s, n in [("A", 100), ("B", 50), ("C", 10)]:
            for _ in range(n):
                rows.append({"stratum": s, "value": rng.randn()})
        return pd.DataFrame(rows)

    def test_exact_split_sizes(self, sample_df):
        train, dev, test = spl.stratified_split(sample_df, "stratum", 10, 10, seed=42)
        assert len(dev) == 10
        assert len(test) == 10
        assert len(train) == 140  # 160 - 10 - 10

    def test_no_overlap(self, sample_df):
        train, dev, test = spl.stratified_split(sample_df, "stratum", 10, 10, seed=42)
        # Indices are reset, so check by value uniqueness
        all_values = pd.concat([train["value"], dev["value"], test["value"]])
        assert all_values.nunique() == len(sample_df)

    def test_all_rows_preserved(self, sample_df):
        train, dev, test = spl.stratified_split(sample_df, "stratum", 10, 10, seed=42)
        assert len(train) + len(dev) + len(test) == len(sample_df)

    def test_strata_represented_in_splits(self, sample_df):
        train, dev, test = spl.stratified_split(sample_df, "stratum", 10, 10, seed=42)
        # All strata with >= 3 rows should be in dev and test
        for s in ["A", "B"]:
            assert (dev["stratum"] == s).any(), f"Stratum {s} missing from dev"
            assert (test["stratum"] == s).any(), f"Stratum {s} missing from test"

    def test_tiny_strata_go_to_train(self):
        """Strata with < 3 rows should go entirely to train."""
        df = pd.DataFrame({
            "stratum": ["big"] * 100 + ["tiny"] * 2,
            "value": range(102),
        })
        train, dev, test = spl.stratified_split(df, "stratum", 5, 5, seed=42)
        tiny_in_dev = (dev["stratum"] == "tiny").sum()
        tiny_in_test = (test["stratum"] == "tiny").sum()
        assert tiny_in_dev == 0
        assert tiny_in_test == 0

    def test_deterministic(self, sample_df):
        t1, d1, _ = spl.stratified_split(sample_df, "stratum", 10, 10, seed=42)
        t2, d2, _ = spl.stratified_split(sample_df, "stratum", 10, 10, seed=42)
        pd.testing.assert_frame_equal(t1, t2)
        pd.testing.assert_frame_equal(d1, d2)


# ===========================================================================
# Step 1: Stratified probe sampling
# ===========================================================================

class TestStratifiedSample:
    @pytest.fixture
    def sample_df(self):
        rng = np.random.RandomState(42)
        rows = []
        for s, n in [("A", 500), ("B", 200), ("C", 30), ("D", 5)]:
            for _ in range(n):
                rows.append({"stratum": s, "value": rng.randn()})
        return pd.DataFrame(rows)

    def test_target_size(self, sample_df):
        result = sub.stratified_sample(sample_df, "stratum", 100, 10, seed=42)
        assert len(result) == 100

    def test_minimum_floor_respected(self, sample_df):
        result = sub.stratified_sample(sample_df, "stratum", 100, 10, seed=42)
        per_stratum = result["stratum"].value_counts()
        # All strata should have >= min(10, stratum_size)
        for s in per_stratum.index:
            expected_min = min(10, sample_df[sample_df["stratum"] == s].shape[0])
            assert per_stratum[s] >= expected_min, \
                f"Stratum {s}: got {per_stratum[s]}, expected >= {expected_min}"

    def test_no_oversample(self, sample_df):
        result = sub.stratified_sample(sample_df, "stratum", 100, 10, seed=42)
        per_stratum = result["stratum"].value_counts()
        pop_sizes = sample_df["stratum"].value_counts()
        for s in per_stratum.index:
            assert per_stratum[s] <= pop_sizes[s], \
                f"Stratum {s} oversampled: {per_stratum[s]} > {pop_sizes[s]}"

    def test_deterministic(self, sample_df):
        r1 = sub.stratified_sample(sample_df, "stratum", 100, 10, seed=42)
        r2 = sub.stratified_sample(sample_df, "stratum", 100, 10, seed=42)
        pd.testing.assert_frame_equal(r1.reset_index(drop=True), r2.reset_index(drop=True))

    def test_tiny_stratum_fully_included(self):
        """A stratum smaller than the floor should be fully included."""
        df = pd.DataFrame({
            "stratum": ["big"] * 500 + ["tiny"] * 3,
            "value": range(503),
        })
        result = sub.stratified_sample(df, "stratum", 50, 10, seed=42)
        tiny_count = (result["stratum"] == "tiny").sum()
        assert tiny_count == 3  # all 3 rows should be included


class TestDifficultyBinProbe:
    """Test the difficulty_bin in subsample_probe.py handles string input."""

    def test_float_input(self):
        assert sub.difficulty_bin(0.0) == "low"
        assert sub.difficulty_bin(0.25) == "medium"
        assert sub.difficulty_bin(0.5) == "high"

    def test_string_float_input(self):
        assert sub.difficulty_bin("0.0") == "low"
        assert sub.difficulty_bin("0.25") == "medium"
        assert sub.difficulty_bin("0.75") == "high"

    def test_non_numeric_string(self):
        assert sub.difficulty_bin("Undergraduate") == "low"
        assert sub.difficulty_bin("") == "low"


# ===========================================================================
# Integration tests (require rebuilt parquets — mark slow)
# ===========================================================================

@pytest.mark.slow
class TestEnrichedParquets:
    """Verify the enriched parquets have correct metadata."""

    def test_drsci_train_schema(self):
        df = pd.read_parquet("data/processed/drsci_train.parquet")
        assert len(df) > 100_000
        ei = df["extra_info"].iloc[0]
        required_keys = {"answer_type", "difficulty", "from", "problem",
                         "subject", "tolerance", "unit"}
        assert required_keys <= set(ei.keys()), f"Missing keys: {required_keys - set(ei.keys())}"
        assert isinstance(ei["difficulty"], float), "difficulty should be float"

    def test_drsci_no_nan_from(self):
        df = pd.read_parquet("data/processed/drsci_train.parquet")
        froms = df["extra_info"].apply(lambda x: x["from"])
        assert froms.notna().all()
        assert (froms != "").all()

    def test_drsci_no_unknown_answer_type(self):
        df = pd.read_parquet("data/processed/drsci_train.parquet")
        atypes = df["extra_info"].apply(lambda x: x["answer_type"])
        assert (atypes != "unknown").all()

    def test_corpus_train_schema(self):
        df = pd.read_parquet("data/processed/corpus_train.parquet")
        ei = df["extra_info"].iloc[0]
        required_keys = {"answer_type", "primary_answer_type", "domain",
                         "domain_coarse", "source", "problem", "tolerance", "unit"}
        assert required_keys <= set(ei.keys()), f"Missing keys: {required_keys - set(ei.keys())}"

    def test_corpus_primary_answer_type_values(self):
        df = pd.read_parquet("data/processed/corpus_train.parquet")
        valid = {"numerical", "expression", "equation", "mcq", "true_false",
                 "interval", "multi-part"}
        pats = df["extra_info"].apply(lambda x: x["primary_answer_type"]).unique()
        assert set(pats) <= valid, f"Unexpected: {set(pats) - valid}"

    def test_corpus_domain_coarse_values(self):
        df = pd.read_parquet("data/processed/corpus_train.parquet")
        valid = {"mechanics", "em_electro", "quantum_modern", "thermo_stat", "optics", "other"}
        dcs = df["extra_info"].apply(lambda x: x["domain_coarse"]).unique()
        assert set(dcs) <= valid, f"Unexpected: {set(dcs) - valid}"

    def test_corpus_data_source_is_actual_source(self):
        df = pd.read_parquet("data/processed/corpus_train.parquet")
        valid = {"UGPhysics", "PHYSICS", "SciBench_RL", "OlympiadBench", "PHYBench"}
        assert set(df["data_source"].unique()) <= valid


@pytest.mark.slow
class TestSplits:
    def test_drsci_split_sizes(self):
        train = pd.read_parquet("data/processed/drsci_train_split.parquet")
        dev = pd.read_parquet("data/processed/drsci_dev.parquet")
        test = pd.read_parquet("data/processed/drsci_test.parquet")
        assert len(dev) == 2000
        assert len(test) == 2000
        assert len(train) + len(dev) + len(test) == 102_563

    def test_corpus_split_sizes(self):
        train = pd.read_parquet("data/processed/corpus_train_split.parquet")
        dev = pd.read_parquet("data/processed/corpus_dev.parquet")
        test = pd.read_parquet("data/processed/corpus_test.parquet")
        assert len(dev) == 200
        assert len(test) == 200
        assert len(train) + len(dev) + len(test) == 6_817


@pytest.mark.slow
class TestProbeSubset:
    def test_probe_size(self):
        df = pd.read_parquet("data/processed/probe_subset.parquet")
        assert len(df) == 2500

    def test_probe_dataset_distribution(self):
        df = pd.read_parquet("data/processed/probe_subset.parquet")
        assert (df["_dataset"] == "drsci").sum() == 2000
        assert (df["_dataset"] == "corpus").sum() == 500

    def test_probe_has_extra_info(self):
        df = pd.read_parquet("data/processed/probe_subset.parquet")
        for _, row in df.head(5).iterrows():
            assert isinstance(row["extra_info"], dict)
            assert "answer_type" in row["extra_info"]
