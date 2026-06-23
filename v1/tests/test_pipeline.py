"""
Unit tests for the ETL pipeline (Bronze → Silver → Gold).
Uses local Parquet files — no network calls required.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
import numpy as np
import pandas as pd

from pipeline.transform import (
    run_silver,
    run_gold,
    _normalize_phase,
    PHASE_MAP,
)


# ─── Test data ────────────────────────────────────────────────────────────────

def _make_bronze_df(n: int = 10) -> pd.DataFrame:
    """Return a minimal Bronze-like DataFrame for testing."""
    phases_cycle = ["PHASE1", "PHASE2", "PHASE3", "PHASE4", "NA"]
    phases = [phases_cycle[i % len(phases_cycle)] for i in range(n)]
    return pd.DataFrame({
        "nctId": [f"NCT{i:08d}" for i in range(n)],
        "briefTitle": [f"Trial {i}" for i in range(n)],
        "officialTitle": [f"Official Trial {i}" for i in range(n)],
        "overallStatus": ["RECRUITING"] * n,
        "phase": phases,
        "startDate": ["2020-01"] * (n - 2) + [None, "1999-01"],
        "completionDate": ["2023-06"] * n,
        "condition": ["Cancer"] * n,
        "searchCondition": ["cancer"] * n,
        "intervention": ["Drug A"] * n,
        "sponsorName": [f"Sponsor_{i % 3}" for i in range(n)],
        "briefSummary": [f"Summary {i}" for i in range(n)],
        "eligibilityCriteria": ["Inclusion: adults only"] * n,
    })


# ─── Phase normalisation ──────────────────────────────────────────────────────

class TestNormalizePhase:
    def test_phase1_maps_correctly(self):
        assert _normalize_phase("PHASE1") == "Phase 1"

    def test_phase2_maps_correctly(self):
        assert _normalize_phase("PHASE2") == "Phase 2"

    def test_early_phase1_becomes_phase1(self):
        assert _normalize_phase("EARLY_PHASE1") == "Phase 1"

    def test_na_maps_to_na(self):
        assert _normalize_phase("NA") == "N/A"

    def test_none_maps_to_na(self):
        assert _normalize_phase(None) == "N/A"

    def test_nan_maps_to_na(self):
        assert _normalize_phase(float("nan")) == "N/A"

    def test_highest_phase_wins_for_multi(self):
        assert _normalize_phase("PHASE1; PHASE3") == "Phase 3"

    def test_unknown_maps_to_na(self):
        assert _normalize_phase("UNKNOWN") == "N/A"


# ─── Silver layer ─────────────────────────────────────────────────────────────

class TestSilverLayer:
    def test_returns_dataframe(self):
        df = _make_bronze_df()
        silver = run_silver(df)
        assert isinstance(silver, pd.DataFrame)

    def test_drops_null_nct_id(self):
        df = _make_bronze_df(10)
        df.loc[0, "nctId"] = None
        silver = run_silver(df)
        assert "NCT00000000" not in silver["nctId"].values

    def test_drops_null_brief_title(self):
        df = _make_bronze_df(10)
        df.loc[1, "briefTitle"] = None
        silver = run_silver(df)
        # 1 null title + 1 out-of-range 1999 date → at most 8 remaining
        assert len(silver) <= 8

    def test_drops_out_of_range_dates(self):
        df = _make_bronze_df(10)
        silver = run_silver(df)
        # Row with startDate=1999-01 should be dropped
        valid = silver.dropna(subset=["startDate"])
        assert all(valid["startDate"].dt.year >= 2000)

    def test_phase_normalised(self):
        df = _make_bronze_df(10)
        silver = run_silver(df)
        valid_phases = {"Phase 1", "Phase 2", "Phase 3", "Phase 4", "N/A"}
        assert set(silver["phase"].unique()).issubset(valid_phases)

    def test_sponsor_filled(self):
        df = _make_bronze_df(10)
        df.loc[0, "sponsorName"] = None
        silver = run_silver(df)
        assert silver["sponsorName"].notna().all()

    def test_output_smaller_than_input_with_bad_rows(self):
        df = _make_bronze_df(10)
        df.loc[0, "nctId"] = None
        silver = run_silver(df)
        assert len(silver) < len(df)


# ─── Gold layer ───────────────────────────────────────────────────────────────

class TestGoldLayer:
    def _get_gold(self) -> dict:
        df = _make_bronze_df(20)
        silver = run_silver(df)
        return run_gold(silver)

    def test_returns_all_four_tables(self):
        gold = self._get_gold()
        assert set(gold.keys()) == {"fact_trials", "dim_condition", "dim_sponsor", "dim_phase"}

    def test_fact_trials_has_required_columns(self):
        gold = self._get_gold()
        required = ["nct_id", "brief_title", "phase_id", "condition_id", "sponsor_id", "is_single_sponsor"]
        for col in required:
            assert col in gold["fact_trials"].columns, f"Missing column: {col}"

    def test_no_duplicate_nct_ids(self):
        gold = self._get_gold()
        assert gold["fact_trials"]["nct_id"].nunique() == len(gold["fact_trials"])

    def test_referential_integrity_phase(self):
        gold = self._get_gold()
        fact_phase_ids = set(gold["fact_trials"]["phase_id"].dropna().astype(int))
        dim_phase_ids = set(gold["dim_phase"]["phase_id"].astype(int))
        assert fact_phase_ids.issubset(dim_phase_ids)

    def test_referential_integrity_condition(self):
        gold = self._get_gold()
        fact_ids = set(gold["fact_trials"]["condition_id"].dropna().astype(int))
        dim_ids = set(gold["dim_condition"]["condition_id"].astype(int))
        assert fact_ids.issubset(dim_ids)

    def test_referential_integrity_sponsor(self):
        gold = self._get_gold()
        fact_ids = set(gold["fact_trials"]["sponsor_id"].dropna().astype(int))
        dim_ids = set(gold["dim_sponsor"]["sponsor_id"].astype(int))
        assert fact_ids.issubset(dim_ids)

    def test_is_single_sponsor_no_nulls(self):
        gold = self._get_gold()
        assert gold["fact_trials"]["is_single_sponsor"].notna().all()

    def test_is_single_sponsor_is_bool(self):
        gold = self._get_gold()
        assert gold["fact_trials"]["is_single_sponsor"].dtype == bool


# ─── Bronze layer integration check ──────────────────────────────────────────

class TestBronzeParquetFiles:
    """Verify that real Bronze Parquet files were written by extract.py."""

    def test_bronze_files_exist(self):
        bronze_dir = Path(__file__).parent.parent / "data" / "bronze"
        files = list(bronze_dir.glob("*.parquet"))
        assert len(files) > 0, "No Bronze Parquet files found — run extract.py first"

    def test_bronze_has_required_columns(self):
        bronze_dir = Path(__file__).parent.parent / "data" / "bronze"
        files = sorted(bronze_dir.glob("*.parquet"))
        df = pd.read_parquet(files[-1])
        required = ["nctId", "briefTitle", "condition", "phase", "sponsorName"]
        for col in required:
            assert col in df.columns

    def test_bronze_record_count_nonzero(self):
        bronze_dir = Path(__file__).parent.parent / "data" / "bronze"
        files = sorted(bronze_dir.glob("*.parquet"))
        df = pd.read_parquet(files[-1])
        assert len(df) > 0

    def test_bronze_no_all_null_nct_ids(self):
        bronze_dir = Path(__file__).parent.parent / "data" / "bronze"
        files = sorted(bronze_dir.glob("*.parquet"))
        df = pd.read_parquet(files[-1])
        assert df["nctId"].notna().any()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
