"""
Tests for MP108 Data Checks and Cleaning.
Note: As required by STUDENT_HANDBOOK.md, tests do not depend on external CSV files
and construct small in-memory data fixtures.
"""

import numpy as np
import pandas as pd
import pytest
from src.mp108.data_checks import (
    check_duplicate_rows,
    check_leaking_column,
    check_sentinel_value,
    check_spelling_variants,
    check_stuck_sensor,
    check_target_typos,
    check_unit_mixup,
)
from src.mp108.data_cleaner import clean_dataset
from src.mp108.utils import load_params


def test_params_loading():
    """Test that params.yaml loads correctly and has required keys."""
    params = load_params()
    assert "data" in params
    assert "raw_train_path" in params["data"]
    assert "cleaning" in params
    assert "service" in params


def test_check_duplicate_rows():
    """Test duplicate detection on synthetic data."""
    df_clean = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
    res_clean = check_duplicate_rows(df_clean)
    assert res_clean["present"] is False
    assert res_clean["count"] is None

    df_dups = pd.DataFrame({"a": [1, 2, 2, 3, 3, 3], "b": ["x", "y", "y", "z", "z", "z"]})
    res_dups = check_duplicate_rows(df_dups)
    assert res_dups["present"] is True
    assert res_dups["count"] == 3


def test_check_sentinel_value():
    """Test sentinel value detection on synthetic data."""
    df = pd.DataFrame({"nearby_clinics": [5, 4, 3, 9999] * 10})
    res = check_sentinel_value(df, candidate_cols=["nearby_clinics"])
    assert res["present"] is True
    assert res["column"] == "nearby_clinics"
    assert res["value"] == 9999
    assert res["count"] == 10


def test_check_unit_mixup():
    """Test unit mixup detection across categorical groups."""
    df = pd.DataFrame({
        "locality": ["New Town"] * 20 + ["Old Town"] * 20,
        "distance_to_hospital_km": [5.0] * 20 + [5000.0] * 20,
    })
    res = check_unit_mixup(df, group_col="locality", candidate_cols=["distance_to_hospital_km"])
    assert res["present"] is True
    assert res["column"] == "distance_to_hospital_km"
    assert res["group_value"] == "Old Town"
    assert res["factor"] == 1000


def test_check_stuck_sensor():
    """Test stuck sensor detection over a consecutive date range."""
    dates = pd.date_range("2026-03-01", periods=30, freq="D").strftime("%Y-%m-%d").tolist()
    # 20 consecutive days with stuck reading 134
    readings = [10] * 5 + [134] * 20 + [25] * 5
    df = pd.DataFrame({
        "date": dates,
        "flu_cases_reported": readings,
    })
    res = check_stuck_sensor(df, date_col="date", candidate_cols=["flu_cases_reported"], min_consecutive_days=14)
    assert res["present"] is True
    assert res["column"] == "flu_cases_reported"
    assert res["start_date"] == "2026-03-06"
    assert res["end_date"] == "2026-03-25"


def test_cleaning_fixes_issues():
    """Test that clean_dataset resolves duplicates and unit mixup."""
    df = pd.DataFrame({
        "record_id": ["R1", "R1", "R2"],
        "date": ["2026-01-01", "2026-01-01", "2026-01-02"],
        "locality": ["Old Town", "Old Town", "New Town"],
        "distance_to_hospital_km": [5000.0, 5000.0, 5.0],
        "nearby_clinics": [9999, 9999, 4],
        "flu_cases_reported": [100, 100, 100],
    })
    cleaned = clean_dataset(df, is_train=True)
    assert len(cleaned) == 2
    assert cleaned.loc[cleaned["locality"] == "Old Town", "distance_to_hospital_km"].iloc[0] == 5.0
    assert (cleaned["nearby_clinics"] == 9999).sum() == 0
