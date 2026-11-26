"""
Additional unit tests for MP108 data checks, cleaning pipeline, and utilities.
Uses small in-memory fixtures without external CSV dependencies.
"""

import numpy as np
import pandas as pd
import pytest

from src.mp108.data_checks import (
    check_leaking_column,
    check_spelling_variants,
    check_target_typos,
)
from src.mp108.data_cleaner import clean_dataset
from src.mp108.utils import compute_file_sha256, load_params


def test_check_leaking_column_absent_and_present():
    """Detect missing-from-holdout and high-correlation leaking columns."""
    train_df = pd.DataFrame({
        "temp_c": [20.0, 35.0, 22.0, 31.0, 28.0],
        "stock_lines": [150, 110, 140, 120, 130],
        "units_sold": [10, 40, 25, 15, 35],
    })
    holdout_df = pd.DataFrame({
        "temp_c": [22.0, 27.0],
        "stock_lines": [110, 130],
    })
    res_clean = check_leaking_column(train_df, holdout_df)
    assert res_clean["present"] is False
    assert res_clean["column"] is None

    train_extra = train_df.copy()
    train_extra["post_sale_receipt"] = [100, 400, 250, 150, 350]
    res_missing = check_leaking_column(train_extra, holdout_df)
    assert res_missing["present"] is True
    assert res_missing["column"] == "post_sale_receipt"

    holdout_with_leak = holdout_df.copy()
    holdout_with_leak["post_sale_receipt"] = [150, 250]
    res_corr = check_leaking_column(train_extra, holdout_with_leak, threshold=0.95)
    assert res_corr["present"] is True
    assert res_corr["column"] == "post_sale_receipt"


def test_check_target_typos():
    """Detect isolated 10x target typos while ignoring normal targets."""
    normal_df = pd.DataFrame({"units_sold": [40, 42, 45, 48, 50, 52, 55, 58, 60, 44] * 3})
    assert check_target_typos(normal_df)["present"] is False

    typo_df = pd.DataFrame({"units_sold": [50] * 48 + [500, 520]})
    res = check_target_typos(typo_df)
    assert res["present"] is True
    assert res["count"] == 2


def test_check_spelling_variants():
    """Detect extra misspelled categorical levels compared to clean reference."""
    later_df = pd.DataFrame({
        "shop_type": ["chain", "independent", "hospital_store"],
        "locality": ["Old Town", "New Town", "Cantonment"],
    })
    clean_train = later_df.copy()
    assert check_spelling_variants(clean_train, later_df)["present"] is False

    messy_train = pd.DataFrame({
        "shop_type": ["chain", "Chain", "independent", "hospital_store"],
        "locality": ["Old Town", "New Town", "Cantonment", "Old Town"],
    })
    res = check_spelling_variants(messy_train, later_df)
    assert res["present"] is True
    assert res["column"] == "shop_type"
    assert res["true_levels"] == 3


def test_clean_dataset_stuck_sensor_and_inference_mode():
    """Verify stuck sensor imputation on train and preservation on inference."""
    df = pd.DataFrame({
        "record_id": ["R1", "R2", "R3", "R4"],
        "date": ["2026-03-10", "2026-03-20", "2026-03-20", "2026-04-10"],
        "locality": ["Old Town", "Old Town", "New Town", "Cantonment"],
        "distance_to_hospital_km": [2500.0, 2500.0, 4.0, 3.0],
        "nearby_clinics": [4, 9999, 6, 8],
        "flu_cases_reported": [80, 134, 134, 100],
    })
    original = df.copy()

    cleaned_train = clean_dataset(df, is_train=True)
    # Input DataFrame must not be mutated
    pd.testing.assert_frame_equal(df, original)
    # Stuck period values (134 between 2026-03-15 and 2026-04-04) imputed with non-stuck median (90.0)
    assert cleaned_train.loc[1, "flu_cases_reported"] == pytest.approx(90.0)
    assert cleaned_train.loc[0, "flu_cases_reported"] == 80

    # On inference (is_train=False), duplicates and flu_cases_reported are not dropped/imputed
    dup_df = pd.concat([df, df.iloc[[0]]], ignore_index=True)
    cleaned_later = clean_dataset(dup_df, is_train=False)
    assert len(cleaned_later) == len(dup_df)
    assert cleaned_later.loc[1, "flu_cases_reported"] == 134
    assert cleaned_later.loc[0, "distance_to_hospital_km"] == pytest.approx(2.5)
    assert (cleaned_later["nearby_clinics"] == 9999).sum() == 0


def test_utils_load_params_missing_and_sha256(tmp_path):
    """Verify load_params FileNotFoundError and SHA-256 computation."""
    with pytest.raises(FileNotFoundError):
        load_params(str(tmp_path / "nonexistent.yaml"))

    sample = tmp_path / "sample.txt"
    sample.write_bytes(b"mp108-test")
    digest = compute_file_sha256(str(sample))
    assert len(digest) == 64
    assert digest == compute_file_sha256(str(sample))
