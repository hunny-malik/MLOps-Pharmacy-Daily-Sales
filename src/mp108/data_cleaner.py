"""
MP108 Data Cleaning Module.
Applies corrections for the issues detected in Stage 1:
- Deduplication: remove duplicate rows in train.csv
- Unit mixup: divide distance_to_hospital_km by 1000 for 'Old Town'
- Sentinel value: replace 9999 in nearby_clinics with median
- Stuck sensor: impute the stuck period (2026-03-15 to 2026-04-04) in flu_cases_reported
"""

from pathlib import Path
from typing import Any, Dict, Optional
import numpy as np
import pandas as pd
from src.mp108.utils import load_params


def clean_dataset(
    df: pd.DataFrame,
    is_train: bool = True,
    params: Optional[Dict[str, Any]] = None,
) -> pd.DataFrame:
    """
    Clean dataset according to MP108 problem rules.
    Reads cleaning settings from params.yaml if available, with robust defaults.
    """
    cleaned = df.copy()

    # Load params if not explicitly provided
    if params is None:
        try:
            params = load_params()
        except Exception:
            params = {}

    cleaning_cfg = params.get("cleaning", {})

    # 1. Remove duplicate rows if in train
    if is_train:
        cleaned = cleaned.drop_duplicates().reset_index(drop=True)

    # 2. Fix unit mixup in distance_to_hospital_km (recorded in metres instead of km)
    unit_cfg = cleaning_cfg.get("unit_mixup", {})
    unit_col = unit_cfg.get("column", "distance_to_hospital_km")
    group_val = unit_cfg.get("group_value", "Old Town")
    factor = float(unit_cfg.get("factor", 1000.0))

    if unit_col in cleaned.columns and "locality" in cleaned.columns:
        # Distance > 50 in Old Town is in metres
        mask = (cleaned["locality"] == group_val) & (cleaned[unit_col] > 50.0)
        cleaned.loc[mask, unit_col] = cleaned.loc[mask, unit_col] / factor

    # 3. Handle sentinel values (e.g. 9999 in nearby_clinics)
    sentinel_cfg = cleaning_cfg.get("sentinel_value", {})
    sentinel_col = sentinel_cfg.get("column", "nearby_clinics")
    sentinel_val = sentinel_cfg.get("sentinel", 9999)

    if sentinel_col in cleaned.columns:
        valid_clinics = cleaned.loc[cleaned[sentinel_col] != sentinel_val, sentinel_col]
        median_val = valid_clinics.median() if len(valid_clinics) > 0 else 5
        cleaned.loc[cleaned[sentinel_col] == sentinel_val, sentinel_col] = median_val

    # 4. Handle stuck sensor period in flu_cases_reported if training
    stuck_cfg = cleaning_cfg.get("stuck_sensor", {})
    stuck_col = stuck_cfg.get("column", "flu_cases_reported")
    start_date = stuck_cfg.get("start_date", "2026-03-15")
    end_date = stuck_cfg.get("end_date", "2026-04-04")

    if is_train and stuck_col in cleaned.columns and "date" in cleaned.columns:
        stuck_mask = (cleaned["date"] >= start_date) & (cleaned["date"] <= end_date)
        valid_flu = cleaned.loc[~stuck_mask, stuck_col]
        median_flu = valid_flu.median() if len(valid_flu) > 0 else 100
        # Impute stuck value of 134 during this window
        cleaned.loc[stuck_mask & (cleaned[stuck_col] == 134), stuck_col] = median_flu

    return cleaned


def save_cleaned_data(params_path: str = "params.yaml") -> None:
    """Read raw train.csv and later.csv, clean them, and save to processed directory."""
    config = load_params(params_path)
    train_df = pd.read_csv(config["data"]["raw_train_path"])
    later_df = pd.read_csv(config["data"]["raw_later_path"])

    cleaned_train = clean_dataset(train_df, is_train=True, params=config)
    cleaned_later = clean_dataset(later_df, is_train=False, params=config)

    out_dir = Path("data/processed")
    out_dir.mkdir(parents=True, exist_ok=True)

    cleaned_train.to_csv(config["data"]["cleaned_train_path"], index=False)
    cleaned_later.to_csv(config["data"]["cleaned_later_path"], index=False)
    print(f"Saved cleaned train data ({len(cleaned_train)} rows) to {config['data']['cleaned_train_path']}")
    print(f"Saved cleaned later data ({len(cleaned_later)} rows) to {config['data']['cleaned_later_path']}")


if __name__ == "__main__":
    save_cleaned_data()
