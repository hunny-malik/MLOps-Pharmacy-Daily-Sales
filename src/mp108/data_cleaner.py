"""
MP108 Data Cleaning Module.
Applies corrections for the issues detected in Stage 1:
- Deduplication: remove 26 duplicate rows
- Unit mixup: divide distance_to_hospital_km by 1000 for 'Old Town'
- Sentinel value: replace 9999 in nearby_clinics with median / NaN
- Stuck sensor: handle or flag the stuck period in flu_cases_reported
"""

from pathlib import Path
from typing import Optional
import numpy as np
import pandas as pd
from src.mp108.utils import load_params


def clean_dataset(df: pd.DataFrame, is_train: bool = True) -> pd.DataFrame:
    """Clean dataset according to MP108 problem rules."""
    cleaned = df.copy()

    # 1. Remove duplicate rows if in train
    if is_train:
        cleaned = cleaned.drop_duplicates().reset_index(drop=True)

    # 2. Fix unit mixup in Old Town distance_to_hospital_km (recorded in metres instead of km)
    if "distance_to_hospital_km" in cleaned.columns and "locality" in cleaned.columns:
        mask = (cleaned["locality"] == "Old Town") & (cleaned["distance_to_hospital_km"] > 50)
        cleaned.loc[mask, "distance_to_hospital_km"] = cleaned.loc[mask, "distance_to_hospital_km"] / 1000.0

    # 3. Handle sentinel values (9999 in nearby_clinics)
    if "nearby_clinics" in cleaned.columns:
        valid_clinics = cleaned.loc[cleaned["nearby_clinics"] != 9999, "nearby_clinics"]
        median_val = valid_clinics.median() if len(valid_clinics) > 0 else 5
        cleaned.loc[cleaned["nearby_clinics"] == 9999, "nearby_clinics"] = median_val

    # 4. Handle stuck sensor period in flu_cases_reported if training
    # During 2026-03-15 to 2026-04-04, reading was stuck at 134
    if is_train and "flu_cases_reported" in cleaned.columns and "date" in cleaned.columns:
        stuck_mask = (cleaned["date"] >= "2026-03-15") & (cleaned["date"] <= "2026-04-04")
        # For rows with stuck reading, impute or interpolate based on neighbouring dates
        valid_flu = cleaned.loc[~stuck_mask, "flu_cases_reported"]
        median_flu = valid_flu.median()
        # Alternatively, we can leave or smooth it for regression
        cleaned.loc[stuck_mask & (cleaned["flu_cases_reported"] == 134), "flu_cases_reported"] = median_flu

    return cleaned


def save_cleaned_data(params_path: str = "params.yaml") -> None:
    """Read raw train.csv and later.csv, clean them, and save to processed directory."""
    config = load_params(params_path)
    train_df = pd.read_csv(config["data"]["raw_train_path"])
    later_df = pd.read_csv(config["data"]["raw_later_path"])

    cleaned_train = clean_dataset(train_df, is_train=True)
    cleaned_later = clean_dataset(later_df, is_train=False)

    out_dir = Path("data/processed")
    out_dir.mkdir(parents=True, exist_ok=True)

    cleaned_train.to_csv(config["data"]["cleaned_train_path"], index=False)
    cleaned_later.to_csv(config["data"]["cleaned_later_path"], index=False)
    print(f"Saved cleaned train data ({len(cleaned_train)} rows) to {config['data']['cleaned_train_path']}")
    print(f"Saved cleaned later data ({len(cleaned_later)} rows) to {config['data']['cleaned_later_path']}")


if __name__ == "__main__":
    save_cleaned_data()
