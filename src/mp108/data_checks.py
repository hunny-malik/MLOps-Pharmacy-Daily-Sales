"""
MP108 Data Quality and Anomaly Detection Module.
Detects the 7 kinds of data problems specified in STUDENT_HANDBOOK.md:
1. leaking_column
2. unit_mixup
3. sentinel_value
4. duplicate_rows
5. stuck_sensor
6. target_typos
7. spelling_variants
"""

from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from src.mp108.utils import load_params


def check_leaking_column(
    train_df: pd.DataFrame,
    holdout_df: pd.DataFrame,
    target_col: str = "units_sold",
    threshold: float = 0.95,
) -> Dict[str, Any]:
    """Check for leaking columns missing from holdout or almost identical to target."""
    train_cols = set(train_df.columns)
    holdout_cols = set(holdout_df.columns)
    missing_from_holdout = train_cols - holdout_cols - {target_col}

    numeric_cols = train_df.select_dtypes(include=[np.number]).columns
    high_corr_col = None
    if target_col in train_df.columns:
        corrs = train_df[numeric_cols].corr()[target_col].drop(target_col, errors="ignore")
        for col, corr_val in corrs.items():
            if abs(corr_val) >= threshold:
                high_corr_col = col
                break

    if missing_from_holdout:
        leaking = list(missing_from_holdout)[0]
        return {"present": True, "column": leaking}
    elif high_corr_col:
        return {"present": True, "column": high_corr_col}
    else:
        return {"present": False, "column": None}


def check_unit_mixup(
    train_df: pd.DataFrame,
    group_col: str = "locality",
    candidate_cols: Optional[List[str]] = None,
    ratio_threshold: float = 50.0,
) -> Dict[str, Any]:
    """Check if one group recorded a numeric column in a different unit."""
    if candidate_cols is None:
        candidate_cols = [
            "temp_c",
            "distance_to_hospital_km",
            "stock_lines",
            "discount_pct",
            "flu_cases_reported",
            "nearby_clinics",
        ]

    for col in candidate_cols:
        if col not in train_df.columns:
            continue
        group_means = train_df.groupby(group_col)[col].mean()
        min_mean = group_means.min()
        max_mean = group_means.max()

        if min_mean > 0 and (max_mean / min_mean) > ratio_threshold:
            affected_group = group_means.idxmax()
            ratio = max_mean / min_mean
            # Determine exact standard unit factor (10, 12, 60, 100, 1000)
            standard_factors = [10, 12, 60, 100, 1000]
            factor = min(standard_factors, key=lambda f: abs(f - ratio))
            return {
                "present": True,
                "column": col,
                "group_value": str(affected_group),
                "factor": int(factor),
            }

    return {"present": False, "column": None, "group_value": None, "factor": None}


def check_sentinel_value(
    train_df: pd.DataFrame,
    candidate_cols: Optional[List[str]] = None,
    sentinels: Optional[List[int]] = None,
) -> Dict[str, Any]:
    """Check for impossible sentinel numbers like -1, -999, 999, 9999."""
    if sentinels is None:
        sentinels = [-1, -99, -999, 999, 9999, 99999]

    if candidate_cols is None:
        candidate_cols = train_df.select_dtypes(include=[np.number]).columns.tolist()

    for col in candidate_cols:
        for s in sentinels:
            cnt = int((train_df[col] == s).sum())
            if cnt >= 5:
                return {
                    "present": True,
                    "column": col,
                    "value": int(s),
                    "count": cnt,
                }

    return {"present": False, "column": None, "value": None, "count": None}


def check_duplicate_rows(train_df: pd.DataFrame) -> Dict[str, Any]:
    """Check for duplicate rows in the dataset."""
    extra_copies = int(train_df.duplicated().sum())
    return {
        "present": extra_copies > 0,
        "count": extra_copies if extra_copies > 0 else None,
    }


def check_stuck_sensor(
    train_df: pd.DataFrame,
    date_col: str = "date",
    candidate_cols: Optional[List[str]] = None,
    min_consecutive_days: int = 14,
) -> Dict[str, Any]:
    """Check if a sensor was stuck sending the exact same reading across many dates."""
    if candidate_cols is None:
        candidate_cols = [
            "temp_c",
            "distance_to_hospital_km",
            "stock_lines",
            "discount_pct",
            "flu_cases_reported",
            "nearby_clinics",
        ]

    train_sorted = train_df.sort_values(by=date_col).reset_index(drop=True)

    for col in candidate_cols:
        if col not in train_sorted.columns:
            continue
        # Find daily unique values
        daily_vals = train_sorted.groupby(date_col)[col].unique()
        # Find dates where all records on that date had a single identical reading
        single_val_dates = [dt for dt, vals in daily_vals.items() if len(vals) == 1]

        # Find long runs of identical single values across dates
        current_run_val = None
        current_run_dates: List[str] = []
        longest_run_val = None
        longest_run_dates: List[str] = []

        for dt in single_val_dates:
            val = daily_vals[dt][0]
            if val == current_run_val:
                current_run_dates.append(dt)
            else:
                if len(current_run_dates) > len(longest_run_dates):
                    longest_run_dates = list(current_run_dates)
                    longest_run_val = current_run_val
                current_run_val = val
                current_run_dates = [dt]

        if len(current_run_dates) > len(longest_run_dates):
            longest_run_dates = list(current_run_dates)
            longest_run_val = current_run_val

        if len(longest_run_dates) >= min_consecutive_days:
            return {
                "present": True,
                "column": col,
                "start_date": str(longest_run_dates[0]),
                "end_date": str(longest_run_dates[-1]),
            }

    return {"present": False, "column": None, "start_date": None, "end_date": None}


def check_target_typos(
    train_df: pd.DataFrame,
    target_col: str = "units_sold",
    gap_factor: float = 3.0,
) -> Dict[str, Any]:
    """Check if a few targets were typed 10 times too big (isolated cluster far above)."""
    if target_col not in train_df.columns:
        return {"present": False, "count": None}

    y = train_df[target_col].dropna().values
    q75 = np.percentile(y, 75)
    q25 = np.percentile(y, 25)
    iqr = q75 - q25

    extreme_threshold = q75 + 5 * iqr
    outliers = y[y > extreme_threshold]

    if len(outliers) > 0 and len(outliers) < (0.05 * len(y)):
        return {"present": True, "count": int(len(outliers))}

    return {"present": False, "count": None}


def check_spelling_variants(
    train_df: pd.DataFrame,
    later_df: pd.DataFrame,
    categorical_cols: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Check if categories in train have misspelled or messy variants compared to clean later.csv."""
    if categorical_cols is None:
        categorical_cols = ["shop_type", "locality"]

    for col in categorical_cols:
        if col in train_df.columns and col in later_df.columns:
            train_levels = set(train_df[col].dropna().unique())
            clean_levels = set(later_df[col].dropna().unique())

            # If train has more messy levels than clean reference
            if len(train_levels) > len(clean_levels):
                return {
                    "present": True,
                    "column": col,
                    "true_levels": len(clean_levels),
                }

    return {"present": False, "column": None, "true_levels": None}


def run_all_checks(params_path: str = "params.yaml") -> Dict[str, Any]:
    """Run all 7 checks on train.csv using parameters from params.yaml."""
    config = load_params(params_path)
    train_path = config["data"]["raw_train_path"]
    later_path = config["data"]["raw_later_path"]
    holdout_path = config["data"]["raw_holdout_path"]

    train_df = pd.read_csv(train_path)
    later_df = pd.read_csv(later_path)
    holdout_df = pd.read_csv(holdout_path)

    results = {
        "leaking_column": check_leaking_column(train_df, holdout_df),
        "unit_mixup": check_unit_mixup(train_df),
        "sentinel_value": check_sentinel_value(train_df),
        "duplicate_rows": check_duplicate_rows(train_df),
        "stuck_sensor": check_stuck_sensor(train_df),
        "target_typos": check_target_typos(train_df),
        "spelling_variants": check_spelling_variants(train_df, later_df),
    }

    return results


if __name__ == "__main__":
    print("==================================================")
    print("Running MP108 Data Checks on train.csv")
    print("==================================================")
    findings = run_all_checks()
    for problem, details in findings.items():
        status = "[PRESENT]" if details["present"] else "[ABSENT] "
        print(f"{status} {problem:18}: {details}")
