"""
MP108 Scikit-Learn Pipeline and Feature Engineering Module.
Defines preprocessing and modeling pipelines for Bhopal pharmacy daily sales prediction.
"""

from typing import List
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.base import BaseEstimator

# Exact feature lists according to PROBLEM.md
NUMERIC_FEATURES: List[str] = [
    "flu_cases_reported",
    "temp_c",
    "nearby_clinics",
    "discount_pct",
    "stock_lines",
    "distance_to_hospital_km",
]

CATEGORICAL_FEATURES: List[str] = [
    "shop_type",
    "locality",
]

TARGET_COLUMN: str = "units_sold"


def get_feature_names() -> List[str]:
    """Return list of all feature column names used for training and inference."""
    return NUMERIC_FEATURES + CATEGORICAL_FEATURES


def get_preprocessor() -> ColumnTransformer:
    """Build ColumnTransformer for numeric scaling and categorical one-hot encoding."""
    return ColumnTransformer(
        transformers=[
            (
                "num",
                StandardScaler(),
                NUMERIC_FEATURES,
            ),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                CATEGORICAL_FEATURES,
            ),
        ],
        remainder="drop",
    )


def create_pipeline(model: BaseEstimator) -> Pipeline:
    """Create an end-to-end Pipeline with preprocessing and regressor model."""
    preprocessor = get_preprocessor()
    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("regressor", model),
        ]
    )
