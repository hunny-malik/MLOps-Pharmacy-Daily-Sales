"""
MP108 Model Training and MLflow Experiment Tracking Module.
Tracks experiments in a local SQLite database (sqlite:///mlflow.db),
logs data_sha256 in every run, evaluates MAE across multiple model settings,
and records model pipelines with serialization_format='cloudpickle'.
"""

import hashlib
from pathlib import Path
from typing import Any, Dict, List, Tuple
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

from src.mp108.data_cleaner import clean_dataset
from src.mp108.pipeline import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    TARGET_COLUMN,
    create_pipeline,
)
from src.mp108.utils import compute_file_sha256, load_params


def get_model_experiments(random_state: int = 42) -> List[Dict[str, Any]]:
    """
    Define 4 distinct model configurations to compare.
    Meets the requirement: 'Try 3+ settings'.
    """
    return [
        {
            "name": "Ridge_Baseline",
            "model_type": "Ridge",
            "model": Ridge(alpha=1.0, random_state=random_state),
            "params": {"alpha": 1.0, "model_type": "Ridge"},
        },
        {
            "name": "RandomForest_Default",
            "model_type": "RandomForestRegressor",
            "model": RandomForestRegressor(
                n_estimators=100,
                max_depth=8,
                random_state=random_state,
            ),
            "params": {
                "n_estimators": 100,
                "max_depth": 8,
                "model_type": "RandomForestRegressor",
            },
        },
        {
            "name": "GradientBoosting_v1",
            "model_type": "GradientBoostingRegressor",
            "model": GradientBoostingRegressor(
                n_estimators=120,
                learning_rate=0.08,
                max_depth=4,
                random_state=random_state,
            ),
            "params": {
                "n_estimators": 120,
                "learning_rate": 0.08,
                "max_depth": 4,
                "model_type": "GradientBoostingRegressor",
            },
        },
        {
            "name": "RandomForest_Tuned",
            "model_type": "RandomForestRegressor",
            "model": RandomForestRegressor(
                n_estimators=150,
                max_depth=12,
                min_samples_split=4,
                random_state=random_state,
            ),
            "params": {
                "n_estimators": 150,
                "max_depth": 12,
                "min_samples_split": 4,
                "model_type": "RandomForestRegressor",
            },
        },
    ]


def run_training(params_path: str = "params.yaml") -> List[Dict[str, Any]]:
    """
    Execute training and MLflow tracking on cleaned train.csv.
    Logs data_sha256, hyperparams, and MAE metrics into sqlite:///mlflow.db.
    """
    config = load_params(params_path)
    train_path = config["data"]["raw_train_path"]
    tracking_uri = config["tracking"]["tracking_uri"]
    experiment_name = config["tracking"]["experiment_name"]
    random_state = config.get("base", {}).get("random_state", 42)
    test_size = config.get("model", {}).get("test_size", 0.2)

    # 1. Compute SHA-256 of raw train.csv (MANDATORY per STUDENT_HANDBOOK.md)
    if not Path(train_path).exists():
        raise FileNotFoundError(
            f"Raw training file not found at '{train_path}'. "
            "Please ensure train.csv is in data/raw/."
        )

    data_sha256 = compute_file_sha256(train_path)
    print(f"[MLflow] Computed data_sha256: {data_sha256}")

    # 2. Load and clean raw train dataset
    raw_df = pd.read_csv(train_path)
    cleaned_df = clean_dataset(raw_df, is_train=True, params=config)
    print(f"[Data] Raw rows: {len(raw_df)}, Cleaned rows: {len(cleaned_df)}")

    # 3. Prepare feature matrices and target vector
    feature_cols = NUMERIC_FEATURES + CATEGORICAL_FEATURES
    X = cleaned_df[feature_cols]
    y = cleaned_df[TARGET_COLUMN]

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    # 4. Set up MLflow SQLite Tracking
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)
    print(f"[MLflow] Tracking to: {tracking_uri}")
    print(f"[MLflow] Experiment: {experiment_name}")

    experiments = get_model_experiments(random_state=random_state)
    run_results: List[Dict[str, Any]] = []

    print("\n" + "=" * 70)
    print("Starting MLflow Model Training Experiments")
    print("=" * 70)

    for exp in experiments:
        exp_name = exp["name"]
        with mlflow.start_run(run_name=exp_name) as run:
            run_id = run.info.run_id
            print(f"\n---> Training: {exp_name} (Run ID: {run_id})")

            # A. Log data fingerprint (MANDATORY)
            mlflow.log_param("data_sha256", data_sha256)

            # B. Log training split settings and model hyperparameters
            mlflow.log_param("test_size", test_size)
            mlflow.log_param("random_state", random_state)
            mlflow.log_params(exp["params"])

            # C. Create and fit end-to-end pipeline
            pipeline = create_pipeline(exp["model"])
            pipeline.fit(X_train, y_train)

            # D. Evaluate MAE metrics
            train_preds = pipeline.predict(X_train)
            val_preds = pipeline.predict(X_val)

            train_mae = float(mean_absolute_error(y_train, train_preds))
            val_mae = float(mean_absolute_error(y_val, val_preds))

            # Metric name MUST contain 'mae'
            mlflow.log_metric("train_mae", train_mae)
            mlflow.log_metric("val_mae", val_mae)
            print(f"     Train MAE: {train_mae:.4f} | Validation MAE: {val_mae:.4f}")

            # E. Log pipeline model artifact with cloudpickle
            mlflow.sklearn.log_model(
                sk_model=pipeline,
                artifact_path="model",
                serialization_format="cloudpickle",
            )

            run_results.append(
                {
                    "name": exp_name,
                    "run_id": run_id,
                    "train_mae": train_mae,
                    "val_mae": val_mae,
                    "pipeline": pipeline,
                }
            )

    # Print summary table
    print("\n" + "=" * 70)
    print("MLflow Tracking Experiments Summary")
    print("=" * 70)
    print(f"{'Run Name':<25} {'Run ID':<34} {'Val MAE':<10}")
    print("-" * 70)
    for res in run_results:
        print(f"{res['name']:<25} {res['run_id']:<34} {res['val_mae']:<10.4f}")

    best_run = min(run_results, key=lambda r: r["val_mae"])
    print("-" * 70)
    print(f"BEST RUN: {best_run['name']} (ID: {best_run['run_id']}) with Val MAE: {best_run['val_mae']:.4f}")
    print("=" * 70)

    return run_results


if __name__ == "__main__":
    run_training()
