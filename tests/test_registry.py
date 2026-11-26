"""
Unit tests for MP108 MLflow Model Registry workflow.
Uses an isolated temporary SQLite MLflow backend and artifact directory.
"""

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pytest
from sklearn.dummy import DummyRegressor

from src.mp108.registry import (
    export_champion_joblib,
    get_best_run,
    get_model_version_by_alias,
    load_model_by_alias,
    promote_to_alias,
    register_best_model,
    register_model_version,
)


def _setup_mlflow_env(tmp_path):
    """Create isolated SQLite tracking URI, artifact root, and experiment."""
    db_path = tmp_path / "mlflow.db"
    artifact_dir = tmp_path / "mlartifacts"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    tracking_uri = f"sqlite:///{db_path}"
    exp_name = "test_mp108_pharmacy"
    model_name = "test-mp108-predictor"

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.create_experiment(exp_name, artifact_location=artifact_dir.as_uri())
    mlflow.set_experiment(exp_name)
    return tracking_uri, exp_name, model_name


def _log_dummy_run(strategy: str, constant: float, mae: float) -> str:
    """Train and log a small DummyRegressor run to the active MLflow experiment."""
    X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    y = np.array([10.0, 20.0, 30.0])
    model = DummyRegressor(strategy=strategy, constant=constant)
    model.fit(X, y)

    with mlflow.start_run() as run:
        mlflow.log_param("strategy", strategy)
        mlflow.log_metric("mae", mae)
        mlflow.sklearn.log_model(model, artifact_path="model")
        return run.info.run_id


def test_get_best_run_selects_lowest_mae(tmp_path):
    """Verify get_best_run returns the run with the lowest MAE."""
    tracking_uri, exp_name, _ = _setup_mlflow_env(tmp_path)
    _log_dummy_run("constant", 15.0, mae=8.5)
    best_run_id = _log_dummy_run("constant", 20.0, mae=3.2)
    _log_dummy_run("constant", 25.0, mae=5.1)

    best = get_best_run(
        experiment_name=exp_name,
        metric_name="mae",
        tracking_uri=tracking_uri,
    )
    assert best.info.run_id == best_run_id
    assert best.data.metrics["mae"] == pytest.approx(3.2)


def test_get_best_run_missing_experiment_or_metric(tmp_path):
    """Verify ValueError when experiment or target metric does not exist."""
    tracking_uri, exp_name, _ = _setup_mlflow_env(tmp_path)
    with pytest.raises(ValueError, match="not found"):
        get_best_run(experiment_name="missing_exp", tracking_uri=tracking_uri)

    _log_dummy_run("mean", 0.0, mae=4.0)
    with pytest.raises(ValueError, match="metric 'rmse'"):
        get_best_run(experiment_name=exp_name, metric_name="rmse", tracking_uri=tracking_uri)


def test_register_best_model_and_champion_alias(tmp_path):
    """Register best run, assign @champion alias, and load model for inference."""
    tracking_uri, exp_name, model_name = _setup_mlflow_env(tmp_path)
    _log_dummy_run("constant", 12.0, mae=9.4)
    best_run_id = _log_dummy_run("constant", 42.0, mae=2.1)

    mv = register_best_model(
        experiment_name=exp_name,
        registered_model_name=model_name,
        metric_name="mae",
        alias="champion",
        tracking_uri=tracking_uri,
    )
    assert mv.run_id == best_run_id
    assert str(mv.version) == "1"

    champ = get_model_version_by_alias(
        alias="champion",
        registered_model_name=model_name,
        tracking_uri=tracking_uri,
    )
    assert champ.version == mv.version
    assert champ.run_id == best_run_id

    loaded = load_model_by_alias(
        alias="champion",
        registered_model_name=model_name,
        tracking_uri=tracking_uri,
    )
    preds = loaded.predict(np.array([[1.0, 2.0], [3.0, 4.0]]))
    np.testing.assert_allclose(preds, [42.0, 42.0])


def test_promote_to_alias_updates_champion_version(tmp_path):
    """Promote a newer model version and verify @champion alias moves cleanly."""
    tracking_uri, exp_name, model_name = _setup_mlflow_env(tmp_path)
    run_v1 = _log_dummy_run("constant", 10.0, mae=5.0)
    mv1 = register_model_version(
        run_id=run_v1,
        registered_model_name=model_name,
        tracking_uri=tracking_uri,
    )
    promote_to_alias(
        version=mv1.version,
        alias="champion",
        registered_model_name=model_name,
        tracking_uri=tracking_uri,
    )

    run_v2 = _log_dummy_run("constant", 99.0, mae=1.5)
    mv2 = register_model_version(
        run_id=run_v2,
        registered_model_name=model_name,
        tracking_uri=tracking_uri,
    )
    promoted = promote_to_alias(
        version=mv2.version,
        alias="champion",
        registered_model_name=model_name,
        tracking_uri=tracking_uri,
    )
    assert str(promoted.version) == "2"
    assert promoted.run_id == run_v2

    loaded = load_model_by_alias(
        alias="champion",
        registered_model_name=model_name,
        tracking_uri=tracking_uri,
    )
    preds = loaded.predict(np.array([[0.0, 0.0]]))
    np.testing.assert_allclose(preds, [99.0])


def test_export_champion_joblib(tmp_path):
    """Export @champion model to disk with joblib and verify predictions."""
    tracking_uri, exp_name, model_name = _setup_mlflow_env(tmp_path)
    _log_dummy_run("constant", 77.0, mae=1.2)
    register_best_model(
        experiment_name=exp_name,
        registered_model_name=model_name,
        metric_name="mae",
        alias="champion",
        tracking_uri=tracking_uri,
    )

    out_file = tmp_path / "models" / "model.joblib"
    saved_path = export_champion_joblib(
        output_path=str(out_file),
        alias="champion",
        registered_model_name=model_name,
        tracking_uri=tracking_uri,
    )
    assert saved_path.exists()
    model = joblib.load(saved_path)
    preds = model.predict(np.array([[1.0, 1.0]]))
    np.testing.assert_allclose(preds, [77.0])
