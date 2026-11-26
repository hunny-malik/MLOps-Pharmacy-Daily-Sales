"""
MP108 MLflow Model Registry Module.
Selects the best tracked run from MLflow, registers the model version,
manages the @champion alias, and loads/exports the champion model.
"""

import argparse
from pathlib import Path
from typing import Dict, Optional, Tuple

import joblib
import mlflow
import mlflow.sklearn
from mlflow.entities import Run
from mlflow.entities.model_registry import ModelVersion
from mlflow.tracking import MlflowClient

from src.mp108.utils import load_params


def _resolve_tracking_config(
    params_path: str = "params.yaml",
    tracking_uri: Optional[str] = None,
    experiment_name: Optional[str] = None,
    registered_model_name: Optional[str] = None,
) -> Tuple[str, str, str]:
    """Resolve tracking URI, experiment name, and registered model name from params.yaml."""
    config = load_params(params_path)
    tracking_cfg = config.get("tracking", {})

    resolved_uri = tracking_uri or tracking_cfg["tracking_uri"]
    resolved_exp = experiment_name or tracking_cfg["experiment_name"]
    resolved_model = registered_model_name or tracking_cfg["registered_model_name"]
    return resolved_uri, resolved_exp, resolved_model


def get_mlflow_client(
    tracking_uri: Optional[str] = None,
    params_path: str = "params.yaml",
) -> MlflowClient:
    """Configure MLflow tracking URI and return an MlflowClient instance."""
    resolved_uri, _, _ = _resolve_tracking_config(
        params_path=params_path,
        tracking_uri=tracking_uri,
    )
    mlflow.set_tracking_uri(resolved_uri)
    return MlflowClient(tracking_uri=resolved_uri)


def get_best_run(
    experiment_name: Optional[str] = None,
    metric_name: str = "mae",
    higher_is_better: bool = False,
    tracking_uri: Optional[str] = None,
    params_path: str = "params.yaml",
) -> Run:
    """Return the finished MLflow run with the best value for metric_name."""
    resolved_uri, resolved_exp, _ = _resolve_tracking_config(
        params_path=params_path,
        tracking_uri=tracking_uri,
        experiment_name=experiment_name,
    )
    client = get_mlflow_client(tracking_uri=resolved_uri, params_path=params_path)

    experiment = client.get_experiment_by_name(resolved_exp)
    if experiment is None:
        raise ValueError(f"MLflow experiment '{resolved_exp}' not found at '{resolved_uri}'.")

    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string="attributes.status = 'FINISHED'",
        max_results=1000,
    )
    eligible = [run for run in runs if metric_name in run.data.metrics]
    if not eligible:
        raise ValueError(
            f"No finished runs with metric '{metric_name}' found in experiment '{resolved_exp}'."
        )

    selector = max if higher_is_better else min
    return selector(eligible, key=lambda r: r.data.metrics[metric_name])


def register_model_version(
    run_id: str,
    registered_model_name: Optional[str] = None,
    artifact_path: str = "model",
    tracking_uri: Optional[str] = None,
    params_path: str = "params.yaml",
    tags: Optional[Dict[str, str]] = None,
) -> ModelVersion:
    """Register a model artifact from a run in the MLflow Model Registry."""
    resolved_uri, _, resolved_model = _resolve_tracking_config(
        params_path=params_path,
        tracking_uri=tracking_uri,
        registered_model_name=registered_model_name,
    )
    client = get_mlflow_client(tracking_uri=resolved_uri, params_path=params_path)

    model_uri = f"runs:/{run_id}/{artifact_path}"
    version = mlflow.register_model(model_uri=model_uri, name=resolved_model)

    if tags:
        for key, value in tags.items():
            client.set_model_version_tag(
                name=resolved_model,
                version=str(version.version),
                key=key,
                value=str(value),
            )

    return client.get_model_version(name=resolved_model, version=str(version.version))


def promote_to_alias(
    version: str,
    alias: str = "champion",
    registered_model_name: Optional[str] = None,
    tracking_uri: Optional[str] = None,
    params_path: str = "params.yaml",
) -> ModelVersion:
    """Assign an alias (default: 'champion') to a registered model version."""
    resolved_uri, _, resolved_model = _resolve_tracking_config(
        params_path=params_path,
        tracking_uri=tracking_uri,
        registered_model_name=registered_model_name,
    )
    client = get_mlflow_client(tracking_uri=resolved_uri, params_path=params_path)
    client.set_registered_model_alias(
        name=resolved_model,
        alias=alias,
        version=str(version),
    )
    return client.get_model_version_by_alias(name=resolved_model, alias=alias)


def get_model_version_by_alias(
    alias: str = "champion",
    registered_model_name: Optional[str] = None,
    tracking_uri: Optional[str] = None,
    params_path: str = "params.yaml",
) -> ModelVersion:
    """Fetch metadata for the model version currently assigned to the given alias."""
    resolved_uri, _, resolved_model = _resolve_tracking_config(
        params_path=params_path,
        tracking_uri=tracking_uri,
        registered_model_name=registered_model_name,
    )
    client = get_mlflow_client(tracking_uri=resolved_uri, params_path=params_path)
    return client.get_model_version_by_alias(name=resolved_model, alias=alias)


def register_best_model(
    experiment_name: Optional[str] = None,
    registered_model_name: Optional[str] = None,
    metric_name: str = "mae",
    higher_is_better: bool = False,
    artifact_path: str = "model",
    alias: str = "champion",
    tracking_uri: Optional[str] = None,
    params_path: str = "params.yaml",
) -> ModelVersion:
    """Select the best run in an experiment, register it, and assign the champion alias."""
    best_run = get_best_run(
        experiment_name=experiment_name,
        metric_name=metric_name,
        higher_is_better=higher_is_better,
        tracking_uri=tracking_uri,
        params_path=params_path,
    )
    version = register_model_version(
        run_id=best_run.info.run_id,
        registered_model_name=registered_model_name,
        artifact_path=artifact_path,
        tracking_uri=tracking_uri,
        params_path=params_path,
        tags={"selection_metric": metric_name, "selection_value": str(best_run.data.metrics[metric_name])},
    )
    return promote_to_alias(
        version=version.version,
        alias=alias,
        registered_model_name=registered_model_name,
        tracking_uri=tracking_uri,
        params_path=params_path,
    )


def load_model_by_alias(
    alias: str = "champion",
    registered_model_name: Optional[str] = None,
    tracking_uri: Optional[str] = None,
    params_path: str = "params.yaml",
):
    """Load a scikit-learn model from the MLflow registry using its alias."""
    resolved_uri, _, resolved_model = _resolve_tracking_config(
        params_path=params_path,
        tracking_uri=tracking_uri,
        registered_model_name=registered_model_name,
    )
    mlflow.set_tracking_uri(resolved_uri)
    model_uri = f"models:/{resolved_model}@{alias}"
    return mlflow.sklearn.load_model(model_uri)


def export_champion_joblib(
    output_path: str = "models/model.joblib",
    alias: str = "champion",
    registered_model_name: Optional[str] = None,
    tracking_uri: Optional[str] = None,
    params_path: str = "params.yaml",
) -> Path:
    """Load the aliased model from MLflow registry and serialize it with joblib."""
    model = load_model_by_alias(
        alias=alias,
        registered_model_name=registered_model_name,
        tracking_uri=tracking_uri,
        params_path=params_path,
    )
    dest = Path(output_path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, dest)
    return dest


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Register and promote MP108 MLflow model.")
    parser.add_argument("--params", default="params.yaml", help="Path to params.yaml")
    parser.add_argument("--run-id", default=None, help="Specific MLflow run ID to register")
    parser.add_argument("--metric", default="mae", help="Metric name used to rank runs")
    parser.add_argument("--higher-is-better", action="store_true", help="Maximize metric instead of minimizing")
    parser.add_argument("--artifact-path", default="model", help="Logged model artifact path inside the run")
    parser.add_argument("--alias", default="champion", help="Registry alias to assign")
    parser.add_argument("--export-joblib", default=None, help="Optional output path for joblib export")
    return parser.parse_args()


def main() -> None:
    """CLI entry point for registering and promoting the champion model."""
    args = _parse_args()
    if args.run_id:
        version = register_model_version(
            run_id=args.run_id,
            artifact_path=args.artifact_path,
            params_path=args.params,
        )
        promoted = promote_to_alias(
            version=version.version,
            alias=args.alias,
            params_path=args.params,
        )
    else:
        promoted = register_best_model(
            metric_name=args.metric,
            higher_is_better=args.higher_is_better,
            artifact_path=args.artifact_path,
            alias=args.alias,
            params_path=args.params,
        )

    print(
        f"Registered '{promoted.name}' version {promoted.version} "
        f"(run_id={promoted.run_id}) with alias @{args.alias}"
    )

    if args.export_joblib:
        exported = export_champion_joblib(
            output_path=args.export_joblib,
            alias=args.alias,
            params_path=args.params,
        )
        print(f"Exported @{args.alias} model to {exported}")


if __name__ == "__main__":
    main()
