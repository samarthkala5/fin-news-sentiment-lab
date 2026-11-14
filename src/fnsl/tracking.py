"""Optional MLflow experiment tracking.

MLflow is a convenience, not a requirement: if it is not installed, or tracking is
disabled (smoke runs, --no-mlflow), every call becomes a no-op and scripts still work.

MLflow 3.x stores runs in a SQLite database by default (the old ./mlruns file store is
in maintenance mode), so both the database and the artifacts live under mlruns/,
which is gitignored. Browse runs with:

    mlflow ui --backend-store-uri sqlite:///mlruns/mlflow.db
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from fnsl import config

EXPERIMENT_NAME = "fin-news-sentiment-lab"


def tracking_uri() -> str:
    """SQLite store inside mlruns/ (absolute path, so it works from any folder)."""
    return "sqlite:///" + (config.MLRUNS_DIR / "mlflow.db").as_posix()


def _flatten(params: dict[str, Any], prefix: str = "") -> dict[str, str]:
    """MLflow params are flat strings: {'a': {'b': 1}} -> {'a.b': '1'}."""
    flat: dict[str, str] = {}
    for key, value in params.items():
        name = f"{prefix}{key}"
        if isinstance(value, dict):
            flat.update(_flatten(value, prefix=f"{name}."))
        else:
            flat[name] = str(value)[:500]
    return flat


class _NoOpRun:
    """Stands in for an MLflow run when tracking is off."""

    active = False

    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None:
        pass

    def log_artifact(self, path: Path) -> None:
        pass


class _MlflowRun:
    active = True

    def __init__(self, mlflow_module: Any) -> None:
        self._mlflow = mlflow_module

    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None:
        clean = {k: float(v) for k, v in metrics.items() if v is not None}
        self._mlflow.log_metrics(clean, step=step)

    def log_artifact(self, path: Path) -> None:
        if Path(path).exists():
            self._mlflow.log_artifact(str(path))


@contextmanager
def mlflow_run(
    run_name: str,
    params: dict[str, Any] | None = None,
    enabled: bool = True,
    tags: dict[str, str] | None = None,
) -> Iterator[_NoOpRun | _MlflowRun]:
    """Open an MLflow run (or a no-op stand-in) and log `params` to it.

    Usage:
        with mlflow_run("tfidf_logreg", params) as run:
            run.log_metrics({"val_macro_f1": 0.71})
    """
    if not enabled:
        yield _NoOpRun()
        return
    try:
        os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")  # silence an import banner
        import mlflow

        config.MLRUNS_DIR.mkdir(parents=True, exist_ok=True)
        mlflow.set_tracking_uri(tracking_uri())
        if mlflow.get_experiment_by_name(EXPERIMENT_NAME) is None:
            mlflow.create_experiment(
                EXPERIMENT_NAME, artifact_location=(config.MLRUNS_DIR / "artifacts").as_uri()
            )
        mlflow.set_experiment(EXPERIMENT_NAME)
    except Exception as err:  # tracking must never break a training run
        print(f"[mlflow] tracking disabled: {type(err).__name__}: {err}")
        yield _NoOpRun()
        return

    with mlflow.start_run(run_name=run_name):
        if params:
            mlflow.log_params(_flatten(params))
        if tags:
            mlflow.set_tags(tags)
        yield _MlflowRun(mlflow)
