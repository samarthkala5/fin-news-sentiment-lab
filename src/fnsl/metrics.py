"""Metrics and the standard results files that every model writes.

results/predictions/<model>.csv   columns: id,text,label,pred
results/metrics/<model>.json      keys:    model, dataset, n_test, macro_f1, accuracy,
                                           per_class_f1, seed, train_time_s, params,
                                           git_commit, timestamp
The JSON also carries a few extra keys (smoke, confusion_matrix, n_unparseable, and
anything passed via `extra`). Tables and resume numbers are built ONLY from these files.
"""

from __future__ import annotations

import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score

from fnsl import config

REQUIRED_METRIC_KEYS = (
    "model",
    "dataset",
    "n_test",
    "macro_f1",
    "accuracy",
    "per_class_f1",
    "seed",
    "train_time_s",
    "params",
    "git_commit",
    "timestamp",
)
PREDICTION_COLUMNS = ["id", "text", "label", "pred"]
LABEL_IDS = list(range(config.NUM_LABELS))


def macro_f1(y_true, y_pred) -> float:
    """Unweighted mean of the per-class F1 scores.

    Every class counts equally, so a model cannot look good by mostly predicting the
    majority class (Neutral). A prediction equal to config.UNPARSEABLE matches no class,
    so it counts as a miss for the true class.
    """
    return float(f1_score(y_true, y_pred, labels=LABEL_IDS, average="macro", zero_division=0))


def compute_metrics(y_true, y_pred) -> dict[str, Any]:
    """Macro-F1, accuracy, per-class F1, confusion matrix and the unparseable count."""
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    if y_true.shape != y_pred.shape:
        raise ValueError(f"y_true has shape {y_true.shape} but y_pred has {y_pred.shape}")
    per_class = f1_score(y_true, y_pred, labels=LABEL_IDS, average=None, zero_division=0)
    return {
        "macro_f1": macro_f1(y_true, y_pred),
        "accuracy": float(np.mean(y_true == y_pred)),
        "per_class_f1": {
            config.ID2LABEL[i]: float(f) for i, f in zip(LABEL_IDS, per_class, strict=True)
        },
        # rows = true label, columns = predicted label (UNPARSEABLE rows are not shown here)
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=LABEL_IDS).tolist(),
        "n_unparseable": int(np.sum(y_pred == config.UNPARSEABLE)),
    }


def git_commit() -> str:
    """Short hash of HEAD, with '-dirty' if tracked code in src/ or scripts/ has
    uncommitted changes. Returns 'unknown' outside a git checkout."""
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=config.ROOT,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        changes = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no", "--", "src", "scripts"],
            cwd=config.ROOT,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return sha + ("-dirty" if changes else "")


def _jsonable(obj: Any) -> Any:
    """json.dump fallback for NumPy scalars/arrays, Paths and sets."""
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, set):
        return sorted(obj)
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def save_results(
    model: str,
    test_df: pd.DataFrame,
    y_pred,
    *,
    params: dict[str, Any],
    train_time_s: float,
    seed: int = config.SEED,
    smoke: bool = False,
    dataset: str | None = None,
    extra: dict[str, Any] | None = None,
    make_figure: bool = True,
) -> dict[str, Any]:
    """Write the predictions CSV, the metrics JSON and a confusion-matrix figure.

    `test_df` must have columns id, text, label (a fixed split, or a subset of one for
    smoke runs). Smoke runs write under results/smoke/ instead of results/.
    `dataset` defaults to the dataset recorded in data/splits/split_report.json.
    Returns the metrics record that was written.
    """
    from fnsl.data import current_dataset_name

    dataset = dataset or current_dataset_name()
    y_pred = np.asarray(y_pred, dtype=int)
    if len(y_pred) != len(test_df):
        raise ValueError(f"{len(y_pred)} predictions for {len(test_df)} test rows")
    dirs = config.output_dirs(smoke)

    predictions = pd.DataFrame(
        {
            "id": test_df["id"].to_numpy(),
            "text": test_df["text"].to_numpy(),
            "label": test_df["label"].to_numpy(dtype=int),
            "pred": y_pred,
        }
    )
    predictions.to_csv(dirs["predictions"] / f"{model}.csv", index=False)

    scores = compute_metrics(predictions["label"], predictions["pred"])
    record: dict[str, Any] = {
        "model": model,
        "dataset": dataset,
        "n_test": int(len(test_df)),
        "macro_f1": scores["macro_f1"],
        "accuracy": scores["accuracy"],
        "per_class_f1": scores["per_class_f1"],
        "seed": int(seed),
        "train_time_s": round(float(train_time_s), 2),
        "params": params,
        "git_commit": git_commit(),
        "timestamp": datetime.now(UTC).isoformat(timespec="seconds"),
        "smoke": bool(smoke),
        "confusion_matrix": scores["confusion_matrix"],
        "n_unparseable": scores["n_unparseable"],
    }
    if extra:
        record.update(extra)

    with open(dirs["metrics"] / f"{model}.json", "w", encoding="utf-8") as f:
        json.dump(record, f, indent=2, default=_jsonable)

    if make_figure:
        title = f"{model} (test, n={len(test_df)})" + (" [SMOKE]" if smoke else "")
        if scores["n_unparseable"]:
            title += f"\n{scores['n_unparseable']} unparseable outputs not shown"
        save_confusion_matrix(
            scores["confusion_matrix"], dirs["figures"] / f"{model}_confusion.png", title
        )
    return record


def save_confusion_matrix(cm, path: Path, title: str = "") -> None:
    """Save a labelled confusion-matrix heatmap (rows = true, columns = predicted)."""
    import matplotlib.pyplot as plt  # imported here so importing fnsl.metrics stays cheap
    from sklearn.metrics import ConfusionMatrixDisplay

    fig, ax = plt.subplots(figsize=(4.8, 4.2))
    ConfusionMatrixDisplay(np.asarray(cm), display_labels=config.LABELS).plot(
        ax=ax, cmap="Blues", values_format="d", colorbar=False
    )
    ax.set_title(title, fontsize=9)
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def is_model_metrics(record: dict[str, Any]) -> bool:
    """True if a JSON record follows the per-model metrics schema (comparison.json,
    kmeans.json and rag_eval.json have their own formats and are skipped)."""
    return all(key in record for key in REQUIRED_METRIC_KEYS)


def load_metrics(metrics_dir: Path | None = None) -> list[dict[str, Any]]:
    """All per-model metrics records in a folder, sorted by model name."""
    metrics_dir = Path(metrics_dir) if metrics_dir is not None else config.METRICS_DIR
    records = []
    for path in sorted(metrics_dir.glob("*.json")):
        with open(path, encoding="utf-8") as f:
            record = json.load(f)
        if is_model_metrics(record):
            records.append(record)
    return sorted(records, key=lambda r: r["model"])


def load_predictions(path: Path) -> pd.DataFrame:
    """Read a predictions CSV and check it has the standard columns."""
    df = pd.read_csv(path, keep_default_na=False, dtype={"text": str})
    if list(df.columns) != PREDICTION_COLUMNS:
        raise ValueError(f"{path} has columns {list(df.columns)}, expected {PREDICTION_COLUMNS}")
    return df
