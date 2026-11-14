"""Project-wide constants: random seed, paths, dataset and label map.

Anything that more than one module needs lives here, so there is exactly one
place to look when a path, the seed or the label order matters.
"""

from __future__ import annotations

import os
import random
import sys
from pathlib import Path

import numpy as np

SEED = 42

# Repo root: src/fnsl/config.py -> parents[2]. Set FNSL_ROOT to override, e.g. if the
# package was installed non-editable and __file__ no longer points into the repo.
ROOT = Path(os.environ.get("FNSL_ROOT", Path(__file__).resolve().parents[2]))

DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"  # gitignored download cache
SPLITS_DIR = DATA_DIR / "splits"  # fixed train/val/test CSVs, committed, read-only
RAG_CORPUS_DIR = DATA_DIR / "rag_corpus"
RAG_EVAL_CSV = DATA_DIR / "rag_eval.csv"

RESULTS_DIR = ROOT / "results"
METRICS_DIR = RESULTS_DIR / "metrics"
PREDICTIONS_DIR = RESULTS_DIR / "predictions"
FIGURES_DIR = RESULTS_DIR / "figures"
SMOKE_DIR = RESULTS_DIR / "smoke"  # gitignored: smoke runs are never final results

CACHE_DIR = ROOT / "cache"  # embeddings, FAISS indexes (gitignored)
CHECKPOINTS_DIR = ROOT / "checkpoints"  # model weights and adapters (gitignored)
MLRUNS_DIR = ROOT / "mlruns"  # local MLflow store (gitignored)

# --- Dataset (verified on Day 1, see README "Dataset") ---------------------------------
DATASET_NAME = "zeroshot/twitter-financial-news-sentiment"
# Pinning the Hub commit makes the download reproducible even if the dataset is edited.
DATASET_REVISION = "ccbe24de388e287beb92dd393a335c376b350ac3"
DATASET_LICENSE = "MIT"

# Fallback: the original takala/financial_phrasebank uses a loading script, which
# `datasets` 4.x no longer runs, so we use a parquet mirror of it.
FALLBACK_DATASET_NAME = "atrost/financial_phrasebank"
FALLBACK_DATASET_REVISION = "fc7fb491db5069dc7494bc0a8c937b3661b8fdce"
FALLBACK_DATASET_LICENSE = "CC BY-NC-SA 3.0 (original Financial PhraseBank)"

# --- Labels ----------------------------------------------------------------------------
# Label ids follow the primary dataset: 0=Bearish, 1=Bullish, 2=Neutral.
LABELS = ["Bearish", "Bullish", "Neutral"]
LABEL2ID = {name: i for i, name in enumerate(LABELS)}
ID2LABEL = dict(enumerate(LABELS))
NUM_LABELS = len(LABELS)
# Prediction value for LLM outputs that map to no label. Always counted as wrong.
UNPARSEABLE = -1

# --- Split -----------------------------------------------------------------------------
SPLIT_NAMES = ("train", "val", "test")
SPLIT_FRACTIONS = {"train": 0.70, "val": 0.15, "test": 0.15}

# --- Smoke runs ------------------------------------------------------------------------
# Every --smoke run uses the same tiny subsets (see data.load_run_splits), so smoke
# predictions from different scripts share test ids and can still be paired.
SMOKE_SIZES = {"train": 200, "val": 64, "test": 64}


def set_seed(seed: int = SEED) -> None:
    """Seed Python, NumPy and any deep-learning framework that is already imported.

    Frameworks are only seeded if they are in sys.modules, so CPU-only code (and CI)
    never pays the cost of importing torch or tensorflow. Training modules import
    their framework at the top, so by the time a script calls set_seed() it is loaded.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    if "torch" in sys.modules:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    if "tensorflow" in sys.modules:
        import tensorflow as tf

        tf.random.set_seed(seed)


def output_dirs(smoke: bool = False) -> dict[str, Path]:
    """Folders a run writes to. Smoke runs go to results/smoke/ so they can never be
    mistaken for (or overwrite) real results."""
    base = SMOKE_DIR if smoke else RESULTS_DIR
    dirs = {
        "metrics": base / "metrics",
        "predictions": base / "predictions",
        "figures": base / "figures",
    }
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    return dirs
