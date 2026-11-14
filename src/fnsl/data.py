"""Load, clean and split the financial-news sentiment data.

The fixed split is created ONCE by scripts/prepare_data.py and saved to data/splits/.
Every model reads those CSV files through load_split(); nobody re-splits.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from fnsl import config

SPLIT_COLUMNS = ["id", "text", "label"]
URL_RE = re.compile(r"https?://\S+")

# The fallback dataset uses 0=negative, 1=neutral, 2=positive. Map onto our label ids.
FALLBACK_LABEL_MAP = {
    0: config.LABEL2ID["Bearish"],
    1: config.LABEL2ID["Neutral"],
    2: config.LABEL2ID["Bullish"],
}


def normalize_whitespace(text: str) -> str:
    """Collapse runs of whitespace (spaces, tabs, newlines) to one space; strip the ends."""
    return " ".join(str(text).split())


def dedup_key(text: str) -> str:
    """Key used to detect duplicates: lowercase, URLs removed, whitespace collapsed.

    Many tweets are the same headline posted with a different t.co short link. If we
    treated those as different texts, near-identical copies could land in both train
    and test, which leaks information and inflates test scores.
    """
    return normalize_whitespace(URL_RE.sub(" ", str(text)).lower())


def _read_csv(path: Path) -> pd.DataFrame:
    # keep_default_na=False: a tweet whose text is literally "NA" or "null" must stay a
    # string, not silently become NaN.
    return pd.read_csv(path, keep_default_na=False, dtype={"text": str})


def load_raw(source: str = "primary", cache_dir: Path | None = None) -> pd.DataFrame:
    """Download the dataset with all Hub splits pooled into one DataFrame.

    Returns columns: text, label (our label ids), source_split (the Hub split name).
    The result is cached as CSV in data/raw/ so later runs work offline.
    """
    if source == "primary":
        name, revision, text_col = config.DATASET_NAME, config.DATASET_REVISION, "text"
        label_map = None
    elif source == "fallback":
        name, revision = config.FALLBACK_DATASET_NAME, config.FALLBACK_DATASET_REVISION
        text_col, label_map = "sentence", FALLBACK_LABEL_MAP
    else:
        raise ValueError(f"source must be 'primary' or 'fallback', got {source!r}")

    cache_dir = Path(cache_dir) if cache_dir is not None else config.RAW_DIR
    cache_file = cache_dir / f"{name.replace('/', '__')}@{revision[:8]}.csv"
    if cache_file.exists():
        return _read_csv(cache_file)

    from datasets import load_dataset  # imported lazily so CI never needs the Hub

    dataset = load_dataset(name, revision=revision)
    frames = []
    for split_name in dataset:
        part = dataset[split_name].to_pandas()
        frames.append(
            pd.DataFrame(
                {"text": part[text_col], "label": part["label"], "source_split": split_name}
            )
        )
    df = pd.concat(frames, ignore_index=True)
    if label_map is not None:
        df["label"] = df["label"].map(label_map)

    cache_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(cache_file, index=False)
    return df


def clean(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    """Normalise whitespace, drop empty texts and (near-)duplicates.

    Returns the cleaned frame with columns id, text, label (ids 0..N-1 in the original
    row order) and a report with how many rows each step removed.
    """
    report = {"n_raw": len(df)}
    df = df.copy()
    df["text"] = df["text"].fillna("").map(normalize_whitespace)

    df = df[df["text"] != ""]
    report["dropped_empty"] = report["n_raw"] - len(df)

    df = df.assign(key=df["text"].map(dedup_key))
    # A tweet that is only a URL has no words left to classify.
    url_only = df["key"] == ""
    report["dropped_url_only"] = int(url_only.sum())
    df = df[~url_only]

    # Same key but different labels: we cannot tell which label is right, so drop the group.
    n_labels = df.groupby("key")["label"].transform("nunique")
    conflicting = n_labels > 1
    report["dropped_conflicting_labels"] = int(conflicting.sum())
    df = df[~conflicting]

    # Same key and same label: keep the first copy only.
    duplicate = df.duplicated("key", keep="first")
    report["dropped_duplicates"] = int(duplicate.sum())
    df = df[~duplicate]

    df = df.reset_index(drop=True)
    df.insert(0, "id", range(len(df)))
    df["label"] = df["label"].astype(int)
    report["n_clean"] = len(df)
    return df[SPLIT_COLUMNS], report


def make_splits(
    df: pd.DataFrame,
    fractions: dict[str, float] | None = None,
    seed: int = config.SEED,
) -> dict[str, pd.DataFrame]:
    """Stratified train/val/test split using two calls to train_test_split.

    Stratifying keeps the class proportions the same in every split. That matters here
    because Neutral is about two thirds of the data: a random split could give a test
    set with noticeably fewer Bearish examples and make scores noisy.
    """
    fractions = fractions or config.SPLIT_FRACTIONS
    if abs(sum(fractions.values()) - 1.0) > 1e-9:
        raise ValueError(f"split fractions must sum to 1, got {fractions}")

    train_val, test = train_test_split(
        df, test_size=fractions["test"], stratify=df["label"], random_state=seed
    )
    # The val fraction is relative to the full data, so rescale it to what is left.
    val_share_of_rest = fractions["val"] / (1.0 - fractions["test"])
    train, val = train_test_split(
        train_val, test_size=val_share_of_rest, stratify=train_val["label"], random_state=seed
    )
    return {
        name: part.sort_values("id").reset_index(drop=True)
        for name, part in {"train": train, "val": val, "test": test}.items()
    }


def save_splits(
    splits: dict[str, pd.DataFrame], out_dir: Path | None = None, overwrite: bool = False
) -> None:
    """Write data/splits/{train,val,test}.csv.

    Refuses to overwrite existing files unless overwrite=True: the split is fixed on
    Day 1 and every later result depends on it staying identical.
    """
    out_dir = Path(out_dir) if out_dir is not None else config.SPLITS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    existing = [out_dir / f"{n}.csv" for n in splits if (out_dir / f"{n}.csv").exists()]
    if existing and not overwrite:
        raise FileExistsError(
            f"Split files already exist ({', '.join(p.name for p in existing)}). "
            "The fixed split is read-only; pass overwrite=True only if you really mean it."
        )
    for name, part in splits.items():
        part[SPLIT_COLUMNS].to_csv(out_dir / f"{name}.csv", index=False)


def load_split(name: str, splits_dir: Path | None = None) -> pd.DataFrame:
    """Read one fixed split and check its columns and labels."""
    if name not in config.SPLIT_NAMES:
        raise ValueError(f"unknown split {name!r}; expected one of {config.SPLIT_NAMES}")
    splits_dir = Path(splits_dir) if splits_dir is not None else config.SPLITS_DIR
    path = splits_dir / f"{name}.csv"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run `python scripts/prepare_data.py` first.")
    df = _read_csv(path)
    if list(df.columns) != SPLIT_COLUMNS:
        raise ValueError(f"{path} has columns {list(df.columns)}, expected {SPLIT_COLUMNS}")
    bad = set(df["label"].unique()) - set(range(config.NUM_LABELS))
    if bad:
        raise ValueError(f"{path} contains unexpected labels {sorted(bad)}")
    return df


def load_splits(splits_dir: Path | None = None) -> dict[str, pd.DataFrame]:
    """Read train, val and test."""
    return {name: load_split(name, splits_dir) for name in config.SPLIT_NAMES}


def load_run_splits(smoke: bool = False, splits_dir: Path | None = None) -> dict[str, pd.DataFrame]:
    """The splits a training script should use.

    Full runs get the complete fixed splits. Smoke runs get small subsets of the same
    files (sizes in config.SMOKE_SIZES); because the subsets are deterministic, every
    script's smoke predictions cover the same test ids.
    """
    splits = load_splits(splits_dir)
    if smoke:
        splits = {name: smoke_subset(df, config.SMOKE_SIZES[name]) for name, df in splits.items()}
    return splits


def split_report_path(splits_dir: Path | None = None) -> Path:
    splits_dir = Path(splits_dir) if splits_dir is not None else config.SPLITS_DIR
    return splits_dir / "split_report.json"


def current_dataset_name(splits_dir: Path | None = None) -> str:
    """Dataset the fixed split was built from (read from split_report.json), so results
    stay correctly labelled even if the fallback dataset had to be used."""
    path = split_report_path(splits_dir)
    if path.exists():
        with open(path, encoding="utf-8") as f:
            return json.load(f).get("dataset", config.DATASET_NAME)
    return config.DATASET_NAME


def class_balance(splits: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Counts and percentages of each label per split (rows = splits)."""
    rows = []
    for name, part in splits.items():
        counts = part["label"].value_counts()
        row = {"split": name, "n": len(part)}
        for label_id, label_name in config.ID2LABEL.items():
            n = int(counts.get(label_id, 0))
            row[label_name] = n
            row[f"{label_name}_%"] = round(100.0 * n / max(len(part), 1), 1)
        rows.append(row)
    return pd.DataFrame(rows).set_index("split")


def smoke_subset(df: pd.DataFrame, n: int, seed: int = config.SEED) -> pd.DataFrame:
    """A small sample for --smoke runs that contains every label when n allows.

    Rows are shuffled, numbered within their label (0, 1, 2, ...), and the n rows with
    the lowest numbers are kept, which interleaves the labels round-robin.
    """
    if n >= len(df):
        return df.copy()
    shuffled = df.sample(frac=1.0, random_state=seed)
    rank_within_label = shuffled.groupby("label").cumcount()
    picked = (
        shuffled.assign(_rank=rank_within_label)
        .sort_values("_rank", kind="stable")
        .head(n)
        .drop(columns="_rank")
    )
    return picked.sort_values("id").reset_index(drop=True)
