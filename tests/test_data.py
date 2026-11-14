"""Tests for cleaning, splitting, and the committed fixed split files."""

from __future__ import annotations

import json

import pandas as pd
import pytest

from fnsl import config, data

# ---------------------------------------------------------------------------------------
# The committed split files (data/splits/) - these are the files every model reads.
# ---------------------------------------------------------------------------------------
needs_splits = pytest.mark.skipif(
    not (config.SPLITS_DIR / "test.csv").exists(),
    reason="run scripts/prepare_data.py first",
)


@pytest.fixture(scope="module")
def splits():
    return data.load_splits()


@pytest.fixture(scope="module")
def report():
    with open(data.split_report_path(), encoding="utf-8") as f:
        return json.load(f)


@needs_splits
def test_split_sizes_add_up(splits, report):
    sizes = {name: len(part) for name, part in splits.items()}
    assert sizes == report["sizes"]
    total = sum(sizes.values())
    assert total == report["cleaning"]["n_clean"]
    # Each split is within 2 rows of its target fraction (the two-stage split rounds twice).
    for name, frac in config.SPLIT_FRACTIONS.items():
        assert abs(sizes[name] - frac * total) <= 2.0, (name, sizes[name], frac * total)


@needs_splits
def test_no_overlap_between_splits(splits):
    names = list(splits)
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            assert not set(splits[a]["id"]) & set(splits[b]["id"]), f"id overlap {a}/{b}"
            assert not set(splits[a]["text"]) & set(splits[b]["text"]), f"text overlap {a}/{b}"
            keys_a = set(splits[a]["text"].map(data.dedup_key))
            keys_b = set(splits[b]["text"].map(data.dedup_key))
            assert not keys_a & keys_b, f"near-duplicate overlap {a}/{b}"


@needs_splits
def test_label_set_and_columns(splits):
    for name, part in splits.items():
        assert list(part.columns) == data.SPLIT_COLUMNS
        assert set(part["label"].unique()) == {0, 1, 2}, name
        assert part["id"].is_unique
        assert (part["text"].str.len() > 0).all()


@needs_splits
def test_splits_are_stratified(splits):
    overall = pd.concat(splits.values())["label"].value_counts(normalize=True)
    for name, part in splits.items():
        share = part["label"].value_counts(normalize=True)
        for label in overall.index:
            assert abs(share[label] - overall[label]) < 0.01, (name, label)


# ---------------------------------------------------------------------------------------
# Unit tests on synthetic data (no files needed).
# ---------------------------------------------------------------------------------------
def test_normalize_whitespace():
    assert data.normalize_whitespace("  a \t b\n\nc  ") == "a b c"


def test_dedup_key_ignores_urls_case_and_spacing():
    a = "$AAPL beats   estimates https://t.co/abc123"
    b = "$aapl BEATS estimates https://t.co/zzz999"
    assert data.dedup_key(a) == data.dedup_key(b) == "$aapl beats estimates"


def test_clean_drops_empty_url_only_conflicts_and_duplicates():
    raw = pd.DataFrame(
        {
            "text": [
                "Stock A rises https://t.co/1",
                "stock a rises https://t.co/2",  # duplicate of row 0 (same label)
                "   ",  # empty
                "https://t.co/only",  # URL only
                "Stock B falls",
                "stock b falls",  # same text, different label -> both dropped
                "Stock C flat",
            ],
            "label": [1, 1, 2, 2, 0, 1, 2],
        }
    )
    clean_df, report = data.clean(raw)
    assert report == {
        "n_raw": 7,
        "dropped_empty": 1,
        "dropped_url_only": 1,
        "dropped_conflicting_labels": 2,
        "dropped_duplicates": 1,
        "n_clean": 2,
    }
    assert list(clean_df["text"]) == ["Stock A rises https://t.co/1", "Stock C flat"]
    assert list(clean_df["id"]) == [0, 1]


def test_make_splits_is_deterministic_and_complete(toy_df):
    first = data.make_splits(toy_df, seed=42)
    second = data.make_splits(toy_df, seed=42)
    for name in config.SPLIT_NAMES:
        pd.testing.assert_frame_equal(first[name], second[name])
    all_ids = sorted(pd.concat(first.values())["id"])
    assert all_ids == sorted(toy_df["id"])


def test_save_splits_refuses_to_overwrite(tmp_path, toy_df):
    splits = data.make_splits(toy_df)
    data.save_splits(splits, tmp_path)
    with pytest.raises(FileExistsError):
        data.save_splits(splits, tmp_path)
    data.save_splits(splits, tmp_path, overwrite=True)  # explicit opt-in works
    assert len(data.load_split("test", tmp_path)) == len(splits["test"])


def test_load_split_keeps_na_like_text(tmp_path):
    df = pd.DataFrame({"id": [0, 1, 2], "text": ["NA", "null", "ok"], "label": [0, 1, 2]})
    df.to_csv(tmp_path / "train.csv", index=False)
    loaded = data.load_split("train", tmp_path)
    assert list(loaded["text"]) == ["NA", "null", "ok"]


def test_smoke_subset_has_all_labels(toy_df):
    small = data.smoke_subset(toy_df, n=6)
    assert len(small) == 6
    assert set(small["label"]) == {0, 1, 2}
    assert data.smoke_subset(toy_df, n=1000).shape == toy_df.shape
