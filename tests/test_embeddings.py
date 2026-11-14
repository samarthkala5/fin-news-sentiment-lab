"""Tests for the cached sentence-embedding wrapper (src/fnsl/embeddings.py).

The cache tests run on CPU without sentence-transformers; the real-model test is heavy.
"""

from __future__ import annotations

import sys
import types

import numpy as np
import pytest

from fnsl.embeddings import DEFAULT_MODEL, cache_path, embed_texts

TEXTS = [
    "Shares jump after the company beats earnings estimates",
    "Stock rallies as quarterly profit tops forecasts",
    "The central bank left interest rates unchanged",
]


def test_cache_path_changes_with_texts_order_model_and_normalize(tmp_path):
    base = cache_path(TEXTS, DEFAULT_MODEL, True, tmp_path)
    assert base == cache_path(list(TEXTS), DEFAULT_MODEL, True, tmp_path)  # deterministic
    assert base.parent == tmp_path and base.name.startswith("all-MiniLM-L6-v2_")
    variants = [
        cache_path(TEXTS[::-1], DEFAULT_MODEL, True, tmp_path),  # same texts, new order
        cache_path(TEXTS[:2], DEFAULT_MODEL, True, tmp_path),
        cache_path(TEXTS, "another/model", True, tmp_path),
        cache_path(TEXTS, DEFAULT_MODEL, False, tmp_path),
    ]
    assert len({base, *variants}) == 1 + len(variants)
    # Texts are kept apart in the hash: ["ab", "c"] is not ["a", "bc"].
    joined_differently = cache_path(["a", "bc"], cache_dir=tmp_path)
    assert cache_path(["ab", "c"], cache_dir=tmp_path) != joined_differently


def test_cache_hit_does_not_need_sentence_transformers(tmp_path, monkeypatch):
    stored = np.arange(6, dtype=np.float32).reshape(3, 2)
    np.save(cache_path(TEXTS, "no-such/model", True, tmp_path), stored)
    # With None in sys.modules, `import sentence_transformers` raises ImportError.
    monkeypatch.setitem(sys.modules, "sentence_transformers", None)
    out = embed_texts(TEXTS, model_name="no-such/model", cache_dir=tmp_path)
    np.testing.assert_array_equal(out, stored)


def test_cache_miss_calls_the_model_once_and_saves_float32(tmp_path, monkeypatch):
    """A fake sentence_transformers module (no torch) records how it is called."""
    calls = []

    class FakeModel:
        def __init__(self, name, **kwargs):
            calls.append(("load", name))

        def encode(self, inputs, batch_size, normalize_embeddings, **kwargs):
            calls.append(("encode", len(inputs), batch_size, normalize_embeddings))
            return np.ones((len(inputs), 4))  # float64, so the float32 cast is checked

    fake_module = types.ModuleType("sentence_transformers")
    fake_module.SentenceTransformer = FakeModel
    monkeypatch.setitem(sys.modules, "sentence_transformers", fake_module)

    first = embed_texts(TEXTS, model_name="fake/model", batch_size=2, cache_dir=tmp_path)
    assert first.dtype == np.float32 and first.shape == (3, 4)
    assert calls == [("load", "fake/model"), ("encode", 3, 2, True)]
    # Exactly one finished cache file and no temporary file left behind.
    expected = cache_path(TEXTS, "fake/model", True, tmp_path)
    assert [p.name for p in tmp_path.iterdir()] == [expected.name]

    # The second call is served from the file: no new load or encode.
    again = embed_texts(TEXTS, model_name="fake/model", batch_size=2, cache_dir=tmp_path)
    np.testing.assert_array_equal(again, first)
    assert len(calls) == 2

    # normalize=False reaches encode() and gets its own cache file.
    embed_texts(TEXTS, model_name="fake/model", cache_dir=tmp_path, normalize=False)
    assert calls[-1] == ("encode", 3, 64, False)
    assert len(list(tmp_path.glob("*.npy"))) == 2


def test_empty_input_raises(tmp_path):
    with pytest.raises(ValueError):
        embed_texts([], cache_dir=tmp_path)


@pytest.mark.heavy
def test_real_model_embeds_normalises_and_caches(tmp_path):
    pytest.importorskip("sentence_transformers")
    first = embed_texts(TEXTS, cache_dir=tmp_path)
    assert first.shape == (3, 384) and first.dtype == np.float32
    np.testing.assert_allclose(np.linalg.norm(first, axis=1), 1.0, atol=1e-5)

    # Rows have length 1, so dot products are cosine similarities: the two earnings
    # headlines should be closer to each other than to the interest-rate headline.
    similarity = first @ first.T
    assert similarity[0, 1] > similarity[0, 2]

    # The second call is served from the one cache file and gives identical values.
    assert len(list(tmp_path.glob("*.npy"))) == 1
    np.testing.assert_array_equal(embed_texts(TEXTS, cache_dir=tmp_path), first)

    # normalize=False is a separate cache entry. all-MiniLM-L6-v2 ends with its own
    # Normalize layer, so even these rows have length 1 and match `first`.
    raw = embed_texts(TEXTS, cache_dir=tmp_path, normalize=False)
    assert len(list(tmp_path.glob("*.npy"))) == 2
    np.testing.assert_allclose(raw / np.linalg.norm(raw, axis=1, keepdims=True), first, atol=1e-5)
