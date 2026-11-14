"""Sentence embeddings with sentence-transformers, cached on disk.

A sentence-embedding model maps each text to a fixed-length vector (384 numbers for
all-MiniLM-L6-v2, a small 6-layer transformer) so that texts with similar meaning end
up close together. Embedding the ~8k training texts takes about a minute on a CPU, so
the result is saved as a .npy file named after a hash of the model name, the settings
and every text: change any of them and the hash (so the file) changes; otherwise a
rerun just loads the file. sentence_transformers (torch) is imported only when needed.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from fnsl import config

DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def cache_path(
    texts: Sequence[str],
    model_name: str = DEFAULT_MODEL,
    normalize: bool = True,
    cache_dir: Path | None = None,
) -> Path:
    """File that holds (or will hold) the embeddings for exactly these inputs.

    The name contains a SHA-256 hash of the model name, the normalize flag and all
    texts in order. JSON-encoding them first keeps the texts separate, so ["ab", "c"]
    and ["a", "bc"] get different hashes. cache_dir defaults to cache/embeddings/.
    """
    payload = json.dumps({"model": model_name, "normalize": normalize, "texts": list(texts)})
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]
    # Readable prefix such as "all-MiniLM-L6-v2"; unusual characters become "_".
    short_name = re.sub(r"[^A-Za-z0-9._-]", "_", model_name.rstrip("/").split("/")[-1])
    cache_dir = Path(cache_dir) if cache_dir is not None else config.CACHE_DIR / "embeddings"
    return cache_dir / f"{short_name}_{digest}.npy"


def embed_texts(
    texts: Sequence[str],
    model_name: str = DEFAULT_MODEL,
    batch_size: int = 64,
    cache_dir: Path | None = None,
    normalize: bool = True,
) -> np.ndarray:
    """Embed texts with a sentence-transformers model, reusing the disk cache if possible.

    Returns a float32 array of shape (len(texts), embedding_dim). With normalize=True
    every row is scaled to length 1, so the dot product of two rows is their cosine
    similarity (all-MiniLM-L6-v2 already ends with a normalisation layer, so for that
    model the flag changes nothing). cache_dir defaults to cache/embeddings/.
    """
    texts = [str(text) for text in texts]
    if not texts:
        raise ValueError("texts is empty")

    path = cache_path(texts, model_name, normalize, cache_dir)
    if path.exists():
        cached = np.load(path)
        if cached.shape[0] == len(texts):
            return cached

    # Heavy import (torch + transformers), so it only happens when we must embed.
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name)
    # Texts are passed positionally: the parameter name changed between versions
    # (it is `inputs` in sentence-transformers 6.x).
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        normalize_embeddings=normalize,
        convert_to_numpy=True,
        show_progress_bar=len(texts) >= 1000,
    )
    embeddings = np.asarray(embeddings, dtype=np.float32)

    path.parent.mkdir(parents=True, exist_ok=True)
    # Write to a temporary file and rename it at the end, so an interrupted run can
    # never leave a half-written cache file behind. The process id in the name gives
    # two runs that embed the same texts at the same time separate temporary files.
    tmp_path = path.with_name(f"{path.stem}.{os.getpid()}.tmp.npy")
    np.save(tmp_path, embeddings)
    tmp_path.replace(path)
    return embeddings
