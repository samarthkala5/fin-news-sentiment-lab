"""Load the RAG corpus, split it into chunks, embed the chunks and index them with FAISS.

Retrieval-augmented generation answers a question in two steps: first *retrieve* the
passages most related to the question, then answer *only from those passages*. This
module prepares the retrieval side:
1. load each .txt file as a LangChain Document (metadata "source" = the file name),
2. split it into overlapping chunks of at most `chunk_size` characters,
3. embed every chunk: an embedding model maps text to a vector so that texts with
   similar meaning end up close together,
4. store the vectors in a FAISS index, which finds the nearest vectors to a query fast.
"""

from __future__ import annotations

import json
import re
import warnings
import zlib
from pathlib import Path
from typing import Any

import numpy as np
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from fnsl import config

# langchain-community 0.4.x warns on import that the package is "being sunset". Its FAISS
# vector store had no official standalone package yet (checked November 2026), so we keep
# using it and hide only that one known warning.
with warnings.catch_warnings():
    warnings.filterwarnings(
        "ignore", message=".*langchain-community.*sunset", category=DeprecationWarning
    )
    from langchain_community.vectorstores import FAISS

DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
# Special --embedding-model value: the offline HashingEmbeddings below (tests, smoke runs).
HASHING_MODEL = "hashing"
SOURCE_PREFIX = "Source:"
# Optional header lines that may follow the Source line, e.g. "Date: March 15, 2020".
HEADER_KEYS = ("Title", "Date", "Publisher")


def load_corpus(corpus_dir: Path | str | None = None) -> list[Document]:
    """Read every .txt file in `corpus_dir` (default data/rag_corpus/) as one Document.

    Only *.txt files are loaded, so SOURCES.md is skipped. If the first line is
    "Source: <URL> ...", it is removed from the text (a URL is noise for retrieval) but
    the URL is kept in metadata["url"]. The "Title:", "Date:" and "Publisher:" lines
    that may follow are moved into metadata too ("title", "date", "publisher"): left in
    the text they would form a chunk of their own that names a document but holds none
    of its content, and such a chunk can outrank the real passages and become the whole
    "answer". metadata["source"] is the file name, which is what answers cite and what
    hit@k compares against.
    """
    corpus_dir = Path(corpus_dir) if corpus_dir is not None else config.RAG_CORPUS_DIR
    paths = sorted(corpus_dir.glob("*.txt"))
    if not paths:
        raise FileNotFoundError(f"no .txt files found in {corpus_dir}")

    docs = []
    for path in paths:
        lines = path.read_text(encoding="utf-8").splitlines()
        metadata = {"source": path.name, "url": ""}
        if lines and lines[0].startswith(SOURCE_PREFIX):
            words = lines.pop(0)[len(SOURCE_PREFIX) :].split()
            metadata["url"] = words[0] if words else ""
        while lines and lines[0].partition(":")[0] in HEADER_KEYS:
            key, _, value = lines.pop(0).partition(":")
            metadata[key.lower()] = value.strip()
        text = "\n".join(lines).strip()
        if text:  # an empty file would give no chunks; skip it
            docs.append(Document(page_content=text, metadata=metadata))
    return docs


def chunk_documents(
    docs: list[Document], chunk_size: int = 500, chunk_overlap: int = 50
) -> list[Document]:
    """Split documents into chunks of at most `chunk_size` characters.

    RecursiveCharacterTextSplitter tries to cut at paragraph breaks first, then line
    breaks, then spaces, and only then inside words, so chunks stay readable. When it
    has to cut inside a paragraph, the next chunk repeats the last words of the previous
    one (at most `chunk_overlap` characters), so the text around the cut keeps some
    context. Chunks that end at a paragraph break usually share nothing.
    Each chunk keeps its document's metadata and gets "chunk" (0, 1, 2, ... within the
    document) and "start_index" (character offset in the document).
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap, add_start_index=True
    )
    chunks = []
    for doc in docs:
        pieces = splitter.split_documents([doc])
        for i, piece in enumerate(pieces):
            piece.metadata["chunk"] = i
        chunks.extend(pieces)
    return chunks


def get_embeddings(model_name: str = DEFAULT_EMBEDDING_MODEL) -> Embeddings:
    """Sentence-transformers embeddings via langchain-huggingface (downloads once).

    Vectors are L2-normalised (length 1). For unit vectors the squared distance is
    |a - b|^2 = 2 - 2*cos(a, b), so ranking chunks by FAISS's L2 distance gives the same
    order as ranking by cosine similarity. Pass "hashing" to get the offline
    HashingEmbeddings instead (no model download, used by tests and smoke runs).
    """
    if model_name == HASHING_MODEL:
        return HashingEmbeddings()
    # Imported here so that CI (no torch / sentence-transformers) can import this module.
    from langchain_huggingface import HuggingFaceEmbeddings

    return HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},  # also used for queries
    )


class HashingEmbeddings(Embeddings):
    """Tiny deterministic embedding for tests and offline runs (NumPy only, no model).

    Each lower-cased word is hashed into one of `dim` buckets and the bucket counts form
    the vector (a "hashed bag of words"), which is then L2-normalised. Texts that share
    words get similar vectors, but synonyms do not ("cut" vs "lower"), so this is only a
    stand-in for a real embedding model. zlib.crc32 is used instead of Python's hash()
    because hash() of a string changes between Python processes.
    """

    # Very common words carry no topic information, so they are ignored.
    STOP_WORDS = frozenset(
        "a an and are as at be by did do does for from has have how in is it its of on or "
        "that the this to was were what when which who why will with".split()
    )

    def __init__(self, dim: int = 1024) -> None:
        self.dim = dim

    def _embed(self, text: str) -> list[float]:
        vector = np.zeros(self.dim, dtype=np.float32)
        for word in re.findall(r"[a-z0-9]+", text.lower()):
            if word not in self.STOP_WORDS:
                vector[zlib.crc32(word.encode("utf-8")) % self.dim] += 1.0
        norm = np.linalg.norm(vector)
        if norm > 0:  # a text with no usable words stays the zero vector
            vector /= norm
        return vector.tolist()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of chunks."""
        return [self._embed(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        """Embed a question (same function as for the chunks)."""
        return self._embed(text)


def build_index(chunks: list[Document], embeddings: Embeddings) -> FAISS:
    """Embed every chunk and store the vectors in an exact (brute-force) FAISS index.

    LangChain's default is faiss.IndexFlatL2: a search compares the query with every
    stored vector, which is exact and fast enough for a few hundred chunks.
    """
    if not chunks:
        raise ValueError("no chunks to index")
    return FAISS.from_documents(chunks, embeddings)


def index_path(name: str = "default") -> Path:
    """Default folder of a saved index: cache/rag_index/<name>/ (cache/ is gitignored)."""
    return config.CACHE_DIR / "rag_index" / name


def save_index(index: FAISS, folder: Path | str, info: dict[str, Any] | None = None) -> Path:
    """Save the index to `folder`: index.faiss (vectors), index.pkl (chunk texts and
    metadata, a Python pickle) and, if given, info.json (how the index was built)."""
    folder = Path(folder)
    index.save_local(str(folder))
    if info is not None:
        with open(folder / "info.json", "w", encoding="utf-8") as f:
            json.dump(info, f, indent=2)
    return folder


def load_index(folder: Path | str, embeddings: Embeddings) -> FAISS:
    """Load an index written by save_index().

    `embeddings` must be the same model the index was built with (see info.json):
    questions are embedded with it, and vectors from different models are not
    comparable. index.pkl is a pickle file, and unpickling can run arbitrary code hidden
    in the file, so LangChain refuses unless allow_dangerous_deserialization=True. That
    is safe here only because we load files this project wrote itself into cache/;
    never load an index.pkl downloaded from somewhere else.
    """
    folder = Path(folder)
    if not (folder / "index.faiss").exists():
        raise FileNotFoundError(f"no saved index in {folder}; run scripts/build_rag.py first")
    return FAISS.load_local(str(folder), embeddings, allow_dangerous_deserialization=True)
