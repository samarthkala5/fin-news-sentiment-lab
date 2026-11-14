"""Evaluate retrieval with hit@k.

hit@k is the fraction of questions whose correct ("gold") source file is among the
source files of the top-k retrieved chunks. Example: if the right file shows up in the
top 3 chunks for 7 of 10 questions, hit@3 = 0.7. hit@1 asks for it at rank 1.
What it measures: whether retrieval puts the right document in front of the generator.
What it does NOT measure: whether the final answer is correct or faithful (the model can
still misread or ignore the chunk), whether the retrieved chunk of the right file
actually contains the answer, or how high the right chunk ranks inside the top k.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import pandas as pd

from fnsl import config
from fnsl.rag.retriever import retrieve

EXAMPLE_PREFIX = "EXAMPLE"
EVAL_COLUMNS = ["question", "source_file"]


def hit_at_k(retrieved_sources: list[list[str]], gold_sources: list[str], k: int) -> float:
    """Fraction of questions whose gold file is among the first k retrieved chunks.

    retrieved_sources[i] lists the source file of each retrieved chunk for question i,
    best first (the same file can appear several times, once per chunk).
    """
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")
    if len(retrieved_sources) != len(gold_sources):
        raise ValueError(
            f"{len(retrieved_sources)} retrieval results for {len(gold_sources)} questions"
        )
    if not gold_sources:
        raise ValueError("no questions to evaluate")
    hits = [
        gold in sources[:k] for sources, gold in zip(retrieved_sources, gold_sources, strict=True)
    ]
    return sum(hits) / len(hits)


def load_eval_questions(
    path: Path | str | None = None, include_examples: bool = False
) -> pd.DataFrame:
    """Read the question,source_file CSV (default data/rag_eval.csv).

    The file is written by hand, so every row is checked: it must have exactly two
    filled-in cells, and a question that contains a comma must be in double quotes
    ("In June 2024, what did the ECB decide?",ecb_....txt). A bad row raises a
    ValueError naming its line, instead of being dropped or cut at the comma without
    notice. Empty lines are skipped. Rows whose question starts with "EXAMPLE" only show
    the format and are dropped, unless include_examples=True (smoke runs).
    """
    path = Path(path) if path is not None else config.RAG_EVAL_CSV
    # utf-8-sig also accepts the byte-order mark that Excel puts at the start of a CSV.
    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = csv.reader(f)
        header = [cell.strip() for cell in next(reader, [])]
        if header != EVAL_COLUMNS:
            raise ValueError(f"{path} has columns {header}, expected {EVAL_COLUMNS}")
        rows = []
        for row in reader:
            cells = [cell.strip() for cell in row]
            if not any(cells):
                continue  # empty line
            if len(cells) != 2 or not all(cells):
                raise ValueError(
                    f"{path}, line {reader.line_num}: expected 2 filled-in columns "
                    f"(question,source_file) but got {row}. Put a question that contains "
                    "a comma in double quotes."
                )
            rows.append(cells)
    df = pd.DataFrame(rows, columns=EVAL_COLUMNS, dtype=str)
    if not include_examples:
        df = df[~df["question"].str.startswith(EXAMPLE_PREFIX)]
    return df.reset_index(drop=True)


def unknown_sources(questions_df: pd.DataFrame, known_sources: set[str]) -> list[str]:
    """Gold file names that are not in the corpus (usually a typo in the CSV). Without
    this check such a question would silently count as a miss."""
    return sorted(set(questions_df["source_file"]) - set(known_sources))


def evaluate_retrieval(
    index: Any, questions_df: pd.DataFrame, ks: tuple[int, ...] = (1, 3)
) -> dict[str, Any]:
    """Retrieve max(ks) chunks per question and compute hit@k for every k in ks.

    Returns {"hit@1": ..., "hit@3": ..., "n_questions": ..., "per_question": [...]},
    where each per-question entry lists the retrieved source files (one per chunk)
    and whether the gold file was hit at each k.
    """
    max_k = max(ks)
    retrieved, per_question = [], []
    for question, gold in zip(questions_df["question"], questions_df["source_file"], strict=True):
        sources = [doc.metadata["source"] for doc, _ in retrieve(index, question, k=max_k)]
        retrieved.append(sources)
        entry: dict[str, Any] = {"question": question, "gold": gold, "retrieved": sources}
        entry.update({f"hit@{k}": gold in sources[:k] for k in ks})
        per_question.append(entry)

    golds = list(questions_df["source_file"])
    result: dict[str, Any] = {f"hit@{k}": hit_at_k(retrieved, golds, k) for k in ks}
    result["n_questions"] = len(golds)
    result["per_question"] = per_question
    return result
