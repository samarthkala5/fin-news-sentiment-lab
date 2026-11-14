"""Evaluate RAG retrieval with hit@1 and hit@k for two chunk sizes (Day 12).

    python scripts/eval_rag.py                            # real run -> results/metrics/
    python scripts/eval_rag.py --smoke --include-examples # pipeline check -> results/smoke/

For each chunk size the corpus is re-chunked and re-indexed, every question in
data/rag_eval.csv is run through the retriever, and hit@1 / hit@k are computed (is the
question's source file among the source files of the top-1 / top-k chunks?). The
better chunk size (highest hit@k, then hit@1, then the smaller size) is reported as
best_chunk_size. Note that it is chosen on the same questions it is scored on, so the
best score is slightly optimistic; both configurations are saved.

Rows starting with "EXAMPLE" only illustrate the format and are skipped unless
--include-examples is given. A run with --smoke or --include-examples always writes to
results/smoke/metrics/rag_eval.json, never to the real results file. --smoke also uses
the offline 'hashing' embeddings unless --embedding-model is given; a real run refuses
them, because the README and RESUME_NUMBERS.md are built from the real results file.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

from fnsl import config
from fnsl.metrics import git_commit
from fnsl.rag import evaluate, ingest

# A real result needs enough questions to mean anything (the plan asks for 20-30).
MIN_REAL_QUESTIONS = 5


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="hit@1 / hit@k retrieval evaluation")
    parser.add_argument("--eval-csv", type=Path, default=config.RAG_EVAL_CSV)
    parser.add_argument("--corpus-dir", type=Path, default=config.RAG_CORPUS_DIR)
    parser.add_argument("--chunk-sizes", type=int, nargs="+", default=[300, 800])
    parser.add_argument("--chunk-overlap", type=int, default=50)
    parser.add_argument("--k", type=int, default=3, help="report hit@1 and hit@k")
    parser.add_argument(
        "--embedding-model",
        default=None,
        help=f"default {ingest.DEFAULT_EMBEDDING_MODEL}; 'hashing' (offline) with --smoke",
    )
    parser.add_argument(
        "--include-examples", action="store_true", help="also use EXAMPLE rows (smoke only)"
    )
    parser.add_argument(
        "--smoke", action="store_true", help="quick offline check; writes to results/smoke/"
    )
    parser.add_argument("--out", type=Path, default=None, help="default results/metrics/")
    args = parser.parse_args(argv)
    if args.k < 1:
        parser.error("--k must be at least 1")
    return args


def output_path(args: argparse.Namespace) -> Path:
    """Where the JSON goes. Smoke and example runs can never write the real results."""
    if args.smoke or args.include_examples:
        if args.out is not None:
            print("--out is ignored for --smoke / --include-examples runs")
        return config.SMOKE_DIR / "metrics" / "rag_eval.json"
    return args.out or config.RESULTS_DIR / "metrics" / "rag_eval.json"


def main(argv: list[str] | None = None) -> dict:
    args = parse_args(argv)
    is_real_run = not (args.smoke or args.include_examples)
    model = args.embedding_model or (
        ingest.HASHING_MODEL if args.smoke else ingest.DEFAULT_EMBEDDING_MODEL
    )
    ks = sorted({1, args.k})
    if is_real_run and model == ingest.HASHING_MODEL:
        sys.exit(
            "'hashing' is an offline stand-in for tests and smoke runs, not a real embedding "
            "model, so it cannot produce results/metrics/rag_eval.json. Add --smoke to try it."
        )

    try:
        questions = evaluate.load_eval_questions(
            args.eval_csv, include_examples=args.include_examples
        )
    except ValueError as err:  # a malformed row in the hand-written CSV
        sys.exit(str(err))
    if is_real_run and len(questions) < MIN_REAL_QUESTIONS:
        sys.exit(
            f"Only {len(questions)} real (non-EXAMPLE) questions in {args.eval_csv}; need at "
            f"least {MIN_REAL_QUESTIONS}. Please write 20-30 questions (columns question,"
            "source_file), each answerable from exactly one file in data/rag_corpus/, then "
            "rerun. Nothing was written. (Use --smoke --include-examples to test the pipeline.)"
        )
    if len(questions) == 0:
        sys.exit(
            f"No questions to evaluate in {args.eval_csv} (EXAMPLE rows are skipped; add "
            "--include-examples to use them for a pipeline check)."
        )

    docs = ingest.load_corpus(args.corpus_dir)
    missing = evaluate.unknown_sources(questions, {doc.metadata["source"] for doc in docs})
    if missing:
        sys.exit(f"source_file values not found in {args.corpus_dir}: {missing}")

    embeddings = ingest.get_embeddings(model)  # loaded once, reused for every chunk size
    configs, per_question = [], []
    for chunk_size in args.chunk_sizes:
        chunks = ingest.chunk_documents(docs, chunk_size, args.chunk_overlap)
        index = ingest.build_index(chunks, embeddings)
        result = evaluate.evaluate_retrieval(index, questions, ks=tuple(ks))
        row = {"chunk_size": chunk_size, "n_chunks": len(chunks)}
        for k in ks:
            row[f"hit@{k}"] = result[f"hit@{k}"]
        configs.append(row)
        per_question += [{"chunk_size": chunk_size, **entry} for entry in result["per_question"]]

    k_max = max(ks)
    best = max(configs, key=lambda c: (c[f"hit@{k_max}"], c["hit@1"], -c["chunk_size"]))
    record = {
        "kind": "rag_eval",
        "n_questions": len(questions),
        "n_documents": len(docs),
        "embedding_model": model,
        "chunk_overlap": args.chunk_overlap,
        "k_values": ks,
        "configs": configs,
        "best_chunk_size": best["chunk_size"],
        "per_question": per_question,
        "smoke": not is_real_run,
        "include_examples": args.include_examples,
        "git_commit": git_commit(),
        "timestamp": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    out = output_path(args)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(record, f, indent=2)

    label = "SMOKE (not a result)" if record["smoke"] else "result"
    print(f"{len(questions)} questions, {len(docs)} documents, embeddings={model} [{label}]")
    for c in configs:
        scores = "  ".join(f"hit@{k}={c[f'hit@{k}']:.3f}" for k in ks)
        print(f"  chunk_size={c['chunk_size']:<5} n_chunks={c['n_chunks']:<4} {scores}")
    print(f"best chunk_size: {record['best_chunk_size']}  -> wrote {out}")
    return record


if __name__ == "__main__":
    main()
