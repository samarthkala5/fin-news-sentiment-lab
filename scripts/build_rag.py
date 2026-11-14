"""Build the RAG index over data/rag_corpus/ and answer one question (Day 11).

    python scripts/build_rag.py                     # default sample question
    python scripts/build_rag.py --question "What did the ECB decide in June 2024?"
    python scripts/build_rag.py --llm-model Qwen/Qwen2.5-0.5B-Instruct   # local generator

Loads the corpus, splits it into chunks, embeds them with a sentence-transformers model
(downloaded once), saves the FAISS index to cache/rag_index/default/, then prints the
top-k retrieved chunks with their source files and an answer that cites its source.
Without --llm-model the answer is retrieval-only (the best chunk plus its file name).
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from fnsl import config
from fnsl.rag import ingest, retriever

DEFAULT_QUESTION = "How did the Federal Reserve respond to the coronavirus in March 2020?"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the RAG index and ask one question")
    parser.add_argument("--corpus-dir", type=Path, default=config.RAG_CORPUS_DIR)
    parser.add_argument("--chunk-size", type=int, default=500, help="max characters per chunk")
    parser.add_argument("--chunk-overlap", type=int, default=50, help="characters shared")
    parser.add_argument(
        "--embedding-model",
        default=ingest.DEFAULT_EMBEDDING_MODEL,
        help="sentence-transformers model, or 'hashing' for offline bag-of-words vectors",
    )
    parser.add_argument(
        "--index-dir", type=Path, default=None, help="default: cache/rag_index/default/"
    )
    parser.add_argument("--question", default=DEFAULT_QUESTION)
    parser.add_argument("--k", type=int, default=3, help="number of chunks to retrieve")
    parser.add_argument(
        "--llm-model",
        default=None,
        help=f"optional local generator, e.g. {retriever.DEFAULT_LLM_MODEL} (downloads ~1 GB)",
    )
    args = parser.parse_args(argv)
    if args.k < 1:
        parser.error("--k must be at least 1")
    return args


def main(argv: list[str] | None = None) -> dict:
    args = parse_args(argv)
    index_dir = args.index_dir or ingest.index_path("default")

    # 1) Load, chunk, embed and index the corpus.
    start = time.perf_counter()
    docs = ingest.load_corpus(args.corpus_dir)
    chunks = ingest.chunk_documents(docs, args.chunk_size, args.chunk_overlap)
    embeddings = ingest.get_embeddings(args.embedding_model)
    index = ingest.build_index(chunks, embeddings)
    info = {
        "embedding_model": args.embedding_model,
        "chunk_size": args.chunk_size,
        "chunk_overlap": args.chunk_overlap,
        "n_documents": len(docs),
        "n_chunks": len(chunks),
        "sources": [doc.metadata["source"] for doc in docs],
    }
    ingest.save_index(index, index_dir, info)
    print(
        f"Indexed {len(docs)} documents as {len(chunks)} chunks "
        f"(chunk_size={args.chunk_size}, overlap={args.chunk_overlap}, "
        f"embeddings={args.embedding_model}) in {time.perf_counter() - start:.1f}s"
    )
    print(f"Saved index to {index_dir}")

    # 2) Ask the question: retrieve chunks, then answer (retrieval-only or with an LLM).
    llm = retriever.load_local_llm(args.llm_model) if args.llm_model else None
    result = retriever.answer_question(index, args.question, k=args.k, llm=llm)

    print(f"\nQuestion: {args.question}")
    # Fewer than k chunks come back if the index holds fewer than k.
    print(f"Top {len(result['hits'])} chunks (score = squared L2 distance, lower is closer):")
    for rank, (doc, score) in enumerate(result["hits"], start=1):
        preview = " ".join(doc.page_content.split())[:300]
        print(
            f"  {rank}. [{doc.metadata['source']}] chunk {doc.metadata['chunk']}  "
            f"score={score:.3f}\n     {preview}"
        )
    print(f"\nAnswer ({'LLM: ' + args.llm_model if llm else 'retrieval-only'}):")
    print(result["answer"])
    return {
        **info,
        "question": args.question,
        "answer": result["answer"],
        "sources": result["sources"],
        "index_dir": str(index_dir),
    }


if __name__ == "__main__":
    main()
