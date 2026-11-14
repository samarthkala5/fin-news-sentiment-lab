"""Retrieve the chunks closest to a question and build an answer that cites them.

Vector search: the question is embedded with the same model as the chunks, and FAISS
returns the k chunks whose vectors are nearest to it. The answer step is kept simple:
without a language model we return the best chunk itself plus its source file; with a
model we run an LCEL chain (prompt | llm | StrOutputParser) whose prompt says to answer
only from the retrieved chunks and to cite their file names in brackets.
Retrieval can fail when the question uses other words than the document ("cut" vs
"lower"), when the answer is split across chunks, or when several documents say almost
the same thing (FOMC statements reuse many sentences from meeting to meeting). A date
that appears only in a file's name or Date line does not help either: those are
metadata, not embedded text, so "What did the Fed decide in October 2026?" has to be
matched on its other words.
"""

from __future__ import annotations

from typing import Any

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

DEFAULT_LLM_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"

RAG_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You answer questions about central-bank monetary policy statements. "
            "Use ONLY the context below. If the context does not contain the answer, "
            "say that you do not know. Cite the source file of every fact in square "
            "brackets, for example [fed_fomc_statement_2024-09-18.txt].\n\n"
            "Context:\n{context}",
        ),
        ("human", "{question}"),
    ]
)


def retrieve(index: Any, question: str, k: int = 3) -> list[tuple[Document, float]]:
    """The k chunks nearest to the question, best first, with their FAISS scores.

    The score is the squared L2 distance between the question vector and the chunk
    vector: lower means more similar. With unit-length embeddings it equals
    2 - 2 * cosine similarity (0 = same direction, 2 = unrelated, 4 = opposite).
    `index` is a langchain_community FAISS vector store.
    """
    hits = index.similarity_search_with_score(question, k=k)
    return [(doc, float(score)) for doc, score in hits]


def unique_sources(docs: list[Document]) -> list[str]:
    """Source file names of the chunks, in rank order, without repeats."""
    sources: list[str] = []
    for doc in docs:
        if doc.metadata["source"] not in sources:
            sources.append(doc.metadata["source"])
    return sources


def format_context(docs: list[Document]) -> str:
    """Join chunks into one context string, each preceded by its [source file] tag, so
    the model can see which file each passage comes from and cite it."""
    return "\n\n".join(f"[{doc.metadata['source']}]\n{doc.page_content}" for doc in docs)


def build_chain(llm: Any) -> Any:
    """LCEL chain: fill the prompt -> run the model -> return its reply as a string.

    The `|` operator connects LangChain "runnables": the output of each step is the
    input of the next. Works with chat models and with plain text-completion LLMs.
    """
    return RAG_PROMPT | llm | StrOutputParser()


def answer_question(index: Any, question: str, k: int = 3, llm: Any = None) -> dict[str, Any]:
    """Retrieve k chunks and answer the question from them.

    Without an LLM (the default) the answer is retrieval-only: the text of the best
    chunk, then "[source: <file>]", then the list of all retrieved source files.
    With an LLM the chunks are passed as context to build_chain(llm), and the model's
    reply is followed by the same "Retrieved sources: ..." line.
    Returns {"question", "answer", "sources", "hits"} where hits are (chunk, score) pairs.
    """
    hits = retrieve(index, question, k)
    docs = [doc for doc, _ in hits]
    sources = unique_sources(docs)
    if not docs:
        answer = "No passages were retrieved, so the question cannot be answered."
    elif llm is None:
        top = docs[0]
        answer = (
            f"{top.page_content}\n[source: {top.metadata['source']}]\n"
            f"Retrieved sources: {', '.join(sources)}"
        )
    else:
        reply = build_chain(llm).invoke({"context": format_context(docs), "question": question})
        # Small models often ignore the "cite in brackets" instruction (Qwen2.5-0.5B did
        # in our test), so we always list the files the context came from ourselves.
        answer = f"{reply.strip()}\nRetrieved sources: {', '.join(sources)}"
    return {"question": question, "answer": answer, "sources": sources, "hits": hits}


def load_local_llm(model_id: str = DEFAULT_LLM_MODEL, max_new_tokens: int = 200) -> Any:
    """Optional small local generator: a Hugging Face instruct model as a LangChain chat model.

    HuggingFacePipeline runs a transformers "text-generation" pipeline on this machine
    (downloads the model once; Qwen2.5-0.5B-Instruct is about 1 GB and runs on a CPU).
    ChatHuggingFace wraps it so the model's own chat template formats the system and
    user messages. return_full_text=False makes the pipeline return only the new text,
    not the prompt; do_sample=False gives the same answer every time (greedy decoding).
    """
    # Heavy imports (transformers, torch) only when a generator is actually requested.
    from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline

    pipeline_llm = HuggingFacePipeline.from_model_id(
        model_id=model_id,
        task="text-generation",
        pipeline_kwargs={
            "max_new_tokens": max_new_tokens,
            "do_sample": False,
            "return_full_text": False,
        },
    )
    # Reuse the pipeline's tokenizer instead of letting ChatHuggingFace load it again.
    return ChatHuggingFace(llm=pipeline_llm, tokenizer=pipeline_llm.pipeline.tokenizer)
