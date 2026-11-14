"""Small LangChain RAG (retrieval-augmented generation) over public central-bank text.

    ingest.py     load data/rag_corpus/*.txt, split into chunks, embed, build a FAISS index
    retriever.py  find the chunks closest to a question and build an answer that cites them
    evaluate.py   hit@k: is the correct source file among the top-k retrieved chunks?

Scripts: scripts/build_rag.py (build the index, ask one question) and scripts/eval_rag.py
(hit@1 / hit@3 on the hand-written questions in data/rag_eval.csv).
"""
