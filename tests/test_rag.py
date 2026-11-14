"""Tests for the RAG pipeline (ingest, retrieval, hit@k evaluation and both scripts).

CPU-only: they use the made-up fixture corpus in tests/fixtures/rag_corpus/ and the
offline HashingEmbeddings, so no model is downloaded. The one test that needs
sentence-transformers is marked heavy.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from langchain_core.documents import Document
from langchain_core.language_models import FakeListChatModel

from fnsl import config
from fnsl.rag import evaluate, ingest, retriever

FIXTURES = Path(__file__).parent / "fixtures" / "rag_corpus"
FIXTURE_FILES = [
    "alpha_bank_statement.txt",
    "beta_bank_statement.txt",
    "delta_short_note.txt",
    "gamma_bank_minutes.txt",
]
# Questions about the fixture docs; each uses words that appear in exactly one file.
FIXTURE_QUESTIONS = [
    ("What happened to the lighthouse rate?", "alpha_bank_statement.txt"),
    ("Did Beta Bank end its saxophone bond purchases?", "beta_bank_statement.txt"),
    ("What did the committee say about the volcano reserve requirement?", "gamma_bank_minutes.txt"),
    ("When does the kiwi window open?", "delta_short_note.txt"),
    ("What does the pineapple index measure?", "alpha_bank_statement.txt"),
]


def _load_script(name: str):
    """Import scripts/<name>.py as a module so tests can call its main()."""
    spec = importlib.util.spec_from_file_location(name, config.ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_eval_csv(path: Path, rows: list[tuple[str, str]]) -> Path:
    pd.DataFrame(rows, columns=["question", "source_file"]).to_csv(path, index=False)
    return path


@pytest.fixture
def fixture_index():
    chunks = ingest.chunk_documents(ingest.load_corpus(FIXTURES), chunk_size=150, chunk_overlap=30)
    return ingest.build_index(chunks, ingest.HashingEmbeddings())


# --- ingest -----------------------------------------------------------------------------


def test_load_corpus_strips_source_line_but_keeps_url():
    docs = ingest.load_corpus(FIXTURES)
    assert [d.metadata["source"] for d in docs] == FIXTURE_FILES
    for doc in docs:
        assert not doc.page_content.startswith("Source:")
        assert doc.metadata["url"].startswith("https://example.org/test-fixtures/")
        assert "TEST FIXTURE" in doc.page_content


def test_load_corpus_skips_sources_md_and_handles_missing_source_line(tmp_path):
    (tmp_path / "SOURCES.md").write_text("| file | url |\n", encoding="utf-8")
    (tmp_path / "plain.txt").write_text("No source line here.\nSecond line.", encoding="utf-8")
    docs = ingest.load_corpus(tmp_path)
    assert len(docs) == 1
    assert docs[0].metadata == {"source": "plain.txt", "url": ""}
    assert docs[0].page_content.startswith("No source line here.")


def test_load_corpus_raises_on_empty_folder(tmp_path):
    with pytest.raises(FileNotFoundError):
        ingest.load_corpus(tmp_path)


def test_load_corpus_moves_title_date_publisher_lines_to_metadata(tmp_path):
    (tmp_path / "doc.txt").write_text(
        "Source: https://example.org/doc (retrieved 2030-01-01)\n"
        "Title: Example Bank statement\nDate: 1 April 2030\nPublisher: Example Bank\n\n"
        "The board kept the walnut rate unchanged.\n",
        encoding="utf-8",
    )
    [doc] = ingest.load_corpus(tmp_path)
    assert doc.page_content == "The board kept the walnut rate unchanged."
    assert doc.metadata == {
        "source": "doc.txt",
        "url": "https://example.org/doc",
        "title": "Example Bank statement",
        "date": "1 April 2030",
        "publisher": "Example Bank",
    }


def test_committed_corpus_has_no_header_only_chunks():
    """The Title/Date/Publisher lines of data/rag_corpus/ files are metadata. Left in the
    text they formed chunks with no statement content, which outranked real passages for
    questions that mention a date and then became the whole retrieval-only answer."""
    if not any(config.RAG_CORPUS_DIR.glob("*.txt")):
        pytest.skip("no corpus in this checkout")
    docs = ingest.load_corpus()
    assert all(doc.metadata.get("title") and doc.metadata.get("date") for doc in docs)
    for chunk_size in (300, 800):
        for chunk in ingest.chunk_documents(docs, chunk_size, 50):
            for line in chunk.page_content.splitlines():
                assert not line.startswith(("Source:", "Title:", "Date:", "Publisher:"))


def test_chunks_respect_size_and_keep_metadata():
    docs = ingest.load_corpus(FIXTURES)
    chunks = ingest.chunk_documents(docs, chunk_size=150, chunk_overlap=30)
    assert all(len(c.page_content) <= 150 for c in chunks)
    # every document gives at least one chunk, and chunks keep their file name and URL
    assert {c.metadata["source"] for c in chunks} == set(FIXTURE_FILES)
    for source in FIXTURE_FILES:
        mine = [c for c in chunks if c.metadata["source"] == source]
        assert [c.metadata["chunk"] for c in mine] == list(range(len(mine)))
        assert all(c.metadata["url"].startswith("https://example.org/") for c in mine)
    # the short note fits in one chunk; the longer fixtures need several
    assert sum(c.metadata["source"] == "delta_short_note.txt" for c in chunks) == 1
    assert sum(c.metadata["source"] == "alpha_bank_statement.txt" for c in chunks) > 1


def test_consecutive_chunks_overlap():
    # One long paragraph of numbered words, so the splitter has to cut between words.
    text = " ".join(f"word{i:03d}" for i in range(200))
    doc = Document(page_content=text, metadata={"source": "numbers.txt"})
    chunks = ingest.chunk_documents([doc], chunk_size=100, chunk_overlap=30)
    assert len(chunks) > 5
    for prev, nxt in zip(chunks, chunks[1:], strict=False):
        first_word_of_next = nxt.page_content.split()[0]
        assert first_word_of_next in prev.page_content.split(), "overlap missing"
        assert nxt.metadata["start_index"] < prev.metadata["start_index"] + len(prev.page_content)


def test_chunk_overlap_larger_than_chunk_size_is_rejected():
    doc = Document(page_content="some text", metadata={"source": "x.txt"})
    with pytest.raises(ValueError):
        ingest.chunk_documents([doc], chunk_size=50, chunk_overlap=80)


def test_hashing_embeddings_are_deterministic_and_normalised():
    emb = ingest.HashingEmbeddings(dim=256)
    a, b = emb.embed_documents(["Rates rose sharply.", "Unrelated text about origami."])
    assert a == emb.embed_query("Rates rose sharply.")  # same text -> same vector
    assert np.linalg.norm(a) == pytest.approx(1.0)
    assert np.linalg.norm(b) == pytest.approx(1.0)
    assert a != b
    # a text with only stop words has no usable words: zero vector, not NaN
    assert emb.embed_query("the of and") == [0.0] * 256


def test_get_embeddings_hashing_needs_no_model():
    assert isinstance(ingest.get_embeddings("hashing"), ingest.HashingEmbeddings)


# --- retrieval --------------------------------------------------------------------------


@pytest.mark.parametrize(("question", "gold"), FIXTURE_QUESTIONS)
def test_distinctive_question_retrieves_right_file_at_k1(fixture_index, question, gold):
    hits = retriever.retrieve(fixture_index, question, k=3)
    assert hits[0][0].metadata["source"] == gold
    scores = [score for _, score in hits]
    assert scores == sorted(scores), "best (smallest distance) first"
    assert all(isinstance(s, float) for s in scores)


def test_answer_without_llm_cites_top_chunk_source(fixture_index):
    result = retriever.answer_question(fixture_index, "When does the kiwi window open?", k=2)
    top_chunk = result["hits"][0][0]
    assert top_chunk.page_content in result["answer"]
    assert "[source: delta_short_note.txt]" in result["answer"]
    assert result["sources"][0] == "delta_short_note.txt"
    assert len(result["hits"]) == 2


def test_answer_with_llm_uses_lcel_chain_and_lists_sources(fixture_index):
    # A fake chat model that ignores its input and gives a fixed reply (no download).
    fake_llm = FakeListChatModel(responses=["It opens on Monday."])
    result = retriever.answer_question(
        fixture_index, "When does the kiwi window open?", k=2, llm=fake_llm
    )
    first_line, sources_line = result["answer"].split("\n")
    assert first_line == "It opens on Monday."
    # even when the model forgets to cite, the retrieved files are listed
    assert sources_line == "Retrieved sources: " + ", ".join(result["sources"])
    assert result["sources"][0] == "delta_short_note.txt"


def test_prompt_contains_tagged_context_and_rules():
    docs = [
        Document(page_content="The kiwi window opens on Monday.", metadata={"source": "d.txt"}),
        Document(page_content="Other text.", metadata={"source": "e.txt"}),
    ]
    context = retriever.format_context(docs)
    assert context.startswith("[d.txt]\nThe kiwi window")
    messages = retriever.RAG_PROMPT.invoke({"context": context, "question": "When?"}).messages
    system, human = messages[0].content, messages[1].content
    assert "ONLY" in system and "square brackets" in system
    assert "[d.txt]" in system and "[e.txt]" in system
    assert human == "When?"


def test_unique_sources_keeps_rank_order():
    docs = [Document(page_content="x", metadata={"source": s}) for s in ["b", "a", "b", "c"]]
    assert retriever.unique_sources(docs) == ["b", "a", "c"]


def test_save_and_load_index_round_trip(fixture_index, tmp_path):
    folder = ingest.save_index(fixture_index, tmp_path / "idx", info={"embedding_model": "x"})
    assert (folder / "index.faiss").exists() and (folder / "index.pkl").exists()
    assert json.loads((folder / "info.json").read_text())["embedding_model"] == "x"
    loaded = ingest.load_index(folder, ingest.HashingEmbeddings())
    question = "saxophone bond purchases"
    before = [d.metadata["source"] for d, _ in retriever.retrieve(fixture_index, question, 3)]
    after = [d.metadata["source"] for d, _ in retriever.retrieve(loaded, question, 3)]
    assert before == after


def test_load_index_missing_folder_gives_clear_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="build_rag"):
        ingest.load_index(tmp_path / "nothing", ingest.HashingEmbeddings())


# --- evaluation -------------------------------------------------------------------------


def test_hit_at_k_matches_hand_computed_values():
    retrieved = [["a", "b", "c"], ["b", "a", "c"], ["c", "c", "b"], ["d", "e", "f"]]
    gold = ["a", "a", "b", "x"]
    # q1 hit at rank 1; q2 at rank 2; q3 at rank 3 (two chunks of "c" first); q4 never.
    assert evaluate.hit_at_k(retrieved, gold, 1) == pytest.approx(1 / 4)
    assert evaluate.hit_at_k(retrieved, gold, 2) == pytest.approx(2 / 4)
    assert evaluate.hit_at_k(retrieved, gold, 3) == pytest.approx(3 / 4)


def test_hit_at_k_rejects_bad_input():
    with pytest.raises(ValueError):
        evaluate.hit_at_k([["a"]], ["a", "b"], 1)  # lengths differ
    with pytest.raises(ValueError):
        evaluate.hit_at_k([["a"]], ["a"], 0)  # k must be >= 1
    with pytest.raises(ValueError):
        evaluate.hit_at_k([], [], 1)  # nothing to evaluate


def test_load_eval_questions_skips_example_rows(tmp_path):
    csv = _write_eval_csv(
        tmp_path / "q.csv",
        [
            ("EXAMPLE: format demo one?", "a.txt"),
            ("Real question one?", "a.txt"),
            ("EXAMPLE: format demo two?", "b.txt"),
            ("  Real question two?  ", "b.txt"),
        ],
    )
    real = evaluate.load_eval_questions(csv)
    assert list(real["question"]) == ["Real question one?", "Real question two?"]
    assert len(evaluate.load_eval_questions(csv, include_examples=True)) == 4


def test_load_eval_questions_rejects_unquoted_comma(tmp_path):
    # pandas would silently use the text before the comma as the row index and keep only
    # " what did the ECB decide?" as the question; the loader must refuse instead.
    csv = tmp_path / "q.csv"
    csv.write_text(
        "question,source_file\nIn June 2024, what did the ECB decide?,a.txt\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="line 2.*double quotes"):
        evaluate.load_eval_questions(csv)


@pytest.mark.parametrize("bad_row", ["A question without a file?", "Half-filled question?,"])
def test_load_eval_questions_rejects_missing_source_file(tmp_path, bad_row):
    csv = tmp_path / "q.csv"
    csv.write_text(f"question,source_file\nFine question?,a.txt\n{bad_row}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="line 3"):
        evaluate.load_eval_questions(csv)


def test_load_eval_questions_accepts_quoted_comma_bom_and_blank_lines(tmp_path):
    csv = tmp_path / "q.csv"
    # utf-8-sig writes the byte-order mark that Excel adds to CSV files
    csv.write_text(
        'question,source_file\n"In June 2024, what did the ECB decide?",a.txt\n\n',
        encoding="utf-8-sig",
    )
    df = evaluate.load_eval_questions(csv)
    assert df.to_dict("records") == [
        {"question": "In June 2024, what did the ECB decide?", "source_file": "a.txt"}
    ]


def test_load_eval_questions_rejects_wrong_header(tmp_path):
    csv = tmp_path / "q.csv"
    csv.write_text("question,file\nQ?,a.txt\n", encoding="utf-8")
    with pytest.raises(ValueError, match="expected"):
        evaluate.load_eval_questions(csv)


def test_committed_eval_csv_is_valid():
    """data/rag_eval.csv parses, EXAMPLE rows are skipped by default, and every
    source_file exists in data/rag_corpus/ (catches typos in hand-written questions)."""
    with_examples = evaluate.load_eval_questions(config.RAG_EVAL_CSV, include_examples=True)
    real = evaluate.load_eval_questions(config.RAG_EVAL_CSV)
    assert not real["question"].str.startswith("EXAMPLE").any()
    assert len(with_examples) >= len(real)
    corpus = {p.name for p in config.RAG_CORPUS_DIR.glob("*.txt")}
    if corpus:  # the corpus may be missing in a fresh checkout without data
        assert evaluate.unknown_sources(with_examples, corpus) == []


def test_unknown_sources_reports_typos():
    df = pd.DataFrame({"question": ["q1", "q2"], "source_file": ["a.txt", "typo.txt"]})
    assert evaluate.unknown_sources(df, {"a.txt", "b.txt"}) == ["typo.txt"]


def test_evaluate_retrieval_per_question_details(fixture_index):
    df = pd.DataFrame(FIXTURE_QUESTIONS, columns=["question", "source_file"])
    result = evaluate.evaluate_retrieval(fixture_index, df, ks=(1, 3))
    assert result["n_questions"] == len(df)
    assert result["hit@1"] == pytest.approx(1.0)  # every fixture question is distinctive
    first = result["per_question"][0]
    assert set(first) == {"question", "gold", "retrieved", "hit@1", "hit@3"}
    assert len(first["retrieved"]) == 3


# --- scripts ----------------------------------------------------------------------------


def test_eval_rag_smoke_writes_only_to_smoke_dir(tmp_results, tmp_path):
    csv = _write_eval_csv(tmp_path / "q.csv", FIXTURE_QUESTIONS[:3])
    script = _load_script("eval_rag")
    record = script.main(
        ["--smoke", "--corpus-dir", str(FIXTURES), "--eval-csv", str(csv)]
        + ["--chunk-sizes", "120", "400"]
    )
    path = tmp_results / "smoke" / "metrics" / "rag_eval.json"
    assert path.exists()
    assert not (tmp_results / "metrics").exists(), "smoke run must not write real results"
    on_disk = json.loads(path.read_text())
    assert on_disk == json.loads(json.dumps(record))
    for key in ("kind", "n_questions", "embedding_model", "chunk_overlap", "k_values"):
        assert key in on_disk
    assert on_disk["kind"] == "rag_eval" and on_disk["smoke"] is True
    assert on_disk["embedding_model"] == "hashing"  # --smoke defaults to offline embeddings
    assert on_disk["k_values"] == [1, 3]
    assert [c["chunk_size"] for c in on_disk["configs"]] == [120, 400]
    assert set(on_disk["configs"][0]) == {"chunk_size", "n_chunks", "hit@1", "hit@3"}
    assert on_disk["best_chunk_size"] in (120, 400)
    assert len(on_disk["per_question"]) == 3 * 2  # every question for both chunk sizes
    assert on_disk["git_commit"] and on_disk["timestamp"]


def test_eval_rag_include_examples_is_forced_into_smoke_dir(tmp_results, tmp_path):
    csv = _write_eval_csv(tmp_path / "q.csv", [("EXAMPLE: " + q, s) for q, s in FIXTURE_QUESTIONS])
    out = tmp_path / "should_not_exist.json"
    script = _load_script("eval_rag")
    script.main(
        ["--include-examples", "--embedding-model", "hashing", "--corpus-dir", str(FIXTURES)]
        + ["--eval-csv", str(csv), "--out", str(out)]
    )
    assert (tmp_results / "smoke" / "metrics" / "rag_eval.json").exists()
    assert not out.exists()
    assert not (tmp_results / "metrics").exists()


def test_eval_rag_refuses_real_run_with_too_few_questions(tmp_results, tmp_path):
    rows = [("EXAMPLE: " + q, s) for q, s in FIXTURE_QUESTIONS] + FIXTURE_QUESTIONS[:2]
    csv = _write_eval_csv(tmp_path / "q.csv", rows)
    script = _load_script("eval_rag")
    # It stops before loading any embedding model, so the default model is never downloaded.
    with pytest.raises(SystemExit) as excinfo:
        script.main(["--corpus-dir", str(FIXTURES), "--eval-csv", str(csv)])
    assert "write" in str(excinfo.value).lower()
    assert not (tmp_results / "metrics" / "rag_eval.json").exists()


def test_eval_rag_real_run_refuses_hashing_embeddings(tmp_results, tmp_path):
    csv = _write_eval_csv(tmp_path / "q.csv", FIXTURE_QUESTIONS)  # 5 real questions
    script = _load_script("eval_rag")
    with pytest.raises(SystemExit, match="hashing"):
        script.main(
            ["--embedding-model", "hashing", "--corpus-dir", str(FIXTURES)]
            + ["--eval-csv", str(csv)]
        )
    assert not (tmp_results / "metrics" / "rag_eval.json").exists()


def test_eval_rag_malformed_csv_exits_with_a_clear_message(tmp_results, tmp_path):
    csv = tmp_path / "q.csv"
    csv.write_text(
        "question,source_file\nIn 2020, how much Treasury debt?,fed.txt\n", encoding="utf-8"
    )
    script = _load_script("eval_rag")
    with pytest.raises(SystemExit, match="double quotes"):
        script.main(["--smoke", "--corpus-dir", str(FIXTURES), "--eval-csv", str(csv)])


@pytest.mark.parametrize("script_name", ["eval_rag", "build_rag"])
def test_scripts_reject_k_below_one(script_name, capsys):
    script = _load_script(script_name)
    with pytest.raises(SystemExit) as excinfo:
        script.parse_args(["--k", "0"])
    assert excinfo.value.code == 2  # argparse usage error
    assert "--k must be at least 1" in capsys.readouterr().err


def test_eval_rag_rejects_unknown_source_file(tmp_results, tmp_path):
    csv = _write_eval_csv(tmp_path / "q.csv", [("What is the kiwi window?", "typo.txt")])
    script = _load_script("eval_rag")
    with pytest.raises(SystemExit, match="typo.txt"):
        script.main(["--smoke", "--corpus-dir", str(FIXTURES), "--eval-csv", str(csv)])


def test_eval_rag_real_run_writes_results_file(tmp_results, tmp_path, monkeypatch):
    # A real run refuses 'hashing', so swap the model loader to keep the test offline.
    monkeypatch.setattr(ingest, "get_embeddings", lambda name: ingest.HashingEmbeddings())
    csv = _write_eval_csv(tmp_path / "q.csv", FIXTURE_QUESTIONS)  # 5 real questions
    script = _load_script("eval_rag")
    record = script.main(
        ["--corpus-dir", str(FIXTURES), "--eval-csv", str(csv), "--chunk-sizes", "150"]
    )
    assert (tmp_results / "metrics" / "rag_eval.json").exists()
    assert record["smoke"] is False and record["n_questions"] == 5
    assert record["embedding_model"] == ingest.DEFAULT_EMBEDDING_MODEL
    assert record["configs"][0]["hit@1"] == pytest.approx(1.0)


def test_build_rag_main_offline(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "CACHE_DIR", tmp_path / "cache")  # default index location
    script = _load_script("build_rag")
    result = script.main(
        ["--embedding-model", "hashing", "--corpus-dir", str(FIXTURES)]
        + ["--question", "Which reserve requirement did the Gamma Bank committee discuss?"]
        + ["--chunk-size", "150", "--chunk-overlap", "30", "--k", "2"]
    )
    index_dir = tmp_path / "cache" / "rag_index" / "default"
    assert result["index_dir"] == str(index_dir)
    assert (index_dir / "index.faiss").exists() and (index_dir / "info.json").exists()
    assert result["sources"][0] == "gamma_bank_minutes.txt"
    assert "[source: gamma_bank_minutes.txt]" in result["answer"]
    assert result["n_documents"] == len(FIXTURE_FILES)


# --- heavy: real sentence-transformers model -------------------------------------------


@pytest.mark.heavy
def test_huggingface_embeddings_are_normalised():
    pytest.importorskip("sentence_transformers")
    pytest.importorskip("langchain_huggingface")
    emb = ingest.get_embeddings(ingest.DEFAULT_EMBEDDING_MODEL)  # downloads ~90 MB once
    rate_hike, rate_rise, recipe = emb.embed_documents(
        [
            "The central bank raised interest rates.",
            "The policy rate was increased by the central bank.",
            "Add two eggs and stir the batter.",
        ]
    )
    assert len(rate_hike) == 384
    assert np.linalg.norm(rate_hike) == pytest.approx(1.0, abs=1e-4)
    assert np.linalg.norm(emb.embed_query("rates")) == pytest.approx(1.0, abs=1e-4)
    # similar meaning -> larger cosine (dot product of unit vectors)
    assert np.dot(rate_hike, rate_rise) > np.dot(rate_hike, recipe)
