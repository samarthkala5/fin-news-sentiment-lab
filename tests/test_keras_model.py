"""Heavy tests for the Keras BiLSTM: text cleaning, the vectoriser, the model, saving and
loading, and scripts/train_keras.py end to end.

The models are tiny (8-dimensional embeddings, 4 LSTM units) and train on the 60-row toy
data, so each test takes seconds on CPU once TensorFlow is imported. Skipped in CI
(`pytest -m "not heavy"`): they need TensorFlow.
"""

from __future__ import annotations

import importlib
import importlib.util
import json

import numpy as np
import pytest

from fnsl import config, data, metrics

pytestmark = pytest.mark.heavy


@pytest.fixture(scope="module")
def keras_model():
    """fnsl.keras_model, imported only when a heavy test runs (TensorFlow is slow to import)."""
    pytest.importorskip("tensorflow")
    return importlib.import_module("fnsl.keras_model")


def _load_script(name: str):
    """Import scripts/<name>.py as a module so tests can call its main()."""
    spec = importlib.util.spec_from_file_location(name, config.ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _tiny_model(keras_model, train_texts, seq_len: int = 8):
    """Vectoriser adapted on `train_texts` plus a very small BiLSTM on top of it."""
    vectorizer = keras_model.build_text_vectorizer(train_texts, max_tokens=1000, seq_len=seq_len)
    return keras_model.build_bilstm_model(vectorizer, embed_dim=8, lstm_units=4)


def test_standardize_text_drops_urls_punctuation_and_empty_texts(keras_model):
    import tensorflow as tf

    out = keras_model.standardize_text(tf.constant(["$AAPL Beats! https://t.co/Ab12Cd", ":)"]))
    words = [text.decode().split() for text in out.numpy()]
    # ":)" has no words left; it becomes a placeholder so no row is pure padding.
    assert words == [["aapl", "beats"], [keras_model.EMPTY_TEXT]]


def test_vectorizer_is_adapted_on_train_texts_only(keras_model, toy_df):
    train, val = toy_df.iloc[:45], toy_df.iloc[45:]
    vectorizer = keras_model.build_text_vectorizer(train["text"], max_tokens=1000, seq_len=8)
    vocabulary = set(vectorizer.get_vocabulary())

    assert {"company", "falls", "upgrade", "item0", "item44"} <= vocabulary
    assert not any(f"item{i}" in vocabulary for i in val["id"]), "val words leaked into vocab"
    ids = vectorizer(keras_model.to_string_array(["item50 falls"])).numpy()
    assert ids.shape == (1, 8)
    assert ids[0, 0] == 1  # "item50" was only in val, so it is the unknown-word id
    assert ids[0, 2:].tolist() == [0] * 6  # padded with 0 up to seq_len


def test_bilstm_fits_one_epoch_and_predicts_probabilities_from_strings(keras_model, toy_df):
    import keras

    keras.utils.set_random_seed(config.SEED)
    train, val = toy_df.iloc[:45], toy_df.iloc[45:]
    model = _tiny_model(keras_model, train["text"])
    history = model.fit(
        keras_model.to_string_array(train["text"]),
        train["label"].to_numpy(),
        epochs=1,
        batch_size=16,
        verbose=0,
    )
    assert {"loss", "accuracy"} <= set(history.history)

    probs = keras_model.predict_proba(model, val["text"])  # raw strings in
    assert probs.shape == (len(val), config.NUM_LABELS)
    assert (probs >= 0).all()
    np.testing.assert_allclose(probs.sum(axis=1), 1.0, atol=1e-5)


def test_padding_is_masked(keras_model, toy_df):
    """Same weights, different padding length: identical outputs only if padding is masked."""
    short = _tiny_model(keras_model, toy_df["text"], seq_len=8)
    long = _tiny_model(keras_model, toy_df["text"], seq_len=16)
    long.set_weights(short.get_weights())
    np.testing.assert_allclose(
        keras_model.predict_proba(short, toy_df["text"]),
        keras_model.predict_proba(long, toy_df["text"]),
        atol=1e-6,
    )


def test_saved_model_reloads_with_same_predictions(keras_model, toy_df, tmp_path):
    """The .keras file keeps the vocabulary, the custom standardize_text and trained weights."""
    model = _tiny_model(keras_model, toy_df["text"])
    texts = keras_model.to_string_array(toy_df["text"])
    model.fit(texts, toy_df["label"].to_numpy(), epochs=1, batch_size=16, verbose=0)
    path = tmp_path / "model.keras"
    model.save(path)
    reloaded = keras_model.load_model(path)
    np.testing.assert_allclose(
        keras_model.predict_proba(model, toy_df["text"]),
        keras_model.predict_proba(reloaded, toy_df["text"]),
        atol=1e-6,
    )


def test_balanced_class_weights_give_each_class_equal_total_weight(keras_model):
    labels = [0, 0, 0, 0, 1, 1, 2, 2]
    weights = keras_model.balanced_class_weights(labels)
    assert weights == pytest.approx({0: 8 / 12, 1: 8 / 6, 2: 8 / 6})
    totals = [labels.count(c) * w for c, w in weights.items()]
    assert totals == pytest.approx([totals[0]] * 3)
    with pytest.raises(ValueError):
        keras_model.balanced_class_weights([0, 0, 1])  # class 2 has no examples


def test_epochs_default_depends_on_smoke(keras_model):
    script = _load_script("train_keras")
    assert script.parse_args([]).epochs == 10
    assert script.parse_args(["--smoke"]).epochs == 1
    assert script.default_run_name(config.SEED) == "keras_bilstm"
    assert script.default_run_name(7) == "keras_bilstm_seed7"


@pytest.mark.skipif(not (config.SPLITS_DIR / "test.csv").exists(), reason="no split files")
def test_train_keras_smoke_writes_schema(keras_model, tmp_results, tmp_path, monkeypatch):
    monkeypatch.setattr(config, "CHECKPOINTS_DIR", tmp_path / "checkpoints")
    script = _load_script("train_keras")
    record = script.main(["--smoke", "--no-mlflow"])

    path = tmp_results / "smoke" / "metrics" / "keras_bilstm.json"
    on_disk = json.loads(path.read_text())
    assert metrics.is_model_metrics(on_disk) and on_disk["smoke"] is True
    assert on_disk["n_test"] == config.SMOKE_SIZES["test"] == record["n_test"]
    params = on_disk["params"]
    assert params["epochs_trained"] == params["best_epoch"] == len(params["history"]) == 1
    assert {"loss", "val_loss", "accuracy", "val_accuracy"} <= set(params["history"][0])
    assert 0.0 <= params["val_macro_f1"] <= 1.0

    smoke_splits = data.load_run_splits(smoke=True)
    predictions = metrics.load_predictions(
        tmp_results / "smoke" / "predictions" / "keras_bilstm.csv"
    )
    assert len(predictions) == config.SMOKE_SIZES["test"]
    # Same ids, same order as the shared smoke test subset, so McNemar can pair models.
    assert predictions["id"].tolist() == smoke_splits["test"]["id"].tolist()
    assert set(predictions["pred"]) <= set(range(config.NUM_LABELS))
    assert not (tmp_results / "metrics").exists(), "smoke run must not write real results"

    # The saved model's vocabulary was learned from the TRAIN texts only.
    import keras

    checkpoint = tmp_path / "checkpoints" / "smoke" / "keras_bilstm.keras"
    assert checkpoint.exists()
    saved = keras_model.load_model(checkpoint)
    (vectorizer,) = [x for x in saved.layers if isinstance(x, keras.layers.TextVectorization)]
    train_only = keras_model.build_text_vectorizer(smoke_splits["train"]["text"])
    assert vectorizer.get_vocabulary() == train_only.get_vocabulary()


@pytest.mark.skipif(not (config.SPLITS_DIR / "test.csv").exists(), reason="no split files")
def test_train_keras_keeps_the_best_val_epoch(keras_model, tmp_results, tmp_path, monkeypatch):
    """With a high learning rate val_loss rises after epoch 1. The saved model, which also
    gives val_macro_f1 and the test predictions, must be the best epoch's, not the last's."""
    monkeypatch.setattr(config, "CHECKPOINTS_DIR", tmp_path / "checkpoints")
    script = _load_script("train_keras")
    argv = ["--smoke", "--no-mlflow", "--epochs", "4", "--patience", "1", "--lr", "0.03"]
    params = script.main(argv)["params"]

    val_losses = [row["val_loss"] for row in params["history"]]
    assert params["best_epoch"] == 1 + int(np.argmin(val_losses))  # history counts from 1
    assert params["best_epoch"] < len(val_losses), "val_loss never rose: test checks nothing"

    # Recompute the class-weighted val loss of the saved model (as fit() weighted it).
    model = keras_model.load_model(tmp_path / "checkpoints" / "smoke" / "keras_bilstm.keras")
    val = data.load_run_splits(smoke=True)["val"]
    y_val = val["label"].to_numpy()
    weight_of = {config.LABEL2ID[name]: w for name, w in params["class_weights"].items()}
    loss = model.evaluate(
        keras_model.to_string_array(val["text"]),
        y_val,
        sample_weight=np.array([weight_of[label] for label in y_val]),
        batch_size=64,
        verbose=0,
        return_dict=True,
    )["loss"]
    assert loss == pytest.approx(val_losses[params["best_epoch"] - 1], rel=1e-4)
    val_pred = keras_model.predict_proba(model, val["text"]).argmax(axis=1)
    assert metrics.macro_f1(y_val, val_pred) == pytest.approx(params["val_macro_f1"])
