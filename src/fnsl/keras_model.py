"""Keras BiLSTM text classifier (TensorFlow backend): the deep-learning baseline of Day 4.

The model takes raw headline strings and returns probabilities for Bearish/Bullish/Neutral.
TextVectorization cleans each text, splits it into words and maps every word to an integer
id from a vocabulary learned on the TRAIN texts only. An Embedding layer turns each id into
a learned dense vector. A bidirectional LSTM reads those vectors left-to-right and
right-to-left and joins its two final states into one summary vector of the text. Dropout
and a softmax layer turn that summary into the three class probabilities.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from pathlib import Path

import keras
import numpy as np
import tensorflow as tf

from fnsl import config

if keras.backend.backend() != "tensorflow":
    # String inputs and TextVectorization inside a model only work with TensorFlow.
    raise ImportError(
        f"fnsl.keras_model needs the TensorFlow backend, but Keras is using "
        f"{keras.backend.backend()!r}. Set the environment variable KERAS_BACKEND=tensorflow."
    )

# The characters that Keras's built-in "lower_and_strip_punctuation" option removes.
PUNCTUATION_REGEX = r'[!"#$%&()\*\+,-\./:;<=>?@\[\\\]^_`{|}~\']'
URL_REGEX = r"https?://\S+"
# Replaces texts that have no words left after cleaning, such as ":)". See standardize_text.
EMPTY_TEXT = "emptytext"


@keras.saving.register_keras_serializable(package="fnsl")
def standardize_text(text: tf.Tensor) -> tf.Tensor:
    """Lowercase, delete URLs, delete punctuation. Runs inside the model on tf.string tensors.

    This is Keras's default "lower_and_strip_punctuation" plus a URL step. 46% of the train
    tweets contain a URL (mostly t.co short links). By default each link becomes its own
    vocabulary entry, nearly always seen just once: 4,292 of the 21,034 distinct train tokens.
    The decorator registers the function with Keras, so a saved .keras model can find it.
    """
    text = tf.strings.lower(text)
    text = tf.strings.regex_replace(text, URL_REGEX, " ")  # before "://" loses its punctuation
    text = tf.strings.regex_replace(text, PUNCTUATION_REGEX, "")
    # A text with no words left would be all padding. The fast cuDNN LSTM used on a GPU
    # rejects such rows, so the text becomes one placeholder (unknown) word instead.
    is_empty = tf.strings.length(tf.strings.strip(text)) == 0
    return tf.where(is_empty, EMPTY_TEXT, text)


def to_string_array(texts: Iterable[str]) -> np.ndarray:
    """1-D NumPy array of Python strings (dtype=object): the text format Keras 3 accepts.

    Keras 3 rejects a plain list ("Unrecognized data type") and a fixed-width string array
    such as np.array(["up"]) ("Invalid dtype: str..."); object arrays become tf.string.
    """
    return np.asarray([str(text) for text in texts], dtype=object)


def token_lengths(texts: Iterable[str]) -> np.ndarray:
    """Number of words in each text after standardize_text (before padding or cutting)."""
    words = tf.strings.split(standardize_text(tf.constant(to_string_array(texts))))
    return words.row_lengths().numpy()


def build_text_vectorizer(
    train_texts: Iterable[str], max_tokens: int = 20000, seq_len: int = 64
) -> keras.layers.TextVectorization:
    """Layer that maps a text to exactly `seq_len` token ids, with a vocabulary from train.

    adapt() counts the words in `train_texts` and keeps the most frequent ones, up to
    `max_tokens` entries including the two reserved ids (0 = padding, 1 = unknown word).
    Pass TRAIN texts only: building the vocabulary from validation or test texts would leak
    information about them into the model. Shorter texts are padded with 0, longer ones cut.
    """
    vectorizer = keras.layers.TextVectorization(
        max_tokens=max_tokens,
        standardize=standardize_text,
        split="whitespace",
        output_mode="int",
        output_sequence_length=seq_len,
    )
    # Pass an array: adapt() treats a pandas Series as a stream of batches and would
    # process it one string at a time.
    vectorizer.adapt(to_string_array(train_texts))
    return vectorizer


def build_bilstm_model(
    vectorizer: keras.layers.TextVectorization,
    embed_dim: int = 128,
    lstm_units: int = 64,
    dropout: float = 0.3,
    num_classes: int = config.NUM_LABELS,
    lr: float = 1e-3,
) -> keras.Model:
    """Compiled model: raw strings in, softmax class probabilities out.

    Shapes for a batch of B texts:
      (B,) strings -> vectorizer -> (B, seq_len) token ids
      -> Embedding -> (B, seq_len, embed_dim) vectors; id 0 (padding) is masked
      -> Bidirectional LSTM -> (B, 2 * lstm_units): forward and backward final states
      -> Dropout (only while training) -> Dense + softmax -> (B, num_classes).
    Trained with Adam on sparse categorical cross-entropy (labels are the integers
    0..num_classes-1, not one-hot vectors); accuracy is tracked as a metric.
    """
    inputs = keras.Input(shape=(), dtype="string", name="text")  # one string per example
    token_ids = vectorizer(inputs)
    # mask_zero=True: id 0 means padding, and the mask makes the LSTM skip those positions,
    # so a text gives the same output however much padding follows it.
    vectors = keras.layers.Embedding(
        input_dim=vectorizer.vocabulary_size(),
        output_dim=embed_dim,
        mask_zero=True,
        name="embedding",
    )(token_ids)
    # One LSTM reads the words first-to-last, a second one last-to-first; their final
    # hidden states (lstm_units numbers each) are concatenated.
    summary = keras.layers.Bidirectional(keras.layers.LSTM(lstm_units), name="bilstm")(vectors)
    summary = keras.layers.Dropout(dropout, name="dropout")(summary)
    outputs = keras.layers.Dense(num_classes, activation="softmax", name="softmax")(summary)

    model = keras.Model(inputs, outputs, name="bilstm_classifier")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def balanced_class_weights(
    labels: Sequence[int] | np.ndarray, num_classes: int = config.NUM_LABELS
) -> dict[int, float]:
    """Loss weight per class: n_samples / (num_classes * n_samples_of_that_class).

    Rare classes get weights above 1 and the majority class below 1, so every class
    contributes the same total weight to the training loss (as in scikit-learn's
    class_weight="balanced"). That matches macro-F1, where all classes count equally.
    """
    labels = np.asarray(labels, dtype=int)
    counts = np.bincount(labels, minlength=num_classes)
    if (counts == 0).any():
        raise ValueError(f"every class needs at least one example, got counts {counts.tolist()}")
    weights = len(labels) / (num_classes * counts)
    return {label: float(weight) for label, weight in enumerate(weights)}


def predict_proba(model: keras.Model, texts: Iterable[str], batch_size: int = 256) -> np.ndarray:
    """Class probabilities for raw texts, shape (n_texts, num_classes); rows sum to 1."""
    return model.predict(to_string_array(texts), batch_size=batch_size, verbose=0)


def load_model(path: str | Path) -> keras.Model:
    """Load a model saved with model.save("<name>.keras").

    The saved TextVectorization layer refers to standardize_text by its registered name,
    so this module must be imported before loading; loading through here guarantees it.
    """
    return keras.saving.load_model(path)
