"""TF-IDF + logistic regression baseline.

TF-IDF turns each text into a sparse vector with one weight per word or word pair
(bigram). A term gets a high weight when it is frequent in this text (TF) but rare
across all texts (IDF), so words like "downgrade" matter more than "the".
Logistic regression then learns one weight per (term, class) and turns the three class
scores into probabilities with a softmax. It is fast, strong on short texts, and easy
to inspect, which makes it the bar every bigger model has to clear.
"""

from __future__ import annotations

import itertools

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from fnsl import config
from fnsl.metrics import macro_f1


def build_pipeline(
    C: float = 1.0,
    class_weight: str | None = None,
    ngram_range: tuple[int, int] = (1, 2),
    min_df: int = 2,
    seed: int = config.SEED,
) -> Pipeline:
    """TF-IDF features (unigrams + bigrams) followed by multinomial logistic regression.

    C is the inverse regularisation strength: small C = simpler model, large C = fits
    the training data more closely. class_weight="balanced" up-weights rare classes,
    which often helps macro-F1 when one class (Neutral) dominates.
    """
    return Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    lowercase=True,
                    ngram_range=ngram_range,
                    min_df=min_df,  # ignore terms seen in fewer than min_df texts
                    sublinear_tf=True,  # use 1 + log(tf): repeated words count less
                ),
            ),
            (
                "clf",
                # L2-regularised by default. (scikit-learn >= 1.8 deprecates `penalty`.)
                LogisticRegression(
                    C=C, class_weight=class_weight, max_iter=2000, random_state=seed
                ),
            ),
        ]
    )


def tune_on_val(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    c_grid: list[float],
    class_weight_grid: list[str | None],
    seed: int = config.SEED,
) -> tuple[dict, list[dict]]:
    """Grid search scored on the validation split only (the test split is never touched).

    Returns the best settings and the validation macro-F1 of every combination tried.
    """
    results = []
    for C, class_weight in itertools.product(c_grid, class_weight_grid):
        model = build_pipeline(C=C, class_weight=class_weight, seed=seed)
        model.fit(train_df["text"], train_df["label"])
        score = macro_f1(val_df["label"], model.predict(val_df["text"]))
        results.append({"C": C, "class_weight": class_weight, "val_macro_f1": score})
    best = max(results, key=lambda r: r["val_macro_f1"])
    return {"C": best["C"], "class_weight": best["class_weight"]}, results
