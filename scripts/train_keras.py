"""Train the Keras BiLSTM baseline (Day 4).

    python scripts/train_keras.py              # full run: CPU, or a GPU via notebooks/colab_runner
    python scripts/train_keras.py --smoke      # 200 train rows, 1 epoch, CPU, about a minute
    python scripts/train_keras.py --seed 7     # run name "keras_bilstm_seed7"

Learns the vocabulary from the train split, trains with early stopping on the VALIDATION
loss (the weights of the best epoch are put back at the end), saves the model to
checkpoints/<run_name>.keras and predicts the test split exactly once. Writes
results/metrics/<run_name>.json, results/predictions/<run_name>.csv and a confusion-matrix
figure (smoke runs write to results/smoke/ and checkpoints/smoke/ instead).
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import keras
import numpy as np
import tensorflow as tf

from fnsl import config, data, metrics
from fnsl.keras_model import (
    balanced_class_weights,
    build_bilstm_model,
    build_text_vectorizer,
    predict_proba,
    to_string_array,
    token_lengths,
)
from fnsl.tracking import mlflow_run

RUN_NAME = "keras_bilstm"

# --seq-len 64: measured on data/splits/train.csv after standardize_text, texts have a median
# of 11 words, a 95th percentile of 20, a 99th percentile of 22 and at most 33. So 64 cuts no
# training text and leaves room for longer inputs. Padding is masked, so a smaller value that
# cuts nothing gives the same predictions; it only runs faster on CPU. Every run prints and
# records how many training texts were longer than --seq-len.
DEFAULT_SEQ_LEN = 64
# --max-tokens 20000: the train split has 16,742 distinct words after standardize_text, so
# this cap (which also counts the padding and unknown-word ids) keeps every one of them.
# Side effect: no training text contains the unknown-word id, so its embedding is never
# trained and keeps its small random starting values, yet words first seen in val or test map
# to it (measured: 10.9% of the val words, in 63.8% of the val texts). A smaller cap turns the
# rarest train words (9,747 are seen only once) into unknown words, so the network learns a
# vector for "unknown"; try that only as a separate run and judge it on val_macro_f1.
DEFAULT_MAX_TOKENS = 20000

# --class-weight balanced (default): Neutral is 65% of the training texts. With an unweighted
# loss the network can reach a low loss by leaning towards Neutral, which costs recall on
# Bearish and Bullish and therefore macro-F1. "balanced" multiplies each example's loss by
# n / (3 * n_class) (train split: Bearish 2.22, Bullish 1.65, Neutral 0.51), so the three
# classes carry the same total weight, just as macro-F1 counts them equally. To check the
# choice, train once with `--class-weight none --run-name keras_bilstm_unweighted` (its own
# name, so the main keras_bilstm files are not overwritten) and compare val_macro_f1 in the
# two metrics files, never their test scores.


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Command-line options. --epochs defaults to 10, or to 1 with --smoke."""
    parser = argparse.ArgumentParser(description="Keras BiLSTM baseline")
    parser.add_argument("--smoke", action="store_true", help="tiny data, 1 epoch; results/smoke/")
    parser.add_argument("--seed", type=int, default=config.SEED)
    parser.add_argument("--epochs", type=int, default=None, help="max epochs; 10 (smoke: 1)")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS, help="vocab cap")
    parser.add_argument("--seq-len", type=int, default=DEFAULT_SEQ_LEN, help="token ids per text")
    parser.add_argument("--embed-dim", type=int, default=128, help="size of each word vector")
    parser.add_argument("--lstm-units", type=int, default=64, help="units per LSTM direction")
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--lr", type=float, default=1e-3, help="Adam learning rate")
    parser.add_argument(
        "--patience", type=int, default=2, help="epochs without a lower val_loss before stopping"
    )
    parser.add_argument("--class-weight", choices=["none", "balanced"], default="balanced")
    parser.add_argument(
        "--run-name",
        default=None,
        help='default "keras_bilstm" ("keras_bilstm_seed<N>" if seed != 42)',
    )
    parser.add_argument("--no-mlflow", action="store_true", help="disable MLflow logging")
    args = parser.parse_args(argv)

    if args.epochs is None:  # the default depends on --smoke
        args.epochs = 1 if args.smoke else 10
    if args.epochs < 1:
        parser.error("--epochs must be at least 1")
    return args


def default_run_name(seed: int) -> str:
    """'keras_bilstm' for the main seed, 'keras_bilstm_seed7' etc. for extra seed runs."""
    return RUN_NAME if seed == config.SEED else f"{RUN_NAME}_seed{seed}"


def checkpoint_path(run_name: str, smoke: bool) -> Path:
    """checkpoints/<run_name>.keras, or checkpoints/smoke/<run_name>.keras for smoke runs."""
    base = config.CHECKPOINTS_DIR / "smoke" if smoke else config.CHECKPOINTS_DIR
    return base / f"{run_name}.keras"


def repo_relative(path: Path) -> str:
    """Path relative to the repo root if it is inside it (keeps results free of local paths)."""
    try:
        return Path(path).resolve().relative_to(config.ROOT).as_posix()
    except ValueError:
        return str(path)


def main(argv: list[str] | None = None) -> dict:
    """Train with early stopping on val, save the model, evaluate once on test.

    Returns the metrics record written by metrics.save_results.
    """
    args = parse_args(argv)
    run_name = args.run_name or default_run_name(args.seed)
    # Seed Python, NumPy and TensorFlow BEFORE building the model: weight initialisation,
    # dropout and the per-epoch shuffling all draw from these generators. On CPU a rerun with
    # the same seed repeats exactly (two smoke runs gave identical predictions). On a GPU the
    # cuDNN LSTM kernels are not bit-for-bit deterministic, so GPU runs can differ slightly
    # even with the same seed.
    keras.utils.set_random_seed(args.seed)
    config.set_seed(args.seed)

    splits = data.load_run_splits(smoke=args.smoke)
    train, val, test = splits["train"], splits["val"], splits["test"]
    device = "GPU" if tf.config.list_physical_devices("GPU") else "CPU"
    print(f"train={len(train)} val={len(val)} test={len(test)} smoke={args.smoke}")
    print(f"device={device} run_name={run_name}")
    if device == "CPU" and not args.smoke:
        print("No GPU found: the full run works on CPU but is faster via colab_runner.ipynb.")

    x_train, y_train = to_string_array(train["text"]), train["label"].to_numpy(dtype=int)
    x_val, y_val = to_string_array(val["text"]), val["label"].to_numpy(dtype=int)

    # The vocabulary is learned from the TRAIN texts only.
    vectorizer = build_text_vectorizer(train["text"], args.max_tokens, args.seq_len)
    model = build_bilstm_model(
        vectorizer,
        embed_dim=args.embed_dim,
        lstm_units=args.lstm_units,
        dropout=args.dropout,
        lr=args.lr,
    )
    model.summary()
    n_truncated = int((token_lengths(train["text"]) > args.seq_len).sum())
    print(
        f"vocabulary: {vectorizer.vocabulary_size()} ids; "
        f"{n_truncated} of {len(train)} train texts are longer than {args.seq_len} words"
    )

    class_weight = balanced_class_weights(y_train) if args.class_weight == "balanced" else None
    validation_data = (x_val, y_val)
    if class_weight is not None:
        print("class weights:", {config.ID2LABEL[c]: round(w, 3) for c, w in class_weight.items()})
        # Weight the validation loss with the same train-based class weights, so early
        # stopping watches the same objective the network is trained on.
        val_weights = np.array([class_weight[label] for label in y_val])
        validation_data = (x_val, y_val, val_weights)

    hyperparams = {
        "epochs": args.epochs,  # maximum; early stopping can end training sooner
        "batch_size": args.batch_size,
        "max_tokens": args.max_tokens,
        "seq_len": args.seq_len,
        "embed_dim": args.embed_dim,
        "lstm_units": args.lstm_units,
        "dropout": args.dropout,
        "lr": args.lr,
        "optimizer": "Adam",
        "loss": "sparse categorical cross-entropy",
        "patience": args.patience,
        "early_stopping": "monitor val_loss, restore_best_weights=True",
        "class_weight": args.class_weight,
        "class_weights": (
            {config.ID2LABEL[c]: w for c, w in class_weight.items()} if class_weight else None
        ),
        "val_loss_weighted": class_weight is not None,
        "vocab_size": int(vectorizer.vocabulary_size()),
        "n_params": int(model.count_params()),
        "n_train": len(train),
        "n_val": len(val),
        "train_texts_truncated": n_truncated,
        "device": device,
        "tensorflow_version": tf.__version__,
        "keras_version": keras.__version__,
    }

    # Early stopping prevents overfitting: after a few epochs the network keeps lowering its
    # training loss by memorising training headlines while the validation loss starts to
    # rise. Training stops once val_loss has not improved for `patience` epochs, and the
    # weights from the best epoch are restored (Keras 3 restores them even if training ran
    # all epochs without stopping early).
    early_stopping = keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=args.patience, restore_best_weights=True
    )
    with mlflow_run(run_name, hyperparams, enabled=not (args.smoke or args.no_mlflow)) as run:
        start = time.perf_counter()
        history = model.fit(
            x_train,
            y_train,
            validation_data=validation_data,
            epochs=args.epochs,
            batch_size=args.batch_size,
            class_weight=class_weight,
            callbacks=[early_stopping],
            verbose=2,  # one line per epoch
        )
        # Wall-clock time of model.fit, including the validation pass after every epoch.
        train_time = time.perf_counter() - start

        epochs_trained = len(history.history["loss"])
        best_epoch = early_stopping.best_epoch + 1  # Keras counts epochs from 0
        per_epoch = [
            {"epoch": i + 1, **{key: float(values[i]) for key, values in history.history.items()}}
            for i in range(epochs_trained)
        ]
        for row in per_epoch:
            run.log_metrics({k: v for k, v in row.items() if k != "epoch"}, step=row["epoch"])

        # Validation macro-F1 of the restored best-epoch weights (model selection = val only).
        val_f1 = metrics.macro_f1(y_val, predict_proba(model, val["text"]).argmax(axis=1))
        print(f"best epoch {best_epoch} of {epochs_trained} trained; val macro-F1 {val_f1:.4f}")

        checkpoint = checkpoint_path(run_name, args.smoke)
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        model.save(checkpoint)  # vocabulary + weights; reload with fnsl.keras_model.load_model

        # The test split is predicted exactly once, with the best-epoch weights.
        test_pred = predict_proba(model, test["text"]).argmax(axis=1)
        params = {
            **hyperparams,
            "epochs_trained": epochs_trained,
            "stopped_early": epochs_trained < args.epochs,
            "best_epoch": best_epoch,
            "val_macro_f1": val_f1,
            "history": per_epoch,
        }
        record = metrics.save_results(
            run_name,
            test,
            test_pred,
            params=params,
            train_time_s=train_time,
            seed=args.seed,
            smoke=args.smoke,
            extra={"checkpoint": repo_relative(checkpoint)},
        )
        run.log_metrics(
            {
                "best_epoch": best_epoch,
                "val_macro_f1": val_f1,
                "test_macro_f1": record["macro_f1"],
                "test_accuracy": record["accuracy"],
                **{f"test_f1_{k}": v for k, v in record["per_class_f1"].items()},
            }
        )
        run.log_artifact(config.output_dirs(args.smoke)["figures"] / f"{run_name}_confusion.png")

    print(
        f"TEST macro-F1 {record['macro_f1']:.4f}  accuracy {record['accuracy']:.4f}  "
        f"per-class {record['per_class_f1']}" + ("  [SMOKE: not a result]" if args.smoke else "")
    )
    return record


if __name__ == "__main__":
    main()
