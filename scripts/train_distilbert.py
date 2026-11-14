"""Fine-tune DistilBERT with an explicit PyTorch training loop (Days 5-6).

    python scripts/train_distilbert.py              # full run: GPU via notebooks/colab_runner.ipynb
    python scripts/train_distilbert.py --smoke      # 200 train rows, 1 epoch, ~1 minute on CPU
    python scripts/train_distilbert.py --seed 7     # run name "distilbert_seed7"
    python scripts/train_distilbert.py --eval-only --checkpoint checkpoints/distilbert/best

Every epoch trains on the train split, then scores the validation split (loss and
macro-F1). The epoch with the best VALIDATION macro-F1 is saved to <output-dir>/best
(model + tokenizer + training_info.json). After the last epoch that best checkpoint is
reloaded and the test split is predicted exactly once. Writes
results/metrics/<run_name>.json, results/predictions/<run_name>.csv and a confusion-matrix
figure (smoke runs write to results/smoke/ and checkpoints/smoke/ instead).
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import pandas as pd
import torch
import transformers
from transformers import AutoTokenizer

from fnsl import config, data, metrics
from fnsl.torch_model import (
    MAX_GRAD_NORM,
    count_truncated,
    get_device,
    load_model,
    make_loader,
    make_optimizer,
    make_scheduler,
    predict,
    train_one_epoch,
)
from fnsl.tracking import mlflow_run

DEFAULT_MODEL = "distilbert/distilbert-base-uncased"
RUN_NAME = "distilbert"
INFO_FILE = "training_info.json"  # saved next to the best checkpoint; read by --eval-only

# --max-length 64: measured with the DistilBERT tokenizer on data/splits/train.csv
# (8,241 texts, counting [CLS] and [SEP]): median 25 tokens, 95th percentile 50, 99th
# percentile 57, longest 82. Only 13 texts (0.16%) are longer than 64, and what gets cut
# off is almost always the tail of a t.co link, which carries no sentiment. Batches are
# padded only to their longest text anyway, so a larger limit would just make the rare
# long batch slower. Every run prints and records how many training texts were truncated.
DEFAULT_MAX_LENGTH = 64


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune DistilBERT (explicit PyTorch loop)")
    parser.add_argument("--smoke", action="store_true", help="tiny data, 1 epoch; results/smoke/")
    parser.add_argument("--seed", type=int, default=config.SEED)
    parser.add_argument("--model-name", default=DEFAULT_MODEL, help="Hub id or local folder")
    parser.add_argument("--epochs", type=int, default=None, help="default 3 (smoke: 1)")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="default 32 (smoke: 16; --eval-only: the batch size the checkpoint was trained with)",
    )
    parser.add_argument("--lr", type=float, default=2e-5, help="peak learning rate")
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.1, help="share of warmup steps")
    parser.add_argument("--max-length", type=int, default=DEFAULT_MAX_LENGTH, help="tokens")
    parser.add_argument(
        "--run-name", default=None, help='default "distilbert" ("distilbert_seed<N>" if seed != 42)'
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="checkpoint folder; default checkpoints/<run_name> (smoke: checkpoints/smoke/...)",
    )
    parser.add_argument(
        "--eval-only", action="store_true", help="skip training; predict test from --checkpoint"
    )
    parser.add_argument("--checkpoint", type=Path, default=None, help="a saved .../best folder")
    parser.add_argument("--no-mlflow", action="store_true", help="disable MLflow logging")
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader worker processes")
    args = parser.parse_args(argv)

    if args.eval_only and args.checkpoint is None:
        parser.error("--eval-only needs --checkpoint PATH, e.g. checkpoints/distilbert/best")
    # Defaults that depend on --smoke.
    if args.epochs is None:
        args.epochs = 1 if args.smoke else 3
    if args.batch_size is None and not args.eval_only:  # --eval-only reads it from the checkpoint
        args.batch_size = 16 if args.smoke else 32
    if args.epochs < 1:
        parser.error("--epochs must be at least 1 (the best epoch is reloaded at the end)")
    return args


def default_run_name(seed: int) -> str:
    """'distilbert' for the main seed, 'distilbert_seed7' etc. for extra seed runs."""
    return RUN_NAME if seed == config.SEED else f"{RUN_NAME}_seed{seed}"


def default_output_dir(run_name: str, smoke: bool) -> Path:
    """checkpoints/<run_name>, or checkpoints/smoke/<run_name> for smoke runs."""
    base = config.CHECKPOINTS_DIR / "smoke" if smoke else config.CHECKPOINTS_DIR
    return base / run_name


def repo_relative(path: Path) -> str:
    """Path relative to the repo root if it is inside it (keeps results free of local paths)."""
    try:
        return Path(path).resolve().relative_to(config.ROOT).as_posix()
    except ValueError:
        return str(path)


def evaluate_on_test(
    checkpoint: Path,
    info: dict,
    test: pd.DataFrame,
    *,
    run_name: str,
    smoke: bool,
    batch_size: int,
    num_workers: int,
) -> dict:
    """Reload a saved checkpoint, predict the test split once and write the results files."""
    device = get_device()
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    model = load_model(checkpoint).to(device)
    loader = make_loader(
        test,
        tokenizer,
        info["params"]["max_length"],  # the length the checkpoint was trained with
        batch_size,
        shuffle=False,  # keep predictions in the same order as the test rows
        seed=info["seed"],
        num_workers=num_workers,
    )
    y_pred, test_loss = predict(model, loader, device)
    return metrics.save_results(
        run_name,
        test,
        y_pred,
        params=info["params"],
        train_time_s=info["train_time_s"],
        seed=info["seed"],
        smoke=smoke,
        extra={
            "test_loss": test_loss,
            "checkpoint": repo_relative(checkpoint),
            # git_commit (added by save_results) is the code that ran this evaluation; with
            # --eval-only that can differ from the code the weights were trained with.
            "train_git_commit": info.get("git_commit"),
        },
    )


def train(args: argparse.Namespace) -> dict:
    """Train, keep the best epoch by validation macro-F1, then evaluate it once on test."""
    run_name = args.run_name or default_run_name(args.seed)
    best_dir = (args.output_dir or default_output_dir(run_name, args.smoke)) / "best"
    # A training_info.json left by an earlier run would describe the wrong weights as soon
    # as this run saves its first epoch (and stay wrong if this run is interrupted), so
    # delete it now. It is written again, for this run, once training has finished.
    (best_dir / INFO_FILE).unlink(missing_ok=True)
    # torch is already imported (by fnsl.torch_model), so set_seed seeds it too. Together
    # with the DataLoader's seeded generator this fixes the head init, dropout and shuffle.
    config.set_seed(args.seed)

    splits = data.load_run_splits(smoke=args.smoke)
    train_df, val_df, test_df = splits["train"], splits["val"], splits["test"]
    device = get_device()
    use_amp = device.type == "cuda"  # mixed precision (float16) only on a GPU
    print(f"train={len(train_df)} val={len(val_df)} test={len(test_df)} smoke={args.smoke}")
    print(f"device={device.type} mixed_precision={use_amp} run_name={run_name}")
    if device.type == "cpu" and not args.smoke:
        print("No GPU found: a full run on CPU is slow; use notebooks/colab_runner.ipynb.")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = load_model(args.model_name).to(device)
    n_truncated = count_truncated(train_df["text"].tolist(), tokenizer, args.max_length)
    print(f"{n_truncated} of {len(train_df)} train texts are longer than {args.max_length} tokens")

    loader_args = {"seed": args.seed, "num_workers": args.num_workers}
    train_loader = make_loader(
        train_df, tokenizer, args.max_length, args.batch_size, shuffle=True, **loader_args
    )
    val_loader = make_loader(
        val_df, tokenizer, args.max_length, args.batch_size, shuffle=False, **loader_args
    )
    num_training_steps = args.epochs * len(train_loader)  # one optimizer step per batch
    optimizer = make_optimizer(model, args.lr, args.weight_decay)
    scheduler = make_scheduler(optimizer, num_training_steps, args.warmup_ratio)
    scaler = torch.amp.GradScaler("cuda") if use_amp else None

    hyperparams = {
        "model_name": args.model_name,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "warmup_ratio": args.warmup_ratio,
        "num_training_steps": num_training_steps,
        "num_warmup_steps": int(args.warmup_ratio * num_training_steps),
        "max_length": args.max_length,
        "max_grad_norm": MAX_GRAD_NORM,
        "optimizer": "AdamW (no weight decay on biases and LayerNorm)",
        "scheduler": "linear warmup, then linear decay to 0",
        "device": device.type,
        "mixed_precision": use_amp,
        "n_train": len(train_df),
        "n_val": len(val_df),
        "train_texts_truncated": n_truncated,
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
    }

    history: list[dict] = []
    best_epoch, best_val_f1 = 0, -1.0
    with mlflow_run(run_name, hyperparams, enabled=not (args.smoke or args.no_mlflow)) as run:
        start = time.perf_counter()
        for epoch in range(1, args.epochs + 1):
            train_loss = train_one_epoch(model, train_loader, optimizer, scheduler, device, scaler)
            val_pred, val_loss = predict(model, val_loader, device)
            val_f1 = metrics.macro_f1(val_df["label"], val_pred)
            history.append(
                {
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "val_macro_f1": val_f1,
                }
            )
            run.log_metrics(
                {"train_loss": train_loss, "val_loss": val_loss, "val_macro_f1": val_f1}, step=epoch
            )
            # Model selection uses the VALIDATION split only; the test split stays unseen.
            is_best = val_f1 > best_val_f1
            if is_best:
                best_epoch, best_val_f1 = epoch, val_f1
                model.save_pretrained(best_dir)
                tokenizer.save_pretrained(best_dir)
            print(
                f"epoch {epoch}/{args.epochs}  train_loss {train_loss:.4f}  "
                f"val_loss {val_loss:.4f}  val_macro_f1 {val_f1:.4f}"
                + ("  (best so far, saved)" if is_best else "")
            )
        # Wall-clock of the whole training loop, incl. per-epoch validation and saving.
        train_time = time.perf_counter() - start

        info = {
            "run_name": run_name,
            "seed": args.seed,
            "smoke": args.smoke,
            "git_commit": metrics.git_commit(),  # the code these weights were trained with
            "train_time_s": train_time,
            "params": {
                **hyperparams,
                "best_epoch": best_epoch,
                "val_macro_f1": best_val_f1,
                "history": history,
            },
        }
        (best_dir / INFO_FILE).write_text(json.dumps(info, indent=2), encoding="utf-8")

        # Reload the BEST epoch's weights (not the last epoch's) and predict test once.
        record = evaluate_on_test(
            best_dir,
            info,
            test_df,
            run_name=run_name,
            smoke=args.smoke,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        )
        run.log_metrics(
            {
                "best_epoch": best_epoch,
                "test_macro_f1": record["macro_f1"],
                "test_accuracy": record["accuracy"],
                **{f"test_f1_{k}": v for k, v in record["per_class_f1"].items()},
            }
        )
        run.log_artifact(config.output_dirs(args.smoke)["figures"] / f"{run_name}_confusion.png")
    return record


def eval_only(args: argparse.Namespace) -> dict:
    """Predict the test split with a checkpoint saved by an earlier training run.

    Hyperparameters, seed, training time, the run name and (unless --batch-size is given)
    the batch size come from the checkpoint's training_info.json, so the metrics file
    matches the run that produced the weights.
    """
    info_path = args.checkpoint / INFO_FILE
    if not info_path.exists():
        raise FileNotFoundError(
            f"{info_path} not found: --checkpoint must be a 'best' folder written by this script"
        )
    info = json.loads(info_path.read_text(encoding="utf-8"))
    run_name = args.run_name or info["run_name"]
    smoke = args.smoke or info["smoke"]  # a smoke checkpoint can only produce smoke results
    config.set_seed(info["seed"])
    test_df = data.load_run_splits(smoke=smoke)["test"]
    print(f"eval-only: {repo_relative(args.checkpoint)} -> {run_name} (smoke={smoke})")
    return evaluate_on_test(
        args.checkpoint,
        info,
        test_df,
        run_name=run_name,
        smoke=smoke,
        # Default: the training run's batch size, so every batch is padded exactly as in
        # that run and the predictions are reproduced exactly.
        batch_size=args.batch_size or info["params"]["batch_size"],
        num_workers=args.num_workers,
    )


def main(argv: list[str] | None = None) -> dict:
    args = parse_args(argv)
    record = eval_only(args) if args.eval_only else train(args)
    print(
        f"TEST macro-F1 {record['macro_f1']:.4f}  accuracy {record['accuracy']:.4f}  "
        f"per-class {record['per_class_f1']}"
        + ("  [SMOKE: not a result]" if record["smoke"] else "")
    )
    return record


if __name__ == "__main__":
    main()
