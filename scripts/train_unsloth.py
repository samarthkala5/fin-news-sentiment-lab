"""LLM sentiment classifier: zero-shot baseline and Unsloth LoRA fine-tune (Days 8-10).

    python scripts/train_unsloth.py --mode zero-shot    # Day 8: base model on 50 test headlines
    python scripts/train_unsloth.py --mode train        # Day 9: LoRA fine-tune, save the adapter
    python scripts/train_unsloth.py --mode eval         # Day 10: adapter on the whole test split
    python scripts/train_unsloth.py --mode all          # the three steps in a row
    python scripts/train_unsloth.py --smoke --mode all  # CPU, tiny model, writes results/smoke/

Unsloth needs an NVIDIA GPU, so real runs go through notebooks/colab_runner.ipynb. --smoke
switches to the plain Hugging Face backend and a 135M-parameter chat model, so the same code
path can be checked on a CPU in about a minute.

Outputs (results/smoke/... for --smoke):
  zero-shot  results/metrics/llm_zeroshot.json, results/predictions/llm_zeroshot.csv
  train      the adapter in --adapter-dir (default checkpoints/llm_lora_adapter) and
             results/metrics/llm_lora_train.json (loss curves; no test metrics)
  eval       results/metrics/llm_lora.json, results/predictions/llm_lora.csv
Unparseable answers count as wrong (pred = -1); the count and examples go into the JSON.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from fnsl import config, data, llm_lora, metrics
from fnsl.tracking import mlflow_run

ZEROSHOT_NAME = "llm_zeroshot"
LORA_NAME = "llm_lora"
TRAIN_LOG_NAME = "llm_lora_train"
MAX_NEW_TOKENS = 5  # a label word is 1-3 tokens for Qwen2.5; more would only show rambling
WARMUP_FRACTION = 0.1  # LR warm-up over the first 10% of steps (Unsloth guide: 5-10%)
ZEROSHOT_N_TEST = 50  # Day 8 of the plan: zero-shot on 50 test headlines
# Unsloth cuts every input to the max_seq_length the model was loaded with. Training rows are
# capped at --max-seq-length on purpose, but a test prompt must never be cut, so zero-shot and
# eval load the model with this roomier limit (headline prompts are under 150 tokens).
INFERENCE_MAX_SEQ_LENGTH = 512

# --smoke: tiny model on CPU, a few optimizer steps, all 64 shared smoke test rows (so smoke
# predictions pair by id with every other script's smoke predictions).
SMOKE_OVERRIDES = {
    "backend": "hf",
    "model_name": llm_lora.SMOKE_MODEL,
    "max_steps": 5,
    "grad_accum": 1,
    "eval_steps": 5,
    "n_val_eval": 16,
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Zero-shot and LoRA fine-tuned LLM sentiment")
    parser.add_argument("--mode", choices=["zero-shot", "train", "eval", "all"], default="all")
    parser.add_argument("--backend", choices=["unsloth", "hf"], default="unsloth")
    parser.add_argument(
        "--model-name",
        default=None,
        help=f"default {llm_lora.DEFAULT_UNSLOTH_MODEL} ({llm_lora.SMOKE_MODEL} with --smoke)",
    )
    parser.add_argument(
        "--n-test",
        type=int,
        default=None,
        help="test rows to predict, 0 = all (default: 50 for zero-shot, all for eval)",
    )
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.0)
    parser.add_argument("--batch-size", type=int, default=8, help="training and generation")
    parser.add_argument("--grad-accum", type=int, default=2)
    parser.add_argument("--max-seq-length", type=int, default=128)
    parser.add_argument("--eval-steps", type=int, default=50, help="val-loss check interval")
    parser.add_argument("--n-val-eval", type=int, default=200, help="val rows for the val loss")
    parser.add_argument(
        "--adapter-dir",
        type=Path,
        default=None,
        help="where the adapter is saved/loaded (default checkpoints/[smoke/]llm_lora_adapter)",
    )
    parser.add_argument("--smoke", action="store_true", help="CPU, tiny model; results/smoke/")
    parser.add_argument("--seed", type=int, default=config.SEED)
    parser.add_argument("--no-mlflow", action="store_true", help="disable MLflow logging")
    return parser.parse_args(argv)


def resolve_settings(args: argparse.Namespace) -> argparse.Namespace:
    """Fill in defaults that depend on other flags (smoke mode, the stage being run)."""
    if args.smoke:
        for key, value in SMOKE_OVERRIDES.items():
            setattr(args, key, value)
    if args.model_name is None:
        args.model_name = llm_lora.DEFAULT_UNSLOTH_MODEL
    # How many test rows each stage predicts (0 = all). Smoke runs use all 64 smoke rows.
    if args.n_test is not None:
        args.n_test_zeroshot = args.n_test_eval = args.n_test
    else:
        args.n_test_zeroshot = 0 if args.smoke else ZEROSHOT_N_TEST
        args.n_test_eval = 0
    # 4-bit weights need a GPU (bitsandbytes / Unsloth); smoke runs stay in float32 on CPU.
    args.load_in_4bit = not args.smoke
    args.inference_max_seq_length = max(INFERENCE_MAX_SEQ_LENGTH, args.max_seq_length)
    # Resolved now (not as an argparse default) so tests can redirect config.CHECKPOINTS_DIR.
    checkpoints = config.CHECKPOINTS_DIR / "smoke" if args.smoke else config.CHECKPOINTS_DIR
    args.adapter_dir = Path(args.adapter_dir or checkpoints / "llm_lora_adapter")
    args.trainer_dir = checkpoints / "llm_lora_trainer"
    args.warmup_steps = max(1, round(WARMUP_FRACTION * args.max_steps))
    return args


def repo_relative(path: Path) -> str:
    """Path relative to the repo root when inside it, so results files (which are committed)
    do not contain machine-specific folders such as /home/<user>/..."""
    try:
        return str(Path(path).resolve().relative_to(config.ROOT.resolve()))
    except ValueError:
        return str(path)


def free_memory() -> None:
    """Release the previous step's model before the next one loads (matters on a 16 GB T4)."""
    gc.collect()
    if "torch" in sys.modules:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def report_outputs(raw_outputs: list[str], preds: list[int]) -> dict[str, Any]:
    """Print and return how many answers could not be parsed, with examples."""
    summary = llm_lora.summarize_outputs(raw_outputs, preds)
    n_bad = sum(p == config.UNPARSEABLE for p in preds)
    print(f"unparseable answers: {n_bad}/{len(preds)} (counted as wrong)")
    for example in summary["unparseable_examples"]:
        print(f"  {example!r}")
    return summary


def run_zero_shot(args: argparse.Namespace, test_df: pd.DataFrame) -> dict[str, Any]:
    """Day 8: the base instruct model, no training, on a fixed random subset of the test split."""
    test = llm_lora.sample_rows(test_df, args.n_test_zeroshot, args.seed)
    params = {
        "model_name": args.model_name,
        "backend": args.backend,
        "load_in_4bit": args.load_in_4bit,
        "n_test": len(test),
        "test_subset_seed": args.seed,
        "prompt_version": llm_lora.PROMPT_VERSION,
        "system_prompt": llm_lora.SYSTEM_PROMPT,
        "max_new_tokens": MAX_NEW_TOKENS,
        "decoding": "greedy",
        "batch_size": args.batch_size,
        "inference_max_seq_length": args.inference_max_seq_length,
    }
    print(f"[zero-shot] {args.model_name} ({args.backend}) on {len(test)} test headlines")
    with mlflow_run(ZEROSHOT_NAME, params, enabled=not (args.smoke or args.no_mlflow)) as run:
        model, tokenizer = llm_lora.load_model(
            args.backend, args.model_name, args.inference_max_seq_length, args.load_in_4bit
        )
        start = time.perf_counter()
        raw_outputs, preds = llm_lora.generate_labels(
            model, tokenizer, test["text"], args.batch_size, MAX_NEW_TOKENS, args.backend
        )
        inference_time = time.perf_counter() - start
        record = metrics.save_results(
            ZEROSHOT_NAME,
            test,
            preds,
            params=params,
            train_time_s=0.0,  # no training: the model is used as downloaded
            seed=args.seed,
            smoke=args.smoke,
            extra={
                "inference_time_s": round(inference_time, 2),
                **report_outputs(raw_outputs, preds),
            },
        )
        run.log_metrics({"test_macro_f1": record["macro_f1"], "test_accuracy": record["accuracy"]})
    del model, tokenizer
    free_memory()
    print(f"[zero-shot] macro-F1 {record['macro_f1']:.4f}  accuracy {record['accuracy']:.4f}")
    return record


def make_train_record(
    args: argparse.Namespace,
    *,
    train_points: list[dict],
    val_points: list[dict],
    train_time_s: float,
    n_trainable: int,
    n_total: int,
    n_train: int,
    n_val_eval: int,
    n_too_long: int,
) -> dict[str, Any]:
    """The training log written to llm_lora_train.json and next to the adapter.

    It deliberately has no macro_f1: it describes training (loss curves, LoRA settings),
    not a test-set result, so metrics.load_metrics() skips it.
    """
    return {
        "model": TRAIN_LOG_NAME,
        "record_type": "training_log",
        "base_model": args.model_name,
        "backend": args.backend,
        "load_in_4bit": args.load_in_4bit,
        "prompt_version": llm_lora.PROMPT_VERSION,
        "lora": {
            "r": args.lora_r,
            "alpha": args.lora_alpha,
            "dropout": args.lora_dropout,
            "target_modules": list(llm_lora.DEFAULT_TARGET_MODULES),
        },
        "trainable_params": n_trainable,
        "total_params": n_total,
        "trainable_pct": round(100.0 * n_trainable / n_total, 3),
        "max_steps": args.max_steps,
        "learning_rate": args.lr,
        "lr_schedule": "linear",
        "warmup_steps": args.warmup_steps,
        "batch_size": args.batch_size,
        "grad_accum": args.grad_accum,
        "effective_batch_size": args.batch_size * args.grad_accum,
        "max_seq_length": args.max_seq_length,
        "n_train": n_train,
        "n_train_too_long": n_too_long,
        "n_val_eval": n_val_eval,
        "eval_steps": args.eval_steps,
        "train_loss": train_points,
        "val_loss": val_points,
        "val_loss_start": val_points[0]["eval_loss"] if val_points else None,
        "val_loss_end": val_points[-1]["eval_loss"] if val_points else None,
        "train_time_s": round(train_time_s, 2),
        "seed": args.seed,
        "smoke": args.smoke,
        "adapter_dir": repo_relative(args.adapter_dir),
        "git_commit": metrics.git_commit(),
        "timestamp": datetime.now(UTC).isoformat(timespec="seconds"),
    }


def run_train(args: argparse.Namespace, splits: dict[str, pd.DataFrame]) -> dict[str, Any]:
    """Day 9: attach LoRA adapters, fine-tune on train, watch the loss on a val subset, save."""
    train_df = splits["train"]
    val_eval = llm_lora.sample_rows(splits["val"], args.n_val_eval, args.seed)
    print(f"[train] {args.model_name} ({args.backend}) train={len(train_df)} val={len(val_eval)}")
    model, tokenizer = llm_lora.load_model(
        args.backend, args.model_name, args.max_seq_length, args.load_in_4bit
    )
    model = llm_lora.add_lora(
        model,
        args.backend,
        r=args.lora_r,
        alpha=args.lora_alpha,
        dropout=args.lora_dropout,
        target_modules=llm_lora.DEFAULT_TARGET_MODULES,
        seed=args.seed,
    )
    n_trainable, n_total = llm_lora.trainable_parameters(model)
    print(f"[train] trainable parameters: {n_trainable:,} of {n_total:,}")
    n_too_long = llm_lora.count_too_long(tokenizer, train_df, args.max_seq_length)
    if n_too_long:
        print(f"[train] WARNING: {n_too_long} train rows exceed {args.max_seq_length} tokens")

    params = {
        "base_model": args.model_name,
        "backend": args.backend,
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "lora_dropout": args.lora_dropout,
        "max_steps": args.max_steps,
        "learning_rate": args.lr,
        "batch_size": args.batch_size,
        "grad_accum": args.grad_accum,
    }
    with mlflow_run(TRAIN_LOG_NAME, params, enabled=not (args.smoke or args.no_mlflow)) as run:
        start = time.perf_counter()
        history = llm_lora.train_lora(
            model,
            tokenizer,
            train_df,
            val_eval,
            output_dir=args.trainer_dir,
            backend=args.backend,
            max_steps=args.max_steps,
            learning_rate=args.lr,
            batch_size=args.batch_size,
            grad_accum=args.grad_accum,
            warmup_steps=args.warmup_steps,
            eval_steps=args.eval_steps,
            logging_steps=1 if args.smoke else 10,
            max_seq_length=args.max_seq_length,
            seed=args.seed,
        )
        train_time = time.perf_counter() - start

        # Only the small adapter (A and B matrices) is saved, plus the tokenizer and its
        # chat template so evaluation formats prompts exactly as training did.
        args.adapter_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(str(args.adapter_dir))
        tokenizer.save_pretrained(str(args.adapter_dir))

        train_points, val_points = llm_lora.split_log_history(history)
        record = make_train_record(
            args,
            train_points=train_points,
            val_points=val_points,
            train_time_s=train_time,
            n_trainable=n_trainable,
            n_total=n_total,
            n_train=len(train_df),
            n_val_eval=len(val_eval),
            n_too_long=n_too_long,
        )
        log_path = config.output_dirs(args.smoke)["metrics"] / f"{TRAIN_LOG_NAME}.json"
        for path in (log_path, args.adapter_dir / llm_lora.TRAIN_INFO_FILE):
            path.write_text(json.dumps(record, indent=2), encoding="utf-8")
        for point in train_points:
            run.log_metrics({"train_loss": point["loss"]}, step=point["step"])
        for point in val_points:
            run.log_metrics({"val_loss": point["eval_loss"]}, step=point["step"])
        run.log_artifact(log_path)
    del model, tokenizer
    free_memory()
    print(
        f"[train] {train_time:.0f}s; val loss {record['val_loss_start']} -> "
        f"{record['val_loss_end']}; adapter saved to {args.adapter_dir}"
    )
    return record


def run_eval(args: argparse.Namespace, test_df: pd.DataFrame) -> dict[str, Any]:
    """Day 10: base model + saved adapter on the test split (once, after training is done)."""
    info = llm_lora.read_adapter_info(args.adapter_dir)  # also checks the prompt version
    train_info = info["train_info"]
    base_model = info["base_model"]
    if base_model != args.model_name:
        print(f"[eval] using the adapter's own base model {base_model}")
    test = llm_lora.sample_rows(test_df, args.n_test_eval, args.seed)
    params = {
        "base_model": base_model,
        "backend": args.backend,
        "train_backend": train_info.get("backend"),
        "load_in_4bit": args.load_in_4bit,
        **info["lora"],  # r, alpha, dropout, target_modules (from adapter_config.json)
        "max_steps": train_info.get("max_steps"),
        "learning_rate": train_info.get("learning_rate"),
        "effective_batch_size": train_info.get("effective_batch_size"),
        "n_test": len(test),
        "test_subset_seed": args.seed,
        "prompt_version": llm_lora.PROMPT_VERSION,
        "max_new_tokens": MAX_NEW_TOKENS,
        "decoding": "greedy",
        "inference_max_seq_length": args.inference_max_seq_length,
        "adapter_dir": repo_relative(args.adapter_dir),
    }
    print(f"[eval] adapter {args.adapter_dir} on {len(test)} test headlines")
    with mlflow_run(LORA_NAME, params, enabled=not (args.smoke or args.no_mlflow)) as run:
        model, tokenizer = llm_lora.load_model(
            args.backend,
            base_model,
            args.inference_max_seq_length,
            args.load_in_4bit,
            adapter_dir=args.adapter_dir,
        )
        start = time.perf_counter()
        raw_outputs, preds = llm_lora.generate_labels(
            model, tokenizer, test["text"], args.batch_size, MAX_NEW_TOKENS, args.backend
        )
        inference_time = time.perf_counter() - start
        record = metrics.save_results(
            LORA_NAME,
            test,
            preds,
            params=params,
            train_time_s=train_info["train_time_s"],  # cost of the LoRA fine-tune
            seed=args.seed,
            smoke=args.smoke,
            extra={
                "inference_time_s": round(inference_time, 2),
                "val_loss_start": train_info.get("val_loss_start"),
                "val_loss_end": train_info.get("val_loss_end"),
                **report_outputs(raw_outputs, preds),
            },
        )
        run.log_metrics({"test_macro_f1": record["macro_f1"], "test_accuracy": record["accuracy"]})
    del model, tokenizer
    free_memory()
    print(f"[eval] macro-F1 {record['macro_f1']:.4f}  accuracy {record['accuracy']:.4f}")
    return record


def main(argv: list[str] | None = None) -> dict[str, dict[str, Any]]:
    args = resolve_settings(parse_args(argv))
    # TRL's SFTTrainer refuses to train when several GPUs are visible to one process (e.g.
    # Kaggle's "GPU T4 x2"); one GPU is plenty here. Must be set before CUDA starts up.
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
    if args.backend == "unsloth":
        # Unsloth patches transformers, trl and peft, so it must be imported before them.
        try:
            import unsloth  # noqa: F401
        except ImportError as err:
            raise SystemExit(
                "Unsloth is not installed (it needs an NVIDIA GPU, see requirements-gpu.txt). "
                "Run this in notebooks/colab_runner.ipynb, or use --smoke / --backend hf on CPU."
            ) from err
    config.set_seed(args.seed)
    splits = data.load_run_splits(smoke=args.smoke)
    print(f"mode={args.mode} backend={args.backend} model={args.model_name} smoke={args.smoke}")

    results: dict[str, dict[str, Any]] = {}
    if args.mode in ("zero-shot", "all"):
        results["zero-shot"] = run_zero_shot(args, splits["test"])
    if args.mode in ("train", "all"):
        results["train"] = run_train(args, splits)
    if args.mode in ("eval", "all"):
        results["eval"] = run_eval(args, splits["test"])
    return results


if __name__ == "__main__":
    main()
