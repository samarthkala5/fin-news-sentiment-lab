"""Render the Hugging Face model card for the fine-tuned DistilBERT; optionally push it (Day 14).

    python scripts/push_to_hub.py            # dry run: write results/model_card/README.md, print it
    python scripts/push_to_hub.py --push --repo-id USER/NAME \\
        --checkpoint checkpoints/distilbert/best

The card (README.md) is built ONLY from files that scripts wrote:
  results/metrics/distilbert.json   test scores, hyperparameters, per-epoch history
  results/metrics/comparison.json   bootstrap 95% CIs and McNemar tests (compare_models.py)
  data/splits/split_report.json     dataset, licence, cleaning counts, split sizes
A missing file is shown as "not available yet", never as a number. From comparison.json
the card uses (anything else is reported as an unexpected format):
  alpha, n_resamples
  models[<model>]: macro_f1, ci_low, ci_high          (percentile bootstrap CI)
  pairs[i]: model_a, model_b, mcnemar{b, c, p_value}, macro_f1_diff{diff, low, high},
            p_holm, significant_holm                  (McNemar p after Holm's correction)

--push uploads the checkpoint folder (model weights, config, tokenizer) and the card with
huggingface_hub, but only if the folder's training_info.json matches distilbert.json (same
training run). The token comes from HF_TOKEN or `hf auth login`; it is never stored here.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from fnsl import config, data

MODEL = "distilbert"  # the metrics file and comparison entry this card describes
BASE_MODEL = "distilbert/distilbert-base-uncased"
# The fine-tuned weights keep the base model's licence (Apache-2.0 on its Hub page).
MODEL_LICENSE = "apache-2.0"
NOT_AVAILABLE = "_Not available yet._"
PAIR_KEYS = {"model_a", "model_b", "mcnemar", "macro_f1_diff", "p_holm", "significant_holm"}
EXAMPLE_TEXT = "$AAPL beats earnings estimates and raises full-year guidance"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render (and optionally push) the model card")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true", help="only write the card (default)")
    mode.add_argument("--push", action="store_true", help="upload model + card to the Hub")
    parser.add_argument("--repo-id", default=None, help="USER/NAME on the Hugging Face Hub")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=config.CHECKPOINTS_DIR / MODEL / "best",
        help="folder written by train_distilbert.py",
    )
    parser.add_argument("--private", action="store_true", help="create the Hub repo as private")
    parser.add_argument("--metrics-dir", type=Path, default=config.METRICS_DIR)
    parser.add_argument("--split-report", type=Path, default=data.split_report_path())
    parser.add_argument("--out", type=Path, default=config.RESULTS_DIR / "model_card" / "README.md")
    args = parser.parse_args(argv)
    if args.push and (args.repo_id is None or args.repo_id.count("/") != 1):
        parser.error("--push needs --repo-id USER/NAME")
    return args


def load_json(path: Path) -> dict[str, Any] | None:
    """The parsed file, or None if it does not exist yet."""
    path = Path(path)
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def f4(value: float) -> str:
    """Scores are shown with 4 decimals everywhere in the card."""
    return f"{value:.4f}"


def comparison_problem(comparison: dict | None, distilbert: dict | None) -> str | None:
    """Why the comparison cannot be shown, or None if it can."""
    rerun = "run `python scripts/compare_models.py`"
    if comparison is None:
        return f"`results/metrics/comparison.json` does not exist yet ({rerun})."
    models, pairs = comparison.get("models"), comparison.get("pairs")
    well_formed = (
        isinstance(models, dict)
        and isinstance(pairs, list)
        and {"alpha", "n_resamples"} <= comparison.keys()
        and all(isinstance(entry, dict) and "macro_f1" in entry for entry in models.values())
        and all(isinstance(pair, dict) and PAIR_KEYS <= pair.keys() for pair in pairs)
    )
    if not well_formed:
        return "`results/metrics/comparison.json` has an unexpected format, so it is not shown."
    if comparison.get("smoke"):
        return "`comparison.json` comes from a smoke run, so it is not shown."
    if distilbert is None:
        return f"`results/metrics/{MODEL}.json` does not exist yet, so nothing can be compared."
    if MODEL not in models:
        return f"`comparison.json` has no DistilBERT entry yet ({rerun})."
    # Same predictions must give the same macro-F1; a mismatch means a stale comparison.
    if abs(models[MODEL]["macro_f1"] - distilbert["macro_f1"]) > 1e-9:
        return f"`comparison.json` is older than `{MODEL}.json` ({rerun})."
    return None


def ci_text(entry: dict) -> str:
    """'0.7123 to 0.7654' from a comparison.json model entry, if it has both bounds."""
    if "ci_low" in entry and "ci_high" in entry:
        return f"{f4(entry['ci_low'])} to {f4(entry['ci_high'])}"
    return "not available"


def ci_level(comparison: dict | None) -> int:
    """Confidence level in percent: 95 for alpha = 0.05 (the project's default)."""
    return round(100 * (1 - comparison["alpha"])) if comparison else 95


def front_matter(distilbert: dict | None, split_report: dict | None) -> str:
    """YAML metadata block that the Hub reads (licence, base model, dataset, task)."""
    base = distilbert["params"].get("model_name", BASE_MODEL) if distilbert else BASE_MODEL
    dataset = split_report["dataset"] if split_report else config.DATASET_NAME
    return "\n".join(
        [
            "---",
            "language: en",
            f"license: {MODEL_LICENSE}",
            "library_name: transformers",
            "pipeline_tag: text-classification",
            f"base_model: {base}",
            "datasets:",
            f"- {dataset}",
            "metrics:",
            "- f1",
            "- accuracy",
            "tags:",
            "- finance",
            "- sentiment-analysis",
            "- text-classification",
            "---",
        ]
    )


def data_section(split_report: dict | None) -> str:
    if split_report is None:
        return f"{NOT_AVAILABLE} (`data/splits/split_report.json` was not found.)"
    dataset = split_report["dataset"]
    cleaning, sizes = split_report["cleaning"], split_report["sizes"]
    balance = split_report["class_balance"]["all"]
    shares = ", ".join(f"{label} {balance[f'{label}_%']}%" for label in config.LABELS)
    return "\n".join(
        [
            f"- Dataset: [`{dataset}`](https://huggingface.co/datasets/{dataset}), Hub revision "
            f"`{split_report['revision']}`, licence **{split_report['license']}**.",
            f"- Cleaning: {cleaning['n_raw']:,} raw posts, {cleaning['n_clean']:,} after removing "
            "URL-only posts, near-duplicates (same text after lower-casing and removing links) "
            "and near-duplicates with conflicting labels.",
            f"- One fixed, stratified split (seed {split_report['seed']}): "
            f"train {sizes['train']:,}, validation {sizes['val']:,}, test {sizes['test']:,}.",
            f"- Class balance (all data): {shares}.",
            "- Epochs were chosen on the validation split only; the test split was used once.",
        ]
    )


# (key in params, label in the card)
HYPERPARAMETERS = [
    ("model_name", "Base model"),
    ("epochs", "Epochs"),
    ("batch_size", "Batch size"),
    ("lr", "Peak learning rate"),
    ("weight_decay", "Weight decay"),
    ("warmup_ratio", "Warmup (share of steps)"),
    ("num_training_steps", "Optimizer steps"),
    ("max_length", "Max sequence length (tokens)"),
    ("max_grad_norm", "Gradient clipping (max norm)"),
    ("optimizer", "Optimizer"),
    ("scheduler", "Learning-rate schedule"),
    ("device", "Device"),
    ("mixed_precision", "Mixed precision (float16)"),
    ("best_epoch", "Best epoch (by validation macro-F1)"),
]


def training_section(distilbert: dict | None) -> str:
    if distilbert is None:
        return f"{NOT_AVAILABLE} (`results/metrics/{MODEL}.json` was not found.)"
    params = distilbert["params"]
    lines = ["| Setting | Value |", "|---|---|"]
    lines += [f"| {label} | {params[key]} |" for key, label in HYPERPARAMETERS if key in params]
    if "val_macro_f1" in params:
        lines.append(f"| Validation macro-F1 at best epoch | {f4(params['val_macro_f1'])} |")
    lines.append(f"| Seed | {distilbert['seed']} |")
    lines.append(f"| Training time (s) | {distilbert['train_time_s']} |")
    if params.get("history"):
        lines += ["", "| Epoch | Train loss | Val loss | Val macro-F1 |", "|---|---|---|---|"]
        lines += [
            f"| {h['epoch']} | {f4(h['train_loss'])} | {f4(h['val_loss'])} | "
            f"{f4(h['val_macro_f1'])} |"
            for h in params["history"]
        ]
    return "\n".join(lines)


def evaluation_section(distilbert: dict | None, comparison: dict | None) -> str:
    if distilbert is None:
        return f"{NOT_AVAILABLE} (`results/metrics/{MODEL}.json` was not found.)"
    problem = comparison_problem(comparison, distilbert)
    ci = ci_text(comparison["models"][MODEL]) if problem is None else "not available"
    level = ci_level(comparison if problem is None else None)
    lines = [
        f"Held-out test split, {distilbert['n_test']} posts, predicted once with the best epoch.",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| Macro-F1 | {f4(distilbert['macro_f1'])} |",
        f"| Macro-F1, {level}% bootstrap CI | {ci} |",
        f"| Accuracy | {f4(distilbert['accuracy'])} |",
    ]
    lines += [
        f"| F1 {label} | {f4(distilbert['per_class_f1'][label])} |" for label in config.LABELS
    ]
    if problem is not None:
        lines += ["", f"Confidence interval: {problem}"]
    return "\n".join(lines)


def comparison_section(distilbert: dict | None, comparison: dict | None) -> str:
    problem = comparison_problem(comparison, distilbert)
    if problem is not None:
        return f"{NOT_AVAILABLE} {problem}"
    level, alpha = ci_level(comparison), comparison["alpha"]
    models = sorted(comparison["models"].items(), key=lambda item: -item[1]["macro_f1"])
    lines = [
        f"Every model is scored on the test split; {level}% CIs come from "
        f"{comparison['n_resamples']} bootstrap resamples of the test posts.",
        "",
        f"| Model | Test macro-F1 | {level}% bootstrap CI |",
        "|---|---|---|",
    ]
    lines += [
        f"| `{name}` | {f4(entry['macro_f1'])} | {ci_text(entry)} |" for name, entry in models
    ]
    pairs = [p for p in comparison["pairs"] if MODEL in (p["model_a"], p["model_b"])]
    if pairs:
        lines += [
            "",
            "Paired tests on the same test posts. The difference is DistilBERT minus the other "
            "model (paired bootstrap). McNemar's test only looks at posts that exactly one of the "
            "two models gets right; Holm's correction adjusts its p-values because several pairs "
            "are tested at once.",
            "",
            f"| DistilBERT vs | Macro-F1 difference ({level}% CI) | Posts only DistilBERT / "
            f"only the other gets right | McNemar p | Holm-adjusted p | "
            f"Significant (Holm, alpha {alpha}) |",
            "|---|---|---|---|---|---|",
        ]
        lines += [pair_row(pair) for pair in pairs]
    # Pairs compare_models.py could not pair up (e.g. different test ids), with its reason.
    for pair in comparison.get("skipped_pairs", []):
        if MODEL in (pair.get("model_a"), pair.get("model_b")):
            other = pair["model_b"] if pair["model_a"] == MODEL else pair["model_a"]
            lines += ["", f"Not compared with `{other}`: {pair.get('reason', 'no reason given')}"]
    return "\n".join(lines)


def pair_row(pair: dict) -> str:
    """One table row for a pair that includes DistilBERT, read as 'DistilBERT vs other'."""
    mcnemar, diff = pair["mcnemar"], pair["macro_f1_diff"]
    if pair["model_a"] == MODEL:
        # comparison.json stores A - B and b = "only A right", c = "only B right".
        other, only_ours, only_theirs = pair["model_b"], mcnemar["b"], mcnemar["c"]
        d, low, high = diff["diff"], diff["low"], diff["high"]
    else:  # DistilBERT is model B: flip the difference and swap the counts
        other, only_ours, only_theirs = pair["model_a"], mcnemar["c"], mcnemar["b"]
        d, low, high = -diff["diff"], -diff["high"], -diff["low"]
    significant = "yes" if pair["significant_holm"] else "no"
    return (
        f"| `{other}` | {d:+.4f} ({low:+.4f} to {high:+.4f}) | {only_ours} / {only_theirs} | "
        f"{mcnemar['p_value']:.3g} | {pair['p_holm']:.3g} | {significant} |"
    )


def usage_section(repo_id: str | None, distilbert: dict | None) -> str:
    model_id = repo_id or "<your-username>/<repo-name>"
    max_length = distilbert["params"].get("max_length") if distilbert else None
    length_args = f", max_length={max_length}" if max_length else ""
    return f"""```python
from transformers import pipeline

classifier = pipeline("text-classification", model="{model_id}")
# Returns one {{'label': ..., 'score': ...}} dict per text.
print(classifier("{EXAMPLE_TEXT}"))
```

Without the pipeline:

```python
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("{model_id}")
model = AutoModelForSequenceClassification.from_pretrained("{model_id}").eval()
texts = ["{EXAMPLE_TEXT}"]
inputs = tokenizer(texts, truncation=True{length_args}, return_tensors="pt")
with torch.no_grad():
    logits = model(**inputs).logits
print(model.config.id2label[int(logits.argmax(dim=-1))])  # Bearish, Bullish or Neutral
```

To reproduce the training run: `python scripts/train_distilbert.py` on a GPU (see
`notebooks/colab_runner.ipynb`); `--smoke` runs a one-minute check on CPU."""


LIMITATIONS = """\
- Trained and tested on short English financial posts (mostly one headline-style tweet).
  It may not transfer to long news articles, other languages, other sources or later periods.
- Only three classes (Bearish, Bullish, Neutral): no sentiment strength, no target company,
  no mixed sentiment. Texts are cut off after the maximum sequence length.
- Labels come from the dataset authors and contain some noise; all scores come from one fixed
  split of one dataset, and the CI only reflects test-set sampling noise.
- Predictions were never checked against market prices. **Do not use it for trading or
  investment decisions.**
- It inherits the biases of DistilBERT's pre-training data (English books and Wikipedia)."""

INTENDED_USE = """\
Research and education: a baseline for comparing financial-sentiment classifiers, and a worked
example of fine-tuning a Transformer with a hand-written PyTorch training loop. Not intended for
automated trading, investment advice, or decisions about people."""


def render_card(
    distilbert: dict | None,
    comparison: dict | None,
    split_report: dict | None,
    repo_id: str | None = None,
) -> str:
    """The full model card (YAML front matter + Markdown). Pass None for a missing file."""
    labels = ", ".join(f"{i} = {name}" for i, name in config.ID2LABEL.items())
    status = (
        ""
        if distilbert is not None
        else "\n> **Status:** the fine-tuned model has not been evaluated yet; scores appear "
        "here once `results/metrics/distilbert.json` exists.\n"
    )
    sections = [
        front_matter(distilbert, split_report),
        "",
        "# DistilBERT fine-tuned for financial news sentiment",
        "",
        "Classifies short English financial news posts as **Bearish**, **Bullish** or "
        "**Neutral**. Part of the *fin-news-sentiment-lab* project, which compares TF-IDF, "
        "Keras, PyTorch and LoRA-fine-tuned LLM models with McNemar tests and bootstrap CIs.",
        "",
        "> This card is generated by `scripts/push_to_hub.py`. Every number is copied from "
        f"`results/metrics/{MODEL}.json`, `results/metrics/comparison.json` or "
        "`data/splits/split_report.json`; anything those files do not contain yet is marked "
        "*not available yet*.",
        status,
        "## Task",
        "",
        f"Single-label text classification with 3 classes ({labels}).",
        "",
        "## Base model",
        "",
        f"[`{BASE_MODEL}`](https://huggingface.co/{BASE_MODEL}) (Apache-2.0), a 6-layer "
        "distilled version of BERT. All weights were fine-tuned together with a new "
        "classification head, using an explicit PyTorch loop: AdamW, linear warmup then "
        "linear decay of the learning rate, gradient clipping, and float16 mixed precision on GPU.",
        "",
        "## Training data",
        "",
        data_section(split_report),
        "",
        "## Training hyperparameters",
        "",
        training_section(distilbert),
        "",
        "## Evaluation",
        "",
        evaluation_section(distilbert, comparison),
        "",
        "## Comparison with the other models",
        "",
        comparison_section(distilbert, comparison),
        "",
        "## Limitations",
        "",
        LIMITATIONS,
        "",
        "## Intended use",
        "",
        INTENDED_USE,
        "",
        "## How to use",
        "",
        usage_section(repo_id, distilbert),
        "",
    ]
    return "\n".join(sections)


def checkpoint_problem(checkpoint: Path, distilbert: dict) -> str | None:
    """Why the weights in `checkpoint` must not be published with this card, or None.

    The card describes results/metrics/distilbert.json, so the uploaded weights must come
    from that same training run. train_distilbert.py saves the run's hyperparameters and
    per-epoch history twice: in training_info.json next to the weights and in the metrics
    file. If the two copies differ, the folder holds another run (an extra seed, a newer
    or an interrupted run) and the card's numbers would not describe it.
    """
    if not (checkpoint / "config.json").exists():
        return f"{checkpoint} is not a saved checkpoint."
    info = load_json(checkpoint / "training_info.json")
    if info is None:
        return f"{checkpoint} has no training_info.json (an interrupted run?)."
    if info.get("params") != distilbert["params"] or info.get("seed") != distilbert["seed"]:
        return f"{checkpoint} comes from a different training run than {MODEL}.json."
    return None


def push_to_hub(card: str, repo_id: str, checkpoint: Path, private: bool) -> str:
    """Create the Hub repo if needed, then upload the checkpoint folder and the card."""
    from huggingface_hub import HfApi, get_token  # only needed for --push

    if get_token() is None:
        raise SystemExit("No Hugging Face token found: run `hf auth login` or set HF_TOKEN.")
    api = HfApi()
    url = api.create_repo(repo_id, private=private, exist_ok=True)
    # Model weights, config.json, tokenizer files and training_info.json.
    api.upload_folder(
        repo_id=repo_id, folder_path=checkpoint, commit_message="Upload fine-tuned DistilBERT"
    )
    api.upload_file(
        path_or_fileobj=card.encode("utf-8"),
        path_in_repo="README.md",
        repo_id=repo_id,
        commit_message="Update model card",
    )
    return str(url)


def main(argv: list[str] | None = None) -> dict:
    args = parse_args(argv)
    paths = {
        "distilbert": args.metrics_dir / f"{MODEL}.json",
        "comparison": args.metrics_dir / "comparison.json",
        "split_report": args.split_report,
    }
    inputs = {name: load_json(path) for name, path in paths.items()}
    if inputs["distilbert"] is not None and inputs["distilbert"].get("smoke"):
        print(f"[model card] {paths['distilbert']} is a smoke run: not used")
        inputs["distilbert"] = None
    for name, path in paths.items():
        if not path.exists():
            print(f"[model card] {name}: not available yet ({path} missing)")

    card = render_card(
        inputs["distilbert"], inputs["comparison"], inputs["split_report"], repo_id=args.repo_id
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(card, encoding="utf-8")
    print(card)
    print(f"[model card] written to {args.out}")

    url = None
    if args.push:
        # Refuse to publish a model whose scores do not exist yet, or weights from another run.
        if inputs["distilbert"] is None:
            raise SystemExit("Refusing to push: no real results/metrics/distilbert.json yet.")
        problem = checkpoint_problem(args.checkpoint, inputs["distilbert"])
        if problem is not None:
            raise SystemExit(f"Refusing to push: {problem}")
        url = push_to_hub(card, args.repo_id, args.checkpoint, args.private)
        print(f"[model card] pushed to {url}")
    return {
        "card_path": str(args.out),
        "available": {name: value is not None for name, value in inputs.items()},
        "pushed_url": url,
    }


if __name__ == "__main__":
    main()
