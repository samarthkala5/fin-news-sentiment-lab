"""Heavy tests for the PyTorch DistilBERT code: batching, optimizer, learning-rate schedule,
the training loop, and scripts/train_distilbert.py end to end.

They use a tiny randomly initialised DistilBERT (1 layer, 32 dimensions) and a 15-token
tokenizer built in memory, so they run in seconds on CPU and need no download.
Skipped in CI (`pytest -m "not heavy"`): they need torch and transformers.
"""

from __future__ import annotations

import importlib.util
import json
import types

import pandas as pd
import pytest

pytestmark = pytest.mark.heavy
torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from transformers import (  # noqa: E402
    BertTokenizer,
    DistilBertConfig,
    DistilBertForSequenceClassification,
    DistilBertTokenizer,
)

from fnsl import config, metrics  # noqa: E402
from fnsl.torch_model import (  # noqa: E402
    TextDataset,
    count_truncated,
    make_loader,
    make_optimizer,
    make_scheduler,
    predict,
    train_one_epoch,
)

CPU = torch.device("cpu")
# Every word the toy data uses; anything else (numbers, "item7") becomes [UNK].
VOCAB = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"] + (
    "company falls plunges cut rises beats upgrade announces meeting update".split()
)


def _load_script(name: str):
    """Import scripts/<name>.py as a module so tests can call its main()."""
    spec = importlib.util.spec_from_file_location(name, config.ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def tokenizer() -> DistilBertTokenizer:
    return DistilBertTokenizer(vocab={token: i for i, token in enumerate(VOCAB)})


def _tiny_model() -> DistilBertForSequenceClassification:
    """A 1-layer, 32-dimension DistilBERT; seeded, so every call gives the same weights."""
    torch.manual_seed(0)
    tiny = DistilBertConfig(
        vocab_size=len(VOCAB),
        dim=32,
        hidden_dim=64,
        n_layers=1,
        n_heads=2,
        max_position_embeddings=64,
        num_labels=config.NUM_LABELS,
    )
    return DistilBertForSequenceClassification(tiny)


@pytest.fixture
def tiny_model() -> DistilBertForSequenceClassification:
    return _tiny_model()


@pytest.fixture
def tiny_model_dir(tmp_path, tiny_model, tokenizer):
    """The tiny model and tokenizer saved like a Hub model, usable as --model-name."""
    path = tmp_path / "tiny-distilbert"
    tiny_model.save_pretrained(path)
    tokenizer.save_pretrained(path)
    return path


@pytest.fixture
def mixed_lengths_df() -> pd.DataFrame:
    """Four texts of 1, 2, 6 and 1 words (3, 4, 8 and 3 tokens with [CLS] and [SEP])."""
    texts = ["falls", "rises beats", "cut cut cut cut cut cut", "update"]
    return pd.DataFrame({"id": range(4), "text": texts, "label": [0, 1, 0, 2]})


def test_text_dataset_returns_one_example_with_its_label():
    encodings = {"input_ids": [[2, 6, 3], [2, 9, 10, 3]], "attention_mask": [[1, 1, 1], [1] * 4]}
    dataset = TextDataset(encodings, [0, 2])
    assert len(dataset) == 2
    assert dataset[1] == {"input_ids": [2, 9, 10, 3], "attention_mask": [1, 1, 1, 1], "labels": 2}
    with pytest.raises(ValueError):
        TextDataset(encodings, [0, 1, 2])


def test_loader_pads_each_batch_only_to_its_longest_text(mixed_lengths_df, tokenizer):
    loader = make_loader(mixed_lengths_df, tokenizer, 64, batch_size=2, shuffle=False, seed=0)
    batches = list(loader)
    assert set(batches[0]) == {"input_ids", "attention_mask", "labels"}
    assert [b["input_ids"].shape[1] for b in batches] == [4, 8], "dynamic padding per batch"
    # The 3-token text is padded with one [PAD]; attention_mask 0 tells the model to ignore it.
    assert batches[0]["attention_mask"].tolist() == [[1, 1, 1, 0], [1, 1, 1, 1]]
    assert batches[0]["input_ids"][0, -1].item() == tokenizer.pad_token_id
    # shuffle=False keeps the DataFrame order, so predictions line up with the rows.
    assert torch.cat([b["labels"] for b in batches]).tolist() == [0, 1, 0, 2]


def test_loader_truncates_long_texts_and_keeps_sep(mixed_lengths_df, tokenizer):
    batches = list(make_loader(mixed_lengths_df, tokenizer, 5, batch_size=4, shuffle=False, seed=0))
    assert batches[0]["input_ids"].shape[1] == 5
    assert batches[0]["input_ids"][2, -1].item() == tokenizer.sep_token_id
    assert count_truncated(mixed_lengths_df["text"].tolist(), tokenizer, max_length=5) == 1


def test_loader_drops_token_type_ids(mixed_lengths_df):
    # transformers 5 loads the DistilBERT checkpoint's tokenizer as a BertTokenizer, which
    # returns token_type_ids. DistilBERT does not use them, so make_loader drops them.
    bert_tokenizer = BertTokenizer(vocab={token: i for i, token in enumerate(VOCAB)})
    assert "token_type_ids" in bert_tokenizer("falls")
    batch = next(iter(make_loader(mixed_lengths_df, bert_tokenizer, 64, 4, False, seed=0)))
    assert "token_type_ids" not in batch


def test_shuffle_order_is_fixed_by_the_seed(toy_df, tokenizer):
    def label_order(seed: int) -> list[int]:
        loader = make_loader(toy_df, tokenizer, 64, batch_size=8, shuffle=True, seed=seed)
        return torch.cat([b["labels"] for b in loader]).tolist()

    assert label_order(1) == label_order(1)
    assert label_order(1) != label_order(2)
    assert label_order(1) != toy_df["label"].tolist(), "shuffle=True must reorder"


def test_optimizer_skips_weight_decay_on_biases_and_layernorm(tiny_model):
    decay, no_decay = make_optimizer(tiny_model, lr=1e-3, weight_decay=0.01).param_groups
    assert decay["weight_decay"] == 0.01 and no_decay["weight_decay"] == 0.0
    assert all(p.ndim >= 2 for p in decay["params"])
    assert all(p.ndim == 1 for p in no_decay["params"])
    layer_norm_scale = tiny_model.distilbert.embeddings.LayerNorm.weight
    assert any(p is layer_norm_scale for p in no_decay["params"])
    n_trainable = sum(1 for p in tiny_model.parameters() if p.requires_grad)
    assert len(decay["params"]) + len(no_decay["params"]) == n_trainable


def test_scheduler_warms_up_then_decays_linearly_to_zero(tiny_model):
    optimizer = make_optimizer(tiny_model, lr=1e-3, weight_decay=0.0)
    scheduler = make_scheduler(optimizer, num_training_steps=20, warmup_ratio=0.1)  # 2 warmup
    lrs = []
    for _ in range(20):
        lrs.append(optimizer.param_groups[0]["lr"])
        optimizer.step()
        scheduler.step()
    assert lrs[0] == 0.0 and lrs[1] == pytest.approx(5e-4), "linear warmup from 0"
    assert lrs[2] == pytest.approx(1e-3) == max(lrs), "peak right after warmup"
    assert all(a >= b for a, b in zip(lrs[2:], lrs[3:], strict=False)), "never rises again"
    assert optimizer.param_groups[0]["lr"] == pytest.approx(0.0), "reaches 0 after the last step"


def test_training_loop_learns_toy_data(toy_df, tiny_model, tokenizer):
    train_loader = make_loader(toy_df, tokenizer, 64, batch_size=8, shuffle=True, seed=0)
    epochs = 10
    optimizer = make_optimizer(tiny_model, lr=5e-3, weight_decay=0.01)
    scheduler = make_scheduler(optimizer, epochs * len(train_loader), warmup_ratio=0.1)
    losses = [
        train_one_epoch(tiny_model, train_loader, optimizer, scheduler, CPU) for _ in range(epochs)
    ]
    assert losses[-1] < 0.5 * losses[0], f"training loss should fall: {losses}"

    eval_loader = make_loader(toy_df, tokenizer, 64, batch_size=16, shuffle=False, seed=0)
    y_pred, eval_loss = predict(tiny_model, eval_loader, CPU)
    assert y_pred.shape == (len(toy_df),)
    assert set(y_pred.tolist()) <= {0, 1, 2}
    assert (y_pred == toy_df["label"].to_numpy()).mean() >= 0.9
    assert eval_loss < losses[0]


# On CPU, autocast(device_type="cuda") warns and switches itself off; that is expected here.
@pytest.mark.filterwarnings("ignore:CUDA is not available")
def test_grad_scaler_branch_gives_the_same_losses_as_plain_float32(toy_df, tokenizer):
    # In real use the GradScaler branch of train_one_epoch (scale, backward, unscale_, clip,
    # step, update) only runs on a GPU. A CPU GradScaler runs the same steps here. Its scale
    # factor is a power of two, so scaling and unscaling are exact and nothing may change.
    def train_losses(scaler) -> list[float]:
        model = _tiny_model()  # same start weights and dropout seed for both runs
        loader = make_loader(toy_df, tokenizer, 64, batch_size=8, shuffle=True, seed=0)
        optimizer = make_optimizer(model, lr=5e-3, weight_decay=0.01)
        scheduler = make_scheduler(optimizer, 3 * len(loader), warmup_ratio=0.1)
        return [train_one_epoch(model, loader, optimizer, scheduler, CPU, scaler) for _ in range(3)]

    scaler = torch.amp.GradScaler("cpu")
    assert scaler.is_enabled()
    assert train_losses(scaler) == train_losses(None)


def test_predict_turns_dropout_off(toy_df, tiny_model, tokenizer):
    loader = make_loader(toy_df, tokenizer, 64, batch_size=16, shuffle=False, seed=0)
    batch = next(iter(loader))
    tiny_model.train()  # dropout on: the same batch gives different outputs each time
    assert not torch.equal(tiny_model(**batch).logits, tiny_model(**batch).logits)

    first_pred, first_loss = predict(tiny_model, loader, CPU)
    second_pred, second_loss = predict(tiny_model, loader, CPU)
    assert not tiny_model.training, "predict() leaves the model in eval mode"
    assert first_loss == second_loss and (first_pred == second_pred).all()


def test_script_defaults_and_argument_checks():
    script = _load_script("train_distilbert")
    assert script.default_run_name(config.SEED) == "distilbert"
    assert script.default_run_name(7) == "distilbert_seed7"
    smoke_dir = script.default_output_dir("distilbert", smoke=True)
    assert smoke_dir == config.CHECKPOINTS_DIR / "smoke" / "distilbert"
    full = script.parse_args([])
    assert (full.epochs, full.batch_size, full.max_length, full.lr) == (3, 32, 64, 2e-5)
    smoke = script.parse_args(["--smoke"])
    assert (smoke.epochs, smoke.batch_size) == (1, 16)
    # --eval-only takes the batch size from the checkpoint's training_info.json
    assert script.parse_args(["--eval-only", "--checkpoint", "best"]).batch_size is None
    with pytest.raises(SystemExit):
        script.parse_args(["--eval-only"])  # needs --checkpoint


@pytest.mark.skipif(not (config.SPLITS_DIR / "test.csv").exists(), reason="no split files")
def test_train_script_smoke_then_eval_only(tmp_results, tmp_path, tiny_model_dir, monkeypatch):
    script = _load_script("train_distilbert")
    checkpoints = tmp_path / "checkpoints"
    record = script.main(
        ["--smoke", "--no-mlflow", "--model-name", str(tiny_model_dir)]
        + ["--output-dir", str(checkpoints), "--lr", "1e-3"]
    )

    smoke_dir = tmp_results / "smoke"
    on_disk = json.loads((smoke_dir / "metrics" / "distilbert.json").read_text())
    assert all(key in on_disk for key in metrics.REQUIRED_METRIC_KEYS)
    assert on_disk["smoke"] is True and on_disk["n_test"] == config.SMOKE_SIZES["test"]
    params = on_disk["params"]
    assert params["epochs"] == 1 and params["best_epoch"] == 1 and len(params["history"]) == 1
    assert params["val_macro_f1"] == params["history"][0]["val_macro_f1"]
    assert params["device"] == "cpu" and params["mixed_precision"] is False
    assert params["max_length"] == 64 and params["n_train"] == config.SMOKE_SIZES["train"]
    predictions = metrics.load_predictions(smoke_dir / "predictions" / "distilbert.csv")
    assert len(predictions) == config.SMOKE_SIZES["test"]
    for name in ("config.json", "model.safetensors", "tokenizer.json", "training_info.json"):
        assert (checkpoints / "best" / name).exists(), name
    assert not (tmp_results / "metrics").exists(), "smoke run must not write real results"

    # --eval-only reloads the saved best checkpoint and reproduces the test predictions.
    # The smoke flag comes from the checkpoint, so it still writes to results/smoke/.
    # Record the batch size it predicts with: it must be the training run's (16 for smoke).
    batch_sizes, real_make_loader = [], script.make_loader

    def recording_make_loader(df, tokenizer, max_length, batch_size, **kwargs):
        batch_sizes.append(batch_size)
        return real_make_loader(df, tokenizer, max_length, batch_size, **kwargs)

    monkeypatch.setattr(script, "make_loader", recording_make_loader)
    again = script.main(
        ["--eval-only", "--checkpoint", str(checkpoints / "best"), "--run-name", "reloaded"]
    )
    assert batch_sizes == [params["batch_size"]] == [16]
    assert again["smoke"] is True
    assert again["params"] == record["params"] and again["train_time_s"] == record["train_time_s"]
    reloaded = metrics.load_predictions(smoke_dir / "predictions" / "reloaded.csv")
    assert reloaded["pred"].tolist() == predictions["pred"].tolist()
    assert again["test_loss"] == record["test_loss"]
    # The training commit travels with the checkpoint, so a re-evaluation still names it.
    info = json.loads((checkpoints / "best" / "training_info.json").read_text())
    assert info["git_commit"] and again["train_git_commit"] == info["git_commit"]
    assert record["train_git_commit"] == info["git_commit"]
    assert not (tmp_results / "metrics").exists()


@pytest.mark.skipif(not (config.SPLITS_DIR / "test.csv").exists(), reason="no split files")
def test_train_script_reloads_the_best_validation_epoch_not_the_last(
    tmp_results, tmp_path, tiny_model_dir, monkeypatch
):
    script = _load_script("train_distilbert")
    # Fake validation macro-F1 per epoch, so that epoch 2 of 3 is the best one.
    val_scores = iter([0.2, 0.9, 0.1])
    fake_metrics = types.SimpleNamespace(
        macro_f1=lambda y_true, y_pred: next(val_scores),
        save_results=metrics.save_results,
        git_commit=metrics.git_commit,
    )
    monkeypatch.setattr(script, "metrics", fake_metrics)
    # Remember the weights after every epoch.
    weights_after_epoch = []
    real_train_one_epoch = script.train_one_epoch

    def train_and_remember(model, *args):
        loss = real_train_one_epoch(model, *args)
        weights_after_epoch.append({k: v.clone() for k, v in model.state_dict().items()})
        return loss

    monkeypatch.setattr(script, "train_one_epoch", train_and_remember)

    checkpoints = tmp_path / "checkpoints"
    record = script.main(
        ["--smoke", "--no-mlflow", "--epochs", "3", "--model-name", str(tiny_model_dir)]
        + ["--output-dir", str(checkpoints), "--lr", "1e-3"]
    )
    params = record["params"]
    assert params["best_epoch"] == 2 and params["val_macro_f1"] == 0.9
    assert [h["val_macro_f1"] for h in params["history"]] == [0.2, 0.9, 0.1]
    # The saved (and reloaded for the test split) weights are epoch 2's, not epoch 3's.
    saved = DistilBertForSequenceClassification.from_pretrained(checkpoints / "best").state_dict()
    epoch_2, epoch_3 = weights_after_epoch[1], weights_after_epoch[2]
    assert all(torch.equal(saved[name], weights) for name, weights in epoch_2.items())
    assert not all(torch.equal(saved[name], weights) for name, weights in epoch_3.items())


@pytest.mark.skipif(not (config.SPLITS_DIR / "test.csv").exists(), reason="no split files")
def test_interrupted_run_cannot_reuse_an_older_training_info(
    tmp_results, tmp_path, tiny_model_dir, monkeypatch
):
    script = _load_script("train_distilbert")
    checkpoints = tmp_path / "checkpoints"
    args = ["--smoke", "--no-mlflow", "--model-name", str(tiny_model_dir)]
    args += ["--output-dir", str(checkpoints)]
    script.main(args)  # an earlier run that finished: best/ has weights + training_info.json
    assert (checkpoints / "best" / "training_info.json").exists()

    # A new run saves its epoch 1 into the same best/ folder, then dies in epoch 2.
    real_train_one_epoch, calls = script.train_one_epoch, []

    def disconnect_in_epoch_2(*args):
        calls.append(1)
        if len(calls) == 2:
            raise KeyboardInterrupt("simulated Colab disconnect")
        return real_train_one_epoch(*args)

    monkeypatch.setattr(script, "train_one_epoch", disconnect_in_epoch_2)
    with pytest.raises(KeyboardInterrupt):
        script.main(args + ["--epochs", "2", "--lr", "5e-3"])
    # The old training_info.json is gone, so the new weights cannot be reported with the
    # old run's hyperparameters and history.
    assert not (checkpoints / "best" / "training_info.json").exists()
    with pytest.raises(FileNotFoundError, match="training_info.json"):
        script.main(["--eval-only", "--checkpoint", str(checkpoints / "best")])


def test_eval_only_rejects_a_folder_without_training_info(tmp_results, tiny_model_dir):
    script = _load_script("train_distilbert")
    with pytest.raises(FileNotFoundError, match="training_info.json"):
        script.main(["--eval-only", "--checkpoint", str(tiny_model_dir)])
