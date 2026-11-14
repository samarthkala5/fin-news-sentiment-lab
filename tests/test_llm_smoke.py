"""Heavy tests: the LoRA pipeline end to end with a tiny chat model on CPU.

They need torch, transformers, peft and trl, and download HuggingFaceTB/SmolLM2-135M-Instruct
(about 270 MB) once. Skipped in CI (`pytest -m "not heavy"`).
"""

from __future__ import annotations

import importlib.util
import json

import pytest

from fnsl import config, metrics

pytestmark = pytest.mark.heavy


def _load_script(name: str):
    """Import scripts/<name>.py as a module so tests can call its main()."""
    spec = importlib.util.spec_from_file_location(name, config.ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _lora_b_matrices(model) -> list:
    return [p for name, p in model.named_parameters() if "lora_B" in name]


def test_lora_attach_train_generate_and_reload(tmp_path, toy_df):
    torch = pytest.importorskip("torch")
    for package in ("transformers", "peft", "trl"):
        pytest.importorskip(package)
    from fnsl import llm_lora

    model, tokenizer = llm_lora.load_model("hf", llm_lora.SMOKE_MODEL, load_in_4bit=False)
    model = llm_lora.add_lora(model, "hf", r=4, alpha=8, seed=0)
    trainable, total = llm_lora.trainable_parameters(model)
    assert 0 < trainable < 0.05 * total, "LoRA trains only a small fraction of the weights"
    # B starts at zero, so before training the adapter changes nothing.
    assert all(torch.count_nonzero(b) == 0 for b in _lora_b_matrices(model))

    history = llm_lora.train_lora(
        model,
        tokenizer,
        toy_df.iloc[:16],
        toy_df.iloc[16:24],
        output_dir=tmp_path / "trainer",
        max_steps=2,
        batch_size=4,
        grad_accum=1,
        warmup_steps=0,
        eval_steps=2,
        logging_steps=1,
    )
    train_points, val_points = llm_lora.split_log_history(history)
    assert [p["step"] for p in train_points] == [1, 2]
    assert [p["step"] for p in val_points] == [0, 2], "val loss before and after training"
    assert any(torch.count_nonzero(b) > 0 for b in _lora_b_matrices(model)), "B was updated"

    texts = list(toy_df["text"][:3])
    raw, labels = llm_lora.generate_labels(model, tokenizer, texts, batch_size=2)
    assert len(raw) == 3 and labels == [llm_lora.parse_label(r) for r in raw]
    assert set(labels) <= {config.UNPARSEABLE, *range(config.NUM_LABELS)}

    # Save only the adapter, load it onto a fresh base model: same weights, same answers.
    adapter_dir = tmp_path / "adapter"
    model.save_pretrained(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))
    reloaded, reloaded_tokenizer = llm_lora.load_model(
        "hf", llm_lora.SMOKE_MODEL, load_in_4bit=False, adapter_dir=adapter_dir
    )
    raw_again, _ = llm_lora.generate_labels(reloaded, reloaded_tokenizer, texts, batch_size=2)
    assert raw_again == raw


def test_trl_trains_on_exactly_the_prompt_tokens_used_at_inference(tmp_path, toy_df):
    """TRL's tokens before the answer must equal what generate_labels sends to the model."""
    for package in ("torch", "transformers", "peft", "trl"):
        pytest.importorskip(package)
    from trl import SFTConfig

    from fnsl import llm_lora

    model, tokenizer = llm_lora.load_model("hf", llm_lora.SMOKE_MODEL, load_in_4bit=False)
    model = llm_lora.add_lora(model, "hf", r=4, alpha=8, seed=0)
    args = SFTConfig(
        output_dir=str(tmp_path / "trainer"),
        max_length=128,
        completion_only_loss=True,
        gradient_checkpointing=False,
        bf16=False,
        fp16=False,
        report_to="none",
    )
    rows = toy_df.iloc[:6]
    trainer = llm_lora.sft_trainer_class(use_cuda=False)(
        model=model,
        args=args,
        train_dataset=llm_lora.to_sft_dataset(rows),
        processing_class=tokenizer,
    )
    # trainer.train_dataset is TRL's tokenized data, in the original row order
    for example, text, label in zip(
        trainer.train_dataset, rows["text"], rows["label"], strict=True
    ):
        answer_start = next(i for i, x in enumerate(example["labels"]) if x != -100)
        prompt = tokenizer.apply_chat_template(
            [llm_lora.build_messages(text)], add_generation_prompt=True, return_dict=True
        )["input_ids"][0]
        assert example["input_ids"][:answer_start] == prompt
        # the loss covers the label word AND the end-of-turn token (pad == eos for SmolLM2)
        answer = tokenizer.decode(example["input_ids"][answer_start:])
        assert answer.startswith(config.ID2LABEL[int(label)] + tokenizer.eos_token)


def test_generate_labels_refuses_prompts_longer_than_max_seq_length(toy_df):
    """Unsloth would silently cut such prompts, so generate_labels must raise instead."""
    for package in ("torch", "transformers", "peft"):
        pytest.importorskip(package)
    from fnsl import llm_lora

    model, tokenizer = llm_lora.load_model("hf", llm_lora.SMOKE_MODEL, load_in_4bit=False)
    model.max_seq_length = 16  # the attribute Unsloth sets on the models it loads
    with pytest.raises(ValueError, match="max_seq_length"):
        llm_lora.generate_labels(model, tokenizer, list(toy_df["text"][:2]), batch_size=2)


@pytest.mark.skipif(not (config.SPLITS_DIR / "test.csv").exists(), reason="no split files")
def test_train_unsloth_smoke_all_writes_schema(tmp_results, tmp_path, monkeypatch):
    for package in ("torch", "transformers", "peft", "trl"):
        pytest.importorskip(package)
    monkeypatch.setattr(config, "CHECKPOINTS_DIR", tmp_path / "checkpoints")
    script = _load_script("train_unsloth")
    results = script.main(["--smoke", "--mode", "all", "--no-mlflow"])
    assert set(results) == {"zero-shot", "train", "eval"}

    smoke = tmp_results / "smoke"
    for name in ("llm_zeroshot", "llm_lora"):
        record = json.loads((smoke / "metrics" / f"{name}.json").read_text())
        assert metrics.is_model_metrics(record) and record["smoke"] is True
        assert record["n_test"] == config.SMOKE_SIZES["test"]
        assert len(record["unparseable_examples"]) <= 10
        predictions = metrics.load_predictions(smoke / "predictions" / f"{name}.csv")
        assert set(predictions["pred"]) <= {config.UNPARSEABLE, *range(config.NUM_LABELS)}

    lora = json.loads((smoke / "metrics" / "llm_lora.json").read_text())
    assert {"r", "alpha", "dropout", "target_modules", "base_model", "max_steps"} <= set(
        lora["params"]
    )
    train_log = json.loads((smoke / "metrics" / "llm_lora_train.json").read_text())
    assert "macro_f1" not in train_log and train_log["val_loss"] and train_log["train_loss"]
    adapter_dir = tmp_path / "checkpoints" / "smoke" / "llm_lora_adapter"
    assert (adapter_dir / "adapter_config.json").exists()
    assert not (tmp_results / "metrics").exists(), "smoke runs must not write real results"
