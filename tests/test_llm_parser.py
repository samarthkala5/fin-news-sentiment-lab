"""CPU tests for the LLM prompt, the answer parser and the data/log helpers (no torch needed)."""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys

import pytest

from fnsl import config, metrics
from fnsl.llm_lora import (
    PROMPT_VERSION,
    SMOKE_MODEL,
    SYSTEM_PROMPT,
    TRAIN_INFO_FILE,
    build_messages,
    parse_label,
    read_adapter_info,
    sample_rows,
    split_log_history,
    summarize_outputs,
    to_sft_dataset,
)

BEARISH = config.LABEL2ID["Bearish"]
BULLISH = config.LABEL2ID["Bullish"]
NEUTRAL = config.LABEL2ID["Neutral"]
UNPARSEABLE = config.UNPARSEABLE


def _load_script(name: str):
    """Import scripts/<name>.py as a module so tests can call its functions."""
    spec = importlib.util.spec_from_file_location(name, config.ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    ("output", "expected"),
    [
        ("Bullish", BULLISH),
        (" bearish.", BEARISH),
        ("**Neutral**", NEUTRAL),
        ("The sentiment is bullish", BULLISH),
        ("Positive", BULLISH),
        ("negative", BEARISH),
        ("NEUTRAL\n", NEUTRAL),
        ("Bullish! Definitely bullish.", BULLISH),  # the same label twice is still one label
        ("Positive (bullish)", BULLISH),  # a synonym and its label agree
        ("bullish or bearish", UNPARSEABLE),  # two different labels: ambiguous
        ("Bearish/Neutral", UNPARSEABLE),
        ("", UNPARSEABLE),
        ("I don't know", UNPARSEABLE),
        ("Bullishness", UNPARSEABLE),  # whole words only
        ("Bull", UNPARSEABLE),
    ],
)
def test_parse_label(output, expected):
    assert parse_label(output) == expected


def test_every_training_answer_parses_back_to_its_label():
    for label_id in range(config.NUM_LABELS):
        answer = build_messages("some headline", label_id)[-1]["content"]
        assert parse_label(answer) == label_id


def test_inference_prompt_is_the_training_prompt_without_the_answer():
    text = "$AAPL beats estimates, raises guidance"
    train = build_messages(text, BULLISH)
    infer = build_messages(text)
    assert train[:-1] == infer, "training and inference must share the exact same prompt"
    assert train[-1] == {"role": "assistant", "content": "Bullish"}
    assert infer[0] == {"role": "system", "content": SYSTEM_PROMPT}
    assert infer[-1]["role"] == "user" and text in infer[-1]["content"]


def test_to_sft_dataset_is_conversational_prompt_completion(toy_df):
    dataset = to_sft_dataset(toy_df)
    assert dataset.column_names == ["prompt", "completion"]
    assert len(dataset) == len(toy_df)
    for i in (0, 1, 2):
        row = dataset[i]
        assert row["prompt"] == build_messages(toy_df["text"][i])
        label_word = config.ID2LABEL[int(toy_df["label"][i])]
        assert row["completion"] == [{"role": "assistant", "content": label_word}]


def test_sample_rows_is_fixed_and_sorted(toy_df):
    assert len(sample_rows(toy_df, 0)) == len(toy_df), "0 means all rows"
    assert len(sample_rows(toy_df, 10_000)) == len(toy_df)
    subset = sample_rows(toy_df, 10, seed=1)
    assert len(subset) == 10 and subset["id"].is_monotonic_increasing
    assert subset.equals(sample_rows(toy_df, 10, seed=1)), "same seed, same rows"


def test_split_log_history_separates_train_and_val_loss():
    history = [
        {"eval_loss": 0.9, "step": 0, "epoch": 0},
        {"loss": 0.8, "learning_rate": 1e-4, "step": 10},
        {"loss": 0.5, "step": 20},
        {"eval_loss": 0.4, "step": 20},
        {"train_loss": 0.65, "train_runtime": 12.0, "step": 20},  # end-of-training summary
    ]
    train, val = split_log_history(history)
    assert train == [{"step": 10, "loss": 0.8}, {"step": 20, "loss": 0.5}]
    assert val == [{"step": 0, "eval_loss": 0.9}, {"step": 20, "eval_loss": 0.4}]


def test_summarize_outputs_lists_distinct_unparseable_answers():
    raw = ["Bullish", "The headline is", "The headline is", "Neutral", "maybe?"]
    summary = summarize_outputs(raw, [parse_label(r) for r in raw])
    assert summary["unparseable_examples"] == ["The headline is", "maybe?"]
    assert summary["most_common_outputs"][0] == ("The headline is", 2)


def _fake_adapter(path, prompt_version=PROMPT_VERSION):
    path.mkdir()
    adapter_config = {
        "base_model_name_or_path": SMOKE_MODEL,
        "r": 8,
        "lora_alpha": 16,
        "lora_dropout": 0.0,
        "target_modules": ["v_proj", "q_proj"],
    }
    (path / "adapter_config.json").write_text(json.dumps(adapter_config))
    train_info = {"prompt_version": prompt_version, "max_steps": 3, "train_time_s": 1.5}
    (path / TRAIN_INFO_FILE).write_text(json.dumps(train_info))
    return path


def test_read_adapter_info(tmp_path):
    info = read_adapter_info(_fake_adapter(tmp_path / "adapter"))
    assert info["base_model"] == SMOKE_MODEL
    assert info["lora"] == {
        "r": 8,
        "alpha": 16,
        "dropout": 0.0,
        "target_modules": ["q_proj", "v_proj"],
    }
    assert info["train_info"]["max_steps"] == 3


def test_read_adapter_info_refuses_a_different_prompt_version(tmp_path):
    adapter = _fake_adapter(tmp_path / "adapter", prompt_version="v0-old")
    with pytest.raises(ValueError, match="prompt"):
        read_adapter_info(adapter)


def test_read_adapter_info_needs_a_trained_adapter(tmp_path):
    with pytest.raises(FileNotFoundError):
        read_adapter_info(tmp_path / "missing")


def test_smoke_settings_force_cpu_backend_and_tiny_model(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "CHECKPOINTS_DIR", tmp_path / "checkpoints")
    script = _load_script("train_unsloth")
    args = script.resolve_settings(script.parse_args(["--smoke", "--model-name", "big/model"]))
    assert args.backend == "hf" and args.model_name == SMOKE_MODEL
    assert args.max_steps <= 10 and not args.load_in_4bit
    assert args.n_test_zeroshot == args.n_test_eval == 0, "smoke uses all shared smoke test rows"
    assert args.adapter_dir == tmp_path / "checkpoints" / "smoke" / "llm_lora_adapter"


def test_default_settings_follow_the_plan(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "CHECKPOINTS_DIR", tmp_path / "checkpoints")
    script = _load_script("train_unsloth")
    args = script.resolve_settings(script.parse_args([]))
    assert args.backend == "unsloth" and args.load_in_4bit
    assert args.n_test_zeroshot == 50 and args.n_test_eval == 0
    assert (args.max_steps, args.lr, args.lora_r, args.lora_alpha) == (300, 2e-4, 16, 16)
    assert args.adapter_dir == tmp_path / "checkpoints" / "llm_lora_adapter"


def test_inference_loads_use_a_limit_that_never_cuts_test_prompts(monkeypatch, tmp_path):
    """Unsloth cuts inputs to max_seq_length, so zero-shot/eval must not reuse the training cap."""
    monkeypatch.setattr(config, "CHECKPOINTS_DIR", tmp_path / "checkpoints")
    script = _load_script("train_unsloth")
    args = script.resolve_settings(script.parse_args(["--max-seq-length", "64"]))
    assert args.max_seq_length == 64
    assert args.inference_max_seq_length == script.INFERENCE_MAX_SEQ_LENGTH >= 256
    args = script.resolve_settings(script.parse_args(["--max-seq-length", "1024"]))
    assert args.inference_max_seq_length == 1024


def test_train_log_is_not_mistaken_for_a_test_result(tmp_path):
    script = _load_script("train_unsloth")
    args = script.resolve_settings(script.parse_args(["--smoke", "--adapter-dir", str(tmp_path)]))
    record = script.make_train_record(
        args,
        train_points=[{"step": 1, "loss": 0.9}],
        val_points=[{"step": 0, "eval_loss": 1.0}, {"step": 5, "eval_loss": 0.6}],
        train_time_s=3.0,
        n_trainable=10,
        n_total=1000,
        n_train=200,
        n_val_eval=16,
        n_too_long=0,
    )
    assert "macro_f1" not in record
    assert not metrics.is_model_metrics(record), "load_metrics() must skip the training log"
    assert record["lora"]["r"] == args.lora_r and record["val_loss_end"] == 0.6
    json.dumps(record)  # must be valid JSON


def test_results_store_repo_relative_adapter_paths(tmp_path):
    script = _load_script("train_unsloth")
    inside = config.ROOT / "checkpoints" / "llm_lora_adapter"
    assert script.repo_relative(inside) == "checkpoints/llm_lora_adapter"
    assert script.repo_relative(tmp_path / "adapter") == str(tmp_path / "adapter")


def test_importing_the_llm_code_does_not_import_heavy_libraries():
    """CI installs only requirements.txt, so torch & co. must be imported lazily."""
    code = (
        "import importlib.util, sys\n"
        "import fnsl.llm_lora\n"
        "spec = importlib.util.spec_from_file_location('s', 'scripts/train_unsloth.py')\n"
        "spec.loader.exec_module(importlib.util.module_from_spec(spec))\n"
        "heavy = ['torch', 'transformers', 'trl', 'peft', 'unsloth', 'tensorflow']\n"
        "print(','.join(m for m in heavy if m in sys.modules))\n"
    )
    env = {**os.environ, "PYTHONPATH": str(config.ROOT / "src")}
    result = subprocess.run(
        [sys.executable, "-c", code], cwd=config.ROOT, env=env, capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "", f"heavy modules imported: {result.stdout.strip()}"
