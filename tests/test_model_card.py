"""Tests for the model card renderer in scripts/push_to_hub.py (CPU only, no network)."""

from __future__ import annotations

import copy
import importlib.util
import json
import os
import re
import subprocess
import sys

import pytest

from fnsl import config

# A whole number with exactly 4 decimals, the way the card prints scores (not a prefix of a
# longer number such as the p-value 0.000369).
SCORE_RE = re.compile(r"(?<![\d.])\d\.\d{4}(?![\d])")

FAKE_METRICS = {
    "model": "distilbert",
    "dataset": "zeroshot/twitter-financial-news-sentiment",
    "n_test": 165,
    "macro_f1": 0.8123,
    "accuracy": 0.8642,
    "per_class_f1": {"Bearish": 0.7011, "Bullish": 0.7522, "Neutral": 0.9033},
    "seed": 42,
    "train_time_s": 321.5,
    "params": {
        "model_name": "distilbert/distilbert-base-uncased",
        "epochs": 2,
        "batch_size": 32,
        "lr": 2e-05,
        "weight_decay": 0.01,
        "max_length": 64,
        "device": "cuda",
        "mixed_precision": True,
        "best_epoch": 2,
        "val_macro_f1": 0.8011,
        "history": [
            {"epoch": 1, "train_loss": 0.6111, "val_loss": 0.5222, "val_macro_f1": 0.7733},
            {"epoch": 2, "train_loss": 0.3444, "val_loss": 0.4555, "val_macro_f1": 0.8011},
        ],
    },
    "git_commit": "abc1234",
    "timestamp": "2026-11-20T10:00:00+00:00",
    "smoke": False,
}


def _pair(model_a, model_b, b, c, p_value, p_holm, diff, low, high) -> dict:
    """One entry of comparison.json "pairs", in the layout scripts/compare_models.py writes."""
    return {
        "model_a": model_a,
        "model_b": model_b,
        "n": 165,
        "mcnemar": {"b": b, "c": c, "statistic": 1.0, "p_value": p_value, "method": "exact"},
        "macro_f1_diff": {"diff": diff, "low": low, "high": high, "p_value": 0.5},
        "significant_at_0.05": p_value < 0.05,
        "p_holm": p_holm,
        "significant_holm": p_holm < 0.05,
    }


FAKE_COMPARISON = {
    "kind": "comparison",
    "n_resamples": 2000,
    "alpha": 0.05,
    "seed": 42,
    "smoke": False,
    "models": {
        "distilbert": {"n": 165, "macro_f1": 0.8123, "ci_low": 0.7911, "ci_high": 0.8302},
        "llm_lora": {"n": 165, "macro_f1": 0.8334, "ci_low": 0.8101, "ci_high": 0.8555},
        "keras_bilstm": {"n": 165, "macro_f1": 0.7766, "ci_low": 0.7502, "ci_high": 0.8013},
        "tfidf_logreg": {"n": 165, "macro_f1": 0.7389, "ci_low": 0.7155, "ci_high": 0.7601},
    },
    "pairs": [
        _pair("distilbert", "tfidf_logreg", 30, 12, 0.000123, 0.000369, 0.0734, 0.0412, 0.1056),
        _pair("keras_bilstm", "tfidf_logreg", 21, 15, 0.4444, 0.8888, 0.0377, -0.0102, 0.0851),
        # DistilBERT as model B: the card must flip the difference and swap b and c.
        _pair("llm_lora", "distilbert", 25, 19, 0.4512, 0.9024, 0.0211, -0.0105, 0.0533),
    ],
    "skipped_pairs": [
        {"model_a": "distilbert", "model_b": "keras_cnn", "reason": "different test ids"}
    ],
}
FAKE_SPLIT_REPORT = {
    "dataset": "zeroshot/twitter-financial-news-sentiment",
    "revision": "abc123",
    "license": "MIT",
    "seed": 42,
    "cleaning": {"n_raw": 1200, "n_clean": 1100},
    "sizes": {"train": 770, "val": 165, "test": 165},
    "class_balance": {"all": {"Bearish_%": 15.5, "Bullish_%": 20.5, "Neutral_%": 64.0}},
}


def _load_script(name: str):
    """Import scripts/<name>.py as a module so tests can call its functions."""
    spec = importlib.util.spec_from_file_location(name, config.ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def card_script():
    return _load_script("push_to_hub")


def _floats(obj) -> set[float]:
    """Every float anywhere inside a JSON-like object."""
    if isinstance(obj, float):
        return {obj}
    if isinstance(obj, dict):
        return set().union(*(_floats(v) for v in obj.values())) if obj else set()
    if isinstance(obj, list):
        return set().union(*(_floats(v) for v in obj)) if obj else set()
    return set()


def test_card_without_results_shows_no_scores(card_script):
    card = card_script.render_card(None, None, None)
    assert card.startswith("---\n") and "pipeline_tag: text-classification" in card
    for heading in ("## Training data", "## Training hyperparameters", "## Evaluation"):
        section = card.split(heading, 1)[1].split("\n## ", 1)[0]
        assert "Not available yet" in section, heading
    assert "comparison.json` does not exist yet" in card
    assert not SCORE_RE.search(card), "no score may appear without a results file"
    assert "Do not use it for trading" in card and "## Intended use" in card


def test_card_numbers_come_only_from_the_input_files(card_script):
    card = card_script.render_card(
        FAKE_METRICS, FAKE_COMPARISON, FAKE_SPLIT_REPORT, repo_id="someone/fin-distilbert"
    )
    assert "| Macro-F1 | 0.8123 |" in card
    assert "| Macro-F1, 95% bootstrap CI | 0.7911 to 0.8302 |" in card
    assert "| F1 Bearish | 0.7011 |" in card and "| Accuracy | 0.8642 |" in card
    assert "| `tfidf_logreg` | 0.7389 | 0.7155 to 0.7601 |" in card
    pair_rows = card.split("| DistilBERT vs |", 1)[1]
    expected_rows = [
        "| `tfidf_logreg` | +0.0734 (+0.0412 to +0.1056) | 30 / 12 | 0.000123 | 0.000369 | yes |",
        # llm_lora was model A: the difference is flipped and b, c are swapped
        "| `llm_lora` | -0.0211 (-0.0533 to +0.0105) | 19 / 25 | 0.451 | 0.902 | no |",
    ]
    for row in expected_rows:
        assert row in pair_rows
    assert "0.0377" not in card and "0.444" not in card, "pairs without DistilBERT are not listed"
    assert "Not compared with `keras_cnn`: different test ids" in card
    assert "| 2 | 0.3444 | 0.4555 | 0.8011 |" in card  # per-epoch history
    assert "| Peak learning rate | 2e-05 |" in card and "| Best epoch" in card
    assert "licence **MIT**" in card and "train 770, validation 165, test 165" in card
    assert "1,200 raw posts" in card and "Bearish 15.5%" in card
    assert 'model="someone/fin-distilbert"' in card and "max_length=64" in card
    assert "Status:" not in card

    # Every 4-decimal number in the card must be (up to sign) a value from an input file.
    inputs = [FAKE_METRICS, FAKE_COMPARISON, FAKE_SPLIT_REPORT]
    allowed = {f"{abs(x):.4f}" for x in _floats(inputs)}
    shown = set(SCORE_RE.findall(card))
    assert shown and shown <= allowed, f"numbers not found in any input file: {shown - allowed}"


def test_stale_or_unexpected_comparison_is_not_shown(card_script):
    stale = copy.deepcopy(FAKE_COMPARISON)
    stale["models"]["distilbert"]["macro_f1"] = 0.7999  # computed from older predictions
    card = card_script.render_card(FAKE_METRICS, stale, FAKE_SPLIT_REPORT)
    assert "older than `distilbert.json`" in card
    assert "0.7911" not in card and "0.7999" not in card

    smoke = {**FAKE_COMPARISON, "smoke": True}
    card = card_script.render_card(FAKE_METRICS, smoke, FAKE_SPLIT_REPORT)
    assert "smoke run" in card and "0.7911" not in card

    # A different layout (here: flat p-values per pair) must not be half-read.
    old_layout = {
        "alpha": 0.05,
        "n_resamples": 1000,
        "models": FAKE_COMPARISON["models"],
        "pairs": [{"model_a": "distilbert", "model_b": "tfidf_logreg", "p_value": 0.01}],
    }
    for unexpected in ({"bootstrap": {"distilbert": [0.79, 0.83]}}, old_layout):
        card = card_script.render_card(FAKE_METRICS, unexpected, FAKE_SPLIT_REPORT)
        assert "unexpected format" in card and "0.7911" not in card
        assert "| Macro-F1 | 0.8123 |" in card, "the metrics file is still shown"

    no_entry = {**FAKE_COMPARISON, "models": {"tfidf_logreg": {"macro_f1": 0.7389}}, "pairs": []}
    card = card_script.render_card(FAKE_METRICS, no_entry, FAKE_SPLIT_REPORT)
    assert "no DistilBERT entry" in card and "0.7389" not in card

    # Without distilbert.json the comparison cannot be checked for staleness: not shown.
    card = card_script.render_card(None, FAKE_COMPARISON, FAKE_SPLIT_REPORT)
    section = card.split("## Comparison with the other models", 1)[1].split("\n## ", 1)[0]
    assert "distilbert.json` does not exist yet" in section
    assert not SCORE_RE.search(card), "no score may appear without distilbert.json"


def test_dry_run_writes_card_and_ignores_smoke_metrics(card_script, tmp_path, capsys):
    metrics_dir = tmp_path / "metrics"
    metrics_dir.mkdir()
    smoke_record = {**FAKE_METRICS, "smoke": True}
    (metrics_dir / "distilbert.json").write_text(json.dumps(smoke_record))
    out = tmp_path / "model_card" / "README.md"

    result = card_script.main(
        ["--dry-run", "--metrics-dir", str(metrics_dir), "--out", str(out)]
        + ["--split-report", str(tmp_path / "missing.json")]
    )
    assert out.exists() and out.read_text() in capsys.readouterr().out
    assert result["available"] == {"distilbert": False, "comparison": False, "split_report": False}
    assert result["pushed_url"] is None
    assert "0.8123" not in out.read_text(), "smoke numbers must never reach the card"


def test_push_refuses_without_real_results(card_script, tmp_path):
    with pytest.raises(SystemExit, match="Refusing to push"):
        card_script.main(
            ["--push", "--repo-id", "someone/fin-distilbert", "--metrics-dir", str(tmp_path)]
            + ["--out", str(tmp_path / "README.md")]
        )
    with pytest.raises(SystemExit):
        card_script.parse_args(["--push"])  # needs --repo-id USER/NAME


def test_push_uploads_only_a_checkpoint_from_the_same_run(card_script, tmp_path, monkeypatch):
    hub = pytest.importorskip("huggingface_hub")
    metrics_dir, checkpoint = tmp_path / "metrics", tmp_path / "best"
    metrics_dir.mkdir()
    checkpoint.mkdir()
    (metrics_dir / "distilbert.json").write_text(json.dumps(FAKE_METRICS))
    (checkpoint / "config.json").write_text("{}")
    args = ["--push", "--repo-id", "someone/fin-distilbert", "--checkpoint", str(checkpoint)]
    args += ["--metrics-dir", str(metrics_dir), "--out", str(tmp_path / "README.md")]

    # A stand-in for the Hub client that records the uploads instead of making them.
    uploads = []

    class FakeHfApi:
        def create_repo(self, repo_id, private, exist_ok):
            return f"https://huggingface.co/{repo_id}"

        def upload_folder(self, **kwargs):
            uploads.append(kwargs)

        def upload_file(self, **kwargs):
            uploads.append(kwargs)

    monkeypatch.setattr(hub, "HfApi", FakeHfApi)
    monkeypatch.setattr(hub, "get_token", lambda: "not-a-real-token")

    # The weights must come from the run that wrote distilbert.json.
    with pytest.raises(SystemExit, match="no training_info.json"):
        card_script.main(args)
    other_run = {"seed": 42, "params": {**FAKE_METRICS["params"], "lr": 5e-5}}
    (checkpoint / "training_info.json").write_text(json.dumps(other_run))
    with pytest.raises(SystemExit, match="different training run"):
        card_script.main(args)
    assert uploads == [], "nothing may be uploaded after a refusal"

    same_run = {"run_name": "distilbert", "seed": 42, "params": FAKE_METRICS["params"]}
    (checkpoint / "training_info.json").write_text(json.dumps(same_run))
    result = card_script.main(args)
    assert result["pushed_url"] == "https://huggingface.co/someone/fin-distilbert"
    folder, card = uploads
    assert folder["repo_id"] == "someone/fin-distilbert" and folder["folder_path"] == checkpoint
    assert card["path_in_repo"] == "README.md"
    assert b"| Macro-F1 | 0.8123 |" in card["path_or_fileobj"]


def test_rendering_needs_no_heavy_libraries():
    # A fresh interpreter, because other tests in this session may have imported torch.
    code = (
        "import importlib.util, sys\n"
        "spec = importlib.util.spec_from_file_location('card', sys.argv[1])\n"
        "module = importlib.util.module_from_spec(spec)\n"
        "spec.loader.exec_module(module)\n"
        "module.render_card(None, None, None)\n"
        "heavy = ('torch', 'transformers', 'huggingface_hub')\n"
        "print(sorted(name for name in heavy if name in sys.modules))\n"
    )
    script = config.ROOT / "scripts" / "push_to_hub.py"
    env = {**os.environ, "PYTHONPATH": str(config.ROOT / "src")}
    result = subprocess.run(
        [sys.executable, "-c", code, str(script)], capture_output=True, text=True, env=env
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "[]"
