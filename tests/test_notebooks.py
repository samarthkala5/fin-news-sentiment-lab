"""Checks for the two notebooks in notebooks/ (CPU only, no network).

The notebooks are committed without outputs, must not leak machine-specific paths or
secrets, and may only call scripts and flags that exist. Most checks read the .ipynb JSON
directly, so they run in CI too. Schema validation needs nbformat, and running the report
notebook needs nbclient + ipykernel; those tests are skipped when the package is missing.
"""

from __future__ import annotations

import functools
import importlib.util
import json
import re
import shlex

import pytest

from fnsl import config, metrics

NOTEBOOKS_DIR = config.ROOT / "notebooks"
NOTEBOOK_NAMES = ["colab_runner.ipynb", "kaggle_report.ipynb"]

# Signs of a path that only exists on one machine (Linux home, root, macOS home, Windows).
LOCAL_PATH_MARKERS = ["/home/", "/root/", "/Users/", "C:\\"]
TOKEN_PATTERNS = [
    re.compile(r"hf_[A-Za-z0-9]{30,}"),  # Hugging Face access token
    re.compile(r"gh[pousr]_[A-Za-z0-9]{30,}"),  # GitHub token
    re.compile(r"github_pat_[A-Za-z0-9_]{30,}"),  # GitHub fine-grained token
    re.compile(r"sk-[A-Za-z0-9_-]{20,}"),  # OpenAI-style API key
    re.compile(r"AKIA[0-9A-Z]{16}"),  # AWS access key id
    re.compile(r"https?://[^/\s:@]+:[^/\s@]+@"),  # user:password inside a URL
    # a hard-coded secret such as HF_TOKEN = "abcdefgh..."
    re.compile(r"(?i)\w*(token|secret|password|api_key)\w*\s*=\s*['\"][^'\"]{8,}['\"]"),
]
SCRIPT_RE = re.compile(r"scripts/(\w+)\.py")
# "python scripts/<name>.py" plus the flag-like words after it (stops at a comment, a quote,
# a backtick or a placeholder such as <folder>).
COMMAND_RE = re.compile(r"python scripts/(\w+)\.py((?:[ \t]+[-\w./=]+)*)")
# Scripts that import torch or tensorflow as soon as they are loaded.
HEAVY_SCRIPTS = {"train_distilbert": "torch", "train_keras": "tensorflow"}


def load_notebook(name: str) -> dict:
    """The notebook's raw JSON (plain json, so no nbformat is needed)."""
    with open(NOTEBOOKS_DIR / name, encoding="utf-8") as f:
        return json.load(f)


def cell_text(cell: dict) -> str:
    """A cell's source; the JSON stores it as a list of lines or as one string."""
    source = cell["source"]
    return source if isinstance(source, str) else "".join(source)


def notebook_text(nb: dict) -> str:
    """Every cell's source plus the notebook metadata, for text searches."""
    return "\n".join(cell_text(c) for c in nb["cells"]) + "\n" + json.dumps(nb["metadata"])


def notebook_commands() -> list[tuple[str, str, str]]:
    """(notebook, script, arguments) for every `python scripts/...` line in the notebooks."""
    found = set()
    for name in NOTEBOOK_NAMES:
        if not (NOTEBOOKS_DIR / name).exists():
            continue  # test_notebook_exists reports this
        for cell in load_notebook(name)["cells"]:
            for script, args in COMMAND_RE.findall(cell_text(cell)):
                found.add((name, script, args.strip()))
    return sorted(found)


def command_param(name: str, script: str, args: str):
    """A pytest parameter for one command; commands of torch/TF scripts are marked heavy."""
    command_id = f"{name.split('.')[0]}: {script} {args}".strip()
    marks = [pytest.mark.heavy] if script in HEAVY_SCRIPTS else []
    return pytest.param(script, args, id=command_id, marks=marks)


@functools.cache
def load_script(name: str):
    """Import scripts/<name>.py as a module (without running its main())."""
    spec = importlib.util.spec_from_file_location(name, config.ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(params=NOTEBOOK_NAMES)
def notebook(request) -> tuple[str, dict]:
    return request.param, load_notebook(request.param)


@pytest.mark.parametrize("name", NOTEBOOK_NAMES)
def test_notebook_exists(name):
    assert (NOTEBOOKS_DIR / name).is_file()


@pytest.mark.parametrize("name", NOTEBOOK_NAMES)
def test_valid_nbformat_v4(name):
    nbformat = pytest.importorskip("nbformat")
    nb = nbformat.read(NOTEBOOKS_DIR / name, as_version=nbformat.NO_CONVERT)
    assert nb.nbformat == 4
    nbformat.validate(nb)  # raises ValidationError if the file breaks the schema


def test_structure(notebook):
    name, nb = notebook
    assert nb["nbformat"] == 4
    assert nb["metadata"]["kernelspec"]["name"] == "python3"
    kinds = [cell["cell_type"] for cell in nb["cells"]]
    assert set(kinds) <= {"markdown", "code"}
    assert kinds[0] == "markdown", "start with a title cell"
    # Every step is explained: no code cell directly follows another code cell.
    for i in range(1, len(kinds)):
        assert not (kinds[i] == "code" and kinds[i - 1] == "code"), (
            f"{name}: code cell {i} needs a markdown cell before it"
        )


def test_no_outputs(notebook):
    name, nb = notebook
    for i, cell in enumerate(nb["cells"]):
        if cell["cell_type"] != "code":
            continue
        assert cell["outputs"] == [], f"{name}: cell {i} has outputs; clear them before commit"
        assert cell["execution_count"] is None, f"{name}: cell {i} has an execution count"
        assert "execution" not in cell.get("metadata", {}), f"{name}: cell {i} has run timings"


def test_no_local_paths(notebook):
    name, nb = notebook
    text = notebook_text(nb)
    for marker in LOCAL_PATH_MARKERS:
        assert marker not in text, f"{name} contains a machine-specific path ({marker!r})"


def test_no_token_like_strings(notebook):
    name, nb = notebook
    text = notebook_text(nb)
    for pattern in TOKEN_PATTERNS:
        match = pattern.search(text)
        assert match is None, f"{name} contains something that looks like a secret: {match[0]!r}"


def test_mentioned_scripts_exist(notebook):
    name, nb = notebook
    scripts = set(SCRIPT_RE.findall(notebook_text(nb)))
    assert scripts, f"{name} mentions no scripts"
    missing = sorted(s for s in scripts if not (config.ROOT / "scripts" / f"{s}.py").exists())
    assert not missing, f"{name} mentions scripts that do not exist: {missing}"


def test_colab_runner_covers_the_gpu_steps():
    nb = load_notebook("colab_runner.ipynb")
    code = "\n".join(cell_text(c) for c in nb["cells"] if c["cell_type"] == "code")
    expected = [
        "nvidia-smi",
        "git clone --depth 1 -b $BRANCH $REPO_URL",
        "pip install -q -r requirements-gpu.txt",
        "pip install -q -e . --no-deps",
        "python scripts/train_distilbert.py --smoke",
        "python scripts/train_distilbert.py --seed 43",
        "python scripts/train_keras.py",
        "python scripts/train_unsloth.py --mode zero-shot",
        "python scripts/train_unsloth.py --mode train",
        "python scripts/train_unsloth.py --mode eval",
        "python scripts/train_unsloth.py --mode zero-shot --n-test 0",
        "python scripts/compare_models.py",
        "python scripts/make_results_table.py",
        "files.download",
        "/kaggle/working",
    ]
    missing = [snippet for snippet in expected if snippet not in code]
    assert not missing, f"colab_runner.ipynb lacks: {missing}"
    assert "drive.mount" not in notebook_text(nb), "the runner must not mount private storage"


def test_kaggle_report_writes_nothing():
    nb = load_notebook("kaggle_report.ipynb")
    code = "\n".join(cell_text(c) for c in nb["cells"] if c["cell_type"] == "code")
    for call in [".to_csv(", ".savefig(", ".write_text(", ".write_bytes(", "mkdir(", '"w"']:
        assert call not in code, f"the report must only read files, found {call!r}"
    assert "results/smoke" not in code


@pytest.mark.parametrize(("script", "args"), [command_param(*c) for c in notebook_commands()])
def test_notebook_command_flags_exist(script, args):
    """Each command's flags are accepted by the script's own argument parser."""
    if script in HEAVY_SCRIPTS:
        pytest.importorskip(HEAVY_SCRIPTS[script])
    module = load_script(script)
    try:
        module.parse_args(shlex.split(args))
    except SystemExit as err:  # argparse exits on an unknown flag or a bad value
        pytest.fail(f"`python scripts/{script}.py {args}` is rejected (exit {err.code})")


def output_text(nb) -> str:
    """Printed text and plain-text displays of all cells of an executed notebook."""
    parts = []
    for cell in nb.cells:
        for output in cell.get("outputs", []):
            parts.append(output.get("text", ""))
            parts.append(output.get("data", {}).get("text/plain", ""))
    return "\n".join(parts)


@pytest.mark.parametrize("with_results", [False, True], ids=["no-results", "repo-results"])
def test_kaggle_report_runs(with_results, tmp_path, monkeypatch):
    """Run the report in a real kernel (in memory; the file on disk is not changed).

    Without results every section must print 'not run yet' instead of failing; with the
    repo's committed files it must run cleanly and list every model.
    """
    nbformat = pytest.importorskip("nbformat")
    nbclient = pytest.importorskip("nbclient")
    pytest.importorskip("ipykernel")
    if with_results:
        monkeypatch.delenv("FNSL_ROOT", raising=False)
    else:
        # fnsl.config takes the repo root from FNSL_ROOT and the kernel inherits this
        # process's environment, so every data and results path points to an empty folder.
        monkeypatch.setenv("FNSL_ROOT", str(tmp_path))
    nb = nbformat.read(NOTEBOOKS_DIR / "kaggle_report.ipynb", as_version=4)
    client = nbclient.NotebookClient(
        nb, timeout=300, kernel_name="python3", resources={"metadata": {"path": str(config.ROOT)}}
    )
    client.execute()  # raises CellExecutionError if any cell fails
    printed = output_text(nb)

    if with_results:
        assert "split_report.json is missing" not in printed
        for record in metrics.load_metrics():
            if not record.get("smoke"):
                assert record["model"] in printed
    else:
        for name in ["split_report.json", "comparison.json", "kmeans.json", "rag_eval.json"]:
            assert f"{name} is missing" in printed
        assert "not run yet: the split CSVs" in printed
        assert not any(tmp_path.iterdir()), "the report must not create files"
