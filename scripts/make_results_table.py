"""Build the README tables and RESUME_NUMBERS.md from the results files (Days 6, 7, 14).

    python scripts/make_results_table.py                    # print the tables, write nothing
    python scripts/make_results_table.py --update-readme    # + README sections, RESUME_NUMBERS.md
    python scripts/make_results_table.py --update-readme --count-tests   # + count pytest tests

Reads results/metrics/*.json (per-model metrics, comparison.json, kmeans.json,
rag_eval.json), data/splits/split_report.json and data/splits/test.csv. With
--update-readme only the text between the <!-- BEGIN:NAME --> and <!-- END:NAME -->
markers of README.md is replaced, and RESUME_NUMBERS.md is regenerated (unless
--no-resume). A missing file gives a "run <script>" message, never a number, and files
written by --smoke runs are ignored.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from fnsl import config, data, metrics, reporting

# Last line of `pytest --collect-only -q`: "96 tests collected in 5.1s" or, with -m,
# "81/96 tests collected (15 deselected) in 5.3s". The first number is what was selected.
COLLECTED_RE = re.compile(r"(\d+)(?:/\d+)? tests? collected")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="README tables and RESUME_NUMBERS.md")
    parser.add_argument("--metrics-dir", type=Path, default=None, help="default results/metrics")
    parser.add_argument(
        "--comparison", type=Path, default=None, help="default <metrics-dir>/comparison.json"
    )
    parser.add_argument(
        "--update-readme",
        action="store_true",
        help="rewrite the marked README sections and RESUME_NUMBERS.md",
    )
    parser.add_argument("--readme", type=Path, default=None, help="default README.md")
    parser.add_argument("--resume-out", type=Path, default=None, help="default RESUME_NUMBERS.md")
    parser.add_argument(
        "--no-resume", action="store_true", help="with --update-readme: skip RESUME_NUMBERS.md"
    )
    parser.add_argument(
        "--count-tests", action="store_true", help="count tests with pytest --collect-only"
    )
    return parser.parse_args(argv)


def load_result(path: Path) -> dict[str, Any] | None:
    """A results JSON, or None if the file is missing or was written by a smoke run."""
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        record = json.load(f)
    if record.get("smoke"):
        print(f"ignoring {reporting.display_path(path)}: written by a --smoke run")
        return None
    return record


def full_test_size() -> int | None:
    """Rows in the fixed test split (data/splits/test.csv), or None if it is missing."""
    try:
        return len(data.load_split("test"))
    except FileNotFoundError:
        return None


def parse_collected(output: str) -> int | None:
    """Number of selected tests in `pytest --collect-only -q` output, None if not found."""
    matches = COLLECTED_RE.findall(output)
    return int(matches[-1]) if matches else None


def count_tests() -> dict[str, Any]:
    """Count all tests and the CPU-only ones CI runs. Collecting does not run them.

    Returns {"all": n, "not_heavy": m}, or {"error": message} if collection fails (for
    example a test file with a syntax error), so a broken count is never reported.
    """
    counts = {}
    for key, marker_args in (("all", []), ("not_heavy", ["-m", "not heavy"])):
        proc = subprocess.run(
            [sys.executable, "-m", "pytest", "--collect-only", "-q", *marker_args],
            cwd=config.ROOT,
            capture_output=True,
            text=True,
        )
        n = parse_collected(proc.stdout)
        if proc.returncode != 0 or n is None:
            last_line = (proc.stdout.strip().splitlines() or ["no output"])[-1]
            return {"error": f"pytest --collect-only failed (exit {proc.returncode}): {last_line}"}
        counts[key] = n
    return counts


def build_sections(
    records: list[dict],
    comparison: dict | None,
    kmeans: dict | None,
    rag: dict | None,
    split_report: dict | None,
    n_test_full: int | None,
    skipped_smoke: list[str],
) -> dict[str, str]:
    """Markdown for every README section, keyed by marker name."""
    return {
        "DATASET": reporting.dataset_section(split_report),
        "RESULTS": reporting.results_section(records, comparison, n_test_full, skipped_smoke),
        "COMPARISON": reporting.comparison_section(comparison, records),
        "KMEANS": reporting.kmeans_section(kmeans),
        "RAG": reporting.rag_section(rag),
    }


def update_readme(path: Path, sections: dict[str, str]) -> bool:
    """Replace the marked sections of README.md; returns True if the file changed.

    All replacements are done in memory first, so if any marker is missing nothing is
    written. newline="" keeps the file's line endings exactly as they are.
    """
    with open(path, encoding="utf-8", newline="") as f:
        old = f.read()
    new = old
    for name, content in sections.items():
        new = reporting.replace_section(new, name, content)
    if new != old:
        with open(path, "w", encoding="utf-8", newline="") as f:
            f.write(new)
    return new != old


def main(argv: list[str] | None = None) -> dict:
    args = parse_args(argv)
    metrics_dir = args.metrics_dir or config.METRICS_DIR
    comparison_path = args.comparison or metrics_dir / "comparison.json"
    kmeans_path, rag_path = metrics_dir / "kmeans.json", metrics_dir / "rag_eval.json"
    readme = args.readme or config.ROOT / "README.md"
    resume_out = args.resume_out or config.ROOT / "RESUME_NUMBERS.md"

    # Per-model metrics; smoke records are never results.
    all_records = metrics.load_metrics(metrics_dir) if metrics_dir.exists() else []
    records = [r for r in all_records if not r.get("smoke")]
    skipped_smoke = [r["model"] for r in all_records if r.get("smoke")]
    comparison = load_result(comparison_path)
    kmeans, rag = load_result(kmeans_path), load_result(rag_path)
    split_report = load_result(data.split_report_path())
    n_test_full = full_test_size()

    sections = build_sections(
        records, comparison, kmeans, rag, split_report, n_test_full, skipped_smoke
    )
    for name, content in sections.items():
        print(f"\n## {name}\n\n{content}")

    result: dict[str, Any] = {"sections": sections, "readme_updated": False, "resume": None}
    if args.count_tests and (args.no_resume or not args.update_readme):
        print("\n--count-tests is ignored: the count only goes into RESUME_NUMBERS.md")
    if not args.update_readme:
        print("\n(print only: pass --update-readme to write README.md and RESUME_NUMBERS.md)")
        return result

    try:
        result["readme_updated"] = update_readme(readme, sections)
    except ValueError as err:
        sys.exit(f"{readme} was NOT changed: {err}")
    print(f"\n{readme}: {'updated' if result['readme_updated'] else 'already up to date'}")

    if not args.no_resume:
        test_counts = count_tests() if args.count_tests else None
        text = reporting.resume_numbers_md(
            records={r["model"]: r for r in records},
            comparison=comparison,
            rag=rag,
            n_test_full=n_test_full,
            test_counts=test_counts,
            paths={
                "metrics_dir": reporting.display_path(metrics_dir),
                "comparison": reporting.display_path(comparison_path),
                "rag": reporting.display_path(rag_path),
            },
            commit=metrics.git_commit(),
            timestamp=datetime.now(UTC).isoformat(timespec="seconds"),
        )
        resume_out.write_text(text, encoding="utf-8")
        result["resume"] = text
        result["test_counts"] = test_counts
        print(f"wrote {resume_out}")
    return result


if __name__ == "__main__":
    main()
