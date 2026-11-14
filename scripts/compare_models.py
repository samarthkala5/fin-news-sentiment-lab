"""Compare all models' test predictions with bootstrap CIs and paired tests (Day 7).

    python scripts/compare_models.py            # results/predictions/*.csv -> results/metrics/
    python scripts/compare_models.py --smoke    # results/smoke/predictions -> results/smoke/
    python scripts/compare_models.py --models tfidf_logreg distilbert

Per model: macro-F1 and accuracy with percentile bootstrap 95% CIs.
Per pair of models scored on exactly the same test ids: McNemar's test (error rates), a
paired bootstrap of the macro-F1 difference, and Holm-adjusted McNemar p-values over all
pairs. Pairs that cannot be paired (different ids, labels that disagree) are listed under
skipped_pairs with the reason, and unreadable files under skipped_models: nothing is
dropped silently. Seed reruns of a model (distilbert_seed43, ...) are summarised as mean
and standard deviation of macro-F1 under seed_runs instead of being tested pairwise.
Writes results/metrics/comparison.json. Predictions under results/smoke/ always count as
a smoke run and write results/smoke/metrics/comparison.json instead.
"""

from __future__ import annotations

import argparse
import itertools
import json
import re
import statistics
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from fnsl import config, metrics
from fnsl.reporting import display_path
from fnsl.stats import accuracy, bootstrap_ci, holm_adjust, mcnemar_test, paired_bootstrap_diff

# Allowed values: labels are class ids 0=Bearish, 1=Bullish, 2=Neutral; a prediction may
# also be config.UNPARSEABLE (-1, an LLM answer that matched no label, counted as wrong).
LABEL_IDS = list(range(config.NUM_LABELS))
PRED_IDS = [*LABEL_IDS, config.UNPARSEABLE]

# Training scripts name a rerun with another seed "<model>_seed<N>"; the default run (seed
# config.SEED = 42) keeps the plain name. Example: distilbert, distilbert_seed43, ..._seed44.
SEED_RUN_RE = re.compile(r"^(?P<base>.+)_seed(?P<seed>\d+)$")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="McNemar + bootstrap comparison of models")
    parser.add_argument(
        "--predictions-dir", type=Path, default=None, help="default results/predictions"
    )
    parser.add_argument(
        "--out", type=Path, default=None, help="default results/metrics/comparison.json"
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="read results/smoke/predictions, write results/smoke/metrics/comparison.json",
    )
    parser.add_argument("--models", nargs="+", default=None, help="only these model names")
    parser.add_argument("--n-resamples", type=int, default=2000)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=config.SEED)
    return parser.parse_args(argv)


def resolve_paths(args: argparse.Namespace) -> tuple[Path, Path, bool]:
    """Input folder, output file and whether this is a smoke run.

    Smoke predictions (--smoke, or a --predictions-dir inside results/smoke/) always write
    results/smoke/metrics/comparison.json, so they can never overwrite or pose as the
    real comparison.
    """
    smoke_out = config.SMOKE_DIR / "metrics" / "comparison.json"
    if args.smoke:
        if args.predictions_dir is not None or args.out is not None:
            print("--smoke: ignoring --predictions-dir/--out, using results/smoke/")
        return config.SMOKE_DIR / "predictions", smoke_out, True
    pred_dir = args.predictions_dir or config.PREDICTIONS_DIR
    if pred_dir.resolve().is_relative_to(config.SMOKE_DIR.resolve()):
        print(f"{display_path(pred_dir)} holds smoke predictions: treating this as --smoke")
        return pred_dir, smoke_out, True
    return pred_dir, args.out or config.METRICS_DIR / "comparison.json", False


def load_all_predictions(
    pred_dir: Path, models: list[str] | None = None
) -> tuple[dict[str, pd.DataFrame], list[dict[str, str]]]:
    """Read <model>.csv for every model (or only `models`).

    Returns {model: predictions} and a list of files that could not be used, each with
    the reason (wrong columns, no rows, repeated ids, values that are not class ids).
    """
    paths = sorted(pred_dir.glob("*.csv"))
    if models is not None:
        missing = sorted(set(models) - {p.stem for p in paths})
        if missing:
            sys.exit(f"No predictions file for {missing} in {pred_dir}")
        paths = [p for p in paths if p.stem in models]

    predictions, skipped = {}, []
    for path in paths:
        try:
            df = metrics.load_predictions(path)
        except ValueError as err:  # wrong columns; keep the message free of local paths
            skipped.append({"model": path.stem, "reason": str(err).replace(str(path), path.name)})
            continue
        # isin() is False for text ("Bullish"), empty cells and non-whole numbers (1.5).
        bad_values = ~df["label"].isin(LABEL_IDS) | ~df["pred"].isin(PRED_IDS)
        if len(df) == 0:
            skipped.append({"model": path.stem, "reason": "file has no rows"})
        elif df["id"].duplicated().any():
            n_dup = int(df["id"].duplicated().sum())
            skipped.append({"model": path.stem, "reason": f"{n_dup} repeated ids"})
        elif bad_values.any():
            reason = (
                f"{int(bad_values.sum())} rows where label is not 0/1/2 or pred is not "
                f"0/1/2/{config.UNPARSEABLE}"
            )
            skipped.append({"model": path.stem, "reason": reason})
        else:
            predictions[path.stem] = df.astype({"label": int, "pred": int})
    return predictions, skipped


def summarize_model(df: pd.DataFrame, n_resamples: int, alpha: float, seed: int) -> dict:
    """Macro-F1 and accuracy of one model, each with a bootstrap CI."""
    y_true, y_pred = df["label"].to_numpy(), df["pred"].to_numpy()
    f1 = bootstrap_ci(y_true, y_pred, n_resamples=n_resamples, alpha=alpha, seed=seed)
    acc = bootstrap_ci(
        y_true, y_pred, metric=accuracy, n_resamples=n_resamples, alpha=alpha, seed=seed
    )
    return {
        "n": len(df),
        "macro_f1": f1["point"],
        "ci_low": f1["low"],
        "ci_high": f1["high"],
        "accuracy": acc["point"],
        "acc_ci_low": acc["low"],
        "acc_ci_high": acc["high"],
    }


def align_pair(df_a: pd.DataFrame, df_b: pd.DataFrame) -> tuple[pd.DataFrame | None, str]:
    """Line up two models' predictions row by row via the test id.

    Returns (merged frame with columns id, label, pred_a, pred_b; "") when the two files
    cover exactly the same ids with the same labels, otherwise (None, reason).
    """
    ids_a, ids_b = set(df_a["id"]), set(df_b["id"])
    if ids_a != ids_b:
        return None, (
            f"different test ids ({len(ids_a)} vs {len(ids_b)} rows, "
            f"{len(ids_a & ids_b)} in common): predictions cannot be paired"
        )
    merged = df_a[["id", "label", "pred"]].merge(
        df_b[["id", "label", "pred"]], on="id", suffixes=("_a", "_b"), validate="one_to_one"
    )
    n_disagree = int((merged["label_a"] != merged["label_b"]).sum())
    if n_disagree:
        return None, f"true labels disagree on {n_disagree} ids: the files are inconsistent"
    merged = merged.rename(columns={"label_a": "label"}).drop(columns="label_b")
    return merged.sort_values("id").reset_index(drop=True), ""


def compare_pair(merged: pd.DataFrame, n_resamples: int, alpha: float, seed: int) -> dict:
    """McNemar's test and the paired bootstrap macro-F1 difference for one aligned pair."""
    y_true = merged["label"].to_numpy()
    pred_a, pred_b = merged["pred_a"].to_numpy(), merged["pred_b"].to_numpy()
    return {
        "n": len(merged),
        "mcnemar": mcnemar_test(y_true, pred_a, pred_b),
        "macro_f1_diff": paired_bootstrap_diff(
            y_true, pred_a, pred_b, n_resamples=n_resamples, alpha=alpha, seed=seed
        ),
    }


def seed_rerun_base(name: str, names) -> str | None:
    """The base model if `name` is a seed rerun ("<base>_seed<N>") of a model in `names`."""
    match = SEED_RUN_RE.match(name)
    if match and match["base"] in names:
        return match["base"]
    return None


def seed_summary(models: dict[str, dict]) -> dict[str, dict]:
    """Mean and standard deviation of test macro-F1 over seed reruns of the same model.

    This is PROJECT_PLAN.md Day 7's optional "rerun DistilBERT with 3 seeds and report
    mean ± std". It shows how much the score moves just from the random seed (weight
    initialisation, data order). The std is the sample standard deviation (n - 1); with
    only 2-3 runs it is a rough guide, not a precise estimate.
    """
    seeds: dict[str, dict[str, int]] = {}  # base model -> {run name: seed}
    for name in models:
        base = seed_rerun_base(name, models)
        if base is not None:
            seeds.setdefault(base, {base: config.SEED})[name] = int(SEED_RUN_RE.match(name)["seed"])
    summary = {}
    for base, runs in sorted(seeds.items()):
        scores = [models[name]["macro_f1"] for name in runs]
        summary[base] = {
            "runs": [
                {
                    "model": name,
                    "seed": s,
                    "n": models[name]["n"],
                    "macro_f1": models[name]["macro_f1"],
                }
                for name, s in runs.items()
            ],
            "n_runs": len(scores),
            "macro_f1_mean": statistics.mean(scores),
            "macro_f1_std": statistics.stdev(scores),
        }
    return summary


def build_comparison(
    predictions: dict[str, pd.DataFrame], n_resamples: int, alpha: float, seed: int
) -> dict[str, Any]:
    """Per-model CIs, every comparable pair, the skipped pairs, and seed-rerun summaries."""
    models = {
        name: summarize_model(df, n_resamples, alpha, seed)
        for name, df in sorted(predictions.items())
    }
    # Seed reruns get a CI each but no pairwise tests: testing a model against a retrained
    # copy of itself answers no question and only makes the Holm correction stricter.
    primary = [name for name in sorted(predictions) if seed_rerun_base(name, predictions) is None]
    pairs, skipped_pairs = [], []
    for name_a, name_b in itertools.combinations(primary, 2):
        merged, reason = align_pair(predictions[name_a], predictions[name_b])
        if merged is None:
            skipped_pairs.append({"model_a": name_a, "model_b": name_b, "reason": reason})
            continue
        result = compare_pair(merged, n_resamples, alpha, seed)
        pairs.append({"model_a": name_a, "model_b": name_b, **result})

    # Correct the McNemar p-values for running several tests at once.
    p_holm = holm_adjust([pair["mcnemar"]["p_value"] for pair in pairs])
    for pair, p_adj in zip(pairs, p_holm, strict=True):
        # The key name fixes the level at 0.05 (raw p-value); Holm uses --alpha.
        pair["significant_at_0.05"] = pair["mcnemar"]["p_value"] < 0.05
        pair["p_holm"] = p_adj
        pair["significant_holm"] = p_adj < alpha
    return {
        "models": models,
        "pairs": pairs,
        "skipped_pairs": skipped_pairs,
        "seed_runs": seed_summary(models),
    }


def print_summary(result: dict[str, Any], out: Path) -> None:
    """Readable console version of comparison.json."""
    level = round(100 * (1 - result["alpha"]))
    print(
        f"\nModels ({result['n_resamples']} bootstrap resamples, {level}% percentile CIs, "
        f"seed {result['seed']}{', SMOKE' if result['smoke'] else ''}):"
    )
    for name, m in result["models"].items():
        print(
            f"  {name:<20} n={m['n']:<5} macro-F1 {m['macro_f1']:.3f} "
            f"[{m['ci_low']:.3f}, {m['ci_high']:.3f}]  accuracy {m['accuracy']:.3f} "
            f"[{m['acc_ci_low']:.3f}, {m['acc_ci_high']:.3f}]"
        )
    for item in result["skipped_models"]:
        print(f"  SKIPPED file {item['model']}: {item['reason']}")

    if result["pairs"]:
        print("\nPairs (b = only A right, c = only B right; diff = macro-F1 A - B):")
    else:
        print("\nNo pairs compared: need at least two models with identical test ids.")
    for pair in result["pairs"]:
        mc, diff = pair["mcnemar"], pair["macro_f1_diff"]
        verdict = "significant" if pair["significant_holm"] else "not significant"
        print(
            f"  {pair['model_a']} vs {pair['model_b']}: n={pair['n']} b={mc['b']} c={mc['c']} "
            f"McNemar p={mc['p_value']:.4f} ({mc['method']}), Holm p={pair['p_holm']:.4f} -> "
            f"{verdict}; diff {diff['diff']:+.3f} [{diff['low']:+.3f}, {diff['high']:+.3f}] "
            f"(bootstrap p={diff['p_value']:.4f})"
        )
    for item in result["skipped_pairs"]:
        print(f"  SKIPPED pair {item['model_a']} vs {item['model_b']}: {item['reason']}")
    for base, group in result["seed_runs"].items():
        seeds = ", ".join(str(run["seed"]) for run in group["runs"])
        print(
            f"\nSeed runs of {base} (seeds {seeds}): macro-F1 {group['macro_f1_mean']:.3f} "
            f"+/- {group['macro_f1_std']:.3f} (mean +/- std over {group['n_runs']} runs)"
        )
    print(f"\nwrote {display_path(out)}")


def main(argv: list[str] | None = None) -> dict:
    args = parse_args(argv)
    pred_dir, out, smoke = resolve_paths(args)
    predictions, skipped_models = load_all_predictions(pred_dir, args.models)
    if not predictions:
        sys.exit(
            f"No usable predictions CSVs in {pred_dir} (skipped: {skipped_models}). "
            "Run a training script first, e.g. python scripts/run_baseline.py"
        )
    names = ", ".join(sorted(predictions))
    print(f"comparing {len(predictions)} model(s) from {display_path(pred_dir)}: {names}")

    comparison = build_comparison(predictions, args.n_resamples, args.alpha, args.seed)
    result = {
        "kind": "comparison",
        "n_resamples": args.n_resamples,
        "alpha": args.alpha,
        "seed": args.seed,
        "smoke": smoke,
        "predictions_dir": display_path(pred_dir),
        "git_commit": metrics.git_commit(),
        "timestamp": datetime.now(UTC).isoformat(timespec="seconds"),
        **comparison,
        "skipped_models": skipped_models,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print_summary(result, out)
    return result


if __name__ == "__main__":
    main()
