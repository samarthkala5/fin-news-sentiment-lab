"""TF-IDF + logistic regression baseline (Day 2).

    python scripts/run_baseline.py            # tune on val, evaluate once on test
    python scripts/run_baseline.py --smoke    # tiny subsets, writes to results/smoke/

Fits on train, picks C and class_weight by validation macro-F1, refits the best
setting on train, and evaluates it once on the test split. Writes
results/metrics/tfidf_logreg.json, results/predictions/tfidf_logreg.csv and a
confusion-matrix figure, and logs the run to MLflow (mlruns/).
"""

from __future__ import annotations

import argparse
import time

from fnsl import config, data, metrics
from fnsl.baselines import build_pipeline, tune_on_val
from fnsl.tracking import mlflow_run

MODEL_NAME = "tfidf_logreg"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="TF-IDF + logistic regression baseline")
    parser.add_argument("--smoke", action="store_true", help="tiny data; results/smoke/")
    parser.add_argument("--seed", type=int, default=config.SEED)
    parser.add_argument("--c-grid", type=float, nargs="+", default=[0.1, 0.3, 1.0, 3.0, 10.0])
    parser.add_argument("--no-mlflow", action="store_true", help="disable MLflow logging")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> dict:
    args = parse_args(argv)
    config.set_seed(args.seed)
    splits = data.load_run_splits(smoke=args.smoke)
    train, val, test = splits["train"], splits["val"], splits["test"]
    print(f"train={len(train)} val={len(val)} test={len(test)} smoke={args.smoke}")

    # 1) Choose hyperparameters on the validation split.
    best, grid = tune_on_val(
        train, val, c_grid=args.c_grid, class_weight_grid=[None, "balanced"], seed=args.seed
    )
    best_val = max(r["val_macro_f1"] for r in grid)
    for r in grid:
        weight = str(r["class_weight"])
        print(f"  C={r['C']:<5} class_weight={weight:<8} val_macro_f1={r['val_macro_f1']:.4f}")
    print(f"best on val: {best} (val macro-F1 {best_val:.4f})")

    # 2) Refit the chosen setting on train and evaluate ONCE on test.
    params = {
        **best,
        "ngram_range": [1, 2],
        "min_df": 2,
        "sublinear_tf": True,
        "max_iter": 2000,
        "val_macro_f1": best_val,
    }
    with mlflow_run(MODEL_NAME, params, enabled=not (args.smoke or args.no_mlflow)) as run:
        start = time.perf_counter()
        model = build_pipeline(C=best["C"], class_weight=best["class_weight"], seed=args.seed)
        model.fit(train["text"], train["label"])
        train_time = time.perf_counter() - start

        record = metrics.save_results(
            MODEL_NAME,
            test,
            model.predict(test["text"]),
            params=params,
            train_time_s=train_time,
            seed=args.seed,
            smoke=args.smoke,
            extra={"val_grid": grid, "n_features": len(model.named_steps["tfidf"].vocabulary_)},
        )
        run.log_metrics(
            {
                "val_macro_f1": best_val,
                "test_macro_f1": record["macro_f1"],
                "test_accuracy": record["accuracy"],
                **{f"test_f1_{k}": v for k, v in record["per_class_f1"].items()},
            }
        )
        run.log_artifact(config.output_dirs(args.smoke)["figures"] / f"{MODEL_NAME}_confusion.png")

    print(
        f"TEST macro-F1 {record['macro_f1']:.4f}  accuracy {record['accuracy']:.4f}  "
        f"per-class {record['per_class_f1']}"
    )
    return record


if __name__ == "__main__":
    main()
