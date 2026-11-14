"""Create the fixed train/val/test split. Run this ONCE (Day 1).

    python scripts/prepare_data.py                    # primary dataset
    python scripts/prepare_data.py --source fallback  # Financial PhraseBank mirror

Steps: download (cached in data/raw/) -> clean -> seed-42 stratified 70/15/15 split ->
data/splits/{train,val,test}.csv plus data/splits/split_report.json (cleaning counts,
sizes, class balance). Existing split files are never overwritten unless you pass
--overwrite, because every result in this repo depends on the split staying fixed.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from fnsl import config, data


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--source", choices=["primary", "fallback"], default="primary")
    parser.add_argument("--seed", type=int, default=config.SEED)
    parser.add_argument("--out-dir", type=Path, default=config.SPLITS_DIR)
    parser.add_argument(
        "--overwrite", action="store_true", help="replace existing split files (avoid!)"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config.set_seed(args.seed)

    if args.source == "primary":
        name, revision, license_ = (
            config.DATASET_NAME,
            config.DATASET_REVISION,
            config.DATASET_LICENSE,
        )
    else:
        name, revision, license_ = (
            config.FALLBACK_DATASET_NAME,
            config.FALLBACK_DATASET_REVISION,
            config.FALLBACK_DATASET_LICENSE,
        )

    raw = data.load_raw(args.source)
    print(f"Loaded {len(raw)} rows from {name} @ {revision[:8]}")
    print("Rows per Hub split:", raw["source_split"].value_counts().to_dict())

    clean_df, cleaning = data.clean(raw)
    print("Cleaning report:", cleaning)

    splits = data.make_splits(clean_df, seed=args.seed)
    data.save_splits(splits, args.out_dir, overwrite=args.overwrite)

    balance = data.class_balance({"all": clean_df, **splits})
    print("\nClass balance (counts and %):")
    print(balance.to_string())

    report = {
        "dataset": name,
        "revision": revision,
        "license": license_,
        "seed": args.seed,
        "fractions": config.SPLIT_FRACTIONS,
        "cleaning": cleaning,
        "sizes": {split: len(part) for split, part in splits.items()},
        "class_balance": balance.to_dict(orient="index"),
        "labels": config.ID2LABEL,
    }
    report_path = data.split_report_path(args.out_dir)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"\nWrote splits and {report_path.relative_to(config.ROOT)}")


if __name__ == "__main__":
    main()
