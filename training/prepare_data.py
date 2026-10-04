"""Validate PlantVillage folders, remove exact duplicates, and create splits."""

import argparse
import hashlib
import json
import os
import random
from collections import defaultdict
from pathlib import Path

import pandas as pd
from PIL import Image, UnidentifiedImageError
from dotenv import load_dotenv

from training.common import ARTIFACTS_DIR, DATA_DIR, save_metadata


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SEED = 42
load_dotenv(Path(__file__).resolve().parents[1] / ".env")


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as file_obj:
        for chunk in iter(lambda: file_obj.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_image(path):
    try:
        with Image.open(path) as image:
            image.verify()
        return True, None
    except (OSError, UnidentifiedImageError, ValueError) as exc:
        return False, str(exc)


def split_one_class(items, rng):
    """Return a deterministic, approximately 70/15/15 split for one class."""
    items = list(items)
    rng.shuffle(items)
    n_items = len(items)
    test_count = max(1, round(n_items * 0.15))
    validation_count = max(1, round(n_items * 0.15))
    while test_count + validation_count >= n_items:
        if validation_count >= test_count and validation_count > 1:
            validation_count -= 1
        elif test_count > 1:
            test_count -= 1
        else:
            break
    train_count = n_items - test_count - validation_count
    if train_count < 1:
        raise ValueError("Each class needs at least three unique, readable images.")
    return {
        "train": items[:train_count],
        "validation": items[train_count : train_count + validation_count],
        "test": items[train_count + validation_count :],
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    default_data_dir = os.environ.get("DATASET_DIR") or str(DATA_DIR / "raw")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(default_data_dir),
        help="Dataset directory (default: DATASET_DIR or data/raw).",
    )
    parser.add_argument(
        "--max-per-class",
        type=int,
        default=200,
        help="Maximum unique images sampled per class (default: 200).",
    )
    args = parser.parse_args(argv)
    root = args.data_dir.expanduser().resolve()
    if not root.is_dir():
        raise SystemExit(
            f"Dataset directory not found: {root}. Obtain PlantVillage and set DATASET_DIR."
        )
    if args.max_per_class < 3:
        raise SystemExit("--max-per-class must be at least 3.")

    unreadable = []
    digest_groups = defaultdict(list)
    class_dirs = sorted(path for path in root.iterdir() if path.is_dir())
    for class_dir in class_dirs:
        for path in sorted(class_dir.rglob("*")):
            if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            is_valid, error = validate_image(path)
            if not is_valid:
                unreadable.append(
                    {"path": str(path.relative_to(root)), "error": error}
                )
                continue
            digest = sha256_file(path)
            digest_groups[digest].append(
                {
                    "path": str(path.relative_to(root)),
                    "label": class_dir.name,
                    "sha256": digest,
                }
            )

    duplicate_rows = []
    unique_by_class = defaultdict(list)
    for digest, group in digest_groups.items():
        if len(group) > 1:
            duplicate_rows.extend(
                {
                    "sha256": digest,
                    "path": item["path"],
                    "label": item["label"],
                    "action": "excluded duplicate group",
                }
                for item in group
            )
            continue
        item = group[0]
        unique_by_class[item["label"]].append(item)

    rng = random.Random(SEED)
    manifest_rows = []
    class_names = []
    class_counts = {}
    skipped_classes = []
    for class_dir in class_dirs:
        label = class_dir.name
        class_items = unique_by_class.get(label, [])
        if len(class_items) < 3:
            skipped_classes.append({"label": label, "unique_readable_images": len(class_items)})
            continue
        if len(class_items) > args.max_per_class:
            class_items = rng.sample(class_items, args.max_per_class)
        partitions = split_one_class(class_items, rng)
        class_names.append(label)
        class_counts[label] = {
            split: len(items) for split, items in partitions.items()
        }
        for split, split_items in partitions.items():
            manifest_rows.extend(
                {
                    "path": item["path"],
                    "sha256": item["sha256"],
                    "label": label,
                    "split": split,
                }
                for item in split_items
            )

    if len(class_names) < 2:
        raise SystemExit(
            "Need at least two class folders with three or more unique, readable images."
        )

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(manifest_rows).to_csv(DATA_DIR / "splits.csv", index=False)
    pd.DataFrame(duplicate_rows, columns=["sha256", "path", "label", "action"]).to_csv(
        DATA_DIR / "duplicates.csv", index=False
    )
    pd.DataFrame(unreadable, columns=["path", "error"]).to_csv(
        DATA_DIR / "unreadable_files.csv", index=False
    )
    manifest_fingerprint = sha256_file(DATA_DIR / "splits.csv")
    config = {
        "dataset_name": "PlantVillage",
        "dataset_root": str(root),
        "seed": SEED,
        "manifest_sha256": manifest_fingerprint,
        "max_per_class": args.max_per_class,
        "requested_split_ratio": {"train": 0.70, "validation": 0.15, "test": 0.15},
        "class_names": class_names,
        "class_counts": class_counts,
        "num_images": len(manifest_rows),
        "exact_duplicate_files_excluded": len(duplicate_rows),
        "unreadable_files": len(unreadable),
        "skipped_classes": skipped_classes,
    }
    (DATA_DIR / "dataset_config.json").write_text(
        json.dumps(config, indent=2), encoding="utf-8"
    )
    (ARTIFACTS_DIR / "class_names.json").write_text(
        json.dumps(class_names, indent=2), encoding="utf-8"
    )
    save_metadata(class_names, config)
    print(f"Found {len(class_names)} classes and {len(manifest_rows)} selected images.")
    print(f"Excluded {len(duplicate_rows)} duplicate files and {len(unreadable)} unreadable files.")
    print(f"Split manifest: {DATA_DIR / 'splits.csv'}")
    print(f"Dataset reports: {DATA_DIR / 'duplicates.csv'}, {DATA_DIR / 'unreadable_files.csv'}")


if __name__ == "__main__":
    main()
