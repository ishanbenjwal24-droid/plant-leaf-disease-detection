"""Train and evaluate the HOG + HSV + LinearSVC baseline."""

import time
from pathlib import Path

import joblib
import numpy as np
from PIL import Image, ImageOps
from sklearn.svm import LinearSVC

from training.common import (
    ARTIFACTS_DIR,
    load_class_names,
    load_dataset_config,
    load_splits,
    save_metadata,
    save_metrics,
)
from training.features import extract_features


def load_feature_rows(frame, data_root, label_to_id):
    features, labels = [], []
    for row in frame.itertuples(index=False):
        image_path = data_root / row.path
        with Image.open(image_path) as source:
            image = ImageOps.exif_transpose(source).convert("RGB")
            features.append(extract_features(image))
        labels.append(label_to_id[row.label])
    return np.asarray(features, dtype=np.float32), np.asarray(labels, dtype=np.int64)


def main():
    frame = load_splits()
    class_names = load_class_names()
    dataset_config = load_dataset_config()
    data_root = Path(dataset_config["dataset_root"])
    label_to_id = {name: index for index, name in enumerate(class_names)}
    train_frame = frame[frame["split"] == "train"]
    test_frame = frame[frame["split"] == "test"]

    print(f"Extracting ML features from {len(train_frame)} training images...")
    x_train, y_train = load_feature_rows(train_frame, data_root, label_to_id)
    print(f"Extracting ML features from {len(test_frame)} test images...")
    x_test, y_test = load_feature_rows(test_frame, data_root, label_to_id)

    model = LinearSVC(class_weight="balanced", random_state=42, max_iter=5000)
    model.fit(x_train, y_train)

    predictions = []
    start = time.perf_counter()
    for sample in x_test:
        predictions.append(int(model.predict(sample.reshape(1, -1))[0]))
    inference_ms = (time.perf_counter() - start) * 1000 / max(1, len(x_test))
    predictions = np.asarray(predictions, dtype=np.int64)

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    model_path = ARTIFACTS_DIR / "ml_model.joblib"
    joblib.dump(model, model_path)
    save_metrics(
        "ml", y_test, predictions, class_names, inference_ms, model_path,
        dataset_config["manifest_sha256"],
    )
    save_metadata(
        class_names,
        dataset_config,
        "ml",
        {
            "artifact": model_path.name,
            "version": f"ml-{model_path.stat().st_mtime_ns}",
            "algorithm": "LinearSVC over HOG and HSV mean/std features",
            "dataset_fingerprint": dataset_config["manifest_sha256"],
        },
    )
    print(f"Saved ML model to {model_path}")


if __name__ == "__main__":
    main()
