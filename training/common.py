"""Shared paths and evaluation helpers for model training."""

import json
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env")


def project_path(variable, default):
    configured = os.getenv(variable)
    path = Path(configured) if configured else Path(default)
    return path if path.is_absolute() else PROJECT_ROOT / path


DATA_DIR = project_path("DATA_DIR", "data")
ARTIFACTS_DIR = project_path("ARTIFACTS_DIR", "artifacts")
SPLITS_PATH = DATA_DIR / "splits.csv"
CLASS_NAMES_PATH = ARTIFACTS_DIR / "class_names.json"
METADATA_PATH = ARTIFACTS_DIR / "metadata.json"


def load_splits():
    if not SPLITS_PATH.exists():
        raise FileNotFoundError(
            f"{SPLITS_PATH} is missing. Run `python -m training.prepare_data` first."
        )
    import pandas as pd

    return pd.read_csv(SPLITS_PATH)


def load_class_names():
    if not CLASS_NAMES_PATH.exists():
        raise FileNotFoundError(
            f"{CLASS_NAMES_PATH} is missing. Run `python -m training.prepare_data` first."
        )
    return json.loads(CLASS_NAMES_PATH.read_text(encoding="utf-8"))


def load_dataset_config():
    config_path = DATA_DIR / "dataset_config.json"
    if not config_path.exists():
        raise FileNotFoundError(
            f"{config_path} is missing. Run `python -m training.prepare_data` first."
        )
    return json.loads(config_path.read_text(encoding="utf-8"))


def save_metadata(class_names, dataset_config, model_name=None, model_details=None):
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    metadata = {}
    if METADATA_PATH.exists():
        metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    metadata.update(
        {
            "project": "Plant Leaf Disease Detection",
            "class_names": list(class_names),
            "dataset": dataset_config,
            "image_preprocessing": {
                "orientation": "EXIF-transposed",
                "color_mode": "RGB",
                "ml_size": [96, 96],
                "cnn_size": [128, 128],
                "cnn_scale": "pixel values divided by 255",
            },
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
    )
    if model_name and model_details:
        metadata.setdefault("models", {})[model_name] = model_details
    METADATA_PATH.write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def save_metrics(
    model_name, y_true, y_pred, class_names, inference_ms, model_path, dataset_fingerprint
):
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    metrics = {
        "available": True,
        "model": model_name,
        "dataset_fingerprint": dataset_fingerprint,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_precision": 0.0,
        "macro_recall": 0.0,
        "macro_f1": 0.0,
        "average_inference_ms_per_image": float(inference_ms),
        "model_size_bytes": int(Path(model_path).stat().st_size),
        "classification_report": classification_report(
            y_true,
            y_pred,
            labels=list(range(len(class_names))),
            target_names=class_names,
            output_dict=True,
            zero_division=0,
        ),
        "confusion_matrix": confusion_matrix(
            y_true, y_pred, labels=list(range(len(class_names)))
        ).tolist(),
        "class_names": list(class_names),
    }
    summary = metrics["classification_report"].get("macro avg", {})
    metrics["macro_precision"] = float(summary.get("precision", 0.0))
    metrics["macro_recall"] = float(summary.get("recall", 0.0))
    metrics["macro_f1"] = float(summary.get("f1-score", 0.0))
    path = ARTIFACTS_DIR / f"{model_name}_metrics.json"
    path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(
        f"{model_name}: accuracy={metrics['accuracy']:.4f}, "
        f"macro F1={metrics['macro_f1']:.4f}, "
        f"avg inference={inference_ms:.2f} ms/image"
    )
    print(f"Saved metrics to {path}")
    return metrics


def top_scores(decision_values, class_indices):
    """Map LinearSVC decision values to ordered class-index/score pairs."""
    values = np.asarray(decision_values).reshape(-1)
    indices = np.asarray(class_indices, dtype=int)
    if len(indices) == 2 and len(values) == 1:
        values = np.asarray([-values[0], values[0]], dtype=float)
    return sorted(
        ((int(class_id), float(score)) for class_id, score in zip(indices, values)),
        key=lambda item: item[1],
        reverse=True,
    )
