"""Lazy model loading and consistent ML/CNN inference."""

import json
import time
from pathlib import Path

import joblib
import numpy as np
from PIL import Image, ImageOps

from training.common import top_scores
from training.features import extract_features


def display_class(name):
    return name.replace("___", " — ").replace("_", " ")


class ArtifactUnavailable(Exception):
    pass


class InferenceService:
    def __init__(self, artifacts_dir):
        self.artifacts_dir = Path(artifacts_dir)
        self._loaded = {}

    def class_names(self):
        path = self.artifacts_dir / "class_names.json"
        if not path.exists():
            return []
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return []

    def model_status(self):
        metadata = self._metadata()
        fingerprint = metadata.get("dataset", {}).get("manifest_sha256")
        models = metadata.get("models", {})
        return {
            model_name: (
                (self.artifacts_dir / filename).is_file()
                and fingerprint is not None
                and models.get(model_name, {}).get("dataset_fingerprint") == fingerprint
            )
            for model_name, filename in {
                "ml": "ml_model.joblib",
                "cnn": "cnn_model.keras",
            }.items()
        }

    def _metadata(self):
        path = self.artifacts_dir / "metadata.json"
        if not path.exists():
            return {}
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}

    def _dataset_fingerprint(self):
        return self._metadata().get("dataset", {}).get("manifest_sha256")

    def metrics(self):
        output = {}
        for model_name in ("ml", "cnn"):
            path = self.artifacts_dir / f"{model_name}_metrics.json"
            if not path.exists():
                output[model_name] = {"available": False, "metrics": None}
                continue
            try:
                metric_data = json.loads(path.read_text(encoding="utf-8"))
                is_current = (
                    metric_data.get("dataset_fingerprint") is not None
                    and metric_data.get("dataset_fingerprint") == self._dataset_fingerprint()
                )
                output[model_name] = {
                    "available": is_current,
                    "metrics": metric_data if is_current else None,
                }
            except (OSError, json.JSONDecodeError):
                output[model_name] = {"available": False, "metrics": None}
        return output

    def model_version(self, model_name):
        return self._metadata().get("models", {}).get(model_name, {}).get("version")

    def _load_model(self, model_name):
        if model_name in self._loaded:
            return self._loaded[model_name]
        filenames = {"ml": "ml_model.joblib", "cnn": "cnn_model.keras"}
        if model_name not in filenames:
            raise ValueError("Model must be 'ml' or 'cnn'.")
        model_path = self.artifacts_dir / filenames[model_name]
        if not model_path.is_file():
            raise ArtifactUnavailable(
                f"The {model_name.upper()} model is not trained yet. Follow the README training command."
            )
        if not self.model_status()[model_name]:
            raise ArtifactUnavailable(
                f"The {model_name.upper()} artifact was trained on a different dataset split. "
                "Rerun its training command after preparing the current dataset."
            )
        try:
            if model_name == "ml":
                model = joblib.load(model_path)
            else:
                import tensorflow as tf

                model = tf.keras.models.load_model(model_path)
        except Exception as exc:
            raise ArtifactUnavailable(
                f"Could not load the {model_name.upper()} artifact. Retrain it using the README instructions."
            ) from exc
        self._loaded[model_name] = model
        return model

    def predict(self, model_name, image):
        class_names = self.class_names()
        if not class_names:
            raise ArtifactUnavailable(
                "Class names are missing. Run the dataset preparation command in the README first."
            )
        model = self._load_model(model_name)
        start = time.perf_counter()
        if model_name == "ml":
            image = ImageOps.exif_transpose(image).convert("RGB")
            feature_row = extract_features(image).reshape(1, -1)
            decision = model.decision_function(feature_row)[0]
            ranked = top_scores(decision, model.classes_)
            ranked = [item for item in ranked if 0 <= item[0] < len(class_names)]
        else:
            image = ImageOps.exif_transpose(image).convert("RGB")
            resized = image.resize((128, 128), Image.Resampling.BILINEAR)
            batch = np.asarray(resized, dtype=np.float32)[None, ...] / 255.0
            scores = np.asarray(model(batch, training=False).numpy()[0]).reshape(-1)
            ranked = sorted(
                enumerate(scores.tolist()), key=lambda item: item[1], reverse=True
            )
        inference_ms = (time.perf_counter() - start) * 1000
        if not ranked:
            raise ArtifactUnavailable("The model returned no valid class scores.")
        best_class, best_score = ranked[0]
        top_classes = [
            {
                "class_name": class_names[class_id],
                "display_name": display_class(class_names[class_id]),
                "score": float(score),
            }
            for class_id, score in ranked[:3]
        ]
        return {
            "model": model_name,
            "predicted_class": class_names[best_class],
            "display_name": display_class(class_names[best_class]),
            "class_score": float(best_score),
            "top_classes": top_classes,
            "inference_ms": float(inference_ms),
            "model_version": self.model_version(model_name),
        }
