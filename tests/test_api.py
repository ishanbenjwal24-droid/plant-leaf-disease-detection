from io import BytesIO

from fastapi.testclient import TestClient
from PIL import Image

from backend.app.config import Settings
from backend.app.main import create_app


def make_image():
    buffer = BytesIO()
    Image.new("RGB", (20, 20), (80, 130, 50)).save(buffer, format="PNG")
    return buffer.getvalue()


class FakeInferenceService:
    def model_status(self):
        return {"ml": True, "cnn": True}

    def class_names(self):
        return ["Apple___healthy", "Apple___scab"]

    def metrics(self):
        return {
            "ml": {"available": False, "metrics": None},
            "cnn": {"available": False, "metrics": None},
        }

    def predict(self, model_name, image):
        class_name = "Apple___healthy" if model_name == "ml" else "Apple___scab"
        return {
            "model": model_name,
            "predicted_class": class_name,
            "display_name": class_name.replace("___", " — "),
            "class_score": 0.71,
            "top_classes": [
                {
                    "class_name": class_name,
                    "display_name": class_name.replace("___", " — "),
                    "score": 0.71,
                }
            ],
            "inference_ms": 2.1,
            "model_version": "test-model",
        }


def client_for(tmp_path, max_upload_bytes=1024):
    settings = Settings(
        project_root=tmp_path,
        artifacts_dir=tmp_path / "artifacts",
        database_path=tmp_path / "data" / "test.sqlite3",
        max_upload_bytes=max_upload_bytes,
        max_image_side=100,
        max_image_pixels=10_000,
        allowed_origins=("http://localhost:5173",),
    )
    return TestClient(create_app(settings, FakeInferenceService()))


def test_status_metrics_and_classes(tmp_path):
    client = client_for(tmp_path)
    assert client.get("/api/health").json()["status"] == "ok"
    assert client.get("/api/classes").json()["classes"] == [
        "Apple___healthy",
        "Apple___scab",
    ]
    metrics = client.get("/api/metrics").json()
    assert metrics["models"]["ml"]["metrics"] is None
    assert metrics["comparison"] is None


def test_compare_creates_metadata_history_without_image(tmp_path):
    client = client_for(tmp_path)
    response = client.post(
        "/api/compare",
        files={"file": ("leaf.png", make_image(), "image/png")},
    )
    assert response.status_code == 200
    result = response.json()
    assert result["ml"]["model"] == "ml"
    assert result["cnn"]["model"] == "cnn"
    history = client.get("/api/history").json()
    assert len(history) == 1
    assert history[0]["mode"] == "compare"
    assert len(history[0]["predictions"]) == 2
    assert client.delete("/api/history").json()["deleted"] == 1


def test_upload_validation_and_model_validation(tmp_path):
    client = client_for(tmp_path)
    corrupt = client.post(
        "/api/compare",
        files={"file": ("bad.png", b"bad", "image/png")},
    )
    assert corrupt.status_code == 400
    wrong_model = client.post(
        "/api/predict",
        data={"model": "unknown"},
        files={"file": ("leaf.png", make_image(), "image/png")},
    )
    assert wrong_model.status_code == 422


def test_upload_size_limit(tmp_path):
    client = client_for(tmp_path, max_upload_bytes=64)
    response = client.post(
        "/api/compare",
        files={"file": ("leaf.png", make_image(), "image/png")},
    )
    assert response.status_code == 413
