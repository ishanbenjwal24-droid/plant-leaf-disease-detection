"""FastAPI application for model inference, metrics, and prediction history."""

import json
import time

from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from starlette.concurrency import run_in_threadpool

from backend.app.config import Settings
from backend.app.db import PredictionHistory, create_session_factory
from backend.app.image_validation import ImageValidationError, decode_image
from backend.app.inference import ArtifactUnavailable, InferenceService
from backend.app.schemas import (
    ClassesResponse,
    CompareResponse,
    HealthResponse,
    HistoryItem,
    PredictionResult,
)


def create_app(settings=None, inference_service=None):
    settings = settings or Settings.from_env()
    service = inference_service or InferenceService(settings.artifacts_dir)
    session_factory = create_session_factory(settings.database_path)
    app = FastAPI(
        title="Plant Leaf Disease Detection API",
        version="1.0.0",
        description=(
            "Educational image classification API comparing HOG/LinearSVC ML "
            "with a CNN trained on PlantVillage. Predictions are not diagnoses."
        ),
    )
    app.state.settings = settings
    app.state.inference = service
    app.state.session_factory = session_factory
    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(settings.allowed_origins),
        allow_credentials=False,
        allow_methods=["GET", "POST", "DELETE"],
        allow_headers=["Content-Type"],
    )

    async def read_image(file):
        try:
            data = await file.read(settings.max_upload_bytes + 1)
        finally:
            await file.close()
        if len(data) > settings.max_upload_bytes:
            raise HTTPException(
                status_code=413,
                detail=f"Image exceeds the {settings.max_upload_bytes // (1024 * 1024)} MB upload limit.",
            )
        try:
            return decode_image(
                data,
                max_image_side=settings.max_image_side,
                max_image_pixels=settings.max_image_pixels,
            )
        except ImageValidationError as exc:
            raise HTTPException(status_code=exc.status_code, detail=str(exc)) from exc

    def add_history(mode, predictions, latency_ms):
        with session_factory() as session:
            record = PredictionHistory(
                mode=mode,
                predictions_json=json.dumps(predictions),
                latency_ms=float(latency_ms),
            )
            session.add(record)
            session.commit()
            session.refresh(record)
            return record

    async def run_prediction(model_name, image):
        try:
            return await run_in_threadpool(service.predict, model_name, image)
        except ArtifactUnavailable as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(
                status_code=500,
                detail="Prediction failed. Check the model artifacts and backend logs.",
            ) from exc

    @app.get("/api/health", response_model=HealthResponse, tags=["status"])
    def health():
        return {"status": "ok", "models": service.model_status()}

    @app.get("/api/classes", response_model=ClassesResponse, tags=["project"])
    def classes():
        names = service.class_names()
        return {"available": bool(names), "classes": names}

    @app.get("/api/metrics", tags=["evaluation"])
    def metrics():
        models = service.metrics()
        ml = models["ml"].get("metrics") if models["ml"]["available"] else None
        cnn = models["cnn"].get("metrics") if models["cnn"]["available"] else None
        comparison = None
        if ml is not None and cnn is not None:
            comparison = {
                "accuracy_difference_cnn_minus_ml": float(
                    cnn.get("accuracy", 0) - ml.get("accuracy", 0)
                ),
                "macro_f1_difference_cnn_minus_ml": float(
                    cnn.get("macro_f1", 0) - ml.get("macro_f1", 0)
                ),
            }
        return {"models": models, "comparison": comparison}

    @app.post("/api/predict", response_model=PredictionResult, tags=["prediction"])
    async def predict(
        file: UploadFile = File(...), model_name: str = Form(..., alias="model")
    ):
        if model_name not in {"ml", "cnn"}:
            raise HTTPException(status_code=422, detail="Model must be 'ml' or 'cnn'.")
        image = await read_image(file)
        prediction = await run_prediction(model_name, image)
        add_history(model_name, [prediction], prediction["inference_ms"])
        return prediction

    @app.post("/api/compare", response_model=CompareResponse, tags=["prediction"])
    async def compare(file: UploadFile = File(...)):
        image = await read_image(file)
        total_start = time.perf_counter()
        ml_prediction = await run_prediction("ml", image)
        cnn_prediction = await run_prediction("cnn", image)
        total_ms = (time.perf_counter() - total_start) * 1000
        add_history(
            "compare",
            [ml_prediction, cnn_prediction],
            total_ms,
        )
        return {
            "ml": ml_prediction,
            "cnn": cnn_prediction,
            "total_inference_ms": total_ms,
        }

    @app.get("/api/history", response_model=list[HistoryItem], tags=["history"])
    def history(limit: int = Query(default=50, ge=1, le=200)):
        with session_factory() as session:
            rows = (
                session.query(PredictionHistory)
                .order_by(PredictionHistory.created_at.desc(), PredictionHistory.id.desc())
                .limit(limit)
                .all()
            )
            return [
                {
                    "id": row.id,
                    "created_at": row.created_at.isoformat(),
                    "mode": row.mode,
                    "predictions": row.predictions,
                    "latency_ms": row.latency_ms,
                }
                for row in rows
            ]

    @app.delete("/api/history", tags=["history"])
    def clear_history():
        with session_factory() as session:
            deleted = session.query(PredictionHistory).delete(synchronize_session=False)
            session.commit()
        return {"deleted": deleted}

    return app


app = create_app()
