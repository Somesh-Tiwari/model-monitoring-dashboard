import json
import math
import os
import time
from pathlib import Path
from typing import Any

import joblib
import psutil
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from app.modeling import DATASET_NAME, load_split, train_model

app = FastAPI(
    title="Model Monitoring API",
    description="Prediction and deployment monitoring for the Wine classification model.",
    version="2.0.0",
)
started_at = time.monotonic()
APP_DIR = Path(__file__).resolve().parent
ARTIFACT_DIR = APP_DIR / "artifacts"
PRODUCTION_MODEL_PATH = ARTIFACT_DIR / "production.joblib"
MONITORING_DIR = Path(os.environ.get("MONITORING_DATA_DIR", str(ARTIFACT_DIR)))
MONITORING_REPORT_PATH = MONITORING_DIR / "deployment_report.json"
MODEL_VERSION_PATH = ARTIFACT_DIR / "production-version.json"

_, _, TEST_FEATURES, _, FEATURE_NAMES, CLASS_NAMES = load_split()


def load_production_model():
    if PRODUCTION_MODEL_PATH.exists():
        return joblib.load(PRODUCTION_MODEL_PATH)
    # Source runs before the first Jenkins deployment use the fixed baseline.
    return train_model(random_state=42)


production_model = load_production_model()
try:
    ACTIVE_MODEL_VERSION = json.loads(MODEL_VERSION_PATH.read_text(encoding="utf-8"))["version"]
except (OSError, json.JSONDecodeError, KeyError):
    ACTIVE_MODEL_VERSION = "bootstrap-seed-42"
metrics: dict[str, Any] = {
    "total_requests": 0,
    "average_latency_ms": 0.0,
    "recent_predictions": [],
    "errors": 0,
    "failed_predictions": 0,
}
demo_index = 0


class PredictionRequest(BaseModel):
    features: list[float] = Field(..., min_length=1, description="Wine feature values")


def read_deployment_report():
    try:
        with MONITORING_REPORT_PATH.open(encoding="utf-8") as report_file:
            return json.load(report_file)
    except (OSError, json.JSONDecodeError):
        return None


def run_prediction(features: list[float]):
    if len(features) != len(FEATURE_NAMES):
        raise HTTPException(status_code=422, detail=f"Expected {len(FEATURE_NAMES)} numeric features")
    if any(not math.isfinite(value) for value in features):
        raise HTTPException(status_code=422, detail="Features must be finite numeric values")
    start = time.perf_counter()
    try:
        predicted = int(production_model.predict([features])[0])
    except Exception as error:
        metrics["errors"] += 1
        metrics["failed_predictions"] += 1
        return JSONResponse(status_code=500, content={"error": "Prediction failed", "detail": str(error)})
    latency = round((time.perf_counter() - start) * 1000, 2)
    metrics["total_requests"] += 1
    previous_total = metrics["average_latency_ms"] * (metrics["total_requests"] - 1)
    metrics["average_latency_ms"] = round((previous_total + latency) / metrics["total_requests"], 2)
    metrics["recent_predictions"].append(predicted)
    del metrics["recent_predictions"][:-20]
    return {"prediction": predicted, "prediction_label": CLASS_NAMES[predicted], "latency_ms": latency}


@app.post("/predict")
def predict(request: PredictionRequest):
    return run_prediction(request.features)


@app.post("/api/demo-prediction")
def demo_prediction():
    """Run one predefined held-out input; its target remains server-side."""
    global demo_index
    features = [float(value) for value in TEST_FEATURES[demo_index % len(TEST_FEATURES)]]
    demo_index += 1
    return run_prediction(features)


@app.get("/api/model")
def get_model():
    return {
        "name": "Wine Classifier",
        "dataset": DATASET_NAME,
        "feature_count": len(FEATURE_NAMES),
        "feature_names": FEATURE_NAMES,
        "class_names": CLASS_NAMES,
    }


@app.get("/api/metrics")
def get_metrics():
    report = read_deployment_report()
    candidate = report.get("candidate", {}) if report else {}
    active_result = candidate if report and candidate.get("version") == ACTIVE_MODEL_VERSION else (report or {}).get("previous_production", {})
    quality = active_result.get("metrics", {})
    return {
        "model": {
            "name": "Wine Classifier",
            "dataset": DATASET_NAME,
            "total_requests": metrics["total_requests"],
            "average_latency_ms": metrics["average_latency_ms"],
            "recent_predictions": metrics["recent_predictions"],
            "errors": metrics["errors"],
            "quality": {
                "accuracy": quality.get("accuracy"),
                "precision": quality.get("precision"),
                "recall": quality.get("recall"),
                "f1_score": quality.get("f1_score"),
                "evaluated_inputs": quality.get("evaluated_inputs", 0),
            },
        },
        "deployment": report,
        "system_health": {
            "cpu_percent": psutil.cpu_percent(interval=None),
            "memory_percent": psutil.virtual_memory().percent,
            "uptime_seconds": round(time.monotonic() - started_at, 1),
            "api_status": "Operational",
        },
    }


static_dir = APP_DIR / "static"
app.mount("/static", StaticFiles(directory=static_dir), name="static")


@app.get("/")
def read_root():
    return FileResponse(static_dir / "index.html")
