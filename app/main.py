import math
import os
import time
from typing import Any

import psutil
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from sklearn.datasets import load_breast_cancer, load_digits, load_wine
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

app = FastAPI(title="Model Monitoring API")
started_at = time.monotonic()

DATASET_DEFINITIONS = {
    "model_1": {"name": "Wine Classifier", "dataset": "Wine", "loader": load_wine},
    "model_2": {"name": "Breast Cancer Classifier", "dataset": "Breast Cancer", "loader": load_breast_cancer},
    "model_3": {"name": "Handwritten Digit Classifier", "dataset": "Digits", "loader": load_digits},
}


def create_model_metrics() -> dict[str, Any]:
    return {
        "total_requests": 0,
        "average_latency_ms": 0.0,
        "recent_predictions": [],
        "evaluated_predictions": [],
        "error_count": 0,
        "failed_predictions": 0,
        "invalid_outputs": 0,
        "missing_values": 0,
    }


def build_model_registry() -> dict[str, dict[str, Any]]:
    registry = {}
    for model_id, definition in DATASET_DEFINITIONS.items():
        dataset = definition["loader"]()
        classifier = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
        classifier.fit(dataset.data, dataset.target)
        labels = [int(label) for label in classifier.classes_]
        samples = []
        for label in labels:
            sample_index = next(index for index, target in enumerate(dataset.target) if target == label)
            samples.append({"actual_class": label, "features": [float(value) for value in dataset.data[sample_index]]})
        registry[model_id] = {
            "id": model_id,
            "name": definition["name"],
            "dataset": definition["dataset"],
            "model": classifier,
            "labels": labels,
            "feature_count": int(dataset.data.shape[1]),
            "samples": samples,
            "metrics": create_model_metrics(),
        }
    return registry


models = build_model_registry()
global_errors = {"total": 0, "missing_values": 0}


class PredictionRequest(BaseModel):
    model_id: str = Field(..., description="Registered model identifier")
    features: list[float] = Field(..., min_length=1, description="Feature vector for the selected model")
    actual_class: int | None = Field(default=None, description="Optional ground-truth class for quality evaluation")


@app.exception_handler(RequestValidationError)
async def handle_validation_error(request: Request, exc: RequestValidationError):
    missing = sum(1 for error in exc.errors() if error.get("type") in {"missing", "list_type"})
    global_errors["total"] += 1
    global_errors["missing_values"] += missing
    return JSONResponse(status_code=422, content={"detail": exc.errors(), "error": "Invalid or incomplete inference payload"})


def record_model_error(model_metrics: dict[str, Any], error_type: str):
    model_metrics["error_count"] += 1
    model_metrics[error_type] += 1


def model_quality(model_metrics: dict[str, Any], labels: list[int]) -> dict[str, Any]:
    evaluated = model_metrics["evaluated_predictions"]
    actual = [item["actual"] for item in evaluated]
    predicted = [item["predicted"] for item in evaluated]
    if not evaluated:
        return {"evaluated_requests": 0, "accuracy": None, "precision": None, "recall": None, "f1_score": None}
    return {
        "evaluated_requests": len(evaluated),
        "accuracy": round(accuracy_score(actual, predicted), 4),
        "precision": round(precision_score(actual, predicted, labels=labels, average="macro", zero_division=0), 4),
        "recall": round(recall_score(actual, predicted, labels=labels, average="macro", zero_division=0), 4),
        "f1_score": round(f1_score(actual, predicted, labels=labels, average="macro", zero_division=0), 4),
    }


@app.post("/predict")
def predict(request: PredictionRequest):
    selected_model = models.get(request.model_id)
    if selected_model is None:
        global_errors["total"] += 1
        return JSONResponse(status_code=404, content={"error": "Unknown model", "model_id": request.model_id})

    model_metrics = selected_model["metrics"]
    if len(request.features) != selected_model["feature_count"]:
        record_model_error(model_metrics, "missing_values")
        return JSONResponse(status_code=422, content={"error": "Incorrect feature count", "expected": selected_model["feature_count"]})
    if any(not math.isfinite(value) for value in request.features):
        record_model_error(model_metrics, "missing_values")
        return JSONResponse(status_code=422, content={"error": "Features must contain finite numeric values"})
    if request.actual_class is not None and request.actual_class not in selected_model["labels"]:
        record_model_error(model_metrics, "invalid_outputs")
        return JSONResponse(status_code=422, content={"error": "Ground-truth class is not valid for this model"})

    started_prediction = time.perf_counter()
    try:
        prediction = int(selected_model["model"].predict([request.features])[0])
    except Exception as error:
        record_model_error(model_metrics, "failed_predictions")
        return JSONResponse(status_code=500, content={"error": "Prediction failed", "detail": str(error)})

    latency = round((time.perf_counter() - started_prediction) * 1000, 2)
    model_metrics["total_requests"] += 1
    previous_total = model_metrics["average_latency_ms"] * (model_metrics["total_requests"] - 1)
    model_metrics["average_latency_ms"] = round((previous_total + latency) / model_metrics["total_requests"], 2)

    if prediction not in selected_model["labels"]:
        record_model_error(model_metrics, "invalid_outputs")
    elif request.actual_class is not None:
        model_metrics["evaluated_predictions"].append({"actual": request.actual_class, "predicted": prediction})
        if len(model_metrics["evaluated_predictions"]) > 500:
            model_metrics["evaluated_predictions"].pop(0)

    model_metrics["recent_predictions"].append(prediction)
    if len(model_metrics["recent_predictions"]) > 20:
        model_metrics["recent_predictions"].pop(0)
    return {"model_id": request.model_id, "prediction": prediction, "latency_ms": latency}


def public_model(model: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": model["id"],
        "name": model["name"],
        "dataset": model["dataset"],
        "feature_count": model["feature_count"],
        "class_count": len(model["labels"]),
        "samples": model["samples"],
    }


@app.get("/api/models")
def get_models():
    return {"models": [public_model(model) for model in models.values()]}


@app.get("/api/metrics")
def get_metrics():
    model_metrics = {}
    for model_id, model in models.items():
        current = model["metrics"]
        model_metrics[model_id] = {
            "id": model_id,
            "name": model["name"],
            "dataset": model["dataset"],
            "feature_count": model["feature_count"],
            "class_count": len(model["labels"]),
            "total_requests": current["total_requests"],
            "average_latency_ms": current["average_latency_ms"],
            "recent_predictions": current["recent_predictions"],
            "quality": model_quality(current, model["labels"]),
            "errors": {
                "total": current["error_count"],
                "failed_predictions": current["failed_predictions"],
                "invalid_outputs": current["invalid_outputs"],
                "missing_values": current["missing_values"],
            },
        }
    return {
        "models": model_metrics,
        "global_errors": global_errors,
        "system_health": {
            "cpu_percent": psutil.cpu_percent(interval=None),
            "memory_percent": psutil.virtual_memory().percent,
            "uptime_seconds": round(time.monotonic() - started_at, 1),
            "api_status": "Operational",
        },
    }


static_dir = os.path.join(os.path.dirname(__file__), "static")
app.mount("/static", StaticFiles(directory=static_dir), name="static")


@app.get("/")
def read_root():
    return FileResponse(os.path.join(static_dir, "index.html"))
