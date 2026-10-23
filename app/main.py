import json
import math
import os
import time
from pathlib import Path
from typing import Any, Literal

import joblib
import pandas as pd
import psutil
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, mean_squared_error, precision_score, r2_score, recall_score

from app.modeling import DATASET_NAME, FEATURE_NAMES, TARGET_NAME, load_split, model_matches_dataset, point_band, train_model

app = FastAPI(
    title="Fantasy Player Points Monitoring API",
    description="Prediction and deployment monitoring for the fixed 2026–2027 FPL player dataset.",
    version="3.0.0",
)
started_at = time.monotonic()
APP_DIR = Path(__file__).resolve().parent
ARTIFACT_DIR = APP_DIR / "artifacts"
PRODUCTION_MODEL_PATH = ARTIFACT_DIR / "production.joblib"
MONITORING_DIR = Path(os.environ.get("MONITORING_DATA_DIR", str(ARTIFACT_DIR)))
MONITORING_REPORT_PATH = MONITORING_DIR / "deployment_report.json"
MONITORING_HISTORY_PATH = MONITORING_DIR / "comparison_history.json"
MODEL_VERSION_PATH = ARTIFACT_DIR / "production-version.json"

_, _, _, _, TEST_FEATURES, TEST_POINTS, _, _, TEST_PLAYERS = load_split()


def load_production_model():
    if PRODUCTION_MODEL_PATH.exists():
        try:
            model = joblib.load(PRODUCTION_MODEL_PATH)
            if model_matches_dataset(model):
                return model
        except (OSError, EOFError, ValueError):
            pass
    return train_model(random_state=42)


production_model = load_production_model()
initial_test_predictions = production_model.predict(TEST_FEATURES)
initial_actual_bands = [point_band(value) for value in TEST_POINTS]
initial_predicted_bands = [point_band(value) for value in initial_test_predictions]
INITIAL_TEST_METRICS = {
    "accuracy": round(float(accuracy_score(initial_actual_bands, initial_predicted_bands)), 4),
    "precision": round(float(precision_score(initial_actual_bands, initial_predicted_bands, average="macro", zero_division=0)), 4),
    "recall": round(float(recall_score(initial_actual_bands, initial_predicted_bands, average="macro", zero_division=0)), 4),
    "f1_score": round(float(f1_score(initial_actual_bands, initial_predicted_bands, average="macro", zero_division=0)), 4),
    "mae": round(float(mean_absolute_error(TEST_POINTS, initial_test_predictions)), 3),
    "rmse": round(float(mean_squared_error(TEST_POINTS, initial_test_predictions) ** 0.5), 3),
    "r2_score": round(float(r2_score(TEST_POINTS, initial_test_predictions)), 4),
    "evaluated_inputs": len(TEST_POINTS),
}
try:
    ACTIVE_MODEL_VERSION = json.loads(MODEL_VERSION_PATH.read_text(encoding="utf-8"))["version"]
except (OSError, json.JSONDecodeError, KeyError):
    ACTIVE_MODEL_VERSION = "fpl-bootstrap-seed-42"
metrics: dict[str, Any] = {
    "total_requests": 0,
    "average_latency_ms": 0.0,
    "recent_predictions": [],
    "errors": 0,
    "failed_predictions": 0,
}
demo_index = 0


class PredictionRequest(BaseModel):
    position: Literal["GKP", "DEF", "MID", "FWD"]
    now_cost: float = Field(..., ge=0, le=300, description="FPL cost in tenths of £1m")
    minutes: float = Field(..., ge=0, le=6000)
    starts: float = Field(..., ge=0, le=100)
    expected_goal_involvements: float = Field(..., ge=0)
    expected_goals: float = Field(..., ge=0)
    expected_assists: float = Field(..., ge=0)

    class Config:
        extra = "forbid"


def read_deployment_report():
    try:
        with MONITORING_REPORT_PATH.open(encoding="utf-8") as report_file:
            report = json.load(report_file)
            return report if report.get("dataset") == DATASET_NAME else None
    except (OSError, json.JSONDecodeError):
        return None


def read_comparison_history():
    try:
        with MONITORING_HISTORY_PATH.open(encoding="utf-8") as history_file:
            history = json.load(history_file)
            return history.get("runs", []) if history.get("dataset") == DATASET_NAME else []
    except (OSError, json.JSONDecodeError):
        return []


def run_prediction(features: dict[str, Any], *, actual_points: float | None = None, player: dict | None = None):
    for name in FEATURE_NAMES:
        if name != "position" and not math.isfinite(float(features[name])):
            raise HTTPException(status_code=422, detail=f"{name} must be a finite number")
    start = time.perf_counter()
    try:
        prediction = float(production_model.predict(pd.DataFrame([features], columns=FEATURE_NAMES))[0])
    except Exception as error:
        metrics["errors"] += 1
        metrics["failed_predictions"] += 1
        return JSONResponse(status_code=500, content={"error": "Prediction failed", "detail": str(error)})
    latency = round((time.perf_counter() - start) * 1000, 2)
    metrics["total_requests"] += 1
    previous_total = metrics["average_latency_ms"] * (metrics["total_requests"] - 1)
    metrics["average_latency_ms"] = round((previous_total + latency) / metrics["total_requests"], 2)
    metrics["recent_predictions"].append(round(prediction, 2))
    del metrics["recent_predictions"][:-20]
    result = {"predicted_total_points": round(prediction, 2), "latency_ms": latency}
    if actual_points is not None and player is not None:
        result.update({
            "player_id": player["id"],
            "player_name": player["name"],
            "actual_total_points": round(actual_points, 2),
            "absolute_error": round(abs(actual_points - prediction), 2),
        })
    return result


@app.post("/predict")
def predict(request: PredictionRequest):
    return run_prediction(request.model_dump())


@app.post("/api/demo-prediction")
def demo_prediction():
    """Score one held-out player whose actual points are stored in the fixed CSV."""
    global demo_index
    index = demo_index % len(TEST_FEATURES)
    features = TEST_FEATURES.iloc[index].to_dict()
    demo_index += 1
    return run_prediction(features, actual_points=TEST_POINTS[index], player=TEST_PLAYERS[index])


@app.get("/api/model")
def get_model():
    return {
        "name": "FPL Total Points Regressor",
        "dataset": DATASET_NAME,
        "features": FEATURE_NAMES,
        "target": TARGET_NAME,
    }


@app.get("/api/metrics")
def get_metrics():
    report = read_deployment_report()
    candidate = report.get("candidate", {}) if report else {}
    active_result = candidate if report and candidate.get("version") == ACTIVE_MODEL_VERSION else (report or {}).get("previous_production", {})
    quality = active_result.get("metrics", {}) or INITIAL_TEST_METRICS
    return {
        "model": {
            "name": "FPL Total Points Regressor",
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
                "mae": quality.get("mae"),
                "rmse": quality.get("rmse"),
                "r2_score": quality.get("r2_score"),
                "evaluated_inputs": quality.get("evaluated_inputs", 0),
            },
        },
        "deployment": report,
        "comparison_history": read_comparison_history(),
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
