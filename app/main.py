# app/main.py
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import joblib
import time
import os
import random

app = FastAPI(title="Model Monitoring API")

# Load model
model_path = os.path.join(os.path.dirname(__file__), "model.joblib")
model = joblib.load(model_path)

# Telemetry data store
metrics = {
    "total_requests": 0,
    "average_latency_ms": 0.0,
    "recent_predictions": []
}

class IrisFeatures(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

@app.post("/predict")
def predict(features: IrisFeatures):
    start_time = time.time()
    
    data = [[features.sepal_length, features.sepal_width, features.petal_length, features.petal_width]]
    prediction = int(model.predict(data)[0])
    
    latency = (time.time() - start_time) * 1000 + random.uniform(10, 50)
    
    metrics["total_requests"] += 1
    total_latency = (metrics["average_latency_ms"] * (metrics["total_requests"] - 1)) + latency
    metrics["average_latency_ms"] = round(total_latency / metrics["total_requests"], 2)
    metrics["recent_predictions"].append(prediction)
    
    if len(metrics["recent_predictions"]) > 20:
        metrics["recent_predictions"].pop(0)
        
    return {"prediction": prediction, "latency_ms": round(latency, 2)}

@app.get("/api/metrics")
def get_metrics():
    return metrics

# Mount static folder
static_dir = os.path.join(os.path.dirname(__file__), "static")
app.mount("/static", StaticFiles(directory=static_dir), name="static")

@app.get("/")
def read_root():
    return FileResponse(os.path.join(static_dir, "index.html"))