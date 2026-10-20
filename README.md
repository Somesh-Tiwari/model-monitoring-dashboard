# Model Monitoring Dashboard
AI DevOps pipeline project featuring FastAPI, Docker, Jenkins, and automated UI testing via Selenium.

## Monitoring signals

The dashboard reports live inference quality and system health through `/api/metrics`.
The dashboard monitors three bundled scikit-learn datasets: Wine, Breast Cancer, and Digits. Select Model 1, Model 2, or Model 3 in the dashboard to view isolated traffic and quality metrics for that model.

Quality metrics are calculated from prediction requests that include an `actual_class` value alongside the selected model's feature vector. Requests without ground truth still contribute to traffic, latency, and prediction-stream metrics, but are excluded from accuracy, precision, recall, and F1-score.

The API also counts failed predictions, invalid outputs, and missing request values, and reports CPU usage, memory usage, API uptime, and average response latency.

## Add your own dataset

Use **Add a dataset** in the dashboard or send a `multipart/form-data` request to `POST /api/models/upload` with `file` (CSV) and `target_column` fields. The CSV must have numeric feature columns, at least four complete rows, and a target with at least two classes. Files are limited to 10 MB. The endpoint trains a Random Forest in memory and registers it for inference and monitoring until the service restarts.

```bash
curl -X POST http://localhost:8000/api/models/upload \
  -F "file=@dataset.csv" \
  -F "target_column=species"
```

New models appear in `GET /api/models` and can be used with the existing `POST /predict` endpoint. Interactive API documentation is available at `/docs`; the OpenAPI schema is at `/openapi.json`.

## Jenkins and Docker Hub

The Jenkins pipeline checks out the project, installs and validates the app, runs the existing Selenium UI check, and builds versioned and `latest` local Docker tags. To publish, add a Jenkins username/password credential with ID `dockerhub-credentials` (use a Docker Hub access token as the password), then enable `PUBLISH_TO_DOCKER_HUB` and set `DOCKER_HUB_REPOSITORY` to `username/repository` when starting a build. Credentials stay in Jenkins and are not stored in this repository. The pipeline expects a Linux Jenkins agent with Python, Node.js/npm, Docker, curl, and Chrome/Chromium available.
