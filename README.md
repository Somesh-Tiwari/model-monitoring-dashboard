Model Monitoring Dashboard
AI DevOps pipeline project featuring FastAPI, Docker, Jenkins, and automated UI testing via Selenium.

## Monitoring signals

The dashboard reports live inference quality and system health through `/api/metrics`.
The dashboard monitors three bundled scikit-learn datasets: Wine, Breast Cancer, and Digits. Select Model 1, Model 2, or Model 3 in the dashboard to view isolated traffic and quality metrics for that model.

Quality metrics are calculated from prediction requests that include an `actual_class` value alongside the selected model's feature vector. Requests without ground truth still contribute to traffic, latency, and prediction-stream metrics, but are excluded from accuracy, precision, recall, and F1-score.

The API also counts failed predictions, invalid outputs, and missing request values, and reports CPU usage, memory usage, API uptime, and average response latency.