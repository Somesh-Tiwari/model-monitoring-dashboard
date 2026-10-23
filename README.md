# Model Monitoring Dashboard

A single-model machine-learning demo with a FastAPI dashboard, fixed evaluation data, and a Windows Jenkins retraining/deployment pipeline.

## Scope and workflow

- The application serves one model: a Random Forest classifier trained on scikit-learn's fixed Wine dataset.
- The dashboard has no custom dataset upload and accepts no target or ground-truth values from prediction users.
- Runtime inference is feature-only. The dashboard's demo button asks the server to run the next predefined Wine test input; its label remains server-side.
- Retraining uses the fixed training portion of the dataset. Jenkins compares the current production model and candidate on the same four rows from a deterministic held-out split.
- Jenkins promotes and deploys the candidate only when its accuracy is greater than or equal to production accuracy. If it is lower, production stays in place.
- Accuracy, precision, recall, F1, four per-input outcomes, and the deployment decision are written to the Docker volume `model-dashboard-monitoring`. The dashboard reads that report at `/api/metrics` and displays it.

## Run locally from source

In PowerShell:

```powershell
cd D:\model-monitoring-dashboard
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m app.train
python -m uvicorn app.main:app --reload --port 8000
```

Open <http://localhost:8000>. Stop the server with **Ctrl+C**. The local baseline model is created at `app/artifacts/production.joblib`; this generated file is ignored by Git.

## API

- `POST /predict` takes only `{"features": [13 numeric Wine feature values]}`. It does not accept a target or actual class.
- `POST /api/demo-prediction` runs the next server-selected fixed test input without returning its expected class.
- `GET /api/model` returns the fixed model and feature names.
- `GET /api/metrics` returns live inference/system metrics and the latest Jenkins evaluation/deployment report.
- Interactive API docs: <http://localhost:8000/docs>.

## Docker

To build and run from source:

```powershell
docker build -t model-dashboard:local .
docker volume create model-dashboard-monitoring
docker run --rm -p 8000:8000 --mount source=model-dashboard-monitoring,target=/data -e MONITORING_DATA_DIR=/data model-dashboard:local
```

The named volume retains Jenkins comparison reports when containers are replaced. To run the previously published Docker Hub image, use `deeppatil069/model-monitoring-dashboard:latest`; it must be rebuilt and republished to include these newer code changes.

## Jenkins on Windows

Use a **Pipeline** job configured as **Pipeline script from SCM**, with this repository and branch. `Jenkinsfile` polls SCM every five minutes, checks out commits, and runs `jenkins-build.bat`. The paths near the top of the Jenkinsfile must match the Python and Docker executable locations on the Windows Jenkins machine.

The build script:

1. Creates a workspace virtual environment and installs requirements.
2. Checks Python syntax and retrains a candidate on the fixed Wine training split.
3. Evaluates production and candidate on the same four fixed test inputs, records the comparison, and updates the monitoring Docker volume.
4. Stops before image build/deployment if candidate accuracy is lower.
5. If approved, builds the Docker image, replaces the named local dashboard container on port 8000, and optionally publishes to Docker Hub.

The previous approved model is persisted under `%JENKINS_HOME%\model-dashboard\production.joblib`, outside the disposable Jenkins workspace. Keep that file and the `model-dashboard-monitoring` Docker volume to preserve production comparison state and dashboard reports between builds. The first run creates a seed-42 baseline and compares a seed-42 candidate.

To enable Docker Hub publishing, create a Jenkins username/password credential with ID `dockerhub-credentials` (Docker Hub username and an access token as the password). Start a build with `PUBLISH_TO_DOCKER_HUB` enabled and set `DOCKERHUB_REPOSITORY` to your namespace/repository. Publishing happens only after the accuracy gate approves.
