# Fantasy Player Points Monitoring Dashboard

A single-model FastAPI dashboard and Windows Jenkins workflow for estimating Fantasy Premier League players' current-season total points from a fixed 2026–2027 player snapshot.

## Dataset and model

- The fixed source is `app/data/fpl_2026_2027.csv`, copied from the CSV supplied for this project. The file contains one row per player and a `total_points` value to use as the known outcome.
- The regressor estimates `total_points` from `position`, `now_cost`, `minutes`, `starts`, `expected_goal_involvements`, `expected_goals`, and `expected_assists`. Cost uses the dataset's FPL scale (tenths of £1m).
- It excludes `total_points` and direct point-scoring outcomes such as goals, assists, bonus, and clean sheets from the inputs.
- The app creates deterministic, position-stratified training (70%), validation (15%), and test (15%) splits by player. Jenkins uses the full validation split for the deployment gate; test metrics are reported separately.
- Regression monitoring reports MAE, RMSE, and R² on the full test split. Jenkins promotes only when candidate validation MAE is no higher than production validation MAE.
- This is a retrospective evaluation of one season snapshot. It does not measure week-ahead forecasting. That requires dated snapshots and later-week points as future labels.

## Run locally from source

In PowerShell:

```powershell
cd D:\model-monitoring-dashboard
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python -m app.train
python -m uvicorn app.main:app --reload --port 8000
```

Open <http://localhost:8000>. Enter player snapshot features to get an estimate without entering actual points. **Evaluate held-out player** compares one player in the fixed test split with that row's stored `total_points`. API documentation is at <http://localhost:8000/docs>.

## API

- `POST /predict` accepts position, cost, minutes, starts, expected goal involvements, expected goals, and expected assists. It does not accept actual points.
- `POST /api/demo-prediction` evaluates the next held-out player and returns prediction, actual total points, and absolute error.
- `GET /api/model` describes the fixed dataset, inputs, and target.
- `GET /api/metrics` returns inference health and the latest compatible Jenkins comparison, if one exists.

## Docker

```powershell
docker build -t model-dashboard:local .
docker volume create model-dashboard-monitoring
docker run --rm -p 8000:8000 --mount source=model-dashboard-monitoring,target=/data -e MONITORING_DATA_DIR=/data model-dashboard:local
```

Open <http://localhost:8000>. The named volume preserves Jenkins comparison reports when containers are replaced.

## Jenkins UI on Windows

Configure a Freestyle project in Jenkins UI with the Git repository and branch. Enable **Poll SCM** with `H/5 * * * *` if Jenkins should check for commits about every five minutes. Add an **Execute Windows batch command** build step:

```bat
set "PYTHON_EXE=C:\Users\Lenovo\AppData\Local\Programs\Python\Python313\python.exe"
set "DOCKER_EXE=C:\Users\Lenovo\AppData\Local\Programs\DockerDesktop\resources\bin\docker.exe"
call jenkins-build.bat
```

Change the executable paths to match your computer. The batch script installs dependencies, retrains and evaluates on the full fixed validation and test splits, and builds/runs the Docker image only when candidate validation MAE passes the production gate. Docker Hub publishing is optional and requires Jenkins credentials and repository settings.

Jenkins stores its approved production model outside its disposable workspace at `%JENKINS_HOME%\model-dashboard\production.joblib`. Keep that file and the `model-dashboard-monitoring` Docker volume to retain model and report history. Artifacts from the former Wine and student-grade models are treated as incompatible with this dataset.
