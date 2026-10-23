@echo off
setlocal

if not defined PYTHON_EXE (
    echo ERROR: Set PYTHON_EXE to the full path of python.exe.
    exit /b 1
)
if not defined DOCKER_EXE (
    echo ERROR: Set DOCKER_EXE to the full path of docker.exe.
    exit /b 1
)
if not defined PRODUCTION_MODEL_PATH if defined JENKINS_HOME set "PRODUCTION_MODEL_PATH=%JENKINS_HOME%\model-dashboard\production.joblib"
if not defined PRODUCTION_MODEL_PATH set "PRODUCTION_MODEL_PATH=%CD%\jenkins-data\production.joblib"

echo Creating Jenkins workspace virtual environment...
"%PYTHON_EXE%" -m venv .venv
if errorlevel 1 exit /b 1

echo Installing application dependencies...
.venv\Scripts\python.exe -m pip install --upgrade pip
if errorlevel 1 exit /b 1
.venv\Scripts\python.exe -m pip install -r requirements.txt
if errorlevel 1 exit /b 1

echo Checking Python syntax...
.venv\Scripts\python.exe -m py_compile app\main.py app\modeling.py app\retrain.py app\train.py app\promote.py app\mark_deployed.py
if errorlevel 1 exit /b 1

echo Retraining and comparing the production and candidate models on four fixed Wine test inputs...
.venv\Scripts\python.exe -m app.retrain
if errorlevel 1 exit /b 1

echo Saving comparison report to the dashboard monitoring volume...
"%DOCKER_EXE%" volume create model-dashboard-monitoring >nul
if errorlevel 1 exit /b 1
powershell.exe -NoProfile -ExecutionPolicy Bypass -Command "& $env:DOCKER_EXE run --rm --mount 'type=volume,source=model-dashboard-monitoring,target=/data' --mount ('type=bind,source=' + (Resolve-Path 'app\artifacts\deployment_report.json').Path + ',target=/tmp/deployment_report.json,readonly') alpine:3.20 cp /tmp/deployment_report.json /data/deployment_report.json"
if errorlevel 1 exit /b 1

.venv\Scripts\python.exe -c "import json; print(str(json.load(open(r'app\artifacts\deployment_report.json', encoding='utf-8'))['deployment_approved']).lower())" > deployment-approved.txt
if errorlevel 1 exit /b 1
set /p DEPLOY_APPROVED=<deployment-approved.txt
del /q deployment-approved.txt
if /I not "%DEPLOY_APPROVED%"=="true" goto deployment_skipped

set "IMAGE_TAG=%BUILD_NUMBER%"
if not defined IMAGE_TAG set "IMAGE_TAG=local"

echo Candidate passed the accuracy gate. Building Docker image model-dashboard:%IMAGE_TAG%...
"%DOCKER_EXE%" build -t model-dashboard:%IMAGE_TAG% .
if errorlevel 1 exit /b 1
"%DOCKER_EXE%" tag model-dashboard:%IMAGE_TAG% model-dashboard:latest
if errorlevel 1 exit /b 1

if not defined DOCKERHUB_REPOSITORY goto deploy_local
if not defined DOCKERHUB_USER (
    echo ERROR: Set DOCKERHUB_USER using Jenkins Credentials Binding.
    exit /b 1
)
if not defined DOCKERHUB_TOKEN (
    echo ERROR: Set DOCKERHUB_TOKEN using Jenkins Credentials Binding.
    exit /b 1
)
echo Publishing approved model image to Docker Hub...
powershell.exe -NoProfile -ExecutionPolicy Bypass -Command "$env:DOCKERHUB_TOKEN | & $env:DOCKER_EXE login --username $env:DOCKERHUB_USER --password-stdin"
if errorlevel 1 exit /b 1
"%DOCKER_EXE%" tag model-dashboard:%IMAGE_TAG% "%DOCKERHUB_REPOSITORY%:%IMAGE_TAG%"
if errorlevel 1 exit /b 1
"%DOCKER_EXE%" tag model-dashboard:%IMAGE_TAG% "%DOCKERHUB_REPOSITORY%:latest"
if errorlevel 1 exit /b 1
"%DOCKER_EXE%" push "%DOCKERHUB_REPOSITORY%:%IMAGE_TAG%"
if errorlevel 1 exit /b 1
"%DOCKER_EXE%" push "%DOCKERHUB_REPOSITORY%:latest"
if errorlevel 1 exit /b 1
"%DOCKER_EXE%" logout

:deploy_local
echo Replacing the local dashboard container with the approved image...
for /f %%I in ('"%DOCKER_EXE%" ps -q --filter "publish=8000"') do "%DOCKER_EXE%" stop %%I
"%DOCKER_EXE%" stop model-dashboard-app 2>nul
"%DOCKER_EXE%" rm model-dashboard-app 2>nul
"%DOCKER_EXE%" run -d --name model-dashboard-app --restart unless-stopped -p 8000:8000 --mount source=model-dashboard-monitoring,target=/data -e MONITORING_DATA_DIR=/data model-dashboard:%IMAGE_TAG%
if errorlevel 1 exit /b 1
.venv\Scripts\python.exe -m app.mark_deployed
if errorlevel 1 exit /b 1
powershell.exe -NoProfile -ExecutionPolicy Bypass -Command "& $env:DOCKER_EXE run --rm --mount 'type=volume,source=model-dashboard-monitoring,target=/data' --mount ('type=bind,source=' + (Resolve-Path 'app\artifacts\deployment_report.json').Path + ',target=/tmp/deployment_report.json,readonly') alpine:3.20 cp /tmp/deployment_report.json /data/deployment_report.json"
if errorlevel 1 exit /b 1
.venv\Scripts\python.exe -m app.promote
if errorlevel 1 exit /b 1
echo Deployment approved and available at http://localhost:8000
goto build_complete

:deployment_skipped
echo Candidate did not meet production accuracy. Production model and running container were retained.
echo Jenkins comparison report has been saved to the dashboard monitoring volume.

:build_complete
echo Retraining workflow completed.
exit /b 0
