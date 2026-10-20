@echo off
setlocal

if not defined PYTHON_EXE (
    echo ERROR: Set PYTHON_EXE to the full path of python.exe in the Jenkins job.
    exit /b 1
)

if not defined DOCKER_EXE (
    echo ERROR: Set DOCKER_EXE to the full path of docker.exe in the Jenkins job.
    exit /b 1
)

echo Creating project virtual environment...
"%PYTHON_EXE%" -m venv .venv
if errorlevel 1 exit /b 1

echo Installing application dependencies...
.venv\Scripts\python.exe -m pip install -r requirements.txt
if errorlevel 1 exit /b 1

echo Checking Python syntax...
.venv\Scripts\python.exe -m py_compile app\main.py app\train.py
if errorlevel 1 exit /b 1

echo Training the bundled models...
.venv\Scripts\python.exe app\train.py
if errorlevel 1 exit /b 1

echo Checking Docker access for the Jenkins service account...
"%DOCKER_EXE%" version
if errorlevel 1 exit /b 1

set "IMAGE_TAG=%BUILD_NUMBER%"
if not defined IMAGE_TAG set "IMAGE_TAG=local"

echo Building model-dashboard:%IMAGE_TAG%...
"%DOCKER_EXE%" build -t model-dashboard:%IMAGE_TAG% .
if errorlevel 1 exit /b 1

echo Tagging model-dashboard:latest...
"%DOCKER_EXE%" tag model-dashboard:%IMAGE_TAG% model-dashboard:latest
if errorlevel 1 exit /b 1

if not defined DOCKERHUB_REPOSITORY goto build_complete
if not defined DOCKERHUB_USER (
    echo ERROR: Set DOCKERHUB_USER using Jenkins Credentials Binding.
    exit /b 1
)
if not defined DOCKERHUB_TOKEN (
    echo ERROR: Set DOCKERHUB_TOKEN using Jenkins Credentials Binding.
    exit /b 1
)

echo Logging in to Docker Hub...
powershell.exe -NoProfile -ExecutionPolicy Bypass -Command "$env:DOCKERHUB_TOKEN | & $env:DOCKER_EXE login --username $env:DOCKERHUB_USER --password-stdin"
if errorlevel 1 exit /b 1

echo Tagging Docker Hub image...
"%DOCKER_EXE%" tag model-dashboard:%IMAGE_TAG% "%DOCKERHUB_REPOSITORY%:%IMAGE_TAG%"
if errorlevel 1 exit /b 1
"%DOCKER_EXE%" tag model-dashboard:%IMAGE_TAG% "%DOCKERHUB_REPOSITORY%:latest"
if errorlevel 1 exit /b 1

echo Pushing versioned and latest image tags...
"%DOCKER_EXE%" push "%DOCKERHUB_REPOSITORY%:%IMAGE_TAG%"
if errorlevel 1 exit /b 1
"%DOCKER_EXE%" push "%DOCKERHUB_REPOSITORY%:latest"
if errorlevel 1 exit /b 1
"%DOCKER_EXE%" logout

:build_complete

echo Build completed successfully.
exit /b 0
