pipeline {
    agent any

    options {
        timestamps()
        skipDefaultCheckout(true)
    }

    parameters {
        booleanParam(name: 'PUBLISH_TO_DOCKER_HUB', defaultValue: false, description: 'Push the built image to Docker Hub')
        string(name: 'DOCKER_HUB_REPOSITORY', defaultValue: '', description: 'Docker Hub repository, for example username/model-dashboard. Required when publishing.')
    }

    environment {
        LOCAL_IMAGE = 'model-dashboard'
        DOCKER_HUB_CREDENTIALS = 'dockerhub-credentials'
    }

    stages {
        stage('Checkout') {
            steps { checkout scm }
        }

        stage('Install and validate') {
            steps {
                sh 'python3 -m venv .venv'
                sh '.venv/bin/python -m pip install --upgrade pip'
                sh '.venv/bin/python -m pip install -r requirements.txt'
                sh '.venv/bin/python -m py_compile app/main.py app/train.py'
                sh '.venv/bin/python app/train.py'
            }
        }

        stage('Dashboard UI checks') {
            steps {
                sh '''
                    .venv/bin/python -m uvicorn app.main:app --host 127.0.0.1 --port 8000 > uvicorn.log 2>&1 &
                    server_pid=$!
                    trap "kill $server_pid 2>/dev/null || true" EXIT
                    for attempt in $(seq 1 30); do
                        if curl --silent --fail http://127.0.0.1:8000/api/models >/dev/null; then break; fi
                        sleep 1
                    done
                    npm --prefix tests ci
                    npm --prefix tests test
                '''
            }
        }

        stage('Build container') {
            steps {
                sh 'docker build -t "$LOCAL_IMAGE:$BUILD_NUMBER" -t "$LOCAL_IMAGE:latest" .'
            }
        }

        stage('Publish to Docker Hub') {
            when { expression { return params.PUBLISH_TO_DOCKER_HUB } }
            steps {
                script {
                    if (!params.DOCKER_HUB_REPOSITORY?.trim()) {
                        error('Set DOCKER_HUB_REPOSITORY before enabling PUBLISH_TO_DOCKER_HUB.')
                    }
                }
                withCredentials([usernamePassword(credentialsId: env.DOCKER_HUB_CREDENTIALS, usernameVariable: 'DOCKERHUB_USER', passwordVariable: 'DOCKERHUB_TOKEN')]) {
                    sh '''
                        set +x
                        echo "$DOCKERHUB_TOKEN" | docker login --username "$DOCKERHUB_USER" --password-stdin
                        docker tag "$LOCAL_IMAGE:$BUILD_NUMBER" "$DOCKER_HUB_REPOSITORY:$BUILD_NUMBER"
                        docker tag "$LOCAL_IMAGE:$BUILD_NUMBER" "$DOCKER_HUB_REPOSITORY:latest"
                        docker push "$DOCKER_HUB_REPOSITORY:$BUILD_NUMBER"
                        docker push "$DOCKER_HUB_REPOSITORY:latest"
                        docker logout
                    '''
                }
            }
        }
    }

    post {
        always {
            sh 'docker logout >/dev/null 2>&1 || true'
            archiveArtifacts artifacts: 'uvicorn.log', allowEmptyArchive: true
        }
        success { echo 'Dashboard pipeline completed successfully.' }
        failure { echo 'Pipeline failed. Review the stage log and archived server log.' }
    }
}
