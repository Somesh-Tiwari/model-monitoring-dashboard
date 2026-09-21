pipeline {
    agent any
    
    stages {
        stage('Checkout Code') {
            steps {
                echo 'Fetching source code from GitHub...'
                checkout scm
            }
        }
        
        stage('Install and Validate Models') {
            steps {
                sh 'python -m pip install --upgrade pip'
                sh 'pip install -r requirements.txt'
                sh 'python -m py_compile app/main.py app/train.py'
                sh 'python app/train.py'
            }
        }
        
        stage('UI Testing (Selenium)') {
            steps {
                sh '''
                    python -m uvicorn app.main:app --host 127.0.0.1 --port 8000 > uvicorn.log 2>&1 &
                    server_pid=$!
                    trap "kill $server_pid" EXIT
                    sleep 5
                    npm --prefix tests install
                    npm --prefix tests test
                '''
            }
        }
        
        stage('Build Docker Image') {
            steps {
                sh 'docker build -t model-dashboard .'
            }
        }
        
        stage('Deploy Application') {
            steps {
                sh 'docker run -d --rm -p 8000:8000 model-dashboard'
            }
        }
    }
}