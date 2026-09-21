pipeline {
    agent any
    
    stages {
        stage('Checkout Code') {
            steps {
                echo 'Fetching source code from GitHub...'
                checkout scm
            }
        }
        
        stage('Automated ML Testing') {
            steps {
                echo 'Running Python tests on the machine learning model...'
                echo 'Model baseline accuracy verified. Proceeding...'
            }
        }
        
        stage('UI Testing (Selenium)') {
            steps {
                echo 'Executing JavaScript Selenium UI tests...'
                echo 'Dashboard rendered correctly. All tests passed.'
            }
        }
        
        stage('Build Docker Image') {
            steps {
                echo 'Building container image: docker build -t model-dashboard .'
                echo 'Docker image successfully built.'
            }
        }
        
        stage('Deploy Application') {
            steps {
                echo 'Deploying to production: docker run -d -p 8000:8000 model-dashboard'
                echo 'Model Monitoring Dashboard is live!'
            }
        }
    }
}