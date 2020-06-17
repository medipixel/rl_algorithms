pipeline {
    agent { 
        dockerfile{
            filename "Dockerfile"
            args "-v /home/mpadmin/.ssh/:/root/.ssh/"
        }
    }
  stages {
    stage('Test') {
      steps {
        echo 'Testing...'
        sh 'make dev'
        sh 'make test'
        sh 'make integration-test'
      }
    }
  }
}