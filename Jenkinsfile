pipeline {
  agent {
    docker {
      image 'medipixel/rl_algorithms'
    }

  }
  stages {
    stage('Test') {
      steps {
        echo 'Testing...'
        sh '''make dev
make test'''
      }
    }
  }
}