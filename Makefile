format:
	black . --exclude checkpoint wandb
	isort . --skip checkpoint --skip wandb --skip data

test:
	black . --check
	isort . --check --diff --skip checkpoint --skip wandb --skip data
	env PYTHONPATH=. pytest --pylint --flake8 --cov=tests --ignore=checkpoint --ignore=data --ignore=wandb --ignore tests/integration --ignore example

integration-test:
	env PYTHONPATH=. pytest tests/integration --cov=tests

docker-push:
	docker build -t medipixel/rl_algorithms .
	docker push medipixel/rl_algorithms

dev:
	pip install -U -r requirements.txt
	pip install -U -r requirements-dev.txt
	pre-commit install
	python setup.py develop

dep:
	pip install -U -r requirements.txt
	python setup.py install

jenkins-dev:
	pip install -U -r requirements-dev.txt
	python setup.py develop