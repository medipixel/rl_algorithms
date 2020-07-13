test:
	black . --check
	isort -y --check-only --skip checkpoint --skip wandb
	env PYTHONPATH=. pytest --pylint --flake8 --cov=tests --ignore=checkpoint --ignore=data --ignore=wandb --ignore tests/integration

integration-test:
	env PYTHONPATH=. pytest tests/integration --cov=tests

format:
	black . --exclude checkpoint wandb
	isort -y --skip checkpoint --skip wandb

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
