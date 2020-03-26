test:
	black . --check
	isort -y --check-only --skip checkpoint --skip wandb
	env PYTHONPATH=. pytest --pylint --flake8 --mypy --ignore=checkpoint --ignore=wandb

format:
	black . --exclude checkpoint wandb
	isort -y --skip checkpoint --skip wandb

docker-push:
	docker build -t medipixel/rl_algorithms .
	docker push medipixel/rl_algorithms

dev:
	pip install -r requirements.txt
	pip install -r requirements-dev.txt
	pre-commit install
	python setup.py develop

dep:
	pip install -r -U requirements.txt
	python setup.py install
