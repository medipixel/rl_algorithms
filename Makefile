test:
	black . --check --exclude checkpoint wandb
	isort -y --check-only --skip checkpoint --skip wandb
	env PYTHONPATH=. pytest --pylint --flake8 --mypy --ignore=checkpoint --ignore=wandb

format:
	black . --exclude checkpoint wandb
	isort -y --skip checkpoint --skip wandb

dev:
	pip install -r requirements-dev.txt
	pre-commit install

dep:
	pip install -r requirements.txt
