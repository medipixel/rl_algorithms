test:
	black . --check --exclude checkpoint
	isort -y --check-only --skip checkpoint
	env PYTHONPATH=. pytest --pylint --flake8 --mypy --ignore=checkpoint

format:
	black .
	isort -y

dev:
	pip install -r requirements-dev.txt
	pre-commit install

dep:
	pip install -r requirements.txt
