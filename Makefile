test:
	python setup.py install
	black . --check
	isort -y --check-only --skip checkpoint --skip wandb
	env PYTHONPATH=. pytest --pylint --flake8 --mypy --ignore=checkpoint --ignore=wandb --cov=tests

format:
	black . --exclude checkpoint wandb
	isort -y --skip checkpoint --skip wandb

dev:
	pip install -r requirements.txt
	pip install -r requirements-dev.txt
	pre-commit install
	python setup.py develop

dep:
	pip install -r requirements.txt
	python setup.py install
