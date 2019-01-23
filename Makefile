test:
	pytest --flake8 --mypy # --pylint --cov=algorithms

format:
	black .
	isort -y

dev:
	pip install -r requirements-dev.txt
	pre-commit install

dep:
	pip install -r requirements.txt
