test:
	pytest --flake8 # --mypy --pylint --cov=algorithms

format:
	isort -y

dev:
	pip install -r requirements-dev.txt

dep:
	pip install -r requirements.txt
