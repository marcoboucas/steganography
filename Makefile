SOURCES = src

lint:
	python -m black $(SOURCES)
	python -m isort $(SOURCES)

	python -m pylint $(SOURCES)
	python -m flake8 $(SOURCES)

install:
	pip install -r requirements.txt

install-dev: install
	pip install -r requirements-dev.txt
	pip install pre-commit
	pre-commit install

test:
	pytest .