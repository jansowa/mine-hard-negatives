.PHONY: test lint typecheck check format

test:
	pytest

lint:
	ruff check app_code tests

typecheck:
	mypy

check: lint typecheck test

format:
	ruff format app_code tests
