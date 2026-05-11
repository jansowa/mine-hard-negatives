.PHONY: test lint typecheck check format

VENV_BIN ?= .venv/bin
PYTEST ?= $(if $(wildcard $(VENV_BIN)/pytest),$(VENV_BIN)/pytest,pytest)
RUFF ?= $(if $(wildcard $(VENV_BIN)/ruff),$(VENV_BIN)/ruff,ruff)
MYPY ?= $(if $(wildcard $(VENV_BIN)/mypy),$(VENV_BIN)/mypy,mypy)

test:
	$(PYTEST)

lint:
	$(RUFF) check app_code tests

typecheck:
	$(MYPY)

check: lint typecheck test

format:
	$(RUFF) format app_code tests
