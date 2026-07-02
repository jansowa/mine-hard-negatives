.PHONY: test lint typecheck check format compile-requirements compile-requirements-lancedb compile-runtime-requirements

VENV_BIN ?= .venv/bin
PYTEST ?= $(if $(wildcard $(VENV_BIN)/pytest),$(VENV_BIN)/pytest,pytest)
RUFF ?= $(if $(wildcard $(VENV_BIN)/ruff),$(VENV_BIN)/ruff,ruff)
MYPY ?= $(if $(wildcard $(VENV_BIN)/mypy),$(VENV_BIN)/mypy,mypy)
UV ?= uv
PYTHON_VERSION ?= 3.12.3
TORCH_BACKEND ?= cu128

test:
	$(PYTEST)

lint:
	$(RUFF) check app_code tests

typecheck:
	$(MYPY)

check: lint typecheck test

format:
	$(RUFF) format app_code tests

compile-requirements-lancedb:
	$(UV) pip compile requirements-lancedb.in -o requirements-lancedb.txt \
		--python-version $(PYTHON_VERSION) \
		--torch-backend $(TORCH_BACKEND) \
		--build-constraint build-constraints.txt \
		--emit-index-url \
		--emit-find-links

compile-requirements:
	$(UV) pip compile requirements.in -o requirements.txt \
		--python-version $(PYTHON_VERSION) \
		--torch-backend $(TORCH_BACKEND) \
		--build-constraint build-constraints.txt \
		--emit-index-url \
		--emit-find-links

compile-runtime-requirements: compile-requirements-lancedb compile-requirements
