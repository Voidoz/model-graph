UV ?= uv
PYTEST ?= $(UV) run pytest
RUFF ?= $(UV) run ruff
MYPY ?= $(UV) run mypy

.PHONY: sync format lint type test live-test check clean

sync:
	$(UV) sync

format:
	$(RUFF) format src tests
	$(RUFF) check --fix src tests

lint:
	$(RUFF) format --check src tests
	$(RUFF) check src tests

type:
	$(MYPY) src

test:
	$(PYTEST) -q

live-test:
	MG_TEST_OLLAMA=1 $(PYTEST) tests/live -q

check: lint type test

clean:
	rm -rf .pytest_cache .ruff_cache .mypy_cache htmlcov .coverage dist build
	find . -name '__pycache__' -type d -prune -exec rm -rf {} +
