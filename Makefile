PYTHON ?= python3
PORT ?= 5000
BLACK ?= black
RUFF ?= ruff
MYPY ?= mypy

.PHONY: setup compile test test-fast coverage typecheck lint format format-check run run-headless run-web smoke-train clean

setup:
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -e ".[web,test,dev]"

compile:
	$(PYTHON) -m compileall -q src main.py config.py tests

test:
	$(PYTHON) -m pytest

test-fast:
	$(PYTHON) -m pytest -m "not slow"

coverage:
	$(PYTHON) -m pytest --cov=src --cov-report=term-missing

typecheck:
	$(MYPY) --config-file=mypy.ini src main.py config.py

lint:
	$(RUFF) check .

format:
	$(BLACK) src tests main.py config.py benchmark.py evaluate_checkpoints.py experiment_runner.py
	$(RUFF) check --fix .

format-check:
	$(BLACK) --check src tests main.py config.py benchmark.py evaluate_checkpoints.py experiment_runner.py
	$(RUFF) check .

run:
	$(PYTHON) main.py

run-headless:
	$(PYTHON) main.py --headless --cpu

run-web:
	$(PYTHON) main.py --headless --web --port $(PORT)

smoke-train:
	$(PYTHON) main.py --headless --cpu --episodes 1

clean:
	find . -type d -name '__pycache__' -prune -exec rm -rf {} +
	rm -rf .pytest_cache .mypy_cache .ruff_cache htmlcov .coverage
