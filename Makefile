.PHONY: test coverage typecheck format format-check audit check

PYTHON ?= python

test:
	$(PYTHON) -m pytest -q

coverage:
	$(PYTHON) -m pytest --cov=src --cov=main --cov-report=term-missing:skip-covered --cov-fail-under=40 -q

typecheck:
	$(PYTHON) -m mypy --config-file mypy.ini src/ai src/utils

format:
	$(PYTHON) -m black main.py config.py src tests

format-check:
	$(PYTHON) -m black --check main.py config.py src tests

audit:
	$(PYTHON) -m pip_audit -r requirements.txt

check: format-check typecheck coverage
