.PHONY: test coverage typecheck dashboard-test format format-check audit build check

PYTHON ?= python

test:
	$(PYTHON) -m pytest -q

coverage:
	$(PYTHON) -m pytest --cov=src --cov=main --cov-report=term-missing:skip-covered --cov-fail-under=40 -q

typecheck:
	$(PYTHON) -m mypy --config-file mypy.ini --follow-imports=silent src/ai/agent.py src/ai/network.py src/utils src/app/training_runtime.py src/web/model_service.py src/web/game_stats_service.py

dashboard-test:
	node --test tests/js/*.test.mjs

format:
	$(PYTHON) -m black main.py config.py src tests

format-check:
	$(PYTHON) -m black --check main.py config.py src tests

audit:
	$(PYTHON) -m pip_audit -r requirements.txt

build:
	$(PYTHON) -m build

check: format-check typecheck dashboard-test coverage
