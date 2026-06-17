.PHONY: setup test coverage typecheck typecheck-audit dashboard-test dashboard-smoke perf-smoke format format-check lint audit size-check build build-if-available release-config hygiene check verify

PYTHON ?= python

setup:
	$(PYTHON) scripts/bootstrap_dev.py

test:
	$(PYTHON) -m pytest -q

coverage:
	$(PYTHON) -m pytest --cov=src --cov=main --cov-report=term-missing:skip-covered --cov-fail-under=75 -q

typecheck:
	$(PYTHON) -m mypy --config-file mypy.ini --follow-imports=silent src main.py config.py

typecheck-audit:
	$(MAKE) typecheck

dashboard-test:
	node --test tests/js/*.test.mjs

dashboard-smoke:
	npm run dashboard-smoke

perf-smoke:
	$(PYTHON) -m pytest -q tests/test_performance_budgets.py

format:
	$(PYTHON) -m black main.py config.py src tests

format-check:
	$(PYTHON) -m black --check main.py config.py src tests

lint:
	$(PYTHON) .github/scripts/run_ruff.py check main.py config.py src tests

audit:
	# CVE-2025-3000 currently has no patched torch release on PyPI.
	# Keep auditing all other advisories and remove this once torch ships a fix.
	$(PYTHON) .github/scripts/run_dependency_audit.py -r requirements.txt --ignore-vuln CVE-2025-3000

build:
	$(PYTHON) -m build

build-if-available:
	@if $(PYTHON) -c "import build" >/dev/null 2>&1; then \
		$(PYTHON) -m build; \
	else \
		echo "python-build package not installed; skipping package build in local verify"; \
	fi

release-config:
	$(PYTHON) .github/scripts/check_release_config.py

hygiene:
	$(PYTHON) .github/scripts/check_repo_hygiene.py

size-check:
	$(PYTHON) .github/scripts/check_file_size.py --max-lines 1000

check: format-check lint typecheck dashboard-test perf-smoke coverage

verify: check dashboard-smoke typecheck-audit release-config hygiene size-check audit build-if-available
