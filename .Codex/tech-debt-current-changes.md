# Technical Debt Analysis - Current Change Files

**Scope:** `origin/main...HEAD` on `codex/validate-grade-items`  
**HEAD:** `b707805 test: broaden app confidence coverage`  
**Generated:** 2026-06-02  
**Validation:** `make check` passed: Black clean, focused mypy clean, 7 dashboard JS tests passed, 535 Python tests passed, total coverage 50.61%.

## Executive Summary

This PR closes the original grade-report items and materially improves reliability, security posture, and test coverage. The main remaining debt is not in the newly extracted helpers; it is in the still-large orchestration and UI surfaces that the PR had to work around.

Changed scope is broad: 68 files, 8,972 insertions, 5,033 deletions. The highest-maintenance files in the changed set are:

| File | Lines | Primary Debt |
|---|---:|---|
| `main.py` | 4,456 | App shell still mixes launcher, interactive loop, headless loop, model IO, dashboard wiring |
| `src/web/static/app.js` | 3,901 | Single global dashboard script with direct DOM/event/control flows |
| `src/web/server.py` | 1,966 | Route registration, Socket.IO controls, metrics, model stats in one server class |
| `src/game/space_invaders.py` | 1,976 | Large game object with rendering, collision, wave logic, rewards |
| `tests/test_web_server.py` | 875 | Useful coverage, but now a large integration-test file with many responsibilities |

## Debt Inventory

| # | Category | Debt Item | Evidence | Risk | Remediation |
|---|---|---|---|---|---|
| 1 | Architecture | `main.py` remains a god module | 4,456 lines; `GameApp` 2,094 lines; `HeadlessTrainer` 1,519 lines; `main()` complexity 43 | High | Continue extracting app modes into `src/app/interactive.py`, `src/app/headless.py`, `src/app/launcher.py` |
| 2 | Architecture | `src/web/server.py` still owns too many concerns | 1,966 lines; `MetricsPublisher` 745 lines; `WebDashboard` 836 lines; `_register_socket_events` complexity 51 | High | Split `routes.py`, `socket_controls.py`, `metrics_publisher.py`, `game_stats_service.py` |
| 3 | Frontend | Dashboard JS is still monolithic | `src/web/static/app.js` is 3,901 lines with global state and inline event entrypoints | High | Introduce modules for `socket`, `models`, `controls`, `charts`, `nn_inspection` |
| 4 | Testing | Critical app shell remains lightly covered | Coverage: `main.py` 6%; `src/game/menu.py` 8%; `src/visualizer/pause_menu.py` 8% | Medium | Add focused lifecycle tests before larger refactors; avoid broad UI snapshots |
| 5 | Game Code | Game implementations combine simulation, rendering, reward logic | `SpaceInvaders` 1,107-line class; `Breakout` 693-line class; `Snake` 618-line class | Medium | Extract collision/reward/state encoders per game as pure helpers |
| 6 | Test Quality | Web server tests are growing into a second god file | `tests/test_web_server.py` 875 lines; integration class 549 lines | Medium | Split into `test_web_routes.py`, `test_web_socket_controls.py`, `test_nn_inspection.py` |
| 7 | Packaging | Some entrypoint/test path mutation remains | `main.py` still mutates `sys.path`; several older tests insert repo root | Low | Add package entrypoint and rely on test runner import path |
| 8 | Dependency/CI | Dependency audit is visible but non-blocking | CI `dependency-audit` uses `continue-on-error: true` | Medium | Keep non-blocking until constraints stabilize, then fail on high/critical CVEs |
| 9 | Security | Trusted checkpoint fallback still exists by design | `Agent.load()` and explicit startup loaders opt into `allow_unsafe_fallback=True` for trusted local dirs | Medium | Add explicit "legacy checkpoint" UX/CLI path and make normal resume restricted where possible |
| 10 | Performance | Large NN/dashboard payload work is tested but not benchmarked in CI | Smoke budget exists, no historical perf trend | Low | Add optional benchmark job or perf artifact for NN snapshot and vector env loop |

## Metrics Dashboard

```yaml
changed_files: 68
diff:
  insertions: 8972
  deletions: 5033
validation:
  python_tests: 535
  dashboard_js_tests: 7
  coverage_total: 50.61
hotspots:
  main_py:
    lines: 4456
    long_functions_over_50_lines: 27
    complex_functions_over_10: 12
    god_classes: [GameApp, HeadlessTrainer]
  web_server_py:
    lines: 1966
    long_functions_over_50_lines: 12
    complex_functions_over_10: 5
    god_classes: [MetricsPublisher, WebDashboard]
  dashboard_app_js:
    lines: 3901
coverage_gaps:
  main_py: 6
  game_menu_py: 8
  pause_menu_py: 8
  snake_py: 48
  space_invaders_py: 51
```

## Impact Analysis

Assumption for cost estimates: one developer hour is valued at `$150`.

| Debt | Development Cost | Quality Cost | Estimated Annual Cost |
|---|---:|---:|---:|
| `main.py` app-shell coupling | 2-4 extra hours per lifecycle/dashboard change because visual and headless paths must both be checked | Missed mode-specific regressions | `$18k-$36k` |
| Monolithic dashboard JS | 2 extra hours per dashboard control/model/chart change | Higher chance of frontend-only regressions escaping pytest | `$12k-$24k` |
| Web server class coupling | 1-3 extra hours per route/control/model stats change | Control/security regressions cluster in one file | `$12k-$27k` |
| Low `main.py` coverage | Slower review due to manual reasoning | Regressions in launcher/headless startup can ship | `$15k-$30k` |
| Large game classes | 2+ hours extra for reward/collision tuning per game | Physics/reward fixes can affect rendering/state | `$10k-$20k` |

## Prioritized Roadmap

### Quick Wins - This Sprint

1. Split `tests/test_web_server.py` by concern.
   - Effort: 2-4 hours
   - Benefit: Faster review, easier targeted test additions
   - ROI: High

2. Extract `game_stats_service.py` from `WebDashboard`.
   - Effort: 3-5 hours
   - Benefit: Removes checkpoint scanning from route body; easier security tests
   - ROI: High

3. Move remaining `main.py` model-list/inspect helpers to `src/app/model_management.py`.
   - Effort: 4-6 hours
   - Benefit: Reduces `main.py` and removes another CLI/app-shell responsibility
   - ROI: Medium-high

4. Add package-style launch documentation and remove test `sys.path.insert` churn.
   - Effort: 2-3 hours
   - Benefit: Cleaner imports and less environment fragility
   - ROI: Medium

### Month 1

1. Extract `socket_controls.py` with one function per action.
   - Target: reduce `_register_socket_events` complexity from 51 to under 15.

2. Start dashboard JS module split with `models.js` and `controls.js`.
   - Target: make destructive flows testable without loading the full dashboard script.

3. Move `GameApp` and `HeadlessTrainer` shared callback wiring into `src/app/training_runtime.py` or a small `DashboardCallbacks` helper.
   - Target: one source of truth for pause/save/load/start-fresh controls.

### Quarter

1. Decompose each large game into pure state/reward/collision helpers.
   - Start with `SpaceInvaders` because it is the largest and most complex.

2. Raise coverage target from 40% to 55%, then 65%.
   - Do this after moving high-risk logic into smaller modules; do not force coverage by testing huge render bodies.

3. Introduce optional Playwright smoke tests for the real dashboard page.
   - Keep local/unit tests fast; browser tests can run as a separate CI job.

## Implementation Guide

Use strangler-style extraction:

1. Add a new helper/service with the current behavior copied behind tests.
2. Route the existing class/function through the helper.
3. Add focused tests against the helper contract.
4. Delete duplicate branches only after both interactive and headless paths use the helper.

Best first target:

```text
src/web/server.py
  -> src/web/game_stats_service.py
  -> tests/test_game_stats_service.py
```

This has low UI risk and removes checkpoint scanning from the Flask route.

## Prevention Plan

- Add a lightweight complexity script or `radon` gate for changed Python files.
- Track maximum file length for source files; warn over 1,000 lines, fail over 2,000 for new files.
- Keep `make check` as the PR gate, but ratchet `--cov-fail-under` from 40 to 55 after this PR settles.
- Require a test file split when any single test module crosses 800 lines.
- Keep dependency audit non-blocking now, then fail on high/critical vulnerabilities after the constraints file is stable.

## Current Status

This PR is healthier than the starting point: the original grade items are closed, CI exists, test coverage is above 50%, model operations are safer, and the neural-net/dashboard contract is much better protected. The next ROI is structural cleanup, especially splitting the app shell and dashboard surfaces so future fixes do not require touching thousand-line files.
