# NN-Game1 Architecture

This repo has five main runtime surfaces. Keep changes close to the surface they affect, then use the matching validation command before opening a PR.

## Application Entrypoint

- `main.py` owns command dispatch, interactive game startup, headless training startup, and web mode startup.
- `src/app/cli.py` owns argument parsing.
- `src/app/training_runtime.py` owns shared training/runtime helpers used outside the main loop.

Use this area for app lifecycle, CLI, launch, save/stop, and runtime coordination changes.

## Games

- `src/game/base_game.py` defines shared game expectations.
- `src/game/breakout.py`, `src/game/snake.py`, `src/game/pong.py`, `src/game/asteroids.py`, and `src/game/space_invaders.py` implement game-specific state, rewards, actions, and rendering.
- `src/game/menu.py` and `src/visualizer/pause_menu.py` cover local UI overlays.

Use this area for gameplay rules, action spaces, reward shaping, game rendering, and human-control behavior.

## AI And Training

- `src/ai/network.py` defines DQN and Dueling DQN networks.
- `src/ai/agent.py`, `src/ai/trainer.py`, and `src/ai/replay_buffer.py` own learning behavior, replay, and optimization.
- `src/ai/evaluator.py` owns evaluation helpers.

Use this area for neural-net architecture, replay behavior, training policy, exploration, and model evaluation.

## Web Dashboard

- `src/web/server.py` serves Flask/Socket.IO routes, dashboard auth, metrics publishing, and live control events.
- `src/web/model_service.py` and `src/web/game_stats_service.py` isolate model and stats filesystem behavior.
- `src/web/static/app.js` is the live dashboard application.
- `src/web/static/dashboard_core.js` contains small dashboard helpers with Node tests.
- `src/web/templates/` contains the dashboard and launcher HTML.

Use this area for browser controls, model management, live metrics, neural-net inspection panels, and launcher behavior.

## Tooling And Release

- `Makefile` defines local validation commands.
- `.github/workflows/ci.yml` runs formatting, ruff linting, typing, dashboard JS tests, Python coverage, Playwright dashboard smoke, build, dependency audit, dependency review, and CodeQL.
- `.github/scripts/check_release_config.py` validates release automation config.
- `.github/scripts/check_repo_hygiene.py` catches tracked scratch files.
- `.github/scripts/check_file_size.py` keeps tracked source files below the configured source-size refactor budgets.
- `.github/scripts/check_audit_waivers.py` fails when an ignored dependency advisory needs review.
- `.github/scripts/check_dependency_files.py` checks dependency-file parity before audits.
- `.github/scripts/run_dependency_audit.py` runs `pip-audit` and installs the audit tool locally if the active environment is missing it.
- `.github/scripts/run_ruff.py` runs ruff and installs the lint tool locally if the active environment is missing it.
- `pyproject.toml`, `requirements.txt`, and `constraints.txt` define packaging and dependency pins.

Use this area for CI, release, dependency, packaging, and repo hygiene changes.

## Validation Commands

```bash
make check
```

Fast local gate: formatting, ruff linting, strict mypy across `src`, `main.py`, and `config.py`, dashboard JS helper tests, performance smoke, and Python coverage with an 80% floor.

```bash
make verify
```

Fuller local gate: `make check`, Playwright dashboard smoke, strict type-audit, release config validation, repo hygiene, source file size budgets, dependency audit, and package build when the `build` package is installed.

```bash
make typecheck-audit
```

Alias for the strict full-app mypy gate.
