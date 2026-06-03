# Bug Hunt: 20 Validated Areas

Branch: `codex/bug-hunt-20`
Baseline: `make check` passed on the branch before this report was written.
Remediation status: all 20 findings have been addressed on this branch.

Grading key:

- Impact: A = highest user/security/correctness risk, B = meaningful risk, C = moderate risk, D = low risk.
- Difficulty: A = easy, B = moderate, C = hard or design-heavy.

| # | Area | Layman explanation | Evidence | Impact | Difficulty |
|---|------|--------------------|----------|--------|------------|
| 1 | Dashboard token can leak to third-party asset hosts | The secret dashboard key is placed in the URL. The dashboard then loads scripts and fonts from external domains, so normal browser referrer behavior can expose the full tokenized URL unless a referrer policy or local assets prevent it. | `src/web/server.py:1193` builds `/?token=...`; `src/web/templates/dashboard.html:8` to `src/web/templates/dashboard.html:15` and `src/web/templates/launcher.html:8` to `src/web/templates/launcher.html:10` load external assets; no `Referrer-Policy` is set. | A / High | B / Medium |
| 2 | Dashboard read APIs are public without the token | Someone who can reach the server cannot open the HTML page without the token, but can still query training status, config, models, game stats, and neural-net layer data directly. | Local probe: `/` returned 401, while `/api/status`, `/api/config`, `/api/models`, `/api/game-stats`, and `/api/layers` all returned 200 without a token. Routes at `src/web/server.py:1343`, `src/web/server.py:1349`, `src/web/server.py:1407`, `src/web/server.py:1454`, and `src/web/server.py:1474` lack `_is_authorized_request()`. | A / High | B / Medium |
| 3 | Socket callback errors expose raw internal details | If a dashboard control callback raises, the exact exception string is sent back to the client. That can reveal local paths, model names, or implementation details. | Local probe returned `{'success': False, 'action': 'save_as', 'error': '/tmp/private/model secret'}`. Source returns `str(exc)` at `src/web/server.py:1232` to `src/web/server.py:1235`. | B / Medium | A / Easy |
| 4 | Dashboard URL is misleading when bound to all interfaces | Running with `--host 0.0.0.0` prints an address users cannot actually open from another machine. They need the real LAN IP, not `0.0.0.0`. | Local probe with `host='0.0.0.0'` printed `http://0.0.0.0:5999/?token=...`; `dashboard_url()` uses `self.host` directly at `src/web/server.py:1193` to `src/web/server.py:1196`; launcher prints it at `main.py:3886`. | B / Medium | A / Easy |
| 5 | Zero-step training crashes after the episode loop | If `MAX_STEPS_PER_EPISODE` is zero or misconfigured, `run_episode()` skips the loop and then reads `info` before it exists. | Repro output: `trainer_zero_steps: UnboundLocalError cannot access local variable 'info' where it is not associated with a value`. Source reads `info` at `src/ai/trainer.py:211` after the loop beginning at `src/ai/trainer.py:178`. | B / Medium | A / Easy |
| 6 | Training brick metrics are Breakout-specific and wrong for other games | Non-Breakout games can report their own progress fields, but `Trainer` only understands `bricks_remaining`; it ignores `info["bricks"]` and uses Breakout brick dimensions. | Repro returned `trainer_info_bricks_metric: 0` even though the dummy game reported `bricks: 7`. Source hardcodes `BRICK_ROWS * BRICK_COLS` and `bricks_remaining` at `src/ai/trainer.py:209` to `src/ai/trainer.py:211`. | B / Medium | B / Medium |
| 7 | Trainer evaluation crashes with zero episodes | Calling evaluation with zero games crashes instead of returning a clear validation error. | Repro output: `trainer_evaluate_zero: ValueError max() iterable argument is empty`. Source divides by `num_episodes` and calls `max(scores)`/`min(scores)` at `src/ai/trainer.py:351` to `src/ai/trainer.py:356`. | B / Medium | A / Easy |
| 8 | Trainer evaluation can hang forever on a stuck game | The training evaluator has no max-step cap, so a game that never returns `done=True` can trap evaluation in an infinite loop. | Source loop is `while not done:` with no step limit at `src/ai/trainer.py:341` to `src/ai/trainer.py:343`; contrast `src/ai/evaluator.py:134`, which has a `steps < max_steps` guard. | A / High | B / Medium |
| 9 | Prioritized replay buffer allows `beta_frames=0` and crashes | A bad replay-buffer schedule value can crash sampling during neural-net learning. | Repro output: `per_beta_zero: ZeroDivisionError division by zero`. Source stores `beta_frames` without validation at `src/ai/replay_buffer.py:344` to `src/ai/replay_buffer.py:377` and divides by it at `src/ai/replay_buffer.py:447` to `src/ai/replay_buffer.py:450`. | B / Medium | A / Easy |
| 10 | Standalone evaluator crashes with zero episodes | The richer evaluator has the same empty-input weakness: a zero-episode evaluation produces NumPy warnings and then a hard failure. | Repro output: `evaluator_zero_episodes: ValueError zero-size array to reduction operation minimum which has no identity`. Source computes reductions and `wins / num_episodes` at `src/ai/evaluator.py:166` to `src/ai/evaluator.py:179`. | B / Medium | A / Easy |
| 11 | Level distribution drops levels above 10 | If a game reaches level 11 or higher, the max level says so but the level distribution table shows zero for every bucket, making evaluation reports misleading. | Repro output: `evaluator_level_12_recorded: 12 {1: 0, ..., 10: 0}`. Source only builds buckets `range(1, 11)` at `src/ai/evaluator.py:157` to `src/ai/evaluator.py:160`. | C / Moderate | A / Easy |
| 12 | Config validation disappears under optimized Python | The app relies on `assert` for important configuration checks. Running Python with optimization disables those checks and allows invalid neural-net/training settings. | Repro with `python -O` printed `optimized_config_allows_invalid: -1 0 []`. Source uses asserts for learning rate, batch size, hidden layers, and other settings at `config.py:518` to `config.py:532`. | B / Medium | A / Easy |
| 13 | Documented audit target is missing its local dependency | `make audit` is advertised in the Makefile but fails in the current dev environment because `pip-audit` is not part of the project dev extras. | Command failed with `No module named pip_audit`. `Makefile:23` to `Makefile:24` calls it, while `pyproject.toml:31` to `pyproject.toml:35` lists dev tools but not `pip-audit`. | B / Medium | A / Easy |
| 14 | Dependency audit is non-blocking in CI | Even if dependency vulnerabilities are found, CI is configured to continue. That can let a risky dependency update merge green. | `.github/workflows/ci.yml:47` to `.github/workflows/ci.yml:63` sets `continue-on-error: true` for the dependency audit job. | A / High | A / Easy |
| 15 | Auto-selected training checkpoint is not compatibility-checked | Explicit model paths are inspected for matching state/action sizes, but the auto-loaded newest checkpoint is returned without inspection. A stale or wrong-game checkpoint can be selected and fail later. | `src/app/training_runtime.py:36` to `src/app/training_runtime.py:51` validates explicit paths; `src/app/training_runtime.py:53` to `src/app/training_runtime.py:71` returns the newest `.pth` directly. | B / Medium | B / Medium |
| 16 | Visual model service has the same unchecked auto-load path | The app-facing model service also validates explicit paths but returns the newest game checkpoint without checking compatibility first. | `src/app/model_service.py:48` to `src/app/model_service.py:63` validates explicit paths; `src/app/model_service.py:65` to `src/app/model_service.py:80` returns the newest `.pth` directly. | B / Medium | B / Medium |
| 17 | Broken checkpoints are listed as usable models | The dashboard model list suppresses checkpoint load failures, then still lists the file with no error marker. Users can see a broken checkpoint as if it were selectable. | Local probe with invalid `broken.pth` printed `broken_model_listed: broken.pth False False`. Source catches all load errors and `pass`es at `src/web/model_service.py:65` to `src/web/model_service.py:80`. | B / Medium | A / Easy |
| 18 | Game stats count corrupt checkpoints as models | The comparison panel increments `model_count` before it knows whether a checkpoint is readable, so corrupt files inflate counts and make the training inventory look healthier than it is. | Local probe with invalid `broken.pth` printed `corrupt_model_count: 1 None`. Source increments at `src/web/game_stats_service.py:37` to `src/web/game_stats_service.py:43` before load failure handling at `src/web/game_stats_service.py:44` to `src/web/game_stats_service.py:53`. | C / Moderate | A / Easy |
| 19 | Neural-net visualization failures are silently hidden | If the neural-net dashboard export breaks, training keeps going but the user gets no log, warning, or error signal explaining why the visualization stopped. | Source catches every exception and does `pass` at `main.py:1986` to `main.py:1996`. | B / Medium | A / Easy |
| 20 | Periodic-save cleanup failures are silently hidden | If old checkpoint deletion fails, the app gives no warning. Disk usage can grow and users may not know cleanup is broken. | Legacy path swallows deletion errors at `main.py:2090` to `main.py:2094`; app service similarly ignores `OSError` at `src/app/model_service.py:160` to `src/app/model_service.py:167`. | C / Moderate | A / Easy |

## Validation Run

Commands and results:

- `git checkout -B codex/bug-hunt-20 origin/main`: created the audit branch from current `origin/main`.
- `make check`: passed.
  - Black format check passed.
  - Focused mypy target passed.
  - 7 dashboard JavaScript tests passed.
  - 550 Python tests passed.
  - Coverage reported 50.94% with a 40% gate.
- `python -m ruff check .`: blocked because `ruff` is not installed in the current environment.
- `python -m pip_audit -r requirements.txt`: failed with `No module named pip_audit`; counted as finding 13 because `make audit` depends on it and project dev extras omit it.
- Local Python probes reproduced findings 2, 3, 4, 5, 6, 7, 9, 10, 11, 12, 17, and 18.

Post-fix validation:

- `python -m pytest tests/test_config.py tests/test_trainer.py tests/test_evaluator.py tests/test_replay_buffer.py tests/test_training_runtime.py tests/test_app_model_service.py tests/test_game_stats_service.py tests/test_model_service.py tests/test_web_routes.py tests/test_web_nn_inspection.py -q`: 176 passed.
- `make check`: passed after remediation.
  - Black format check passed.
  - Focused mypy target passed.
  - 7 dashboard JavaScript tests passed.
  - 571 Python tests passed.
  - Coverage reported 51.59% with a 40% gate.
- `python -m ruff check .`: still blocked in this local environment because `ruff` is not installed.
- CI-equivalent dependency audit from a temporary venv passed: `python -m pip_audit -r requirements.txt` reported no known vulnerabilities, with the expected skip for local project package `nn-game1`.
- `git diff --check`: passed.

## Suggested Order

1. Fix dashboard exposure first: findings 1, 2, 3, and 4.
2. Fix neural-net/training correctness next: findings 5, 6, 7, 8, 9, 10, 11, and 12.
3. Fix checkpoint truthfulness and loading safety: findings 15, 16, 17, and 18.
4. Tighten release/testing gates: findings 13 and 14.
5. Add observability for silent failures: findings 19 and 20.

## Discarded Candidates

- Delete-model path traversal was checked but not included because current tests cover unauthorized deletion and traversal-shaped IDs.
- Save-as filename traversal was checked but not included because the save path normalizes filenames before writing.
- README token guidance was checked but not included because the Safety Notes section already documents tokenized URLs and trusted-network hosting.
