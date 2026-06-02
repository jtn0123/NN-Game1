# Codebase Grade Report

**Project:** NN-Game1
**Audited:** 2026-06-01
**Stack:** Python DQN arcade-training app with PyTorch, NumPy, Pygame, Flask-SocketIO dashboard, vanilla JS frontend, pytest tests

## Summary

| ID | Category | Grade | Items |
|----|----------|-------|-------|
| A | Architecture & Design | C+ | 4 |
| B | Backend Quality | B- | 4 |
| C | Frontend Quality | C+ | 4 |
| D | Testing & Reliability | C | 4 |
| E | Security | C- | 4 |
| F | Dependencies & Tech Currency | C- | 3 |
| G | Performance & Scalability | B- | 3 |
| H | Documentation & Onboarding | C | 3 |
| I | Developer Experience & Tooling | C | 4 |
| **Overall** | | **C+** | **35** |

**Original top 5 highest-leverage fixes:** E1, A1, D1, I1, B1
**Current highest-leverage remaining fixes:** C1, A1, C3, A2, E4

**Implementation progress:**
- 2026-06-01: Implemented E1 by adding checkpoint metadata sidecars and making dashboard model/stat listing read JSON metadata instead of deserializing `.pth` files.
- 2026-06-01: Implemented E2 by defaulting the dashboard to `127.0.0.1` and replacing wildcard Socket.IO CORS with explicit localhost origins.
- 2026-06-01: Implemented E3 for mutating dashboard actions by adding a per-run control token to Socket.IO controls, clear-log events, and model DELETE requests.
- 2026-06-01: Implemented D1/F1/F2/F3/I1 foundations by adding `pyproject.toml`, a compatibility `requirements.txt`, and `Makefile` setup/test/lint/typecheck/run commands.
- 2026-06-01: Completed the first I2/I3 baseline by formatting the codebase, enabling Ruff import/pyflakes checks, fixing mypy errors, and adding format/lint/typecheck gates to CI.
- 2026-06-01: Implemented B2 by validating and normalizing dashboard control payloads before invoking runtime callbacks.
- 2026-06-01: Implemented B3/B4 by making legacy trainer progress extraction game-aware and logging throttled NN visualization emit failures instead of silently swallowing them.
- 2026-06-01: Implemented C4/D3/D4 with server-rendered game options, registry contract tests, dashboard control validation tests, and model metadata sidecar endpoint coverage.
- 2026-06-01: Implemented A4/H1/H2 cleanup by ignoring generated experiment run folders, documenting the experiments directory, refreshing supported-game README text, and removing the broken architecture image.
- 2026-06-01: Advanced B1/A1 by extracting `src/app/model_service.py` for shared model filename normalization, path resolution, training-history payload construction, recent metric helpers, and periodic checkpoint cleanup used by both visual and headless modes.
- 2026-06-01: Implemented A3 by adding a `BaseVecGame` protocol, registering vectorized constructors beside normal game metadata, and making headless vector setup registry-driven.
- 2026-06-01: Implemented C2 by replacing dashboard and launcher inline event handlers with data-action bindings, delegated JS listeners, and a static regression test that rejects inline handler attributes.
- 2026-06-01: Tightened CI compatibility by fixing pygame-stub type errors in visual surfaces/fonts and documenting the remaining implementation backlog below.

**Completed items:** A3, A4, B2, B3, B4, C2, C4, D1, D3, D4, E1, E2, E3, F2, F3, G1, H1, H2, I1, I2, I3.

**Partially completed items:** A1 and B1 were advanced by `src/app/model_service.py`, but `main.py` still needs deeper service extraction and residual visual/headless lifecycle cleanup. F1 was advanced by package metadata/extras, but a lockfile or explicit lock strategy is still open.

**Remaining implementation backlog:**

| ID | Remaining work | Impact | Effort |
|----|----------------|--------|--------|
| C1 | Split `src/web/static/app.js` into focused modules with a small bootstrap entry. | Major | L |
| A1 | Continue extracting `main.py` lifecycle, CLI, web launcher, and training-controller responsibilities beyond the first `ModelService` slice. | Major | L |
| C3 | Add frontend smoke/unit coverage for dashboard state transitions, model empty states, controls, and chart rendering. | Moderate | M |
| A2 | Migrate `config.py` toward scoped config dataclasses while preserving current attribute compatibility. | Moderate | M |
| E4 | Vendor pinned dashboard browser assets or add exact SRI/crossorigin attributes for CDN scripts. | Moderate | S |
| D2 | Add pytest markers/skip guards for heavyweight `torch`, `pygame`, `web`, and slow integration tests. | Moderate | S |
| F1 | Choose and commit a reproducible lock strategy (`uv.lock`, `pip-tools`, or documented equivalent). | Major | M |
| G2 | Move neural-network visualization sampling into a rate-limited publisher with explicit serialization budgets. | Moderate | M |
| G3 | Add an advisory performance benchmark command for headless steps/sec regression tracking. | Minor | S |
| H3 | Mark historical bug/progress reports as snapshots and add a current status note with validation caveats. | Moderate | S |
| I4 | Improve optional dependency first-run errors and dependency-specific test selection. | Moderate | S |

---

## A — Architecture & Design — C+

The project has a useful domain split under `src/game`, `src/ai`, `src/visualizer`, and `src/web`, and the game registry gives new games a clear integration point in `src/game/__init__.py:38-84`. The architecture weakens at the application boundary: `main.py` is over 3,900 lines and owns CLI parsing, visual mode, headless mode, web launcher orchestration, restart logic, save/load, metrics, rendering, and control callbacks. `config.py` also centralizes every game, training, reward, visualization, device, and persistence setting in one dataclass, which makes experimentation easy but pushes unrelated concerns through one global object.

#### A1 — Split `main.py` into application services
- **Where:** `main.py:124-1930`, `main.py:1964-3330`, `main.py:3376-4000`
- **What's wrong:** Two large application classes plus global launch helpers make lifecycle behavior hard to reason about. Visual and headless modes duplicate save/load/config/web-dashboard logic, so fixes can land in one path and miss the other.
- **Impact:** Major — this is the largest source of regression risk and repeated development drag.
- **Fix:** Extract `src/app/cli.py`, `src/app/web_launcher.py`, `src/app/model_service.py`, and `src/app/training_controller.py`. Move duplicated model/dashboard/config callback code behind shared helpers used by both `GameApp` and `HeadlessTrainer`.
- **Effort:** L
- **Grade lift:** C+ → B- (removes the biggest structural bottleneck)

#### A2 — Convert the monolithic config into scoped configs
- **Where:** `config.py:20-520`
- **What's wrong:** One dataclass mixes Breakout, Space Invaders, neural network, reward shaping, web/dashboard, device, logging, and persistence settings. The comments are helpful, but every feature depends on the same global object.
- **Impact:** Moderate — changing one game's tuning or persistence setting risks accidental cross-game behavior.
- **Fix:** Introduce nested dataclasses such as `GameConfig`, `TrainingConfig`, `NetworkConfig`, `DashboardConfig`, and per-game config blocks. Keep a compatibility layer for current attribute names during migration.
- **Effort:** M
- **Grade lift:** C+ → B- (clearer ownership and safer game-specific changes)

#### A3 — Make vectorized environment support a first-class interface
- **Where:** `src/game/breakout.py`, `src/game/space_invaders.py`, `src/game/pong.py`, `src/game/snake.py`, `src/game/asteroids.py`, `main.py:2955-3257`
- **What's wrong:** Each game exposes a `Vec*` class, but there is no base protocol equivalent to `BaseGame` for vectorized environments. `main.py` must know concrete vectorized classes and training assumptions.
- **Impact:** Moderate — adding or changing vectorized games requires edits in multiple places.
- **Fix:** Add `BaseVecGame`/protocol with `reset`, `step`, `close`, and `seed`, then register vectorized constructors beside normal game metadata.
- **Effort:** M
- **Grade lift:** C+ → B- (keeps the registry pattern consistent)

#### A4 — Move generated experiment scripts out of the source checkout
- **Where:** `experiments/run_20251129_171433/*`, `experiments/run_20251129_171513/*`, `experiments/run_20251129_171627/*`
- **What's wrong:** Experiment output scripts live inside the repository tree and are not covered by `.gitignore`. They can clutter diffs and make it harder to distinguish source from run artifacts.
- **Impact:** Minor — mostly repo hygiene, but it will get painful as experiments grow.
- **Fix:** Add an `experiments/README.md` plus ignore generated run directories, or move generated code into `runs/`/`artifacts/` and keep only reproducible experiment definitions in source.
- **Effort:** S
- **Grade lift:** C+ → B- (cleaner project boundaries)

---

## B — Backend Quality — B-

The AI/backend core is stronger than the outer app shell. `src/ai/replay_buffer.py:31-220` uses contiguous NumPy storage and vectorized batch operations, `src/ai/agent.py:145-260` supports dueling DQN, NoisyNets, PER, N-step returns, schedulers, and compile compatibility, and `src/game/base_game.py:20-110` provides a simple game contract. The backend loses points for validation gaps around web-controlled actions, duplicated persistence behavior between visual/headless modes, and a legacy `Trainer` path that still assumes Breakout-specific metrics.

#### B1 — Unify save/load behavior across visual and headless modes
- **Where:** `main.py:1860-1930`, `main.py:2377-2518`, `main.py:3257-3330`
- **What's wrong:** Visual mode sanitizes custom save filenames before adding `.pth`, but headless `_save_model_as` appends `.pth` directly and relies on the caller. Save metadata and dashboard notification paths are also duplicated.
- **Impact:** Major — user-facing model management can diverge between modes.
- **Fix:** Create a `ModelService` that owns filename sanitization, model directory selection, metadata construction, checkpoint cleanup, and dashboard save notifications. Call it from both app classes.
- **Effort:** M
- **Grade lift:** B- → B (removes duplicated behavioral surface)

#### B2 — Validate web control payloads before invoking callbacks
- **Where:** `src/web/server.py:1409-1493`
- **What's wrong:** The Socket.IO `control` handler trusts raw payload fields for actions like speed, config changes, model paths, performance mode, and selected game. Several callbacks then apply values to runtime state.
- **Impact:** Moderate — malformed local UI or socket traffic can cause confusing runtime failures.
- **Fix:** Add per-action validators with allowed action names, numeric ranges, finite-number checks, known game IDs, and model path containment before invoking callbacks.
- **Effort:** S
- **Grade lift:** B- → B (turns dashboard controls into a real API boundary)

#### B3 — Generalize the legacy `Trainer` metrics for non-Breakout games
- **Where:** `src/ai/trainer.py:157-219`
- **What's wrong:** `Trainer.run_episode` computes `bricks_broken` from `BRICK_ROWS * BRICK_COLS` and `info['bricks_remaining']`, which is Breakout-specific despite the generic game interface.
- **Impact:** Moderate — using this trainer with other registered games produces misleading metrics.
- **Fix:** Move game-specific episode summary fields into `BaseGame` info contracts or normalize metrics through a small adapter per game.
- **Effort:** S
- **Grade lift:** B- → B (keeps generic trainer actually generic)

#### B4 — Replace silent exception swallowing in visualization paths with throttled diagnostics
- **Where:** `main.py:1846-1858`, `main.py:2520-2585`
- **What's wrong:** Neural-network visualization emit failures are swallowed silently. That protects training, but it makes dashboard breakage invisible.
- **Impact:** Moderate — dashboard data can disappear with no actionable signal.
- **Fix:** Log the first failure and then throttle repeats by exception type/time window; show a dashboard warning without crashing training.
- **Effort:** S
- **Grade lift:** B- → B (improves observability without risking training uptime)

---

## C — Frontend Quality — C+

The dashboard is feature-rich: charts, model management, console filtering, neural-network canvas, throttled fetches, and adaptive rendering live in `src/web/static/app.js`. The downside is that almost all behavior sits in one large global script, with inline event handlers in templates and no frontend test harness. The HTML also still initializes the game selector with only Breakout before client-side population.

#### C1 — Modularize the dashboard JavaScript
- **Where:** `src/web/static/app.js:1-3830`
- **What's wrong:** Charts, sockets, model management, save dialogs, game switching, neural visualization, config editing, and utility code share global variables and functions.
- **Impact:** Major — frontend changes are high-blast-radius and hard to test.
- **Fix:** Split into modules such as `socketClient.js`, `charts.js`, `models.js`, `nnVisualizer.js`, `controls.js`, and `state.js`. Keep a small bootstrap entry file.
- **Effort:** L
- **Grade lift:** C+ → B- (gives the frontend maintainable boundaries)

#### C2 — Remove inline event handlers from templates
- **Where:** `src/web/templates/dashboard.html:21-28`, `src/web/templates/dashboard.html:48-120`
- **What's wrong:** The template wires behavior through inline `onchange`/`onclick` attributes. This couples markup to global function names and makes CSP hard to adopt.
- **Impact:** Moderate — hurts maintainability and security hardening.
- **Fix:** Add stable data attributes and bind events from the JS bootstrap after DOM load.
- **Effort:** M
- **Grade lift:** C+ → B- (cleaner UI behavior ownership)

#### C3 — Add a frontend test harness for dashboard state transitions
- **Where:** `src/web/static/app.js`, `tests/test_web_server.py`
- **What's wrong:** Backend API behavior is tested, but chart/model/control UI behavior is not covered. Many dashboard bugs would only be caught manually.
- **Impact:** Moderate — dashboard regressions are likely as features grow.
- **Fix:** Add lightweight JS unit tests for pure helpers and Playwright smoke tests for connect, model list empty state, pause/save/reset controls, and chart rendering.
- **Effort:** M
- **Grade lift:** C+ → B- (turns the dashboard into a verifiable surface)

#### C4 — Stop hardcoding initial game options in dashboard HTML
- **Where:** `src/web/templates/dashboard.html:21-25`, `src/game/__init__.py:38-84`
- **What's wrong:** The backend registry exposes five games, but the initial dashboard select contains only Breakout until JS populates/updates it.
- **Impact:** Minor — users can see stale UI before the API populates, and no-JS/failure states are misleading.
- **Fix:** Render game options server-side from `get_all_game_info()` or show a disabled loading state until `/api/games` succeeds.
- **Effort:** S
- **Grade lift:** C+ → B- (small but visible consistency win)

---

## D — Testing & Reliability — C

There are many tests and they target meaningful behavior across games, agent learning, replay buffers, web status, and phase-specific dashboard data. However, test collection currently fails in this worktree because the active interpreter lacks required imports (`torch`, `pygame`, and some web deps), and there is no CI workflow or pinned test environment to prove the suite from a clean checkout. `python3 -m compileall -q src main.py config.py tests` passed, but `pytest --collect-only -q` stopped with 16 import errors.

#### D1 — Make the test environment reproducible and runnable from clean checkout
- **Where:** `requirements.txt:1-24`, `tests/conftest.py:1-28`, missing `pyproject.toml`/CI
- **What's wrong:** `pytest --collect-only -q` fails before running product tests because dependencies are not installed in the active environment. The repo has no lockfile, tox/nox config, or CI to define the expected test matrix.
- **Impact:** Major — test count looks healthy, but the current checkout cannot prove it.
- **Fix:** Add a `pyproject.toml` with test/dev extras, pin a supported Python range, add `nox` or `tox`, and create a CI job that installs `.[test]` and runs pytest.
- **Effort:** M
- **Grade lift:** C → B- (turns tests into a reliable gate)

#### D2 — Mark or isolate heavyweight integration tests
- **Where:** `tests/test_integration.py:50-232`, `tests/test_visualizer.py`, `tests/test_web_server.py`
- **What's wrong:** Tests import heavy runtime dependencies at module import time. There is no clear split between fast unit tests, Pygame tests, PyTorch tests, and web dashboard tests.
- **Impact:** Moderate — contributors cannot run a fast subset without already having the full stack ready.
- **Fix:** Add pytest markers (`unit`, `torch`, `pygame`, `web`, `slow`) and skip/xfail guards at fixture level for unavailable optional subsystems.
- **Effort:** S
- **Grade lift:** C → C+ (improves feedback loop immediately)

#### D3 — Add deterministic smoke tests for each registered game
- **Where:** `src/game/__init__.py:38-84`, `tests/test_*game*.py`
- **What's wrong:** Each game has tests, but there is no single registry-level contract test that instantiates every registered game and verifies `reset`, `step`, `state_size`, `action_size`, and info shape.
- **Impact:** Moderate — new registry entries can drift from the generic training assumptions.
- **Fix:** Parametrize over `list_games()` and assert every game satisfies the `BaseGame` runtime contract in headless mode.
- **Effort:** S
- **Grade lift:** C → C+ (protects the plugin-like architecture)

#### D4 — Add regression tests for dashboard model metadata loading
- **Where:** `src/web/server.py:1200-1257`, `src/ai/agent.py:950-1030`
- **What's wrong:** Model listing and model loading involve checkpoint deserialization, metadata extraction, and compatibility checks, but the dangerous/edge cases are not isolated.
- **Impact:** Moderate — model picker and resume flows are high-value user paths.
- **Fix:** Add tests for corrupted `.pth`, incompatible state/action sizes, metadata-only listing failure behavior, and path validation.
- **Effort:** M
- **Grade lift:** C → C+ (hardens save/resume workflows)

---

## E — Security — C-

This appears to be a local developer/training app, not a hardened public service. Even so, the dashboard binds to all interfaces by default, accepts wildcard CORS, exposes unauthenticated Socket.IO controls, and deserializes PyTorch checkpoints with `weights_only=False`. That combination is the clearest safety risk in the repo.

#### E1 — Stop unsafe checkpoint deserialization during model listing
- **Where:** `src/web/server.py:1200-1257`
- **What's wrong:** `/api/models` calls `torch.load(path, map_location='cpu', weights_only=False)` for every `.pth` to read metadata. PyTorch checkpoints are pickle-based, so a malicious file in `models/` can execute code merely by opening the model picker.
- **Impact:** Major — local code execution risk through a passive dashboard endpoint.
- **Fix:** Store checkpoint metadata in a sidecar JSON file on save, read that for listing, and only load trusted checkpoints during explicit load. If `torch.load` is still needed, use `weights_only=True` where compatible and handle legacy files behind an explicit opt-in.
- **Effort:** M
- **Grade lift:** C- → C+ (removes the sharpest security footgun)

#### E2 — Restrict dashboard exposure by default
- **Where:** `src/web/server.py:1063-1079`, `src/web/server.py:1099-1100`, `src/web/server.py:1576-1584`
- **What's wrong:** `WebDashboard` defaults to `host='0.0.0.0'`, allows `cors_allowed_origins="*"`, and runs Werkzeug with `allow_unsafe_werkzeug=True`.
- **Impact:** Major — anyone on the LAN may be able to hit a control surface intended for the local user.
- **Fix:** Default host to `127.0.0.1`, make LAN binding an explicit CLI flag, restrict CORS to the served origin, and print a warning when exposing to the network.
- **Effort:** S
- **Grade lift:** C- → C (reduces accidental exposure)

#### E3 — Add authorization or a session token for Socket.IO controls
- **Where:** `src/web/server.py:1398-1498`
- **What's wrong:** Any socket client that can connect can send `control` actions such as pause, reset, save, load model, config change, game switch, and save-and-quit.
- **Impact:** Moderate — not catastrophic for local-only use, but risky once bound to LAN.
- **Fix:** Generate a per-run token, inject it into the served page, and require it on Socket.IO control events and mutating REST endpoints.
- **Effort:** M
- **Grade lift:** C- → C+ (protects the control plane)

#### E4 — Replace CDN script dependencies with pinned local assets or integrity checks
- **Where:** `src/web/templates/dashboard.html:7-13`
- **What's wrong:** The dashboard loads Chart.js, chartjs-plugin-zoom, Socket.IO, and fonts from public CDNs without SRI. A CDN/network issue can break or alter the dashboard.
- **Impact:** Moderate — supply-chain and offline reliability risk.
- **Fix:** Vendor pinned assets under `src/web/static/vendor/` or add `integrity`/`crossorigin` attributes with exact versions.
- **Effort:** S
- **Grade lift:** C- → C (small hardening step)

---

## F — Dependencies & Tech Currency — C-

Dependencies are listed clearly, but `requirements.txt` uses broad lower bounds and there is no lockfile or package metadata. The README says Python 3.9+ tested with 3.11, while the current shell is Python 3.14.5 and test collection fails without a managed environment. Tooling packages are listed as optional comments but not separated into extras.

#### F1 — Add package metadata, extras, and a lock strategy
- **Where:** `requirements.txt:1-24`, missing `pyproject.toml`
- **What's wrong:** Runtime, web, test, and dev dependencies are all in one requirements file with broad minimum versions.
- **Impact:** Major — installs can drift and break tests or training unpredictably.
- **Fix:** Add `pyproject.toml` with dependencies and extras like `[project.optional-dependencies] test`, `web`, and `dev`. Use `uv.lock`, `pip-tools`, or a similar lock strategy for reproducible local/CI installs.
- **Effort:** M
- **Grade lift:** C- → C+ (makes environments repeatable)

#### F2 — Pin the supported Python version range
- **Where:** `README.md:116-135`, missing `.python-version`/`pyproject.toml`
- **What's wrong:** Docs say Python 3.9+ and tested with 3.11, but there is no enforced range. The current shell is Python 3.14.5, which may outrun PyTorch/Pygame compatibility.
- **Impact:** Moderate — contributors can land in an interpreter that cannot install or run the stack.
- **Fix:** Set `requires-python`, add `.python-version`, and document the recommended Python version for CPU/MPS training.
- **Effort:** S
- **Grade lift:** C- → C (removes environment ambiguity)

#### F3 — Separate optional web/dev/test dependencies from core training
- **Where:** `requirements.txt:8-24`, `src/web/server.py:44-53`
- **What's wrong:** Flask, Socket.IO, eventlet, pytest, black, and mypy sit beside core training deps even though they serve different workflows.
- **Impact:** Moderate — installing the project for headless training pulls unnecessary packages, while tests still lack a guaranteed dev install path.
- **Fix:** Keep core deps minimal and expose extras: `pip install -e ".[web,test,dev]"`.
- **Effort:** S
- **Grade lift:** C- → C (cleaner install intent)

---

## G — Performance & Scalability — B-

The codebase shows real performance work: replay buffers use contiguous arrays, vectorized environments exist for all registered games, config includes benchmark-derived CPU/MPS notes, and dashboard updates are throttled/adaptive. The main remaining performance risks are avoidable serialization/deserialization work, large monolithic UI updates, and repeated model/dashboard work in training loops.

#### G1 — Avoid checkpoint deserialization in hot dashboard endpoints
- **Where:** `src/web/server.py:1200-1257`, `src/web/server.py:1332-1393`
- **What's wrong:** Model/status endpoints may deserialize checkpoint files to compute metadata. That can be slow for large models and dangerous for security.
- **Impact:** Moderate — model pickers and game stats can stutter or block as checkpoints grow.
- **Fix:** Persist metadata sidecars during `Agent.save` and read lightweight JSON from dashboard endpoints.
- **Effort:** M
- **Grade lift:** B- → B (faster and safer dashboard)

#### G2 — Reduce per-frame neural visualization extraction cost
- **Where:** `main.py:1793-1858`, `main.py:2520-2585`, `src/ai/network.py:197-209`, `src/ai/network.py:414-431`
- **What's wrong:** Visualization enables activation capture, runs Q-value/activation/weight extraction, and emits layer data from training paths. Some throttling exists, but extraction and serialization remain coupled to the trainers.
- **Impact:** Moderate — dashboard mode can steal throughput from training.
- **Fix:** Move NN visualization sampling into a rate-limited publisher service with explicit budgets for activations, weights, and layer analysis.
- **Effort:** M
- **Grade lift:** B- → B (keeps training speed predictable)

#### G3 — Add performance regression benchmarks to CI/local tooling
- **Where:** `benchmark.py`, `config.py:252-281`, missing CI
- **What's wrong:** Performance claims are documented in comments, but there is no repeatable benchmark gate or trend artifact.
- **Impact:** Minor — speed regressions may go unnoticed until long training runs.
- **Fix:** Add a small benchmark command that reports steps/sec for one tiny headless run and store expected thresholds by platform as advisory, not hard-fail, checks.
- **Effort:** S
- **Grade lift:** B- → B (keeps optimization work measurable)

---

## H — Documentation & Onboarding — C

The README is unusually detailed and educational, which fits the project well. It is also stale: it advertises only Breakout and Space Invaders while the registry includes Pong, Snake, and Asteroids, references `docs/architecture.png` although no `docs/` directory is present, and omits newer web-launcher/vectorized workflows in places. Prior bug/progress docs are useful history but not clearly separated from current truth.

#### H1 — Refresh README to match current registered games and workflows
- **Where:** `README.md:1-86`, `src/game/__init__.py:38-84`, `main.py:3376-3518`
- **What's wrong:** README says supported games are Breakout and Space Invaders and shows an older project structure. Current code supports five games, web mode, vectorized envs, model inspection, and more.
- **Impact:** Major — new users will misunderstand what the app can do and where files live.
- **Fix:** Update feature list, project tree, quick-start commands, game list, and mode descriptions from the live registry/CLI.
- **Effort:** S
- **Grade lift:** C → C+ (high user-facing clarity gain)

#### H2 — Restore or remove the missing architecture image
- **Where:** `README.md:7`, missing `docs/architecture.png`
- **What's wrong:** The README embeds an image path that does not exist in the checkout.
- **Impact:** Minor — visible broken documentation asset.
- **Fix:** Add the diagram under `docs/architecture.png` or replace it with a checked-in Mermaid/text diagram.
- **Effort:** S
- **Grade lift:** C → C+ (small polish fix)

#### H3 — Mark historical bug/progress docs as historical
- **Where:** `BUGS_AND_IMPROVEMENTS.md`, `30_BUGS_LIST.md`, `30_MORE_BUGS_LIST.md`, `PROGRESS_REPORT.md`, `PHASE*_SUMMARY.md`
- **What's wrong:** Several bug-list items appear stale or already fixed, while progress docs claim all tests pass despite current collection failing without dependencies.
- **Impact:** Moderate — future agents/contributors can chase outdated findings.
- **Fix:** Add a current `docs/status.md` or top-of-file banners marking old reports as historical snapshots, with dates and validation caveats.
- **Effort:** S
- **Grade lift:** C → C+ (reduces false starts)

---

## I — Developer Experience & Tooling — C

The repo has a useful `.gitignore`, mypy config, compile-clean Python files, and a substantial pytest suite. The weak points are missing project metadata, missing CI, missing formatting/lint commands, no one-command setup/test path, and a detached worktree with no local environment that can run tests out of the box.

#### I1 — Add a one-command development workflow
- **Where:** missing `Makefile`/`justfile`/`noxfile.py`, `requirements.txt:1-24`
- **What's wrong:** There is no canonical command for setup, test, typecheck, lint, run web mode, or run headless smoke training.
- **Impact:** Major — every contributor/agent must rediscover the workflow.
- **Fix:** Add a `justfile` or `Makefile` with `setup`, `test`, `test-fast`, `typecheck`, `lint`, `format`, `run-web`, and `smoke-train` commands.
- **Effort:** S
- **Grade lift:** C → C+ (immediate local velocity improvement)

#### I2 — Add CI for compile, tests, lint, and typecheck
- **Where:** missing `.github/workflows/*`, `mypy.ini:1-17`
- **What's wrong:** No CI workflow exists to prove the suite or enforce formatting/type checks.
- **Impact:** Major — regressions can land silently.
- **Fix:** Add GitHub Actions with Python setup, dependency install, `python -m compileall`, pytest, mypy, and formatter/linter checks.
- **Effort:** M
- **Grade lift:** C → B- (turns local tooling into a gate)

#### I3 — Adopt a real formatter/linter configuration
- **Where:** `requirements.txt:21-24`, missing `ruff.toml`/`pyproject.toml` formatter config
- **What's wrong:** Black and mypy are listed, but there is no Black configuration, Ruff/flake8 config, or documented lint command.
- **Impact:** Moderate — style and basic static checks depend on individual habits.
- **Fix:** Add Ruff plus Black config in `pyproject.toml`, document commands, and run them in CI.
- **Effort:** S
- **Grade lift:** C → C+ (low-cost quality baseline)

#### I4 — Make optional dependency errors quieter and more actionable
- **Where:** `src/web/server.py:44-53`, `config.py:16`, tests importing `torch`/`pygame`
- **What's wrong:** Import-time dependency failures break test collection and print warnings before pytest can classify the missing subsystem.
- **Impact:** Moderate — onboarding feels broken before the user gets a clear setup instruction.
- **Fix:** Move optional imports behind fixtures or command paths, add dependency checks with precise install commands, and mark tests by dependency group.
- **Effort:** S
- **Grade lift:** C → C+ (better first-run experience)
