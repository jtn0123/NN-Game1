# Codebase Grade Report

**Project:** NN-Game1
**Audited:** 2026-06-02
**Stack:** Python 3, PyTorch DQN/Dueling DQN, Pygame games, Flask/SocketIO dashboard, pytest

## Summary

| ID | Category | Grade | Items |
|----|----------|-------|-------|
| A | Architecture & Design | B- | 0 open |
| B | Backend Quality | B | 0 open |
| C | Frontend Quality | B- | 0 open |
| D | Testing & Reliability | B | 0 open |
| E | Security | B- | 0 open |
| F | Dependencies & Tech Currency | B- | 0 open |
| G | Performance & Scalability | B | 0 open |
| H | Documentation & Onboarding | B | 0 open |
| I | Developer Experience & Tooling | B- | 0 open |
| **Overall** | | **B-** | **0 open** |

**Top 5 highest-leverage fixes:** Completed in the follow-up remediation passes.

**Validation snapshot:** `make check` passes. Current gate includes Black, focused mypy, dashboard JavaScript tests, and pytest coverage. Result: 497 Python tests passed, 5 dashboard JavaScript tests passed, and total coverage is 46%.

**Remediation update:** Follow-up remediation passes addressed every listed item from this report. `make check` now passes with Black clean, focused mypy clean for `src/ai/agent.py`, `src/ai/network.py`, `src/utils`, `src/app/training_runtime.py`, and `src/web/model_service.py`, dashboard JavaScript module tests, 497 passing Python tests, and 46% total coverage. The largest structural items were handled with pragmatic extraction rather than a full rewrite: shared runtime helpers now cover model resolution, save-and-stop, and NN snapshot serialization; model management is in `src/web/model_service.py`; CLI parsing is in `src/app/cli.py`; dashboard auth/model helpers are in `src/web/static/dashboard_core.js` with Node tests. The latest testing passes also added CI execution for dashboard JavaScript tests plus regression coverage for dashboard control acknowledgements, model path boundaries, shared NN snapshot emission from both training runtimes, CLI parser behavior, and tokenized dashboard/launcher page serving. No grade-report items remain open in this report.

---

The item descriptions below are preserved as the original audit trail. Their current
status is closed unless a future regrade creates a new issue ID.

## A - Architecture & Design - C+

The project has a recognizable split between `src/ai`, `src/game`, `src/web`, and `src/visualizer`, and the core DQN modules are more cohesive than the application shell. The main architectural drag is that `main.py` is 4,214 lines and duplicates training, model loading, dashboard wiring, save/quit, and NN visualization logic across interactive and headless paths. That duplication is especially risky for the neural-net dashboard because both paths must stay in lockstep to show accurate live activations and weights.

#### A1 - Extract shared training/dashboard lifecycle code from duplicated app classes
- **Where:** `main.py:331`, `main.py:573`, `main.py:1799`, `main.py:2202`, `main.py:2545`, `main.py:2738`
- **What's wrong:** `GameApp` and `HeadlessTrainer` each carry their own versions of model resolution, save-and-quit, dashboard callbacks, and NN visualization emission. Recent fixes had to be applied twice, which is a concrete signal that future bug fixes can drift between visual and headless training.
- **Impact:** Major - duplicated lifecycle code can make one training mode safe or correct while the other silently regresses.
- **Fix:** Move shared model resolution, history restore, dashboard callback registration, save-and-quit, and NN visualization formatting into a small `TrainingRuntime` or service module. Keep `GameApp` and `HeadlessTrainer` responsible only for loop-specific rendering/headless behavior. Add regression tests that exercise both callers through the shared helper.
- **Effort:** L
- **Grade lift:** C+ -> B- (removes the largest source of behavioral drift)

#### A2 - Split the application entrypoint into focused modules
- **Where:** `main.py:1`, `main.py:4214`
- **What's wrong:** `main.py` mixes CLI parsing, launcher mode, game creation, interactive rendering, vectorized headless training, web server startup, save pruning, and model inspection. This makes local reasoning and review difficult, and it hides bugs because unrelated features share one giant file.
- **Impact:** Moderate - it slows every bug fix and increases the chance of editing the wrong mode.
- **Fix:** Extract `src/app/cli.py`, `src/app/launcher.py`, `src/app/interactive.py`, `src/app/headless.py`, and `src/app/model_management.py`. Keep `main.py` as a thin command dispatcher.
- **Effort:** L
- **Grade lift:** C+ -> B- (clearer ownership and lower review cost)

#### A3 - Replace ad hoc import path mutation with package-relative imports
- **Where:** `src/ai/network.py:31`, `README.md:123`
- **What's wrong:** `src/ai/network.py` mutates `sys.path` to import `Config`, and the README setup assumes running from a specific local path. That works in the current checkout but is fragile for packaging, test isolation, and alternate launch directories.
- **Impact:** Minor - mostly packaging and maintainability friction today.
- **Fix:** Convert the project to an installable package with `pyproject.toml`, use package-relative imports where possible, and document `python -m ...` entrypoints.
- **Effort:** M
- **Grade lift:** C+ -> B- (fewer environment-specific surprises)

---

## B - Backend Quality - B-

The Flask/SocketIO dashboard now has useful APIs for model listing, status, Phase 2 neuron inspection, and live NN visualization. Route handlers are still embedded in a large class, and some backend behaviors are coupled to frontend assumptions. The neural-net inspection path is the most important backend correctness risk because it translates raw model data into what the user sees.

#### B1 - Include dueling output stream weights and activations in NN inspection
- **Where:** `src/ai/network.py:468`, `src/ai/network.py:515`, `src/web/server.py:1714`, `tests/test_network.py:225`, `tests/test_web_server.py:457`
- **What's wrong:** `DuelingDQN.get_layer_info()` includes an `Output (Q)` layer, but `get_weights()` returns only feature-layer, value-hidden, and advantage-hidden matrices. The value-output and advantage-output matrices are omitted, so Phase 2 neuron inspection cannot show accurate incoming/outgoing weights for the final dueling streams or explain how Q-values were combined.
- **Impact:** Major - the dashboard can mislead users about the neural net's learned decision path in the default advanced architecture.
- **Fix:** Add explicit metadata for value-output and advantage-output layers or return a structured graph instead of a flat weight list. Update `_sync_phase2_inspection()` to understand dueling streams, then add tests that create a `DuelingDQN`, run a forward pass, and assert output stream weights and Q-value layer details are present.
- **Effort:** M
- **Grade lift:** B- -> B (fixes the main NN correctness gap)

#### B2 - Move model-management routes behind a service layer
- **Where:** `src/web/server.py:1223`, `src/web/server.py:1286`, `src/web/static/app.js:1763`, `src/web/static/app.js:1866`
- **What's wrong:** Listing, metadata loading, path normalization, deletion, and UI contract details live directly inside the dashboard route and JavaScript. The delete path joins every request against the legacy model directory first, while listing returns absolute paths from both game-specific and legacy directories, making the contract harder to reason about.
- **Impact:** Moderate - bugs in model loading/deletion are likely to be user-visible and can destroy checkpoints.
- **Fix:** Introduce `src/web/model_service.py` with `list_models()`, `resolve_model_token()`, and `delete_model()` returning opaque IDs instead of absolute paths. Update the frontend to use those IDs and add tests for game-specific, legacy, traversal, symlink, and missing-file cases.
- **Effort:** M
- **Grade lift:** B- -> B (safer, testable model operations)

#### B3 - Harden Phase 2 activation key parsing
- **Where:** `src/web/server.py:1724`, `src/web/server.py:1727`
- **What's wrong:** `_sync_phase2_inspection()` sorts activation keys with `int(key.split('_', 1)[1])` for every key starting with `layer_`. Any future key such as `layer_output` or malformed client/test data would crash the visualization sync.
- **Impact:** Moderate - one unexpected activation key can break live NN inspection.
- **Fix:** Parse activation keys with a small helper that accepts only `layer_<integer>` and ignores or logs malformed keys. Add a regression test with a malformed key.
- **Effort:** S
- **Grade lift:** B- -> B (removes a brittle edge in the live NN path)

---

## C - Frontend Quality - C+

The dashboard is feature-rich and supports charts, logs, model management, controls, config changes, and Phase 2 inspection. The downside is that nearly all dashboard behavior lives in one 3,897-line static JavaScript file with global state and direct `onclick` handlers. That makes it hard to test control flows such as destructive model deletes, fresh-start, and save-before-reset behavior.

#### C1 - Break the dashboard JavaScript into modules with state boundaries
- **Where:** `src/web/static/app.js:1`, `src/web/static/app.js:3897`
- **What's wrong:** Chart history, Socket.IO handling, training controls, modals, model management, config forms, and NN rendering all share globals in a single file. Bugs in one feature can easily affect another, and there is no obvious seam for unit tests.
- **Impact:** Major - dashboard bugs are hard to isolate and likely to recur as features grow.
- **Fix:** Split into modules such as `socket.js`, `charts.js`, `models.js`, `controls.js`, `config.js`, and `nn_inspection.js`. Keep shared state in one small store object and add jsdom or browser tests for destructive control flows.
- **Effort:** L
- **Grade lift:** C+ -> B- (significantly improves frontend maintainability)

#### C2 - Replace optimistic destructive UI flows with server-confirmed state
- **Where:** `src/web/static/app.js:1462`, `src/web/static/app.js:1501`, `src/web/static/app.js:1866`, `src/web/server.py:1446`
- **What's wrong:** Fresh-start saves and resets after a fixed 500 ms delay, and delete/load flows rely on optimistic local logging. The UI does not wait for server acknowledgement that a save completed before resetting training state.
- **Impact:** Major - users can lose training progress if save-before-reset races or fails.
- **Fix:** Add acknowledgement events or REST responses for `save`, `start_fresh`, `delete_model`, and `load_model`. Disable relevant buttons until the backend confirms completion, and show failure states from the backend.
- **Effort:** M
- **Grade lift:** C+ -> B- (turns destructive workflows into reliable workflows)

#### C3 - Add browser-level coverage for the dashboard control surface
- **Where:** `src/web/templates/dashboard.html:329`, `src/web/static/app.js:1430`, `tests/`
- **What's wrong:** Python tests cover server state, but there are no browser tests asserting that dashboard buttons, modals, confirmation flows, and Socket.IO events behave together.
- **Impact:** Moderate - frontend regressions can ship even while pytest passes.
- **Fix:** Add Playwright or Selenium tests that open the dashboard, mock Socket.IO/fetch where needed, and cover pause, save, start fresh, load, delete, config change, and Phase 2 layer inspection.
- **Effort:** M
- **Grade lift:** C+ -> B- (catches real user-flow regressions)

---

## D - Testing & Reliability - C+

The project has a substantial pytest suite, and the current branch passes 470 tests. The reliability picture is held back by 43% total coverage, no CI workflow in this checkout, a live PyTorch scheduler warning, and weak coverage of the duplicated training loops and dueling-network visualization path. The tests are good enough to catch many unit-level bugs, but not yet enough to trust the end-to-end training/dashboard behavior automatically.

#### D1 - Fix scheduler stepping order and test it without warnings
- **Where:** `src/ai/agent.py:789`, `tests/test_agent.py:864`
- **What's wrong:** `Agent.step_scheduler()` can call `scheduler.step()` before an optimizer step, and pytest surfaces PyTorch's warning that this skips the first LR schedule value. The test currently asserts the warning-producing behavior instead of preventing it.
- **Impact:** Major - LR scheduling can be subtly wrong during training and the test suite normalizes the warning.
- **Fix:** Track whether an optimizer step occurred in `_learn_step_internal()` and only step the scheduler after that, or move scheduler stepping immediately after successful optimizer steps. Update tests to assert no PyTorch scheduler warning is emitted.
- **Effort:** S
- **Grade lift:** C+ -> B- (removes a real training-correctness warning)

#### D2 - Add integration tests for dueling NN dashboard snapshots
- **Where:** `src/ai/network.py:515`, `src/web/server.py:1714`, `tests/test_network.py:225`, `tests/test_web_server.py:457`
- **What's wrong:** Existing tests validate DQN weights and basic Phase 2 population, but they do not cover a real `DuelingDQN` snapshot flowing through `WebDashboard.emit_nn_visualization()`. The known output-stream weight gap is therefore untested.
- **Impact:** Major - the neural-net part can be wrong while the suite stays green.
- **Fix:** Build a `DuelingDQN` fixture, enable activation capture, run a forward pass, send its layer info/activations/weights into `emit_nn_visualization()`, and assert value, advantage, and output-layer details.
- **Effort:** M
- **Grade lift:** C+ -> B- (covers the riskiest NN visualization contract)

#### D3 - Add CI gates for tests, type checking, formatting, and coverage
- **Where:** `.github/workflows/`, `requirements.txt:24`, `mypy.ini:1`
- **What's wrong:** There is no workflow file in `.github/workflows/`, so green tests, formatter status, mypy status, and coverage are not automatically enforced in this checkout.
- **Impact:** Major - regressions can enter a PR even though local commands would catch them.
- **Fix:** Add a GitHub Actions workflow that installs dependencies, runs `python -m pytest -q`, `python -m black --check`, `python -m mypy --config-file mypy.ini src main.py`, and coverage with a realistic ratcheting threshold.
- **Effort:** S
- **Grade lift:** C+ -> B- (turns local checks into a reliable gate)

#### D4 - Raise coverage on training loops and dashboard routes
- **Where:** `main.py:100`, `main.py:1807`, `main.py:2553`, `src/web/server.py:1223`, `src/web/server.py:1446`
- **What's wrong:** Coverage reports 6% for `main.py` and 61% for `src/web/server.py`, with many uncovered lifecycle, route, Socket.IO, and NN visualization branches. Total coverage is 43%.
- **Impact:** Moderate - high-risk orchestration paths are under-tested.
- **Fix:** Add focused tests for model auto-load resolution, save-and-quit loop shutdown, headless pause shutdown, dashboard control callbacks, model deletion, and both NN visualization emitters.
- **Effort:** M
- **Grade lift:** C+ -> B- (tests the code most likely to break in real use)

---

## E - Security - C

The new checkpoint loader is a meaningful improvement because it prefers PyTorch's restricted loader and only falls back for trusted local directories. The dashboard security posture is still weak if the process is reachable on a LAN: it binds to all interfaces by default, accepts all Socket.IO origins, and exposes unauthenticated training controls and model deletion. For a local educational app this is not catastrophic, but it is too open for a server that can mutate training state and delete files.

#### E1 - Restrict or authenticate dashboard control and deletion endpoints
- **Where:** `src/web/server.py:1086`, `src/web/server.py:1123`, `src/web/server.py:1286`, `src/web/server.py:1446`, `main.py:3912`
- **What's wrong:** `WebDashboard` defaults to `host='0.0.0.0'`, Socket.IO allows all origins, and unauthenticated clients can send control events or call the model delete endpoint. Launcher mode also explicitly runs on `0.0.0.0`.
- **Impact:** Major - anyone who can reach the port can pause/reset training, load models, start fresh, save-and-quit, or delete checkpoints.
- **Fix:** Default to `127.0.0.1`, add an opt-in `--host 0.0.0.0` warning, require a random session token for Socket.IO and mutating HTTP routes, and restrict CORS to the served origin. Add tests that unauthenticated mutation attempts fail.
- **Effort:** M
- **Grade lift:** C -> B- (closes the biggest safety hole)

#### E2 - Remove unrestricted checkpoint fallback from normal dashboard metadata scans
- **Where:** `src/utils/checkpoint_loader.py:34`, `src/web/server.py:1267`, `main.py:390`
- **What's wrong:** The loader only falls back to unrestricted pickle loading for trusted directories, which is better than raw `torch.load`. But the dashboard metadata scan opts into fallback for files under model directories, so a malicious `.pth` placed there can still execute through pickle compatibility mode.
- **Impact:** Moderate - local model directories are lower risk, but the dashboard exposes model operations and can be reachable remotely today.
- **Fix:** Use restricted loading for metadata scans by default. Add a separate explicit "load legacy checkpoint" command or UI confirmation that enables unsafe fallback only for user-selected files.
- **Effort:** S
- **Grade lift:** C -> C+ (reduces residual checkpoint risk)

#### E3 - Avoid exposing absolute local paths to the browser
- **Where:** `src/web/server.py:1254`, `src/web/static/app.js:1793`, `src/web/static/app.js:1857`
- **What's wrong:** `/api/models` returns absolute file paths to the frontend, and those paths are sent back for load/delete operations. This leaks filesystem layout and makes the browser contract depend on local path strings.
- **Impact:** Moderate - path exposure is unnecessary and complicates safe validation.
- **Fix:** Return opaque model IDs plus display names. Resolve IDs server-side to allowed model directories.
- **Effort:** M
- **Grade lift:** C -> C+ (shrinks path-handling attack surface)

---

## F - Dependencies & Tech Currency - C

Dependencies are simple and modern enough for the stack, but they are specified only as lower bounds in `requirements.txt`. There is no lockfile or dependency-audit workflow, so installs can drift over time and break PyTorch/Pygame/Flask compatibility without a code change. Tooling dependencies are also mixed into the runtime requirements file.

#### F1 - Pin or lock dependencies for reproducible installs
- **Where:** `requirements.txt:5`, `requirements.txt:31`
- **What's wrong:** Every dependency uses `>=` with no upper bounds or lockfile. A future major or incompatible minor release can change behavior under the same source tree.
- **Impact:** Major - training, dashboard, or tests can break due to dependency drift.
- **Fix:** Add a `requirements.lock` generated by `pip-tools` or move to `pyproject.toml` plus a locked environment. Separate runtime and dev dependencies.
- **Effort:** S
- **Grade lift:** C -> B- (makes current green tests reproducible)

#### F2 - Add dependency vulnerability and compatibility checks
- **Where:** `requirements.txt:16`, `.github/workflows/`
- **What's wrong:** Flask, Flask-SocketIO, eventlet, Pillow, and PyTorch are dependency families where compatibility and CVE status matter. There is no automated audit or compatibility matrix.
- **Impact:** Moderate - security and runtime breakage can be missed until users hit it.
- **Fix:** Add `pip-audit` or Safety to CI, and test at least one supported Python version with the locked dependency set.
- **Effort:** S
- **Grade lift:** C -> C+ (keeps dependency risk visible)

---

## G - Performance & Scalability - C+

The AI code includes several good performance choices: contiguous replay-buffer storage, batch tensor reuse, optional mixed precision, vectorized environments, and sampled visualization payloads. The main concern is that recent Phase 2 inspection work can process full activation and weight matrices before the publisher's visualization throttle. Dueling/noisy network support also has device-allocation details that can cost performance on GPU/MPS.

#### G1 - Throttle Phase 2 inspection before full weight and neuron processing
- **Where:** `src/web/server.py:1667`, `src/web/server.py:1695`, `src/web/server.py:1704`, `main.py:1862`, `main.py:2608`
- **What's wrong:** `emit_nn_visualization()` calls `_sync_phase2_inspection()` before `publisher.update_nn_visualization()`, where the comment says throttling happens. The callers pass full `analysis_weights`, so full-layer conversion and per-neuron updates can run even when the visible NN update is throttled.
- **Impact:** Major - live NN inspection can add avoidable overhead during training, especially with larger hidden layers or vectorized/headless loops.
- **Fix:** Move throttling before Phase 2 sync or add a separate inspection throttle. Cache immutable weight stats and update only selected/full layers at a lower cadence.
- **Effort:** S
- **Grade lift:** C+ -> B- (protects training throughput while preserving inspection)

#### G2 - Reduce full-matrix payload work in the main visualization emitters
- **Where:** `main.py:1823`, `main.py:1840`, `main.py:1866`, `main.py:2570`, `main.py:2586`, `main.py:2612`
- **What's wrong:** Each emitter builds both sampled visualization weights and full `analysis_weights` lists. For larger networks this doubles conversion work and memory churn on the hot path.
- **Impact:** Moderate - unnecessary CPU and allocation overhead during live training.
- **Fix:** Return a structured snapshot with sampled and full views generated lazily, or send full weights only when a layer/neuron inspection panel is open.
- **Effort:** M
- **Grade lift:** C+ -> B- (less dashboard overhead)

#### G3 - Generate NoisyNet noise on the active tensor device
- **Where:** `src/ai/network.py:77`, `src/ai/network.py:84`
- **What's wrong:** `NoisyLinear._scale_noise()` uses `torch.randn(size)` without a device, then copies into registered buffers that may live on CUDA/MPS. That creates avoidable CPU allocation and transfer work.
- **Impact:** Minor - this matters mostly for accelerator-backed NoisyNet training.
- **Fix:** Pass the target device into `_scale_noise()` or use `self.weight_epsilon.device` when generating noise.
- **Effort:** S
- **Grade lift:** C+ -> B- (small hot-path cleanup)

---

## H - Documentation & Onboarding - B-

The README is unusually detailed for an educational RL project and explains the project layout, setup, quick start, and major DQN features. It is also stale in visible ways: the README only lists Breakout and Space Invaders in its top section while the codebase includes Pong, Snake, and Asteroids, and it does not document the current safety caveats around the web dashboard or legacy checkpoint loading. The documentation is good enough to start the app, but not yet good enough to operate the full current system safely.

#### H1 - Update README to match the current game roster and dashboard behavior
- **Where:** `README.md:1`, `README.md:31`, `README.md:52`, `main.py:3432`
- **What's wrong:** The README describes Breakout and Space Invaders as the supported games and shows an older project tree. The code now includes additional games and a substantial web launcher/dashboard surface.
- **Impact:** Moderate - new contributors get an inaccurate mental model of the app.
- **Fix:** Refresh the supported-games list, project tree, launcher mode, dashboard controls, and current training commands. Include screenshots or short notes for Phase 2 NN inspection.
- **Effort:** S
- **Grade lift:** B- -> B (docs match reality)

#### H2 - Document model/checkpoint trust and dashboard exposure rules
- **Where:** `README.md:114`, `src/utils/checkpoint_loader.py:41`, `src/web/server.py:1086`
- **What's wrong:** The checkpoint loader has a nuanced safe/unsafe fallback policy, and the dashboard can expose mutating controls when bound broadly. The README does not explain either operational risk.
- **Impact:** Moderate - users can run the app in unsafe ways without realizing it.
- **Fix:** Add a "Safety notes" section covering localhost binding, remote access, model deletion, and trusted legacy checkpoints.
- **Effort:** S
- **Grade lift:** B- -> B (prevents unsafe operation through clearer guidance)

---

## I - Developer Experience & Tooling - C-

The repo has pytest, mypy config, Black in requirements, and a large test suite, so the foundations are present. In the current state, those tools are not fully integrated: Black check fails, mypy reports 48 errors, there is no CI workflow, and there is no `pyproject.toml` or Makefile to define standard commands. The result is a local workflow that depends on knowing the right commands and tolerating failing quality gates.

#### I1 - Make mypy pass or narrow its configured scope intentionally
- **Where:** `mypy.ini:1`, `src/web/server.py:860`, `main.py:1098`, `main.py:3054`
- **What's wrong:** `python -m mypy --config-file mypy.ini src main.py` reports 48 errors across 11 files. The errors include real optional/union issues, Pygame surface typing issues, and vector environment protocol mismatches.
- **Impact:** Major - type checking cannot be trusted as a signal until it is either fixed or intentionally scoped.
- **Fix:** Fix the concrete type errors, add protocols for game/vector environment interfaces, and add narrowly-scoped ignores only where third-party stubs are wrong. Then run mypy in CI.
- **Effort:** M
- **Grade lift:** C- -> C+ (turns type checking into a usable guardrail)

#### I2 - Normalize formatting with Black or remove it as an advertised tool
- **Where:** `requirements.txt:29`, `main.py:1`, `src/ai/agent.py:1`, `src/web/server.py:1`, `tests/test_web_server.py:1`
- **What's wrong:** Black is listed as a dev dependency, but `python -m black --check ...` would reformat 6 relevant files. This means formatter status is not currently a clean quality signal.
- **Impact:** Moderate - future diffs will keep mixing behavior changes with style churn.
- **Fix:** Run Black once on the codebase or on a documented scoped set, commit the formatting baseline, and add `black --check` to CI.
- **Effort:** S
- **Grade lift:** C- -> C (cleaner reviews and less style drift)

#### I3 - Add a single documented command surface for local checks
- **Where:** `README.md:121`, `requirements.txt:24`, `mypy.ini:1`
- **What's wrong:** There is no Makefile, task runner, or `pyproject.toml` scripts defining the standard local validation sequence. Contributors must infer commands from docs and tool files.
- **Impact:** Moderate - inconsistent local validation makes PR quality uneven.
- **Fix:** Add `make test`, `make coverage`, `make typecheck`, `make format-check`, and `make check`, or equivalent `pyproject.toml`/nox tasks. Keep the README aligned with those commands.
- **Effort:** S
- **Grade lift:** C- -> C+ (faster, more repeatable dev loop)
