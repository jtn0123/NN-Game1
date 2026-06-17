# Codebase Grade Report

**Project:** nn-game1
**Audited:** 2026-06-16
**Scope:** Whole repository, current `origin/main` after PR #20 and release `0.0.4`
**Stack:** Python 3.10-3.12, PyTorch DQN/Dueling DQN, Pygame games, Flask/Socket.IO dashboard, vanilla JS dashboard, pytest, Node test runner, GitHub Actions

## Summary

| ID | Category | Grade | Items |
|----|----------|-------|-------|
| A | Architecture & Design | B+ | 4 |
| B | Backend Quality | B+ | 4 |
| C | Frontend Quality | B | 4 |
| D | Testing & Reliability | A- | 4 |
| E | Security | B | 4 |
| F | Dependencies & Tech Currency | B | 3 |
| G | Performance & Scalability | B+ | 3 |
| H | Documentation & Onboarding | B | 3 |
| I | Developer Experience & Tooling | A- | 3 |
| **Overall** | | **B+** | **32** |

**Top 5 highest-leverage remaining fixes:** E1, C4, B2, D2, A3

## Validation Snapshot

- PR #20 merged and released as `0.0.4`.
- CI on PR #20 passed: Python 3.11, Python 3.12, dashboard JS tests, coverage, mypy, format check, build, dependency audit, dependency review, CodeQL, PR title lint, and CodeRabbit.
- Local pre-merge validation included `make verify`; package build was skipped locally only when `build` was not installed, while CI installed `build` and ran `make build`.
- Current largest files after the expanded refactor pass: `src/web/static/styles.css` 3,290 lines, `src/web/static/app.js` 3,206 lines, `src/app/interactive.py` 1,991 lines, `src/game/space_invaders.py` 1,942 lines, `src/app/headless.py` 1,397 lines.
- Current dependency audit passes with a documented narrow ignore for `CVE-2025-3000`; GitHub Advisory Database currently lists patched versions for that torch advisory as `None`.
- 2026-06-16 follow-up for Architecture, Backend, Frontend, and Performance: added game constructor protocols, extracted socket control routing into `src/web/socket_controls.py`, centralized API error responses, moved chart update modeling into `dashboard_core.js`, and added replay-buffer performance coverage. Validation: `make check` passed.
- 2026-06-16 expanded refactor pass: split `main.py` into `src/app/interactive.py`, `src/app/headless.py`, lifecycle types, and process control; split dashboard routes and metrics publisher out of `src/web/server.py`; extracted chart orchestration into `src/web/static/dashboard_charts.js`; added a repeatable Playwright dashboard smoke command; extracted Space Invaders scoring/difficulty rules into pure helpers with direct tests.

---

## A - Architecture & Design - B+

The architecture is now materially cleaner than the previous grade: game construction moved into `src/app/game_factory.py`, performance presets moved into `src/app/performance_modes.py`, shared web contracts moved into `src/web/contracts.py`, lifecycle classes moved out of `main.py`, and dashboard routes/metrics moved out of `src/web/server.py`. The remaining concentration points are mostly frontend app behavior and per-game domain modules. Overall this is a B+: modular seams exist, CI protects them, and the biggest backend/lifecycle ownership problems have been reduced.

#### ~~A1 - Split the application lifecycle shell~~ ✓ done 2026-06-16
- **Where:** `src/app/interactive.py`, `src/app/headless.py`, `src/app/lifecycle_types.py`, `src/app/process_control.py`, `main.py`.
- **What changed:** `GameApp` and `HeadlessTrainer` now live in mode-specific modules, shared lifecycle types have a dedicated module, restart/process logic is isolated, and `main.py` is back to CLI entrypoint/wiring responsibilities.
- **Impact:** Major - visual, headless, and web-mode lifecycle fixes now have a smaller blast radius.
- **Fix:** Completed with compatibility imports in `main.py` and updated lifecycle tests.
- **Effort:** L
- **Grade lift:** B -> B+ by removing the largest cross-mode coupling point.

#### ~~A2 - Split dashboard server responsibilities~~ ✓ done 2026-06-16
- **Where:** `src/web/routes.py`, `src/web/socket_controls.py`, `src/web/metrics_publisher.py`, `src/web/server.py`.
- **What changed:** Route registration, API error helpers, socket control routing, and metrics publisher/data classes now have dedicated modules. `WebDashboard` remains the composition boundary and compatibility exports preserve existing imports.
- **Impact:** Major - web control, security, and metrics changes are now easier to review independently.
- **Fix:** Completed with route/socket/publisher tests and compatibility wrappers in `src/web/server.py`.
- **Effort:** L
- **Grade lift:** B -> B+ by making dashboard server behavior locally testable and easier to review.

#### A3 - Decompose large game modules
- **Where:** `src/game/space_invaders.py` 1,931 lines, `src/game/asteroids.py` 1,052 lines, `src/game/breakout.py` 965 lines, `src/game/snake.py` 737 lines.
- **What's wrong:** Game rules, collision logic, reward shaping, state encoding, rendering, and human input are still mixed in large per-game modules.
- **Impact:** Moderate - game changes remain high-context and shared improvements are difficult to reuse.
- **Fix:** Start with Space Invaders. Extract pure modules for entities, collision/reward logic, state encoding, and rendering adapters. Preserve the public game class API and migrate tests incrementally.
- **Effort:** L
- **Grade lift:** B -> B+ by reducing local complexity in the largest domain modules.
- **Status:** Partially addressed 2026-06-16 - Space Invaders scoring, pressure, level-speed, and invasion rules now live in `src/game/space_invaders_rules.py` with direct tests. Entity/rendering/state encoding extraction remains.

#### ~~A4 - Strengthen constructor and runtime protocols~~ ✓ done 2026-06-16
- **Where:** `src/app/game_factory.py:35`, `src/app/game_factory.py:48`, `src/app/game_factory.py:63`; `src/game/base_game.py:1`.
- **What's wrong:** `game_factory.py` still needs `type: ignore[call-arg]` because registered game constructors are not expressed through a precise protocol. The registry works, but type checking cannot prove all game classes support the same construction contract.
- **Impact:** Moderate - future games can drift from runtime expectations without immediate type feedback.
- **Fix:** Add constructor protocols for single-game and vector-game classes, update registry return types, and remove the `type: ignore[call-arg]` comments from `game_factory.py`.
- **Effort:** M
- **Grade lift:** B -> B+ by making game runtime boundaries enforceable.

---

## B - Backend Quality - B+

Backend quality is now stronger: dashboard control acknowledgements are explicit, API routes require dashboard tokens, shared contracts exist, route registration is separated, metrics publishing has its own module, and model/game helper services are better isolated. The main remaining limitation is that model/checkpoint ownership is still split between app and web adapters, and some API payload shapes are still informal dictionaries.

#### ~~B1 - Extract socket control handlers into a dedicated module~~ ✓ done 2026-06-16
- **Where:** `src/web/socket_controls.py`, `src/web/server.py`, `tests/test_socket_controls.py`, `tests/test_web_socket_controls.py`.
- **What changed:** Socket control behavior now has one command function per action plus shared ack helpers. `WebDashboard` delegates to the extracted router.
- **Impact:** Major - destructive control workflows are user-visible and now live in smaller, testable units.
- **Fix:** Completed with direct socket-control unit tests and existing Socket.IO integration tests.
- **Effort:** M
- **Grade lift:** B -> B+ by reducing risk in the most stateful backend surface.

#### B2 - Consolidate model service boundaries
- **Where:** `src/app/model_service.py:1`, `src/web/model_service.py:1`, `main.py` model save/load call sites.
- **What's wrong:** Model resolution and checkpoint management are improved, but app-level and web-facing services still split responsibilities. Listing, deletion, loading, history, and serialization are not yet one cohesive contract.
- **Impact:** Moderate - model behavior can still drift between CLI/headless and web dashboard paths.
- **Fix:** Create one app-level checkpoint/model service with web serialization adapters. Keep filesystem safety and opaque IDs in the shared service, then make `src/web/model_service.py` a thin adapter or remove it.
- **Effort:** M
- **Grade lift:** B -> B+ by giving checkpoint behavior one owner.

#### B3 - Make metrics and inspection payloads fully typed
- **Where:** `src/web/contracts.py:1`, `src/web/server.py:1707`, `src/web/server.py:1713`, `src/web/server.py:1719`.
- **What's wrong:** `ControlAck` and game payloads are typed now, but metric, neuron, layer, and chart payloads still rely on informal dictionary shapes.
- **Impact:** Moderate - browser/backend payload drift can still break dashboard panels without type feedback.
- **Fix:** Add `TypedDict` or dataclass payloads for status snapshots, metrics history, layer inspection, neuron inspection, and save events. Use those types in publisher methods and route tests.
- **Effort:** M
- **Grade lift:** B -> B+ by extending the new contract pattern to all dashboard payloads.

#### B4 - Normalize error responses across read APIs
- **Where:** `src/web/server.py:1539`, `tests/test_web_routes.py:158`.
- **What's wrong:** Auth failures now have a stable `{"error": "Unauthorized"}` shape, but not every read endpoint has a documented error schema for malformed state, model lookup errors, or unavailable inspection data.
- **Impact:** Minor - current behavior works, but frontend error handling still depends on endpoint-specific assumptions.
- **Fix:** Add a small response helper for API errors and update route tests to assert error shapes for model, inspection, and config failures.
- **Effort:** S
- **Grade lift:** B -> B+ by making browser error states easier to handle consistently.
- **Status:** Partially addressed 2026-06-16 - `src/web/routes.py` now owns the shared `api_error()` helper and existing protected API/model-delete failures use the stable shape; broader read-endpoint error-schema tests remain.

---

## C - Frontend Quality - B

Frontend quality improved from C+ to B. The launcher is now a static JS/CSS app, dashboard controls wait for server acknowledgements, startup dependency failures degrade visibly, JS tests cover dashboard core/startup fallbacks, chart behavior has a dedicated module, and Playwright covers a real dashboard smoke workflow. The limiting factor is still the large `app.js` script, which owns models, settings, screenshots, comparison stats, logs, and neural-network visualization.

#### C1 - Modularize the dashboard app shell
- **Where:** `src/web/static/app.js:1`; startup at `src/web/static/app.js:228`; file length is 4,099 lines.
- **What's wrong:** Charts, sockets, controls, model saves, settings, screenshot polling, comparison stats, logs, and neural-network visualization share one global script.
- **Impact:** Major - the main user-facing surface remains expensive to reason about and easy to regress.
- **Fix:** Split into modules: `dashboard_boot.js`, `socket_client.js`, `controls.js`, `charts.js`, `models.js`, `screenshots.js`, `game_stats.js`, and `nn_visualizer.js`. Keep `dashboard_core.js` as shared utilities.
- **Effort:** L
- **Grade lift:** B- -> B+ by removing the largest frontend maintainability blocker.
- **Status:** Partially addressed 2026-06-16 - chart orchestration moved to `src/web/static/dashboard_charts.js`; remaining controls/models/screenshots/game-stats/NN visualization still live in `app.js`.

#### ~~C2 - Extract chart setup and update logic~~ ✓ done 2026-06-16
- **Where:** `src/web/static/dashboard_charts.js`, `src/web/static/dashboard_core.js`, `src/web/static/app.js`, `tests/js/dashboard_core.test.mjs`, `tests/js/dashboard_startup.test.mjs`.
- **What changed:** Chart setup/update orchestration moved into `DashboardCharts`; running-average and chart update modeling live in `DashboardCore`; `app.js` now delegates through compatibility wrappers.
- **Impact:** Moderate - chart behavior is localized and large-history update logic is covered by Node tests.
- **Fix:** Completed for the chart subsystem; broader dashboard app-shell modularization remains under C1.
- **Effort:** M
- **Grade lift:** B- -> B by isolating a complex dashboard subsystem.

#### ~~C3 - Add real browser workflow tests~~ ✓ done 2026-06-16
- **Where:** `tests/e2e/dashboard_smoke.mjs`, `tests/e2e/dashboard_smoke_server.py`, `Makefile`, `package.json`.
- **What changed:** A Playwright smoke starts a disposable `WebDashboard`, opens the tokenized URL, verifies Socket.IO connection, clicks Save, opens/closes the model modal, asserts the current game, and fails on browser/script errors.
- **Impact:** Major - the dashboard now has automated coverage for the real browser path.
- **Fix:** Completed as `make dashboard-smoke`; broader launcher and inspection-panel browser coverage can be added later.
- **Effort:** M
- **Grade lift:** B- -> B+ by covering the actual user surface.

#### C4 - Replace remaining dynamic HTML templates with DOM builders
- **Where:** `src/web/static/app.js` dynamic model, stats, and neural inspection rendering paths.
- **What's wrong:** Console rendering is safer and helper escaping exists, but several dynamic UI panels still assemble HTML strings from payloads.
- **Impact:** Moderate - it is easy for future payload fields to bypass escaping or break markup.
- **Fix:** Continue the PR #20 pattern: render dynamic rows and panels with `document.createElement` and `textContent`, and add hostile-string tests for model names, game names, layer labels, and inspection data.
- **Effort:** M
- **Grade lift:** B- -> B by making dynamic rendering consistently safe and testable.

---

## D - Testing & Reliability - A-

Testing and reliability are now strong. CI runs Python 3.11 and 3.12, Black, mypy, dashboard JS tests, coverage, build, dependency audit, dependency review, and CodeQL. Coverage is gated at 70%, PR #20 added regression tests for dashboard startup/launcher/game factory/socket acknowledgements, and the expanded pass added an explicit Playwright dashboard smoke plus focused performance and rule-helper tests. The remaining gap is targeted coverage for omitted manual UI/runtime shells.

#### ~~D1 - Add automated browser smoke tests for dashboard workflows~~ ✓ done 2026-06-16
- **Where:** `tests/e2e/dashboard_smoke.mjs`, `tests/e2e/dashboard_smoke_server.py`, `Makefile`.
- **What changed:** The dashboard browser smoke is now repeatable locally through `make dashboard-smoke`.
- **Impact:** Major - the connected dashboard/save/model-modal workflow no longer relies on manual browser smoke.
- **Fix:** Completed as a separate explicit command so browser availability does not make unit verification flaky.
- **Effort:** M
- **Grade lift:** B+ -> A- by covering the actual frontend/runtime integration path.

#### D2 - Add focused tests for omitted runtime shells
- **Where:** coverage omits `main.py`, `src/app/interactive.py`, `src/app/headless.py`, `src/app/process_control.py`, `src/game/menu.py`, and `src/visualizer/pause_menu.py` in `pyproject.toml`.
- **What's wrong:** The coverage gate is meaningful, but the manual app shell and UI overlays remain excluded from measured coverage.
- **Impact:** Moderate - important launch and UI behavior still needs manual reasoning.
- **Fix:** Add tests around app-mode selection, dashboard callback wiring, pause menu state transitions, and menu command behavior. Revisit coverage omissions once the shell is decomposed.
- **Effort:** M
- **Grade lift:** B+ -> A- by shrinking the unmeasured runtime surface.

#### D3 - Consolidate large test fixtures
- **Where:** `tests/test_replay_buffer.py` 898 lines, `tests/test_agent.py` 892 lines, `tests/test_web_socket_controls.py` 478 lines, `tests/test_web_routes.py` 409 lines.
- **What's wrong:** Coverage is broad, but several test files are large and repeat setup patterns.
- **Impact:** Minor - test maintenance cost rises as coverage grows.
- **Fix:** Extract shared builders for agents, dashboard clients, fake callbacks, and games into focused fixtures. Keep assertions local to each test module.
- **Effort:** S
- **Grade lift:** B+ -> A- by keeping a growing test suite readable.

#### D4 - Add perf regression tests for hot training paths
- **Where:** `main.py:3015`, `src/ai/trainer.py:1`, `src/ai/replay_buffer.py:1`.
- **What's wrong:** Functional coverage is strong, but the highest-throughput training loops do not have lightweight regression budgets.
- **Impact:** Moderate - future refactors can silently reduce training throughput.
- **Fix:** Add non-flaky microbenchmarks or budgeted smoke tests for replay sampling, vectorized stepping, and dashboard payload generation. Keep thresholds loose enough for CI variability.
- **Effort:** M
- **Grade lift:** B+ -> A- by protecting performance-sensitive correctness.

---

## E - Security - B

Security is materially better than the earlier report. The dashboard requires tokens for pages, APIs, and Socket.IO controls; CSP and referrer policies are tested; CodeQL passes; dependency audit runs; dynamic rendering has safer helpers. The main issues are token-in-URL bootstrap, remaining dynamic HTML surfaces, and the documented torch advisory with no patched release.

#### E1 - Replace URL token bootstrap with a less leak-prone flow
- **Where:** dashboard URL/token behavior in `src/web/server.py`, route auth tests in `tests/test_web_routes.py`.
- **What's wrong:** The local dashboard still bootstraps through a `?token=...` URL. Referrer policy and local binding reduce risk, but address-bar tokens are still easier to leak through copy/paste, history, or logs.
- **Impact:** Major - this is the most visible remaining auth hygiene issue.
- **Fix:** Use a short-lived one-time bootstrap token that exchanges into an HttpOnly same-site cookie for browser sessions. Keep header/token auth for API tests and CLI-generated URLs during transition.
- **Effort:** M
- **Grade lift:** B -> B+ by removing the largest token exposure path.

#### E2 - Remove inline script/style requirements from CSP
- **Where:** `src/web/server.py:42`, `src/web/templates/dashboard.html:1`, `src/web/templates/launcher.html:1`.
- **What's wrong:** CSP is now explicit and tested, but `script-src` and `style-src` still include `'unsafe-inline'`.
- **Impact:** Moderate - inline allowances reduce CSP value if any dynamic rendering bug appears.
- **Fix:** Move the remaining inline token assignment into a data attribute or JSON endpoint, move inline styles out of templates, then remove `'unsafe-inline'` from CSP. Update `parse_csp` tests accordingly.
- **Effort:** M
- **Grade lift:** B -> B+ by making CSP a stronger browser defense.

#### E3 - Finish dynamic rendering hardening
- **Where:** `src/web/static/app.js` dynamic model, stats, and neural inspection panels.
- **What's wrong:** Some dynamic browser rendering still uses string templates. Existing escaping is good, but the pattern is not uniform.
- **Impact:** Moderate - future untrusted fields could become XSS or markup-breakage paths.
- **Fix:** Prefer DOM builders for all dynamic rendering and add hostile payload tests around every API-fed panel.
- **Effort:** M
- **Grade lift:** B -> B+ by closing the remaining browser injection surface.

#### E4 - Track the torch no-fix advisory explicitly
- **Where:** `Makefile:26`, `requirements.txt:1`, `constraints.txt:1`.
- **What's wrong:** Dependency audit passes with a narrow ignore for `CVE-2025-3000` because no patched torch release is currently available. This is documented, but it needs periodic review.
- **Impact:** Moderate - ignored advisories can become stale once a patched release exists.
- **Fix:** Add a short dated note in dependency docs or CI output linking the advisory and requiring review when torch releases a patched version. Remove the ignore immediately once a fixed version is available and compatible.
- **Effort:** S
- **Grade lift:** B -> B+ by making the one accepted dependency risk auditable.

---

## F - Dependencies & Tech Currency - B

Dependency hygiene is solid: constraints are present, Dependabot is configured, dependency review runs, and `make audit` is a CI gate. The one caveat is the torch advisory with no fixed release, which is documented and narrowly ignored. Packaging and release automation are current enough for a B, with room to improve lock/refresh workflows.

#### F1 - Add a repeatable constraints refresh workflow
- **Where:** `requirements.txt:1`, `constraints.txt:1`, `pyproject.toml:1`.
- **What's wrong:** The repo has both declared ranges and tested pins, but no command documents how to regenerate `constraints.txt`.
- **Impact:** Moderate - dependency updates can become manual and inconsistent.
- **Fix:** Add a `make constraints` or documented pip-compile workflow that refreshes constraints intentionally, then run `make verify`.
- **Effort:** S
- **Grade lift:** B -> B+ by making dependency maintenance repeatable.

#### F2 - Review torch advisory ignore on a schedule
- **Where:** `Makefile:26`.
- **What's wrong:** `CVE-2025-3000` is ignored because there is no patched torch release right now. The repo needs a reminder to remove the ignore when a fix appears.
- **Impact:** Moderate - accepted dependency risk should not become invisible.
- **Fix:** Add a dated comment in `Makefile` or a dependency note that says when this was reviewed and what release condition removes it. Consider a weekly Dependabot/security review note.
- **Effort:** S
- **Grade lift:** B -> B+ by improving governance around the one dependency exception.

#### F3 - Keep release metadata and docs in sync
- **Where:** `pyproject.toml:1`, `VERSION:1`, `CHANGELOG.md:1`, `README.md:1`.
- **What's wrong:** Release automation updates version metadata and changelog, but README architecture text still describes an older simplified layout and does not include newer `src/app` and `src/web` modules.
- **Impact:** Minor - setup works, but docs can lag behind dependency/runtime structure.
- **Fix:** Update README project structure after significant module moves and include `src/app`, `src/web`, CI, and constraints workflow.
- **Effort:** S
- **Grade lift:** B -> B+ by keeping package and onboarding metadata aligned.

---

## G - Performance & Scalability - B+

The project has good performance-oriented design for a local RL app: vectorized environments exist, replay buffer tests are broad, dashboard startup now fails gracefully, large-history chart modeling is tested, and Space Invaders hot-path formulas are isolated for profiling/tuning. Performance is still not broadly benchmarked in CI, and some large UI/training paths remain difficult to optimize safely.

#### G1 - Add lightweight training-loop performance budgets
- **Where:** `main.py:3015`, `src/ai/trainer.py:1`, `src/ai/replay_buffer.py:1`.
- **What's wrong:** Training hot paths have functional tests but no regression budget.
- **Impact:** Moderate - throughput regressions can slip through while tests still pass.
- **Fix:** Add an optional benchmark or smoke budget for replay sampling and vectorized stepping. Keep it non-flaky and record results as CI artifacts or local docs.
- **Effort:** M
- **Grade lift:** B -> B+ by protecting the highest-value performance path.
- **Status:** Partially addressed 2026-06-16 - added a loose replay-buffer `sample_no_copy` budget test; vectorized stepping and artifact recording remain.

#### G2 - Bound dashboard rendering work as history grows
- **Where:** `src/web/static/app.js` chart and log rendering paths; `MAX_CONSOLE_LOGS` and chart downsampling constants.
- **What's wrong:** The dashboard has log limits and chart downsampling, but chart update/render behavior is still embedded in the large app script and lacks performance tests.
- **Impact:** Moderate - long training sessions can expose UI stalls that unit tests miss.
- **Fix:** Extract chart rendering and add tests for large histories, scrollbar updates, and no-animation update paths. Add one browser smoke with synthetic history size.
- **Effort:** M
- **Grade lift:** B -> B+ by making long-session dashboard performance safer.
- **Status:** Partially addressed 2026-06-16 - large-history chart update modeling is now unit-tested and a real dashboard smoke exists; synthetic long-history browser coverage remains.

#### G3 - Reduce large game update/render coupling
- **Where:** `src/game/space_invaders.py:1`, `src/game/asteroids.py:1`, `src/game/breakout.py:1`.
- **What's wrong:** Simulation, rendering, and state extraction are mixed, so performance tuning risks gameplay regressions.
- **Impact:** Minor - current scale is local, but tuning larger games is harder than necessary.
- **Fix:** Extract pure state/reward/collision helpers and keep rendering adapters thin. Add tests for pure helpers before optimizing.
- **Effort:** L
- **Grade lift:** B -> B+ by making game hot paths easier to profile and tune.
- **Status:** Partially addressed 2026-06-16 - Space Invaders scoring, pressure, level-speed, and invasion formulas moved to pure helpers with tests; broader entity/collision/render decomposition remains.

---

## H - Documentation & Onboarding - B

Documentation is good for a solo/local project: README covers setup, quick start, safety notes, and developer checks; `docs/architecture.md` gives maintainers a current map; changelog and release automation are active. The README still contains an older simplified project tree, and the newer dashboard/app module split is better documented in `docs/architecture.md` than in onboarding.

#### H1 - Refresh README project structure
- **Where:** `README.md:50`, `docs/architecture.md:1`.
- **What's wrong:** README's project tree still centers on `main.py`, `src/game`, `src/ai`, and `src/visualizer`; it omits newer `src/app`, `src/web`, contracts, launcher assets, CI scripts, and constraints workflow.
- **Impact:** Moderate - new contributors may miss the real ownership boundaries.
- **Fix:** Update the README tree to match current directories and link to `docs/architecture.md` for ownership guidance.
- **Effort:** S
- **Grade lift:** B -> B+ by aligning onboarding with current structure.

#### H2 - Document dashboard auth and token behavior
- **Where:** `README.md` safety/developer sections, `docs/architecture.md` Web Dashboard section.
- **What's wrong:** The README notes trusted network exposure, but it does not clearly explain dashboard token bootstrap, API token headers, Socket.IO auth, and URL-token tradeoffs.
- **Impact:** Moderate - users can expose the dashboard without understanding the remaining local-auth risks.
- **Fix:** Add a "Dashboard security model" section covering localhost default, `--host 0.0.0.0`, token URLs, referrer policy, CSP, and trusted-network assumptions.
- **Effort:** S
- **Grade lift:** B -> B+ by making operational security clearer.

#### H3 - Add a dependency maintenance note
- **Where:** `README.md`, `Makefile:26`, `constraints.txt:1`.
- **What's wrong:** Dependency audit behavior is encoded in `Makefile`, but the no-fix torch advisory and constraints refresh process are not explained for maintainers.
- **Impact:** Minor - maintainers may not know why one advisory is ignored or how to refresh pins.
- **Fix:** Add a short dependency maintenance note linking `make audit`, constraints, Dependabot, and the torch advisory exception.
- **Effort:** S
- **Grade lift:** B -> B+ by making dependency operations transparent.

---

## I - Developer Experience & Tooling - A-

DevEx is now one of the stronger areas. `make verify` is a real local gate, CI mirrors it across Python versions, CodeQL and dependency review run, release automation produced `0.0.4`, PR title lint protects semantic releases, and a documented `make dashboard-smoke` command now runs the real browser path. The remaining gaps are mostly convenience and consistency around local environment setup and complexity-growth warnings.

#### I1 - Make local build/audit tooling self-contained
- **Where:** `Makefile:35`, `pyproject.toml:28`, `requirements.txt:1`.
- **What's wrong:** `make verify` skips package build locally if `build` is not installed, while CI installs it. This is pragmatic, but local verification can differ from CI.
- **Impact:** Moderate - developers can miss build issues locally unless their environment has dev extras installed.
- **Fix:** Add `make setup-dev` or document `pip install -r requirements.txt -c constraints.txt` as the required dev setup before `make verify`. Optionally make `build-if-available` print the exact install command.
- **Effort:** S
- **Grade lift:** B+ -> A- by reducing local/CI drift.

#### I2 - Add a changed-file complexity warning
- **Where:** no complexity gate yet; known hotspots are `main.py`, `src/web/server.py`, and `src/web/static/app.js`.
- **What's wrong:** Tooling catches formatting, typing, tests, security, and coverage, but not growth in already-large files.
- **Impact:** Moderate - future work can keep adding to the same large modules without friction.
- **Fix:** Add a lightweight script that warns when changed source files exceed size or complexity thresholds. Start as warning-only to avoid noisy CI failures.
- **Effort:** S
- **Grade lift:** B+ -> A- by steering future work toward smaller modules.

#### ~~I3 - Add a documented browser smoke command~~ ✓ done 2026-06-16
- **Where:** `Makefile`, `package.json`, `tests/e2e/dashboard_smoke.mjs`.
- **What changed:** `make dashboard-smoke` runs the Playwright dashboard smoke through `npm run dashboard-smoke`.
- **Impact:** Moderate - frontend regressions now have a known local browser command before review.
- **Fix:** Completed with a disposable server launcher and browser script.
- **Effort:** M
- **Grade lift:** B+ -> A- by making real UI verification repeatable.
