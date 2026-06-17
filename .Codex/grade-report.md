# Codebase Grade Report

**Project:** nn-game1
**Audited:** 2026-06-17
**Stack:** Python 3.10-3.12 DQN/Pygame training app with Flask/Socket.IO dashboard, vanilla JS frontend modules, Node tests, and Playwright dashboard smoke coverage.

## Summary

| ID | Category | Grade | Items |
|----|----------|-------|-------|
| A | Architecture & Design | B+ | 4 |
| B | Backend Quality | B+ | 4 |
| C | Frontend Quality | B | 4 |
| D | Testing & Reliability | B+ | 4 |
| E | Security | B | 4 |
| F | Dependencies & Tech Currency | B+ | 3 |
| G | Performance & Scalability | B+ | 3 |
| H | Documentation & Onboarding | B | 4 |
| I | Developer Experience & Tooling | A- | 3 |
| **Overall** | | **B+** | **33** |

**Top 5 highest-leverage fixes:** B1, D1, C1, E1, A1

**Validation snapshot:**
- `git diff --quiet HEAD origin/main`: passed; this worktree matches the merged `origin/main` tree.
- `make verify`: passed; black, ruff, mypy, 24 JS tests, 668 pytest tests, 71.86% coverage, release config, hygiene, source-size gate, dependency audit, and package build all passed.
- `make dashboard-smoke`: passed.
- `npm audit --audit-level=moderate`: passed with 0 vulnerabilities.
- `python -m pip_audit -r requirements.txt --ignore-vuln CVE-2025-3000`: passed with no known vulnerabilities found and 1 intentional torch advisory ignore.

**Implementation update for A/B/C/D/G/I request:**
- A: extracted model commands and web launcher orchestration out of `main.py`; added grouped typed `Config` views; moved Breakout entities into `breakout_entities.py`; extracted shared runtime dashboard emission policy.
- B: fixed vectorized new-best dashboard emission, added shared checkpoint discovery, typed route context/handlers more strictly, and replaced the socket control dispatcher chain with a handler table.
- C: replaced raw NN inspection panel HTML insertion with DOM construction, added dialog focus trap/restore behavior, and strengthened CDN failure reporting/pinning.
- D: added focused Python and Node coverage for runtime helpers, vector no-copy reset semantics, config groups, checkpoint discovery, NN panel safety, dialogs, and expanded Playwright dashboard smoke journeys.
- G: bounded initial dashboard history snapshots, fixed no-copy vector reset state behavior for completed envs, and added repeatable performance smoke budgets.
- I: added `make setup`, `scripts/bootstrap_dev.py`, `.pre-commit-config.yaml`, CI Playwright smoke/perf gates, and stricter mypy ratchets for newly typed modules.

**Post-implementation validation:**
- `make verify`: passed on 2026-06-17. This included black, ruff, mypy, 27 Node dashboard tests, performance smoke, 682 pytest tests with 70.46% coverage, Playwright dashboard smoke, release config, repository hygiene, source-size gate, dependency audit, and package build.

---

## A - Architecture & Design - B+

The app now has clear top-level domains: app runtimes, game registry, AI/training, visualizers, and web dashboard. `src/game/base_game.py:55-147` defines the game contract, `src/game/__init__.py:64-125` centralizes registered games, and web concerns are split across `src/web/routes.py`, `src/web/model_service.py`, `src/web/socket_controls.py`, and `src/web/metrics_publisher.py`. The remaining architectural ceiling is that the interactive/headless runtime classes and game files still coordinate many lifecycle concerns in-process.

#### A1 - Split runtime loop orchestration into focused services
- **Where:** `src/app/interactive.py:54-260`, `src/app/headless.py:37-219`, `src/app/headless.py:220-431`, `src/app/headless.py:452-755`
- **What's wrong:** `GameApp` and `HeadlessTrainer` still own configuration mutation, dashboard callback wiring, model loading, training loops, metrics publishing, checkpoint decisions, and shutdown behavior. The files are under the line budget now, but each class still has several reasons to change.
- **Impact:** Major - central runtime changes can accidentally affect save/load, dashboard updates, or training stability.
- **Fix:** Extract a `RuntimeSession`/`TrainingLoop` service for episode stepping and checkpoint cadence, a dashboard binding object for callback registration, and a model lifecycle object for load/save/reset. Keep `GameApp` and `HeadlessTrainer` as composition roots.
- **Effort:** L
- **Grade lift:** B+ -> A- (removes the largest remaining coupling hotspot)

#### A2 - Finish separating game rules from rendering and entities
- **Where:** `src/game/asteroids.py:31-220`, `src/game/breakout.py:31-195`, `src/game/pong.py`, `src/game/snake.py`, `src/game/space_invaders.py:1-878`
- **What's wrong:** Several game modules still mix entities, state transitions, reward shaping, collision logic, human input helpers, and rendering. `space_invaders` has started this split with entity/rendering modules, but the pattern is not consistent across games.
- **Impact:** Moderate - gameplay fixes and performance work are harder to isolate and test than they need to be.
- **Fix:** Establish a common pattern per game: `*_entities.py`, `*_rules.py`, `*_rendering.py`, and a thin game facade implementing `BaseGame`. Start with `asteroids.py` and `breakout.py` because they are the largest remaining game files.
- **Effort:** L
- **Grade lift:** B+ -> A- (makes the game layer easier to extend without regressions)

#### A3 - Break `Config` into nested typed configuration groups
- **Where:** `config.py:20-32`, `config.py:38-163`, `config.py:173-203`, `config.py:213-260`
- **What's wrong:** One `Config` dataclass holds cross-game screen settings, per-game knobs, neural-network settings, training behavior, visualization settings, storage paths, and device selection. Some values are explicitly legacy or approximate, such as `STATE_SIZE`.
- **Impact:** Moderate - unrelated settings are easy to mutate from CLI/dashboard code, and game-specific defaults are harder to validate independently.
- **Fix:** Introduce nested dataclasses such as `GameConfig`, `TrainingConfig`, `NetworkConfig`, `DashboardConfig`, and `StorageConfig`. Add compatibility properties for existing callers, then migrate call sites gradually.
- **Effort:** M
- **Grade lift:** B+ -> A- (reduces cross-domain configuration coupling)

#### A4 - Shrink the launcher compatibility surface
- **Where:** `main.py:75-85`, `main.py:100-203`, `main.py:206-240`
- **What's wrong:** `main.py` still imports internal runtime and game types directly and owns model inspection/listing plus web startup details. This keeps import-time coupling to pygame, torch, and dashboard modules higher than a launcher needs.
- **Impact:** Moderate - packaging and CLI changes are more fragile when the entrypoint remains a broad compatibility layer.
- **Fix:** Move model inspection/listing to `src/app/model_commands.py` and web startup to `src/app/web_launcher.py`. Leave `main.py` as `parse_args`, dispatch, and process exit handling.
- **Effort:** M
- **Grade lift:** B+ -> A- (clarifies the app boundary and reduces import coupling)

---

## B - Backend Quality - B+

Backend and app-service quality is solid. The dashboard has stable error helpers in `src/web/routes.py:19-30`, typed API payloads in `src/web/contracts.py:8-123`, model path helpers in `src/app/model_paths.py:21-79`, and safe checkpoint loading in `src/utils/checkpoint_loader.py:34-66`. Remaining issues are mostly around runtime edge cases, partially typed Flask surfaces, and service ownership boundaries.

#### B1 - Fix vectorized new-best dashboard emission
- **Where:** `src/app/headless.py:560-584`
- **What's wrong:** The vectorized training loop updates `self.best_score` before computing `is_new_best = score > self.best_score`, making the new-best flag false after the update. The dashboard still updates on early episodes and every fifth episode, but it can miss immediate new-best updates in vectorized mode.
- **Impact:** Moderate - users can see stale best-score dashboard state during high-speed vectorized training.
- **Fix:** Compute `is_new_best = score > self.best_score` before mutating `self.best_score`, reuse that boolean for saving and dashboard emission, and add a focused vectorized-loop test.
- **Effort:** S
- **Grade lift:** B+ -> A- (removes a concrete user-visible runtime bug)

#### B2 - Type route handlers and dashboard context more strictly
- **Where:** `src/web/routes.py:33-214`, `src/web/server.py:123-190`, `src/web/socket_controls.py:17-40`, `mypy.ini:4-7`
- **What's wrong:** Mypy passes, but route handler bodies are not checked because untyped functions are still allowed. The live mypy run reported annotation notes for `src/web/routes.py`, and `dashboard: Any` hides contract drift between Flask routes and `WebDashboard`.
- **Impact:** Moderate - API payload shape regressions can slip through type checking even though contracts exist.
- **Fix:** Add return annotations to route hooks and handlers, define a `DashboardRouteContext` protocol for route dependencies, and enable `check_untyped_defs` or `disallow_untyped_defs` for `src/web` first.
- **Effort:** M
- **Grade lift:** B+ -> A- (turns existing contracts into stronger static guarantees)

#### B3 - Consolidate app and web model-service boundaries
- **Where:** `src/app/model_service.py:15-188`, `src/web/model_service.py:19-141`, `src/app/model_paths.py:21-79`
- **What's wrong:** The app model service and web model service now share path rules, but they still split listing, resolution, loading, deletion, cleanup, metadata, and compatibility behavior across two service classes. That makes it easy for web behavior to drift from app save/load expectations.
- **Impact:** Moderate - checkpoint lifecycle bugs are costly because they affect user saves and resume flows.
- **Fix:** Create a shared `CheckpointRepository` for model directories, opaque IDs, compatibility inspection, listing, deletion, and metadata. Keep app/web services as thin adapters around that repository.
- **Effort:** M
- **Grade lift:** B+ -> A- (centralizes the persistence contract)

#### B4 - Replace the control dispatcher chain with a handler table
- **Where:** `src/web/socket_controls.py:285-321`
- **What's wrong:** `dispatch_control` is a long conditional chain even though each action already has a focused handler. Adding new actions requires editing the dispatcher and increases the chance of inconsistent validation or acknowledgement behavior.
- **Impact:** Minor - the current implementation works, but action growth will keep adding friction.
- **Fix:** Build a dictionary from action name to small handler callable, with shared signature normalization for handlers that need `emit_event`. Keep the existing tests as the safety net.
- **Effort:** S
- **Grade lift:** B+ -> A- (small clarity win in a central API surface)

---

## C - Frontend Quality - B

The frontend has improved a lot: dashboard JS and CSS are split, destructive flows use custom dialogs, and core helpers have Node tests. `src/web/templates/dashboard.html:522-532` still loads plain global scripts in a strict order, `src/web/static/dashboard_state.js:5-31` exposes mutable globals for compatibility, and several modules still write HTML strings into the DOM. This is good for a small local app, but not yet an A-level frontend architecture.

#### C1 - Move dashboard scripts to importable modules
- **Where:** `src/web/templates/dashboard.html:522-532`, `src/web/static/dashboard_state.js:5-31`, `src/web/static/app.js:29-63`
- **What's wrong:** Frontend modules communicate through globals and load order instead of explicit imports/exports. This makes tests rely on VM-loaded script order and makes missing-script failures possible at runtime.
- **Impact:** Moderate - frontend behavior remains harder to isolate, tree-check, and refactor safely.
- **Fix:** Convert dashboard scripts to ES modules or a small build step, export explicit APIs, and load one entry module from the template. Keep `dashboard_core.js` CommonJS support only for tests or replace tests with native ESM imports.
- **Effort:** M
- **Grade lift:** B -> B+ (removes the largest frontend maintainability limiter)

#### C2 - Remove raw HTML insertion from inspection panels
- **Where:** `src/web/static/dashboard_nn_panels.js:31-85`, `src/web/static/dashboard_nn_panels.js:220-254`, `src/web/static/dashboard_controls.js:369-374`
- **What's wrong:** Some flows still render template strings through `innerHTML`. Several values are escaped or numeric, and `DashboardCore.modelListHtml` has escape tests, but the pattern leaves future fields vulnerable to accidental unsafe interpolation.
- **Impact:** Moderate - dashboard data is mostly local, but this is a recurring XSS footgun as inspection/model metadata grows.
- **Fix:** Build inspection panels with DOM nodes and `textContent`, or use a tiny safe rendering helper that requires escaped text nodes by default. Add tests that hostile layer/action names and metadata render as text.
- **Effort:** M
- **Grade lift:** B -> B+ (hardens the most fragile rendering pattern)

#### C3 - Add focus trapping and focus restore for dashboard dialogs
- **Where:** `src/web/static/dashboard_dialogs.js:45-114`, `src/web/templates/dashboard.html:508-519`
- **What's wrong:** Dialogs set `role="dialog"`, `aria-modal`, focus a preferred button, and support Escape, but they do not trap Tab focus or restore focus to the opener. The static model browser modal has markup roles but relies on external behavior for complete keyboard ergonomics.
- **Impact:** Moderate - keyboard and assistive-technology users can lose context during destructive controls like start fresh, delete, or save and quit.
- **Fix:** Track the opener before showing each dialog, loop focus within the dialog while open, close on Escape consistently, and restore focus after close. Add a small JS test for focus cycling and restore.
- **Effort:** S
- **Grade lift:** B -> B+ (raises accessibility reliability in core workflows)

#### C4 - Localize third-party browser dependencies or add stronger fallbacks
- **Where:** `src/web/templates/dashboard.html:9-12`, `src/web/static/app.js:162-199`, `tests/js/dashboard_startup.test.mjs:154-178`
- **What's wrong:** Chart.js, chart zoom, and Socket.IO client are loaded from CDNs. Startup tests verify visible errors when Chart.js or Socket.IO are missing, but the dashboard still depends on external network availability unless those scripts are cached.
- **Impact:** Moderate - local training dashboards should stay useful when offline or when a CDN is blocked.
- **Fix:** Vendor or package the browser assets into `src/web/static/vendor/`, pin versions, update CSP to serve them from self, and keep the existing startup-error tests as fallback coverage.
- **Effort:** M
- **Grade lift:** B -> B+ (improves local reliability and supports CSP hardening)

---

## D - Testing & Reliability - B+

The test base is strong for a hobby/learning project: `make verify` passes 668 Python tests and 24 JS tests, with 71.86% coverage and a package build. The repo also has a Playwright dashboard smoke test in `tests/e2e/dashboard_smoke.mjs`. The grade is capped below A because coverage is uneven, several runtime files are omitted from coverage, and the smoke test is not part of `make verify` or CI.

#### D1 - Raise coverage on low-covered game/render/runtime surfaces
- **Where:** coverage output for `src/game/space_invaders_rendering.py`, `src/game/snake.py`, `src/game/asteroids.py`, `src/game/space_invaders_entities.py`; `pyproject.toml:74-85`
- **What's wrong:** Total coverage clears the 70% gate, but some meaningful gameplay/rendering files are far below that and several runtime files are omitted from coverage. This leaves game-specific behavior and visual/runtime edges weaker than the overall number suggests.
- **Impact:** Major - game and runtime regressions are the most likely user-visible failures.
- **Fix:** Add tests for render-helper geometry, snake/asteroids edge cases, and runtime save/reset/pause flows that can run headless. Ratchet coverage by package or file group rather than only total coverage.
- **Effort:** M
- **Grade lift:** B+ -> A- (turns broad coverage into deeper reliability)

#### D2 - Put the Playwright dashboard smoke in CI or `make verify`
- **Where:** `Makefile:31`, `Makefile:49`, `.github/workflows/ci.yml:54-66`, `tests/e2e/dashboard_smoke.mjs:62-135`
- **What's wrong:** `make dashboard-smoke` passes locally, but `make verify` and CI do not run it. The most realistic browser coverage is therefore easy to skip before merging.
- **Impact:** Moderate - dashboard regressions can pass the main gate even when a real browser would fail.
- **Fix:** Add a CI job or `verify-browser` target that installs/uses a stable browser and runs `make dashboard-smoke`. If full CI cost is too high, run it on PRs touching `src/web/**`, `tests/e2e/**`, or templates/static assets.
- **Effort:** S
- **Grade lift:** B+ -> A- (promotes the best frontend reliability check to a merge gate)

#### D3 - Expand browser journeys beyond save/load smoke
- **Where:** `tests/e2e/dashboard_smoke.mjs:97-110`, `src/web/static/dashboard_controls.js:43-107`, `src/web/static/dashboard_settings.js`
- **What's wrong:** The current smoke covers connection, save, load modal, and the selected game dropdown. It does not exercise settings changes, game switching, start-fresh confirmation branches, keyboard dialog behavior, screenshot fallback, or NN inspection panels.
- **Impact:** Moderate - the dashboard's riskiest interaction paths still rely mostly on unit tests and manual confidence.
- **Fix:** Add Playwright journeys for settings apply/reject, start-fresh cancel/save/skip, model delete cancel, game switch cancel, and keyboard-only dialog dismissal. Keep the disposable server fixture so no real models are touched.
- **Effort:** M
- **Grade lift:** B+ -> A- (covers the high-risk user workflows)

#### D4 - Make performance regression tests less anecdotal
- **Where:** `tests/test_performance_budgets.py:12-34`, `benchmark.py:69-105`
- **What's wrong:** There is one loose replay-buffer budget and a rich manual benchmark script, but no tracked benchmark artifact or trend gate for training throughput, vectorized stepping, dashboard payload size, or chart rendering.
- **Impact:** Moderate - performance regressions could land gradually without a clear signal.
- **Fix:** Add a small benchmark target that records replay sampling, single-env step, vectorized step, and dashboard snapshot serialization timings. Store thresholds conservatively and run the quick version in CI.
- **Effort:** M
- **Grade lift:** B+ -> A- (protects the project's performance-oriented features)

---

## E - Security - B

The local dashboard has meaningful protections: session tokens are generated with `secrets.token_urlsafe`, HTTP APIs and Socket.IO controls are token-gated, model IDs are opaque, checkpoint paths are constrained, PyTorch restricted loading is tried first, CodeQL is enabled, and dependency audit runs. This is solid for a trusted local training app. The grade is capped by token-in-URL ergonomics, `unsafe-inline` CSP, lack of rate limiting, and an intentional ignored torch advisory.

#### E1 - Remove `unsafe-inline` from the dashboard CSP
- **Where:** `src/web/server.py:34-44`, `src/web/templates/dashboard.html:12`, `src/web/templates/dashboard.html:14-16`
- **What's wrong:** The CSP currently allows inline scripts and inline styles. That is partly needed by the inline token assignment and template style patterns, but it weakens the XSS blast-radius protection.
- **Impact:** Moderate - CSP is one of the strongest defenses if future dashboard rendering accidentally injects unsafe HTML.
- **Fix:** Move the inline token bootstrap to a data/meta read path, remove inline style dependencies where practical, serve all scripts from `self`, and update tests to assert no `unsafe-inline` remains.
- **Effort:** M
- **Grade lift:** B -> B+ (tightens browser-side containment)

#### E2 - Replace query-string dashboard tokens with a less leaky bootstrap
- **Where:** `src/web/server.py:202-214`, `src/web/routes.py:51-76`, `README.md:613-619`
- **What's wrong:** The dashboard URL includes `?token=...`. The app sets no-referrer headers, but query tokens still land in browser history, copied URLs, local terminal output, and potentially logs.
- **Impact:** Moderate - LAN dashboard use is documented, and leaked tokenized URLs grant control access until the session ends.
- **Fix:** Keep the first open URL as a one-time bootstrap token, set an HttpOnly/SameSite session cookie, redirect to `/`, and require the cookie or header thereafter. Keep `NN_GAME_DASHBOARD_TOKEN` for automation.
- **Effort:** M
- **Grade lift:** B -> B+ (reduces accidental credential leakage)

#### E3 - Add basic throttling for mutating controls
- **Where:** `src/web/server.py:345-360`, `src/web/socket_controls.py:285-321`, `src/web/routes.py:160-183`
- **What's wrong:** Authorized controls can be called repeatedly without rate limiting or per-action cooldowns. The dashboard is local/trusted by design, but actions like save, delete, start fresh, and restart have filesystem or process impact.
- **Impact:** Minor - token auth is the main control, but throttling would limit mistakes, scripts, or repeated clicks.
- **Fix:** Add a small per-token/per-action cooldown for destructive controls and return stable 429/ack errors. Cover save/delete/start-fresh/restart behavior in socket and route tests.
- **Effort:** M
- **Grade lift:** B -> B+ (adds defense in depth around mutating controls)

#### E4 - Add an expiration policy for the ignored torch advisory
- **Where:** `Makefile:20-23`, `README.md:662`, `constraints.txt:20`
- **What's wrong:** `pip-audit` intentionally ignores `CVE-2025-3000` because no patched torch release is available. The reason is documented, but there is no date, owner, or automated reminder to remove the ignore when a fixed torch exists.
- **Impact:** Moderate - ignored advisories can become invisible long-term debt.
- **Fix:** Add a short allowlist file with vulnerability ID, package, rationale, date added, review-by date, and upstream status. Make the audit helper print or fail when the review date expires.
- **Effort:** S
- **Grade lift:** B -> B+ (keeps a known security exception accountable)

---

## F - Dependencies & Tech Currency - B+

Dependency hygiene is good: package metadata lives in `pyproject.toml`, reproducible pins live in `constraints.txt`, the compatibility `requirements.txt` installs extras, Dependabot covers pip and GitHub Actions, dependency review runs on PRs, and `pip-audit` plus `npm audit` pass apart from the documented torch advisory ignore. The main gap is process hardening around upgrades and Node audit automation.

#### F1 - Add a constraints refresh and upgrade playbook
- **Where:** `pyproject.toml:5-37`, `constraints.txt:1-22`, `requirements.txt:1-8`
- **What's wrong:** Runtime dependency ranges and pinned constraints are present, but there is no documented command for regenerating pins, testing upgrades, or deciding when to move Python/Node ranges forward.
- **Impact:** Moderate - dependency freshness will depend on ad hoc updates instead of a repeatable process.
- **Fix:** Add `docs/dependencies.md` or a Make target documenting how to regenerate `constraints.txt`, run `make verify`, run `npm audit`, and review torch/CUDA/PyTorch compatibility.
- **Effort:** S
- **Grade lift:** B+ -> A- (makes future upgrades predictable)

#### F2 - Automate Node audit in the main verification path
- **Where:** `package.json:5-12`, `Makefile:20-23`, `.github/workflows/ci.yml:54-66`
- **What's wrong:** `npm audit --audit-level=moderate` passes, but it is not part of `make verify` or CI. The frontend dependency surface is tiny, so the cost is low.
- **Impact:** Minor - JS dependency risk is low today, but easy automation is missing.
- **Fix:** Add `node-audit` and include it in `make verify` and CI after `npm ci` or equivalent setup.
- **Effort:** S
- **Grade lift:** B+ -> A- (closes a cheap dependency-audit gap)

#### F3 - Separate legacy full-dev install from runtime install guidance
- **Where:** `requirements.txt:1-8`, `pyproject.toml:23-37`, `README.md:656-660`
- **What's wrong:** `pip install -r requirements.txt -c constraints.txt` installs `.[web,test,dev]` for compatibility. That is convenient, but it blurs runtime, web, test, and dev dependencies for users who only want to run the game.
- **Impact:** Minor - larger installs increase setup time and dependency exposure.
- **Fix:** Keep `requirements.txt` as a dev compatibility shim, but document `pip install .`, `pip install .[web]`, and `pip install -r requirements.txt -c constraints.txt` as separate runtime/web/dev paths.
- **Effort:** S
- **Grade lift:** B+ -> A- (sharpens install intent without code churn)

---

## G - Performance & Scalability - B+

The project has real performance thinking: replay buffers use contiguous arrays, performance modes are explicit, vectorized environments exist for multiple games, metrics and NN visualization are throttled, dashboard charts downsample, and a benchmark script exists. The remaining work is mostly about bounding payload growth and making performance claims enforceable in CI.

#### G1 - Bound initial dashboard history payloads
- **Where:** `src/web/metrics_publisher.py:52-67`, `src/web/metrics_publisher.py:336-348`, `src/web/static/dashboard_charts.js:736-750`
- **What's wrong:** `get_snapshot()` sends the full retained history arrays to clients, and the configured history length is 100000 episodes. This is acceptable in many local runs but can become expensive for long training sessions and reconnects.
- **Impact:** Moderate - long-running sessions can pay large serialization, transfer, and chart-update costs at connection time.
- **Fix:** Add query/options for recent-window snapshots, send compact summary stats by default, and let the frontend request older history pages only when the user scrolls back.
- **Effort:** M
- **Grade lift:** B+ -> A- (keeps dashboard cost bounded during long training)

#### G2 - Make vectorized environments truly vectorized where feasible
- **Where:** `src/game/space_invaders_vec.py:95-148`, `src/game/asteroids_vec.py:49-75`, `src/game/base_game.py:214-243`
- **What's wrong:** Vectorized environment classes still loop over individual game objects in Python. That is a useful batching boundary for action selection and replay insertion, but it is not true vectorized physics/collision computation.
- **Impact:** Moderate - high `--vec-envs` settings will eventually hit Python loop overhead.
- **Fix:** Start with one game, likely Breakout or Space Invaders, and move state arrays, collision checks, and reward updates into batched NumPy operations. Keep the current object-loop implementation as a compatibility fallback.
- **Effort:** L
- **Grade lift:** B+ -> A- (unlocks the next training-throughput tier)

#### G3 - Gate quick performance budgets in CI
- **Where:** `tests/test_performance_budgets.py:12-34`, `benchmark.py:69-105`, `src/web/static/dashboard_nn.js:214-236`
- **What's wrong:** Performance-sensitive code has optimizations and manual benchmarks, but CI only has one loose replay-buffer budget. Dashboard render timing and training throughput are not tracked over time.
- **Impact:** Moderate - performance regressions can land slowly and only be noticed during real training.
- **Fix:** Add a `make perf-smoke` target with conservative budgets for replay sampling, vectorized stepping, dashboard snapshot serialization, and NN canvas update model generation. Run it on changed performance-sensitive files.
- **Effort:** M
- **Grade lift:** B+ -> A- (turns performance expectations into a repeatable signal)

---

## H - Documentation & Onboarding - B

The README is thorough, `docs/architecture.md` is current and helpful, and the Makefile exposes practical validation commands. The main issue is stale or noisy docs: the README still shows older project structure and extension steps in places, while multiple historical bug/planning reports remain at the root.

#### H1 - Update stale README architecture and extension sections
- **Where:** `README.md:34-77`, `README.md:666-703`, `docs/architecture.md:13-37`
- **What's wrong:** The README's project tree and "Extending to Other Games" section still describe older structure and say to register games in config, while the real registry lives in `src/game/__init__.py`.
- **Impact:** Moderate - new contributors can follow outdated extension guidance and make changes in the wrong place.
- **Fix:** Replace the old tree with the current app/web/game/AI layout and update extension docs to use `BaseGame`, `GAME_REGISTRY`, vectorized optional support, and matching tests.
- **Effort:** S
- **Grade lift:** B -> B+ (removes the biggest onboarding mismatch)

#### H2 - Move historical bug and planning reports into an archive
- **Where:** `30_BUGS_LIST.md`, `30_MORE_BUGS_LIST.md`, `BUGS_AND_IMPROVEMENTS.md`, `POTENTIAL_BUGS.md`, `PLANNING.md`, `PHASE1_IMPLEMENTATION_SUMMARY.md`, `PHASE2_BACKEND_SUMMARY.md`, `PROGRESS_REPORT.md`
- **What's wrong:** The repo root contains many historical audit/planning artifacts alongside active setup files. They are useful context, but they make it harder to identify the canonical docs.
- **Impact:** Minor - onboarding feels noisier than the current code quality deserves.
- **Fix:** Move historical reports to `docs/archive/` and add a short `docs/archive/README.md` that explains what is historical versus current.
- **Effort:** S
- **Grade lift:** B -> B+ (makes the repo easier to scan)

#### H3 - Expand architecture docs for persistence, dashboard auth, and release flow
- **Where:** `docs/architecture.md:29-50`, `README.md:613-662`, `.github/workflows/release.yml:1-88`
- **What's wrong:** `docs/architecture.md` gives a good map, but it does not explain important contracts such as checkpoint trust boundaries, opaque model IDs, dashboard token flow, release/versioning behavior, or dependency-audit exceptions.
- **Impact:** Moderate - contributors can miss the hidden contracts that matter most for safe changes.
- **Fix:** Add sections for checkpoint lifecycle, dashboard auth/control flow, release automation, dependency audit exceptions, and validation lanes. Link each section to the owning files and tests.
- **Effort:** M
- **Grade lift:** B -> B+ (documents the non-obvious system contracts)

#### H4 - Add a short contributor workflow doc
- **Where:** `README.md:623-662`, `Makefile:1-49`, `.github/workflows/ci.yml:1-75`
- **What's wrong:** Developer commands exist, but there is no compact contributor guide that says which command to run for a one-line docs change, a Python change, a dashboard change, or a release-sensitive change.
- **Impact:** Minor - contributors may over-run or under-run checks.
- **Fix:** Add `CONTRIBUTING.md` with setup, branch naming, validation matrix, PR title convention, and when to run dashboard smoke.
- **Effort:** S
- **Grade lift:** B -> B+ (turns the existing tooling into a clearer workflow)

---

## I - Developer Experience & Tooling - A-

The tooling is strong: `make verify` is meaningful, CI runs Python 3.11 and 3.12, release config is validated, package build is checked, file size is gated, repo hygiene is checked, PR title lint supports semantic release, CodeQL and dependency review are enabled, and helper scripts self-install ruff/pip-audit when needed. The main limitations are stricter typing, browser smoke automation, and local bootstrap polish.

#### I1 - Ratchet mypy strictness by package
- **Where:** `mypy.ini:1-7`, `src/web/routes.py:33-214`, `src/web/server.py:123-190`
- **What's wrong:** Mypy passes, but untyped function bodies are not checked. The live run reported notes for route payload annotations, showing that static checking is not yet as strict as the docs imply.
- **Impact:** Moderate - type drift can hide in route and callback bodies.
- **Fix:** Add package-specific mypy sections: start with `check_untyped_defs = True` for `src/web` and `src/app`, then move toward `disallow_untyped_defs = True` for small helper modules.
- **Effort:** M
- **Grade lift:** A- -> A (makes type checking match the project's maturity)

#### I2 - Add a one-command dev bootstrap
- **Where:** `README.md:656-660`, `package.json:5-12`, `pyproject.toml:23-37`
- **What's wrong:** Setup instructions are clear, but there is no `make setup` or script that installs Python constraints plus Node dependencies and verifies the selected Python/Node versions.
- **Impact:** Minor - first-time setup still requires manual command ordering.
- **Fix:** Add `make setup` or `scripts/bootstrap_dev.py` that checks Python 3.10-3.12, installs `requirements.txt -c constraints.txt`, runs `npm install`, and prints the next validation command.
- **Effort:** S
- **Grade lift:** A- -> A (smooths onboarding without changing runtime code)

#### I3 - Add pre-commit hooks for cheap checks
- **Where:** `Makefile:1-49`, `.github/workflows/ci.yml:37-66`
- **What's wrong:** CI and Make targets are good, but there is no pre-commit configuration for black, ruff, source-size, and basic hygiene. Developers can discover cheap failures only after running larger gates.
- **Impact:** Minor - local feedback can be faster for frequent edits.
- **Fix:** Add `.pre-commit-config.yaml` or a lightweight `make precommit` target using existing commands. Keep it optional, but document it in `CONTRIBUTING.md`.
- **Effort:** S
- **Grade lift:** A- -> A (improves the local edit loop)
