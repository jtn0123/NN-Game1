# Codebase Grade Report

**Project:** nn-game1
**Audited:** 2026-06-17
**Stack:** Python 3.10-3.12 DQN/Pygame training app with Flask/Socket.IO dashboard, vanilla JS frontend modules, Node tests, Playwright dashboard smoke coverage, and GitHub Actions CI.

## Summary

| ID | Category | Grade | Items |
|----|----------|-------|-------|
| A | Architecture & Design | A- | 4 |
| B | Backend Quality | A | 4 |
| C | Frontend Quality | A- | 4 |
| D | Testing & Reliability | A | 4 |
| E | Security | A- | 4 |
| F | Dependencies & Tech Currency | A- | 3 |
| G | Performance & Scalability | A- | 4 |
| H | Documentation & Onboarding | A- | 3 |
| I | Developer Experience & Tooling | A | 3 |
| **Overall** | | **A-** | **33** |

**Top 5 remaining highest-leverage fixes:** A1, A2, A3, C1, C3

**Validation snapshot:**
- `git fetch origin main --prune`: passed; `origin/main` is at merged commit `7d0701b`.
- Worktree branch: `codex/regrade-after-merge`, tracking `origin/main`.
- `make verify`: passed after the coverage-ratchet implementation pass. This included Black, Ruff, mypy, 35 Node dashboard tests, 2 performance smoke tests, 763 pytest tests, 81.43% coverage against an 80% gate, Playwright dashboard smoke, release config, hygiene, granular source-size budgets, dependency audit waiver expiry, pip-audit, npm audit, and package build.
- `npm audit --audit-level=moderate`: passed with 0 vulnerabilities.

**Implementation update for A/B/C/D/F/G/I request:**
- A: extracted dashboard callback binding into `src/app/dashboard_bindings.py`, moved Asteroids entities to `src/game/asteroids_entities.py`, moved the Asteroids manual runner to `scripts/play_asteroids.py`, and added typed config apply methods for grouped settings.
- B: introduced shared `CheckpointRepository`, made app/web model services thinner, removed `WebDashboard` socket-control compatibility wrappers, narrowed socket-control publisher/callback protocols, and added explicit `CommandResult` normalization.
- C: extracted the model-browser DOM renderer into `dashboard_model_list.js`, removed remaining runtime `innerHTML` assignments from dashboard code, added static load-modal Escape/focus-restore handling, and expanded model-list/Playwright coverage.
- D: added coverage for Asteroids entities, Snake full-board food behavior, Space Invaders font caches, process restart orchestration, dashboard binding, checkpoint repository, NN snapshot cadence, and expanded the dashboard smoke journey.
- F: added npm Dependabot, dependency-file checks, dependency refresh/check targets, and an expiring audit-waiver guard for the ignored torch advisory.
- G: throttled full NN analysis payload generation, cached recurring Space Invaders fonts, added incremental chart model updates, and added a `benchmark-quick` target.
- I: added stricter mypy ratchets for new/shared modules, granular source-size budgets, and a `clean` target. Largest tracked source files now pass the configured budgets, with `src/web/static/styles_layout.css` at 982 LOC and `src/game/asteroids.py` down to 709 LOC.

**Implementation update for top-20 request:**
- E: vendored Chart.js, chartjs-plugin-zoom, and Socket.IO under `src/web/static/vendor/`; removed inline dashboard scripts, external script/font origins, and template inline styles; tightened CSP to self-only scripts/styles; replaced visible query-token sessions with an HttpOnly SameSite cookie bootstrap redirect; added destructive-control throttling for delete/save/start-fresh/restart/clear-log flows; and made trusted legacy checkpoint fallback visible in model metadata/UI.
- H: replaced the missing README architecture image with the real architecture note, refreshed the repo tree, documented cookie bootstrap auth, dependency/audit gates, coverage threshold, benchmark target, and the real game registry contract.
- D: added deterministic Snake self-collision/tail-vacate tests, Asteroids ship-collision life/game-over tests, Space Invaders offscreen rendering/effects tests, checkpoint fallback-status tests, model-service security metadata tests, route/socket throttling tests, and frontend legacy-warning/CSP-adjacent tests. The Python coverage gate is now 80%.
- B/C: tightened dashboard route protocols, removed token exposure from rendered HTML, shifted normal browser API/control traffic to cookie auth, removed remaining dashboard template inline visibility, and kept automation token support through `X-Dashboard-Token`.
- Packaging: updated `pyproject.toml` package data so vendored dashboard assets are included in sdist and wheel; `make verify` confirmed the built artifacts include them.

The detailed findings below are preserved for traceability. Items covered by the implementation updates should be treated as completed or materially reduced; the main remaining work is larger runtime/game-module decomposition and frontend module-system cleanup.

---

## A - Architecture & Design - A-

The application has a much cleaner shape after the refactor work: `main.py:73-84` mostly dispatches into app services, `src/game/base_game.py:55-188` defines the shared game contracts, `src/game/__init__.py:64-125` centralizes game registration, and web concerns are split across routes, socket controls, contracts, model service, and metrics publishing. The grade is capped below A because the runtime classes and several game modules still combine orchestration, state mutation, rendering, persistence, and dashboard wiring in large files.

#### A1 - Split runtime loop orchestration into focused services
- **Where:** `src/app/interactive.py:54-260`, `src/app/interactive.py:560-745`, `src/app/headless.py:42-224`, `src/app/headless.py:520-760`
- **What's wrong:** `GameApp` and `HeadlessTrainer` still own configuration mutation, dashboard callback wiring, model lifecycle, episode stepping, metrics, checkpoint cadence, evaluation, and shutdown. These classes are now under the size budget, but each still has several independent reasons to change.
- **Impact:** Major - changes to one runtime concern can still regress save/load, dashboard updates, training stability, or shutdown behavior.
- **Fix:** Extract a `RuntimeSession` or `TrainingLoop` for stepping and episode completion, a dashboard binding object for callback registration and emissions, and a model lifecycle object for load/save/reset. Keep `GameApp` and `HeadlessTrainer` as thin composition roots.
- **Effort:** L
- **Grade lift:** B+ -> A- (removes the largest remaining coupling hotspot)

#### A2 - Normalize game modules around entities, rules, and rendering
- **Where:** `src/game/asteroids.py:31-929`, `src/game/breakout.py:31-868`, `src/game/pong.py:1-835`, `src/game/snake.py:1-739`, `src/game/space_invaders.py:1-878`
- **What's wrong:** Game modules still mix entity types, reward shaping, collision rules, rendering, human controls, and manual demo runners. Space Invaders and Breakout have started extracting helpers, but the pattern is not consistent across games.
- **Impact:** Moderate - gameplay fixes and test additions require reading broad files and can accidentally touch rendering or human-play behavior.
- **Fix:** Standardize each game around a thin `BaseGame` facade plus `*_entities.py`, `*_rules.py`, and `*_rendering.py` modules. Start with Asteroids and Snake because they have the weakest coverage and largest mixed responsibilities.
- **Effort:** L
- **Grade lift:** B+ -> A- (makes game behavior easier to extend and test independently)

#### A3 - Finish the configuration migration to nested typed groups
- **Where:** `config.py:67-697`, `config.py:563-608`
- **What's wrong:** The new typed views are useful, but the mutable `Config` class still stores game settings, network settings, training knobs, reward shaping, paths, logging, visualization, and runtime limits as one flat object. Callers can still mutate unrelated domains directly.
- **Impact:** Moderate - cross-domain config drift remains easy, especially when CLI, dashboard, and game defaults all share the same mutable object.
- **Fix:** Convert the typed views into owned nested dataclasses such as `config.game`, `config.training`, `config.network`, `config.runtime`, and game-specific settings. Add compatibility properties during migration, then move callers to the grouped API.
- **Effort:** M
- **Grade lift:** B+ -> A- (turns the current compatibility views into real boundaries)

#### ~~A4 - Move manual/demo runners out of production game modules~~ ✓ done 2026-06-17
- **Where:** `src/game/asteroids.py:932-983`
- **What's wrong:** Asteroids still contains a manual executable loop at the bottom of the production module. It is useful for development, but it keeps pygame demo concerns in the same file as game logic and entities.
- **Impact:** Minor - low runtime risk, but it adds noise to one of the largest source files and slightly blurs the module purpose.
- **Fix:** Move manual runners to `scripts/play_asteroids.py` or a shared demo launcher. Keep game modules import-only.
- **Effort:** S
- **Grade lift:** B+ -> A- (small cleanup that reinforces the game module boundary)

---

## B - Backend Quality - A

Backend quality is strong for a local app. Routes use typed payload contracts and explicit route-facing protocols, socket controls use a handler table plus explicit command-result normalization, checkpoint paths are constrained, model listing/deletion share a checkpoint repository, destructive controls have a small double-submit guard, and compatibility wrappers have been retired. The remaining backend debt is mostly larger runtime composition, not route/service correctness.

#### ~~B1 - Consolidate app and web checkpoint services~~ ✓ done 2026-06-17
- **Where:** `src/app/model_service.py:16-186`, `src/web/model_service.py:20-129`, `src/app/checkpoint_catalog.py:21-48`
- **What's wrong:** App save/load behavior and web model management share discovery helpers, but listing, deletion, compatibility inspection, path resolution, and metadata handling are still split between two service classes. This leaves room for behavior drift between CLI/runtime and dashboard flows.
- **Impact:** Moderate - checkpoint lifecycle bugs are expensive because they affect user saves, resume behavior, and destructive dashboard actions.
- **Fix:** Create a shared `CheckpointRepository` that owns allowed roots, opaque IDs, discovery, compatibility metadata, deletion, and sidecar lookup. Keep app and web services as thin adapters around that repository.
- **Effort:** M
- **Grade lift:** B+ -> A- (centralizes the persistence contract)

#### ~~B2 - Retire WebDashboard socket-control compatibility wrappers~~ ✓ done 2026-06-17
- **Where:** `src/web/server.py:243-317`, `src/web/socket_controls.py:85-283`, `src/web/socket_controls.py:381-395`
- **What's wrong:** `WebDashboard` still exposes a set of `_handle_*` wrappers that forward into `socket_controls`, even though live dispatch now goes through `dispatch_control`. Some wrappers depend on the imported Flask-SocketIO `emit`, which makes them awkward to reason about outside a socket event.
- **Impact:** Moderate - legacy wrappers increase the API surface and can preserve untested behavior after the real dispatcher changes.
- **Fix:** Remove wrappers that are no longer called, or mark them as deprecated only if external tests still import them. Route all tests through `socket_controls.dispatch_control` or the socket client.
- **Effort:** S
- **Grade lift:** B+ -> A- (shrinks a central backend compatibility surface)

#### ~~B3 - Replace broad `Any` protocol fields with narrower service protocols~~ ✓ done 2026-06-17
- **Where:** `src/web/routes.py:21-33`, `src/web/socket_controls.py:17-40`, `src/web/metrics_publisher.py:43-122`, `src/app/training_runtime.py:26-32`
- **What's wrong:** The code now has protocols, but several protocol attributes are still typed as `Any`, so mypy cannot prove route, publisher, and dashboard contracts end to end.
- **Impact:** Moderate - API drift can still hide behind `Any` even though the modules are otherwise type checked.
- **Fix:** Add small protocols for publisher, model service, config, and dashboard callbacks. Replace `Any` fields incrementally, then keep `disallow_untyped_defs` enabled for `src.web`.
- **Effort:** M
- **Grade lift:** B+ -> A- (turns the current typed shell into stronger static guarantees)

#### ~~B4 - Convert callback side effects into explicit command results~~ ✓ done 2026-06-17
- **Where:** `src/web/socket_controls.py:60-82`, `src/app/interactive.py:230-244`, `src/app/headless.py:173-207`
- **What's wrong:** Control handlers normalize callback return values, but runtime callbacks still rely on side effects on `WebDashboard`, config, and training state. This is readable at small scale, but harder to trace for restart, save-and-quit, load, and start-fresh flows.
- **Impact:** Minor - the current tests cover the main cases, but new controls will keep duplicating callback conventions.
- **Fix:** Introduce explicit command result objects for mutating controls and make runtime adapters translate those results to dashboard logs/events.
- **Effort:** M
- **Grade lift:** B+ -> A- (clarifies the backend control contract)

---

## C - Frontend Quality - A-

The dashboard frontend is functional and tested in both Node and Playwright. It has central action dispatch, safe DOM rendering for NN inspection panels and the model browser, dialog focus trapping, cookie-based auth bootstrap without exposing the token in rendered HTML, self-hosted browser dependencies, and expanded modal/browser smoke coverage. The grade is capped below A because the browser app still depends on globals, strict script order, and very large JS/CSS files.

#### C1 - Move dashboard scripts to importable modules
- **Where:** `src/web/templates/dashboard.html:528-538`, `src/web/static/dashboard_state.js:5-31`, `src/web/static/app.js:29-63`
- **What's wrong:** The frontend still loads many scripts in order and communicates through globals. Tests can exercise helpers, but the runtime module graph is implicit and missing-script failures are still a real class of bug.
- **Impact:** Moderate - frontend refactors remain harder than they should be, and load-order coupling slows future dashboard work.
- **Fix:** Convert static scripts to ESM or add a small build step with one entry module. Export explicit APIs for charts, dialogs, controls, games, NN panels, and startup.
- **Effort:** M
- **Grade lift:** B -> B+ (removes the largest frontend maintainability limiter)

#### ~~C2 - Remove the remaining `innerHTML` rendering paths~~ ✓ done 2026-06-17
- **Where:** `src/web/static/dashboard_controls.js:223-249`, `src/web/static/dashboard_controls.js:369-374`, `src/web/static/dashboard_games.js:18-30`, `src/web/static/dashboard_nn.js:905-918`
- **What's wrong:** Most dynamic rendering is safe now, but a few paths still use `innerHTML` for placeholders, model-list markup, option clearing, and panel cleanup. Some inputs are trusted or escaped, but the pattern invites future unsafe interpolation.
- **Impact:** Moderate - the dashboard is local, but model metadata and inspection content are exactly the places where accidental unsafe rendering tends to return.
- **Fix:** Replace placeholder markup and model list rendering with DOM construction and `replaceChildren`. Keep a tiny helper for clearing children instead of assigning `innerHTML = ''`.
- **Effort:** S
- **Grade lift:** B -> B+ (closes the remaining raw-rendering footgun)

#### C3 - Split large dashboard JS and CSS modules by responsibility
- **Where:** `src/web/static/dashboard_charts.js:1-922`, `src/web/static/dashboard_nn.js:15-833`, `src/web/static/styles_layout.css:1-982`, `src/web/static/styles_controls.css:1-814`
- **What's wrong:** The largest frontend files are still near the 1000-line limit and combine state, rendering, event handling, layout, and styling details. They pass the size gate, but they are not yet comfortable to modify.
- **Impact:** Moderate - future dashboard changes will be slower and more likely to create accidental regressions.
- **Fix:** Split chart model/state from Chart.js rendering, NN geometry from canvas rendering, and layout CSS from component CSS. Keep the existing Node tests around the extracted pure helpers.
- **Effort:** M
- **Grade lift:** B -> B+ (makes dashboard work less fragile)

#### ~~C4 - Finish accessibility behavior for the static model modal~~ ✓ done 2026-06-17
- **Where:** `src/web/templates/dashboard.html:513-525`, `src/web/static/dashboard_controls.js:364-390`, `tests/e2e/dashboard_smoke.mjs:104-108`
- **What's wrong:** Custom action dialogs trap and restore focus, but the persistent load-model modal only toggles a visible class and the smoke test checks basic open/close. It does not yet enforce focus restore, Escape behavior, or keyboard-only traversal for the model browser.
- **Impact:** Minor - this affects a secondary workflow, but it is a visible dashboard control surface.
- **Fix:** Reuse `DashboardDialogs` modal focus helpers or add a small modal controller for `#load-modal`. Add Playwright coverage for keyboard open, Escape close, and focus restoration.
- **Effort:** S
- **Grade lift:** B -> B+ (completes the dashboard modal accessibility story)

---

## D - Testing & Reliability - A

Testing is now a clear strength: `make verify` passes, CI runs the browser smoke, and the coverage gate is 80%. The suite covers Python unit/integration surfaces, Socket.IO controls, frontend helpers, Playwright dashboard smoke, performance budgets, dependency audit, hygiene, file size, and package build. Coverage is now 81.43% with 763 Python tests plus 35 Node tests; remaining work is deeper file-level coverage in the broader game/runtime surfaces and continued runtime coverage ratcheting.

#### D1 - Raise file-level coverage on low-covered game and rendering surfaces ✓ improved 2026-06-17
- **Where:** `src/game/space_invaders_rendering.py:12-240`, `src/game/space_invaders_entities.py:81-180`, `src/game/snake.py:1-739`, `src/game/asteroids.py:31-929`
- **What's wrong:** Total coverage passes at the new 80% floor, and Space Invaders rendering/entity coverage is now strong. The coverage output still shows weaker coverage in broader game-specific files, including roughly 52% for `snake.py`, 55% for `asteroids.py`, and 67% for `pong.py`.
- **Impact:** Major - game behavior and rendering are the most user-visible areas left under-tested.
- **Fix:** Continue adding headless tests for game rules, collision transitions, reward outputs, entity update edge cases, and render-helper geometry. Use pygame surfaces where needed, but keep assertions focused on state changes and bounded drawing behavior.
- **Effort:** M
- **Grade lift:** B+ -> A- (moves reliability from broad to deep)

#### D2 - Bring runtime orchestration back into coverage accounting
- **Where:** `pyproject.toml:74-85`, `src/app/headless.py:42-224`, `src/app/interactive.py:54-260`, `src/app/headless_dashboard.py:25-619`, `src/app/interactive_dashboard.py:22-669`
- **What's wrong:** Coverage currently omits the largest runtime orchestration and dashboard mixin files. That keeps the total gate stable, but it hides risk in save/load, start-fresh, shutdown, and dashboard callback behavior.
- **Impact:** Major - these are the paths users rely on during long training runs.
- **Fix:** Add fast headless tests around runtime setup, callback binding, save-and-quit, start-fresh, load failure, and shutdown behavior. Remove omitted files one group at a time and ratchet the gate after each group.
- **Effort:** M
- **Grade lift:** B+ -> A- (aligns the coverage signal with the real runtime surface)

#### D3 - Expand Playwright journeys for destructive and keyboard workflows ✓ improved 2026-06-17
- **Where:** `tests/e2e/dashboard_smoke.mjs:97-123`, `src/web/static/dashboard_controls.js:364-390`, `src/web/static/dashboard_games.js:49-80`
- **What's wrong:** The smoke test now covers connect, save, model modal, performance mode, settings, and game dropdown value. It still skips destructive confirmation branches, cancel paths, keyboard dialogs, model deletion cancellation, start fresh, and game-switch cancellation.
- **Impact:** Moderate - realistic browser regressions can still land outside the current happy-path smoke.
- **Fix:** Add disposable-server Playwright flows for start-fresh cancel/confirm, model delete cancel, game switch cancel, Escape close, and keyboard-only dialog traversal.
- **Effort:** M
- **Grade lift:** B+ -> A- (guards the dashboard's riskiest user workflows)

#### ~~D4 - Track performance regressions with repeatable benchmark artifacts~~ ✓ done 2026-06-17
- **Where:** `tests/test_performance_budgets.py:14-61`, `benchmark.py:69-180`
- **What's wrong:** The current performance tests catch catastrophic replay-buffer and dashboard payload regressions, and `benchmark.py` has richer scenarios, but there is no stored baseline or trend output for throughput-sensitive code.
- **Impact:** Moderate - gradual training throughput regressions can pass until they become obvious manually.
- **Fix:** Add a quick benchmark JSON artifact for replay sampling, single-env stepping, vectorized stepping, and dashboard snapshot serialization. Keep thresholds loose in CI and compare detailed trends locally.
- **Effort:** M
- **Grade lift:** B+ -> A- (makes performance reliability measurable instead of anecdotal)

---

## E - Security - A-

Security is strong for a local dashboard: sessions generate random tokens, the tokenized URL now bootstraps an HttpOnly SameSite cookie and redirects back to `/`, API requests and socket controls are token/cookie-gated, destructive controls have a small cooldown, model IDs avoid exposing raw paths, self-hosted dashboard assets allow a self-only script/style CSP, checkpoint loading prefers PyTorch's restricted loader, legacy compatibility fallback is surfaced in metadata, CodeQL runs, and Python/Node audits pass. The grade is capped below A by the local-dashboard trust model and the intentional torch advisory waiver.

#### ~~E1 - Remove `unsafe-inline` and third-party script allowances from CSP~~ ✓ done 2026-06-17
- **Where:** `src/web/server.py:34-44`, `src/web/templates/dashboard.html:9-22`
- **What's wrong:** The dashboard CSP allows inline scripts/styles and external CDN origins for Chart.js, chart zoom, Socket.IO, and Google Fonts. Startup has good failure handling, but the browser security boundary is still looser than it needs to be.
- **Impact:** Moderate - CSP is the backstop if future dashboard rendering accidentally injects unsafe content.
- **Fix:** Vendor or package browser assets under `src/web/static/vendor/`, move token bootstrap to a meta/data attribute path, remove inline script/style needs, and assert in tests that CSP no longer includes `unsafe-inline`.
- **Effort:** M
- **Grade lift:** B -> B+ (tightens browser containment and offline reliability)

#### ~~E2 - Replace query-string dashboard tokens with cookie bootstrap~~ ✓ done 2026-06-17
- **Where:** `src/web/server.py:193-214`, `src/web/routes.py:82-107`, `README.md:613-619`
- **What's wrong:** The dashboard URL includes `?token=...`. Referrer controls help, but query tokens can still appear in terminal output, browser history, copied URLs, screenshots, and local logs.
- **Impact:** Moderate - when the dashboard is bound to `0.0.0.0`, a leaked URL grants control access for that session.
- **Fix:** Use the printed URL as a one-time bootstrap token, set an HttpOnly/SameSite session cookie, redirect to `/`, and accept `X-Dashboard-Token` only for explicit automation.
- **Effort:** M
- **Grade lift:** B -> B+ (reduces accidental credential leakage)

#### ~~E3 - Add basic throttling for mutating controls~~ ✓ done 2026-06-17
- **Where:** `src/web/server.py:352-367`, `src/web/socket_controls.py:381-395`, `src/web/routes.py:196-219`
- **What's wrong:** Authorized clients can repeat save, delete, start-fresh, restart, and clear-log actions without per-action cooldowns. Token auth is the main defense, but accidental double-clicks or scripts can still spam filesystem and process-impacting operations.
- **Impact:** Minor - local trusted use keeps this from being severe, but the actions are destructive enough to deserve guardrails.
- **Fix:** Add a small per-token/per-action cooldown and return stable 429 or failed ack responses for throttled actions. Cover delete, save, start-fresh, and restart in route/socket tests.
- **Effort:** S
- **Grade lift:** B -> B+ (adds defense-in-depth around destructive actions)

#### ~~E4 - Make unrestricted checkpoint compatibility explicit and observable~~ ✓ done 2026-06-17
- **Where:** `src/utils/checkpoint_loader.py:34-66`, `src/app/model_service.py:45-50`, `src/app/model_service.py:74-80`
- **What's wrong:** Legacy app loads can opt into unrestricted PyTorch loading for files under trusted model directories. That is reasonable for local backwards compatibility, but users do not get a clear migration path away from unsafe legacy payloads.
- **Impact:** Minor - the fallback is directory-constrained, but checkpoint pickle loading remains a sensitive boundary.
- **Fix:** Add a warning in dashboard/model metadata when a checkpoint required unsafe fallback, and provide a one-time re-save/migration path that writes a restricted-loader-compatible checkpoint.
- **Effort:** M
- **Grade lift:** B -> B+ (keeps compatibility while lowering long-term risk)

---

## F - Dependencies & Tech Currency - A-

Dependency hygiene is strong. Runtime dependencies live in `pyproject.toml:7-28`, `requirements.txt:1-8` keeps historical install compatibility, `constraints.txt:1-23` pins reproducible versions, `package-lock.json:1-80` locks the Playwright dependency, Dependabot covers pip, npm, and GitHub Actions, `make audit` runs pip-audit with an expiring waiver guard, and `npm audit --audit-level=moderate` passed. Remaining work is mostly periodic maintenance rather than missing automation.

#### ~~F1 - Add Dependabot coverage for npm~~ ✓ done 2026-06-17
- **Where:** `.github/dependabot.yml:1-20`, `package.json:11-16`, `package-lock.json:17-79`
- **What's wrong:** Dependabot watches pip and GitHub Actions but not the npm ecosystem. The repo currently has a tiny Node dependency surface, but Playwright updates are important for browser compatibility and security.
- **Impact:** Moderate - stale browser tooling can silently break smoke tests or lag security patches.
- **Fix:** Add a weekly `npm` Dependabot entry for `/`, grouped separately from Python and Actions.
- **Effort:** S
- **Grade lift:** B+ -> A- (closes the last automated dependency-update gap)

#### ~~F2 - Give the ignored torch advisory an expiry workflow~~ ✓ done 2026-06-17
- **Where:** `Makefile:29-32`, `README.md:665`
- **What's wrong:** The dependency audit intentionally ignores `CVE-2025-3000` because no patched torch release is available on PyPI. The reason is documented, but there is no date, issue, or periodic reminder forcing re-evaluation.
- **Impact:** Moderate - permanent ignores can become invisible after the ecosystem catches up.
- **Fix:** Link the ignore to a tracked issue or comment with an expiry date, and add a CI note that fails or warns when the waiver deadline passes.
- **Effort:** S
- **Grade lift:** B+ -> A- (keeps the waiver intentional instead of permanent)

#### ~~F3 - Add a constraints refresh and parity check~~ ✓ done 2026-06-17
- **Where:** `pyproject.toml:7-28`, `requirements.txt:1-8`, `constraints.txt:1-23`, `scripts/bootstrap_dev.py:33-40`
- **What's wrong:** The dependency model is good, but dependency ranges and pinned constraints can drift unless someone manually refreshes them and confirms parity.
- **Impact:** Minor - reproducibility is already decent, but upgrades remain more manual than they need to be.
- **Fix:** Add `make deps-refresh` and `make deps-check` targets that regenerate constraints in a controlled way and verify that pinned packages satisfy `pyproject.toml` extras.
- **Effort:** M
- **Grade lift:** B+ -> A- (improves repeatability for future upgrades)

---

## G - Performance & Scalability - A-

Performance is a clear design concern: the replay buffer uses contiguous NumPy storage in `src/ai/replay_buffer.py:31-82`, batch insert/sample paths exist in `src/ai/replay_buffer.py:131-197` and `src/ai/replay_buffer.py:529-582`, vectorized headless training is implemented in `src/app/headless.py:520-760`, metrics snapshots can be bounded in `src/web/metrics_publisher.py:347-359`, NN analysis payloads are throttled, recurring Space Invaders fonts are cached, chart updates can be incremental, and a quick benchmark target exists. Remaining work is trend storage and deeper browser/module splitting for very long sessions.

#### ~~G1 - Avoid building full NN inspection weights on every emitted snapshot~~ ✓ done 2026-06-17
- **Where:** `src/app/training_runtime.py:134-175`, `src/web/server.py:530-547`, `src/web/metrics_publisher.py:465-527`
- **What's wrong:** `build_nn_snapshot` samples display weights but also converts full analysis weights and activations for inspection on each emitted NN snapshot. The publisher then suppresses weights for normal client updates, but the full Python-side conversion work has already happened.
- **Impact:** Moderate - high-speed training can spend avoidable CPU and serialization time on inspection data the user may not open.
- **Fix:** Make full inspection payload construction lazy or lower-frequency. Emit display snapshots every visual tick, and compute full inspection data only when a panel is open or on a slower cadence.
- **Effort:** M
- **Grade lift:** B+ -> A- (reduces overhead in the live dashboard hot path)

#### ~~G2 - Cache recurring render resources in Space Invaders rendering~~ ✓ done 2026-06-17
- **Where:** `src/game/space_invaders_rendering.py:207-240`, `src/game/space_invaders_entities.py:130-147`
- **What's wrong:** Render paths still construct fonts and temporary drawing resources in per-frame or per-entity methods. That is acceptable at small scale, but it adds avoidable overhead to visual training.
- **Impact:** Moderate - rendering overhead competes with training and dashboard updates in visual mode.
- **Fix:** Cache fonts and reusable surfaces on the game object during initialization or first render. Invalidate only when screen scale or theme settings change.
- **Effort:** S
- **Grade lift:** B+ -> A- (removes low-risk per-frame allocation)

#### ~~G3 - Add trend-oriented benchmark output~~ ✓ done 2026-06-17
- **Where:** `tests/test_performance_budgets.py:14-61`, `benchmark.py:69-180`
- **What's wrong:** The CI budget tests are useful but intentionally loose. The richer benchmark script can save results, but no standard local/CI target records comparable baselines.
- **Impact:** Moderate - gradual throughput changes are hard to explain after several refactors.
- **Fix:** Add `make benchmark-quick` to emit JSON with environment, replay sampling, single-env, vectorized, and dashboard serialization timings. Keep CI thresholds broad, but store artifacts for PR comparison.
- **Effort:** M
- **Grade lift:** B+ -> A- (makes performance changes reviewable)

#### ~~G4 - Make chart updates incremental for long sessions~~ ✓ done 2026-06-17
- **Where:** `src/web/static/dashboard_charts.js:18-25`, `src/web/static/dashboard_charts.js:784-868`
- **What's wrong:** Chart updates rebuild labels and assign full dataset arrays on each update. Downsampling exists, but long sessions still do more work than necessary when only one new episode has arrived.
- **Impact:** Minor - bounded snapshots keep this under control, but very long dashboard sessions can accumulate avoidable browser work.
- **Fix:** Track the last rendered episode and append only new points when the user is at the tail. Fall back to full rebuild when history is replaced or the user changes chart range.
- **Effort:** M
- **Grade lift:** B+ -> A- (smooths the longest-running dashboard sessions)

---

## H - Documentation & Onboarding - A-

The repo has a substantial README, safety notes, setup steps, developer checks, and a dedicated architecture note. The README now points at the real architecture note, shows a current repo map, explains cookie bootstrap auth and trusted checkpoint fallback metadata, documents the validation/audit gates, and describes the real game registry contract. The grade is capped below A mostly because the architecture docs are still hand-maintained rather than generated from source-of-truth tooling.

#### ~~H1 - Fix the missing architecture image and stale README structure~~ ✓ done 2026-06-17
- **Where:** `README.md:7`, `README.md:50-80`, `docs/architecture.md:1-64`
- **What's wrong:** The README references `docs/architecture.png`, but the repo contains `docs/architecture.md` and no architecture image. The README project tree also omits major current areas such as `src/app`, `src/web`, `src/utils`, CI scripts, and E2E tests.
- **Impact:** Moderate - first-time readers get a broken visual cue and an outdated map before they reach the better architecture doc.
- **Fix:** Replace the image link with a link to `docs/architecture.md` or add the generated image. Update the README tree to match the current repo layout.
- **Effort:** S
- **Grade lift:** B -> B+ (removes avoidable onboarding friction)

#### ~~H2 - Update game-extension docs to match the registry contract~~ ✓ done 2026-06-17
- **Where:** `README.md:669-706`, `src/game/__init__.py:58-125`, `src/game/base_game.py:55-188`
- **What's wrong:** The README says to register a new game in config and shows an incomplete `BaseGame` implementation. The real path is the game registry with single and vectorized constructors, metadata, action labels, and optional human-control protocols.
- **Impact:** Moderate - contributors following the docs will implement the wrong integration path.
- **Fix:** Rewrite the section around `BaseGame`, optional `BaseVecGame`, `GAME_REGISTRY`, metadata fields, and the tests to run after adding a game.
- **Effort:** S
- **Grade lift:** B -> B+ (makes extension docs executable)

#### ~~H3 - Keep architecture validation docs aligned with current gates~~ ✓ done 2026-06-17
- **Where:** `docs/architecture.md:52-64`, `Makefile:13-49`
- **What's wrong:** The architecture doc can drift from the Makefile coverage floor and validation mix after ratchets such as the current 80% gate. It should keep mentioning that `make verify` includes Playwright dashboard smoke and performance smoke.
- **Impact:** Minor - the Makefile is the source of truth, but stale docs create needless confusion.
- **Fix:** Update the validation command descriptions after each gate change, or generate that section from the Makefile in a lightweight docs check.
- **Effort:** S
- **Grade lift:** B -> B+ (keeps docs honest after tooling improvements)

---

## I - Developer Experience & Tooling - A

Developer tooling is one of the strongest areas. `Makefile:1-49` provides clear local gates, `scripts/bootstrap_dev.py:33-49` bootstraps Python/Node/hooks, `.pre-commit-config.yaml:1-19` runs core local checks, `mypy.ini:1-38` has stricter per-module ratchets, source-size budgets are now granular, `make clean` handles generated local artifacts, and CI runs Python 3.11 and 3.12 plus Playwright, CodeQL, dependency review, PR-title lint, package build, and release config. Remaining work is only incremental typing expansion.

#### ~~I1 - Expand strict mypy ratchets beyond the current modules~~ ✓ done 2026-06-17
- **Where:** `mypy.ini:1-38`, `src/app/interactive_dashboard.py:22-669`, `src/app/headless_dashboard.py:25-619`, `src/web/metrics_publisher.py:43-122`
- **What's wrong:** Strict mypy is enabled for several newer modules, but globally `disallow_untyped_defs = False` remains in place. Important dashboard mixins and publisher code still rely heavily on `Any`.
- **Impact:** Moderate - type checking is good, but still not as strong as the repo's current architecture wants to be.
- **Fix:** Add strict sections for one module group at a time: metrics publisher, dashboard mixins, app model service, then game factories. Replace `Any` with protocols while keeping CI green.
- **Effort:** M
- **Grade lift:** A- -> A (turns type checking into a broader safety net)

#### ~~I2 - Add more granular file-size budgets for near-limit files~~ ✓ done 2026-06-17
- **Where:** `.github/scripts/check_file_size.py:12-35`, `src/game/asteroids.py:1-983`, `src/web/static/styles_layout.css:1-982`, `src/web/static/dashboard_charts.js:1-922`, `src/web/static/dashboard_nn.js:1-919`
- **What's wrong:** The 1000-line source-size gate passes, but several files are close enough that small future additions can push them over the limit. CSS, JS, game logic, and runtime files do not all need the same ceiling.
- **Impact:** Minor - current tooling catches violations, but only after files are already near the cliff.
- **Fix:** Add optional per-root thresholds, for example 900 for JS/CSS dashboard files and 850 for game modules, then split the current near-limit files gradually.
- **Effort:** S
- **Grade lift:** A- -> A (keeps the refactor budget proactive)

#### ~~I3 - Add a cleanup target for generated local artifacts~~ ✓ done 2026-06-17
- **Where:** `Makefile:1-49`, `.gitignore:1-120`
- **What's wrong:** `make verify` builds package artifacts and leaves ignored local output such as `build/`, `dist/`, and egg-info caches. The worktree remains clean, but developers have to know the cleanup commands manually.
- **Impact:** Minor - not a correctness issue, just local loop polish.
- **Fix:** Add `make clean` that removes build outputs, caches, coverage files, Playwright reports, and egg-info directories without touching models or logs unless explicitly requested.
- **Effort:** S
- **Grade lift:** A- -> A (improves the day-to-day local loop)
