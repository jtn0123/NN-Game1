# Codebase Grade Report

**Project:** nn-game1
**Audited:** 2026-06-25
**Stack:** Python 3.10-3.12 DQN/Pygame training app with Crystal Caves RL experiments, Flask/Socket.IO dashboard, vanilla JS dashboard modules, Node tests, Playwright smoke, and local experiment artifact tooling.

## Summary

| ID | Category | Grade | Items |
|----|----------|-------|-------|
| A | Architecture & Design | B+ | 4 |
| B | Backend Quality | B+ | 3 |
| C | Frontend Quality | B | 3 |
| D | Testing & Reliability | A- | 4 |
| E | Security | B | 3 |
| F | Dependencies & Tech Currency | B+ | 3 |
| G | Performance & Scalability | B+ | 3 |
| H | Documentation & Onboarding | B+ | 3 |
| I | Developer Experience & Tooling | A- | 3 |
| **Overall** | | **B+** | **29** |

**Top 5 highest-leverage fixes:** A1, D2, G1, H1, I1

**Current readiness verdict:** architecture is good enough to move back to NN improvement after a small stabilization pass. Do not keep doing broad file-splitting just to chase 500-line files; remaining simplification work should be tied directly to NN iteration speed, checkpoint safety, or experiment comparability.

**Validation snapshot:**

- Latest current-tree full gate from the persistence cleanup pass: `make verify PYTHON=/Users/justin/.pyenv/versions/3.12.11/bin/python` passed with `1094 passed`, coverage `77.21%`, dashboard smoke clean, dependency audit clean, file-size gate clean, and package build successful.
- Node dependency audit rechecked during this audit: `npm audit --audit-level=moderate` returned `found 0 vulnerabilities`.
- Source-size gate is green, but many files remain in the 850-999 LOC band, so line count is no longer a blocking risk but still shows complexity pockets.

---

## A - Architecture & Design - B+

The architecture is materially better than earlier in the week. NN extension contracts now live in `src/ai/extension_contracts.py`, experiment-only losses moved out of `agent.py` into `src/ai/agent_experiments.py`, PER+n-step is isolated in `src/ai/prioritized_n_step.py`, and the status-session runner has been split into `experiments/cc_status/` modules. The grade is capped below A because runtime orchestration, the core `Agent`, metrics publishing, and several experiment modules still contain large multi-responsibility methods.

#### A1 - Split runtime orchestration from app composition roots
- **Where:** `src/app/headless.py:63`, `src/app/headless.py:278`, `src/app/interactive.py:72`, `src/app/interactive.py:507`
- **What's wrong:** `HeadlessTrainer` and `GameApp` still mix CLI/config mutation, environment creation, model lifecycle, dashboard callbacks, training loops, metrics emission, and shutdown behavior. File size is under the hard 1000-line gate, but the classes still have several reasons to change.
- **Impact:** Major - runtime changes can accidentally affect dashboard state, checkpoint cadence, training speed, or model resume behavior.
- **Fix:** Extract a `TrainingSession` or `RuntimeSession` object that owns episode stepping, checkpoint decisions, and metric emission. Keep `HeadlessTrainer` and `GameApp` as thin composition roots that wire config, dashboard, and UI.
- **Effort:** L
- **Grade lift:** B+ -> A- (removes the largest remaining architecture hotspot)

#### A2 - Keep new NN methods behind provider-style extension points
- **Where:** `src/ai/extension_contracts.py:1`, `src/ai/agent_experiments.py:20`, `src/ai/agent.py:557`
- **What's wrong:** The extension contract exists, but existing route/demo/correction losses still sit in one large mixin and depend on agent internals. A future experiment could still grow `AgentExperimentMixin` instead of registering a standalone provider.
- **Impact:** Moderate - new NN ideas can become hard to compare or remove if they grow the agent surface again.
- **Fix:** Move the next new loss into a provider module that implements `AuxiliaryLossProvider`; do not add another branch to `AgentExperimentMixin` unless it is promoted. Add a provider smoke test and a recipe.
- **Effort:** M
- **Grade lift:** B+ -> A- (keeps future NN expansion plug-in oriented)

#### A3 - Split experiment run modes by lifecycle, not only by idea name
- **Where:** `experiments/cc_status/runs_demo.py:189`, `experiments/cc_status/runs_demo.py:465`, `experiments/cc_status/runs_route.py:217`, `experiments/cc_status/corrections.py:64`
- **What's wrong:** The runner split worked, but some run-mode functions are still 175-309 LOC and own setup, training, selection, eval, reporting, and artifact wiring in one function.
- **Impact:** Moderate - experiment bugs are likely to hide in long run modes, and future NN ideas will copy/paste flow code.
- **Fix:** Extract shared lifecycle helpers for `prepare_trainer`, `train_with_selection`, `evaluate_selected_checkpoint`, and `write_run_artifacts`. Keep recipe-specific code mostly as config and provider installation.
- **Effort:** M
- **Grade lift:** B+ -> A- (reduces duplicated experiment scaffolding)

#### A4 - Finish game-layer separation only where it supports Crystal Caves work
- **Where:** `src/game/crystal_caves.py:286`, `src/game/crystal_caves_logic.py:14`, `src/game/crystal_caves_rendering.py:13`, `src/game/crystal_caves_dressing.py:13`
- **What's wrong:** Crystal Caves has been decomposed, but the facade and mixins still form a large implicit class with game state spread across files. Further splitting every classic game would be expensive and not directly tied to the current NN blocker.
- **Impact:** Minor - this is maintainability friction, but not the main reason NN progress is slow.
- **Fix:** Only split Crystal Caves surfaces that block NN instrumentation or level-gen work. Defer broad refactors of Pong/Snake/Asteroids unless they become active targets.
- **Effort:** M
- **Grade lift:** B+ -> A- (focused cleanup without derailing NN work)

---

## B - Backend Quality - B+

The app/backend layer is solid for a local training tool. Route contracts, model services, restricted checkpoint loading, artifact validation, and dashboard APIs are tested and broadly organized. Remaining backend work is mostly about smaller service boundaries and reducing broad `Any` surfaces around dashboard/runtime integration.

#### B1 - Consolidate checkpoint inspection/listing behind one repository
- **Where:** `src/ai/agent_persistence.py:526`, `src/app/model_service.py:16`, `src/web/model_service.py:21`, `src/app/checkpoint_catalog.py:1`
- **What's wrong:** Checkpoint save/load is now cleaner, but static inspection, app model service, web model service, and checkpoint catalog still split related model lifecycle behavior. This creates drift risk around compatibility, opaque IDs, trusted dirs, and metadata display.
- **Impact:** Moderate - checkpoint regressions are high-cost because they affect resume, selected-eval validation, and dashboard loading.
- **Fix:** Create a shared `CheckpointRepository` for inspect/list/resolve/delete/metadata. Keep app/web services as adapters around it. Start by moving `inspect_model` and `list_models`.
- **Effort:** M
- **Grade lift:** B+ -> A- (centralizes model lifecycle semantics)

#### B2 - Split route registration by API group
- **Where:** `src/web/routes.py:65`
- **What's wrong:** `register_dashboard_routes` still defines all HTTP routes in one 192-line nested function. The protocol typing is better, but adding a route still means editing a broad registration body.
- **Impact:** Moderate - dashboard API changes are easy to review poorly when all routes are adjacent nested closures.
- **Fix:** Split into `register_status_routes`, `register_model_routes`, `register_control_routes`, and `register_game_routes`, each taking the same route context and CSP helpers.
- **Effort:** S
- **Grade lift:** B+ -> A- (small clarity win in a central API layer)

#### B3 - Reduce broad `Any` typing at app/web seams
- **Where:** `src/web/server.py:107`, `src/web/routes.py:16`, `src/app/headless.py:47`, `src/app/interactive.py:54`
- **What's wrong:** Mypy passes, but important integration boundaries still use `Any` for dashboard, config, callbacks, and model-service surfaces. That weakens the value of the new route contracts.
- **Impact:** Moderate - type checking may miss contract drift between runtime, dashboard, and model loading.
- **Fix:** Add small Protocols for dashboard callbacks, model services, and runtime dashboard bindings. Ratchet `check_untyped_defs` or `disallow_untyped_defs` for `src/web` and selected `src/app` modules.
- **Effort:** M
- **Grade lift:** B+ -> A- (turns existing contracts into stronger static checks)

---

## C - Frontend Quality - B

The dashboard is functional and test-covered, with split JS/CSS modules and Playwright smoke coverage. It is still a vanilla global-script application, and the NN dashboard script is right at the line budget. For a local tool this is acceptable, but the frontend is not yet as modular as the Python side.

#### C1 - Split `dashboard_nn.js` into focused NN panels/modules
- **Where:** `src/web/static/dashboard_nn.js:1`
- **What's wrong:** `dashboard_nn.js` is 999 LOC and sits exactly below the file-size gate. It owns NN data rendering, interactions, and panel updates, making it the frontend file most likely to fail the next feature addition.
- **Impact:** Moderate - NN inspection UI changes are likely while improving the agent, and this file is too close to the limit.
- **Fix:** Split into `dashboard_nn_state.js`, `dashboard_nn_render.js`, and `dashboard_nn_events.js`, preserving existing globals only through a thin compatibility export. Move tests with the split.
- **Effort:** M
- **Grade lift:** B -> B+ (removes the biggest frontend hotspot)

#### C2 - Move dashboard scripts toward explicit module imports
- **Where:** `src/web/templates/dashboard.html:522`, `src/web/static/dashboard_state.js:1`, `src/web/static/app.js:1`
- **What's wrong:** Script ordering and global state still define module contracts. Tests work, but the runtime graph is implicit.
- **Impact:** Moderate - frontend refactors are harder and missing-script/load-order failures remain possible.
- **Fix:** Convert to ESM or introduce a minimal build step. Load one dashboard entry module from the template and export/import explicit APIs.
- **Effort:** M
- **Grade lift:** B -> B+ (improves maintainability without changing product behavior)

#### C3 - Remove remaining unsafe HTML rendering patterns as panels grow
- **Where:** `src/web/static/dashboard_nn_panels.js:31`, `src/web/static/dashboard_controls.js:369`
- **What's wrong:** Some rendering paths still use template strings or `innerHTML`. Existing escaping and tests reduce risk, but the pattern is easy to misuse as NN metadata grows.
- **Impact:** Minor - data is local, but this is a recurring dashboard hardening issue.
- **Fix:** Prefer DOM builders and `textContent` for inspection/model metadata, or centralize safe rendering in one helper with hostile-input tests.
- **Effort:** S
- **Grade lift:** B -> B+ (small hardening and maintenance win)

---

## D - Testing & Reliability - A-

This category is now strong. The repo has 1000+ Python tests, JS tests, Playwright dashboard smoke, file-size gates, coverage gates, mypy, Ruff, Black, dependency audit, and package build verification. The remaining gap is not quantity; it is making the tests more targeted at NN correctness and reducing blind spots hidden by total coverage.

#### D1 - Add package/file-group coverage ratchets
- **Where:** `pyproject.toml:74`, coverage output for `src/app/crystal_curriculum.py`, `src/game/space_invaders_rendering.py`, `src/game/snake.py`, `src/game/asteroids.py`
- **What's wrong:** Total coverage is 77.21%, but low-covered files remain masked by stronger files. Several runtime/render/game surfaces are below the reliability level suggested by the aggregate number.
- **Impact:** Moderate - broad coverage can look healthy while specific gameplay/runtime paths remain under-tested.
- **Fix:** Add package-level or file-group coverage floors for `src/ai`, `src/game/crystal_caves*`, `src/web`, and `experiments/cc_status`. Keep a lower floor for legacy games if they are not active.
- **Effort:** M
- **Grade lift:** A- -> A (makes coverage signal harder to game accidentally)

#### D2 - Add NN learning-target contract tests before new algorithm changes
- **Where:** `src/ai/agent.py:557`, `src/ai/replay_buffer.py:713`, `src/ai/prioritized_n_step.py:17`, `src/ai/agent_experiments.py:29`
- **What's wrong:** Replay and persistence are well tested, but future NN expansion needs very direct tests for target computation, auxiliary loss composition, metric history emission, and provider failures.
- **Impact:** Major - a small target/loss bug can invalidate long training runs while all gameplay tests still pass.
- **Fix:** Add tests that build deterministic mini-batches and assert DQN target math, n-step/PER weighting, auxiliary contribution weights, metric history updates, and provider error behavior.
- **Effort:** M
- **Grade lift:** A- -> A (protects the riskiest future work)

#### D3 - Keep Playwright smoke in the main gate and add one NN-inspection journey
- **Where:** `Makefile:49`, `.github/workflows/ci.yml:54`, `tests/e2e/dashboard_smoke.mjs:1`
- **What's wrong:** Dashboard smoke now runs in `make verify` and CI, but it mostly covers dashboard shell, save/load, settings, and performance mode. NN inspection panels are still mostly unit-tested.
- **Impact:** Moderate - the visual NN surface is central to this project and can break in browser-only ways.
- **Fix:** Add a Playwright assertion that the NN inspection panel opens, receives mocked layer/action data, and does not throw console/page errors.
- **Effort:** S
- **Grade lift:** A- -> A (covers a high-value user-facing path)

#### D4 - Keep compatibility tests for checkpoint schema as first-class contracts
- **Where:** `tests/test_agent_persistence_contracts.py:1`, `src/ai/agent_persistence.py:17`
- **What's wrong:** Recent tests cover many save/load behaviors, but this file should become the required landing zone for any checkpoint schema change. Otherwise future feature work may bypass the contract.
- **Impact:** Minor - current state is good, but the risk returns when new metadata/replay fields are added.
- **Fix:** Add a short comment/docstring in the test file listing which schema changes require new contract tests. Do not rely only on round-trip tests.
- **Effort:** S
- **Grade lift:** A- -> A (preserves the hardening work already done)

---

## E - Security - B

Security is reasonable for a local training dashboard. Token auth, no-referrer headers, opaque model IDs, restricted PyTorch checkpoint loading, CodeQL, dependency review, pip audit, and npm audit are all in place. The cap is browser-side hardening: query-string tokens and `unsafe-inline` CSP remain.

#### E1 - Replace query-string dashboard tokens with cookie bootstrap
- **Where:** `src/web/server.py:147`, `src/web/server.py:183`, `src/web/routes.py:65`
- **What's wrong:** Dashboard URLs include `?token=...`. The no-referrer header helps, but tokenized URLs still appear in browser history, terminal output, copied links, and local logs.
- **Impact:** Moderate - LAN dashboard mode is supported, and leaked tokens grant mutating control until the session ends.
- **Fix:** Use a one-time bootstrap URL to set an HttpOnly/SameSite cookie, then redirect to `/` without the query token. Keep header token support for automation.
- **Effort:** M
- **Grade lift:** B -> B+ (reduces accidental credential leakage)

#### E2 - Remove `unsafe-inline` from CSP after vendoring browser assets
- **Where:** `src/web/server.py:31`, `src/web/templates/dashboard.html:9`
- **What's wrong:** The CSP allows inline scripts/styles and external CDN script sources. This weakens the containment value of CSP if a future dashboard render path mishandles HTML.
- **Impact:** Moderate - CSP is a useful backstop for the dashboard's dynamic UI.
- **Fix:** Move token bootstrap to meta/data attributes or cookies, vendor Chart.js/Socket.IO assets locally, remove inline style/script dependencies, and update CSP tests.
- **Effort:** M
- **Grade lift:** B -> B+ (hardens the browser surface)

#### E3 - Add basic cooldowns for destructive dashboard controls
- **Where:** `src/web/socket_controls.py:285`, `src/web/routes.py:160`, `src/web/server.py:332`
- **What's wrong:** Tokenized control APIs can be spammed without rate limiting. This is acceptable for local use but weak for LAN use or accidental repeated clicks.
- **Impact:** Minor - token auth is the main protection, but cooldowns would reduce damage from mistakes.
- **Fix:** Add per-token/per-action cooldowns for save, delete, start-fresh, restart, and load-model controls. Return stable 429 HTTP responses or socket ack errors.
- **Effort:** S
- **Grade lift:** B -> B+ (small resilience gain)

---

## F - Dependencies & Tech Currency - B+

Dependency health is good. Python dependencies are capped, CI tests Python 3.11/3.12, package builds pass, pip audit passes with an intentional torch CVE waiver, npm audit returns 0 vulnerabilities, CodeQL runs, and Dependabot is configured. The remaining work is mostly around making the one known torch advisory and Node 24 requirement easier to manage.

#### F1 - Track the intentional torch advisory as a dated work item
- **Where:** `Makefile:23`, `.github/scripts/run_dependency_audit.py:1`
- **What's wrong:** `CVE-2025-3000` is intentionally ignored because there is no patched torch release. The Makefile includes a TODO date, but the decision should also live in a dependency risk note or issue.
- **Impact:** Moderate - ignored CVEs can become stale if not revisited.
- **Fix:** Add a short dependency-risk note linking the ignore, its reason, and the recheck date. Remove the waiver as soon as a patched torch release is available.
- **Effort:** S
- **Grade lift:** B+ -> A- (keeps security debt visible)

#### F2 - Make Node 24 requirement explicit in setup docs and CI expectations
- **Where:** `package.json:9`, `README.md:129`, `.github/workflows/ci.yml:28`
- **What's wrong:** `package.json` requires Node 24. CI installs it, but the README setup section only calls out Python prerequisites before `make setup`.
- **Impact:** Minor - local dashboard test setup can fail confusingly on older Node versions.
- **Fix:** Add Node 24 to prerequisites and bootstrap error messaging.
- **Effort:** S
- **Grade lift:** B+ -> A- (smooths local setup)

#### F3 - Keep constraints and pyproject dependency ranges synchronized
- **Where:** `requirements.txt:1`, `constraints.txt:1`, `pyproject.toml:10`
- **What's wrong:** The project uses both install requirements and package metadata dependency ranges. This is fine, but drift can cause local dev, CI, and package users to test different dependency sets.
- **Impact:** Minor - dependency drift is a recurring source of hard-to-reproduce test failures.
- **Fix:** Add a small script/check that verifies each runtime dependency appears consistently in `requirements.txt` and `pyproject.toml`, with constraints only pinning where intended.
- **Effort:** S
- **Grade lift:** B+ -> A- (prevents dependency metadata drift)

---

## G - Performance & Scalability - B+

Performance is a real strength for the project: vectorized envs, replay sampling budgets, dashboard payload limits, status-session heartbeats, live metrics, and artifact validation all support fast iteration. The remaining need is trend visibility so regressions are caught before long NN runs are wasted.

#### G1 - Add a repeatable NN-run smoke benchmark before full experiments
- **Where:** `experiments/cc_status_session.py:1`, `experiments/cc_status/training.py:123`, `tests/test_performance_budgets.py:1`
- **What's wrong:** There are performance tests for replay sampling and dashboard snapshot size, but no quick benchmark that catches NN training-loop throughput regressions before launching multi-hour sessions.
- **Impact:** Major - slowdowns in vector env stepping, replay sampling, or dashboard metrics can waste expensive experiment time.
- **Fix:** Add a `make nn-smoke-benchmark` target that runs a tiny status-session recipe and records steps/sec, replay size, loss samples, live metrics writes, and artifact validation time. Store conservative thresholds.
- **Effort:** M
- **Grade lift:** B+ -> A- (protects the most expensive workflow)

#### G2 - Bound dashboard NN payloads the same way history payloads are bounded
- **Where:** `src/web/metrics_publisher.py:156`, `src/web/metrics_publisher_nn.py:13`, `src/web/static/dashboard_nn.js:1`
- **What's wrong:** History snapshots are bounded and tested, but NN inspection payloads can grow as more activations/layer metadata are exposed.
- **Impact:** Moderate - dashboard updates can become a hidden training slowdown.
- **Fix:** Add explicit caps for layer/neuron/action inspection arrays and a payload-size performance test.
- **Effort:** S
- **Grade lift:** B+ -> A- (prevents frontend instrumentation from hurting training)

#### G3 - Keep artifact size budgets for experiment outputs
- **Where:** `.Codex/artifacts/`, `experiments/cc_status/artifacts.py:1`, `CC_NN_CLEANUP_AUDIT.md:1`
- **What's wrong:** The cleanup removed replay-heavy checkpoint bloat, but future runs can still accidentally write huge full checkpoints or redundant traces.
- **Impact:** Moderate - artifact bloat slows comparisons and makes future sessions harder to reason about.
- **Fix:** Add artifact validation warnings for total artifact size, full replay checkpoint presence in rejected runs, and excessive JSONL trace size. Keep selected-checkpoint-only as the default.
- **Effort:** S
- **Grade lift:** B+ -> A- (preserves the cleaned experiment workflow)

---

## H - Documentation & Onboarding - B+

Documentation is much stronger than typical for an experimental RL repo. The Crystal Caves handoff, experiment tracker, metrics review, cleanup audit, and extension architecture all capture hard-earned decisions. The downside is that there are now many overlapping docs, and the root README still reflects the older general arcade-game project more than the current Crystal Caves NN workflow.

#### H1 - Create one current "start here for Crystal Caves NN" index
- **Where:** `CC_NN_HANDOFF.md:1`, `CC_NN_EXPERIMENT_TRACKER.md:1`, `CC_NN_EXTENSION_ARCHITECTURE.md:1`, `CC_NN_CLEANUP_AUDIT.md:1`
- **What's wrong:** The documentation is valuable but scattered. A future AI or human can easily read the wrong doc first and repeat archived work.
- **Impact:** Major - repeated failed NN ideas are one of the project's biggest time costs.
- **Fix:** Add `CC_NN_START_HERE.md` that links the current baseline, promotion gate, recipe commands, no-repeat list, and next recommended experiment. Keep it short and update it after each promoted baseline.
- **Effort:** S
- **Grade lift:** B+ -> A- (makes future sessions faster and less error-prone)

#### H2 - Update README project structure for the current architecture
- **Where:** `README.md:31`, `README.md:48`, `README.md:129`
- **What's wrong:** The README still presents the older simple structure with core files like `agent.py`, `network.py`, and `trainer.py`, but the current repo has web services, app runtimes, Crystal Caves modules, status-session experiments, and architecture docs.
- **Impact:** Moderate - onboarding through README does not match the actual codebase shape.
- **Fix:** Replace the old tree with the current domain layout and add a Crystal Caves experiment workflow subsection pointing to `CC_NN_START_HERE.md`.
- **Effort:** S
- **Grade lift:** B+ -> A- (aligns public docs with current reality)

#### H3 - Mark archived/negative experiments as machine-readable recipes or retired notes
- **Where:** `CC_NN_EXPERIMENT_TRACKER.md:16`, `docs/cc_nn_experiment_tracker/`, `experiments/cc_status/recipes.py`
- **What's wrong:** The no-repeat decisions are documented in prose. That is useful, but not as enforceable as recipe metadata that marks an idea as promoted, diagnostic, retired, or blocked.
- **Impact:** Moderate - future sessions may reintroduce failed settings because the code still supports them.
- **Fix:** Add status metadata to recipes/experiment descriptors and generate a short no-repeat table into docs.
- **Effort:** M
- **Grade lift:** B+ -> A- (turns documentation into workflow guardrails)

---

## I - Developer Experience & Tooling - A-

DevEx is one of the strongest areas now. `make verify` is comprehensive, CI mirrors the important checks, file-size limits are enforced, pre-commit exists, tests are broad, package builds pass, and the experiment runner produces validated artifacts. The remaining work is packaging the current dirty tree and adding a few NN-specific fast lanes.

#### I1 - Stabilize the current architecture branch into a reviewable PR
- **Where:** `git status`, `CC_NN_CLEANUP_AUDIT.md:1`, `.Codex/grade-report.md:1`
- **What's wrong:** The working tree has many modified and untracked files from valuable architecture, metrics, docs, and experiment-runner work. The code verifies, but it is not yet packaged into a clean review unit.
- **Impact:** Major - starting new NN experiments before stabilizing this branch risks mixing architecture changes with result changes.
- **Fix:** Do one PR-prep pass: group changes by durable area, ensure generated artifacts stay ignored, update start-here docs, run `make verify`, then commit/push/open PR. Do not add new NN method changes into the same review.
- **Effort:** M
- **Grade lift:** A- -> A (turns validated local work into durable project state)

#### I2 - Add fast NN-specific make targets
- **Where:** `Makefile:1`, `experiments/cc_status/recipes.py`, `tests/test_cc_status_session.py:1`
- **What's wrong:** `make verify` is excellent but broad. New NN methods need a faster preflight that runs provider tests, recipe expansion, one-episode smoke, artifact validation, and selected checkpoint inspection.
- **Impact:** Moderate - developers may skip expensive checks or run overly broad checks during rapid NN iteration.
- **Fix:** Add `make nn-smoke`, `make nn-provider-test`, and `make cc-status-smoke` targets that call the relevant tests/recipes with the standard Python path.
- **Effort:** S
- **Grade lift:** A- -> A (improves iteration speed for the next phase)

#### I3 - Ratchet type checks onto `experiments/cc_status`
- **Where:** `mypy.ini:1`, `experiments/cc_status/*.py`
- **What's wrong:** The core app is type-checked, but experiment modules still use many `Any`, `noqa`, and dynamic dict payloads. That is understandable for fast research, but the runner is now important infrastructure.
- **Impact:** Moderate - experiment result bugs are expensive because they can invalidate runs without failing app tests.
- **Fix:** Add a gradual mypy target for `experiments/cc_status` starting with artifact/recipe/promotion modules, then expand to run modes after shared lifecycle helpers land.
- **Effort:** M
- **Grade lift:** A- -> A (makes experiment infrastructure more trustworthy)
