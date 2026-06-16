# Frontend and Architecture Expansion

Audited: 2026-06-16

Scope: expansion of the existing grade report's Frontend Quality and Architecture & Design categories.

Baseline grades from `.Codex/grade-report.md`:
- Architecture & Design: B-
- Frontend Quality: C+

## Execution Status - 2026-06-16

Completed in this pass:
- C3, C5, C6, C9, C10, C11, C12
- A3, A4, A9

Substantial first pass, not fully closed:
- C4: added Node/frontend contract coverage and template action coverage; browser-level workflow tests remain.
- A1: extracted game construction into `src/app/game_factory.py`; broad runtime-module split remains.
- A2: shared performance presets extracted; broader trainer lifecycle duplication remains.
- A6/A7: extracted `src/web/contracts.py` and added `/api/performance-modes`; full route/socket/schema split remains.
- A8: `main.py` now uses the app-level model service for model resolution and checkpoint cleanup; web listing/deletion still uses its focused web model service.

Stability follow-up completed:
- Guarded dashboard Socket.IO control callbacks so callback failures return stable error acknowledgements instead of crashing the handler.
- Updated dashboard and launcher control flows to wait for server acknowledgements before showing destructive-action success, redirect, restart, or saved states.
- Ordered dashboard startup so initial status fetches run after DOM/chart initialization.
- Added visible fallback errors for missing dashboard core, Chart.js, or Socket.IO, and guarded chart updates before chart instances exist.

Still open large refactors:
- C1, C2, C7, C8
- A5, A10

## Frontend Quality Expansion

### C1 - Modularize the dashboard app shell

**Where**
- `src/web/static/app.js:12` starts a large set of global mutable dashboard state.
- `src/web/static/app.js:140` bootstraps charts, sockets, screenshot polling, footer timers, config, games, and stats.
- `src/web/static/app.js` is 3,867 lines and owns charts, socket handling, control commands, model UI, screenshots, game comparison, and the neural network visualizer.

**What's wrong**
The dashboard runs as one global script instead of a set of focused modules. Unrelated concerns share state and lifecycle, which makes it hard to reason about regressions.

**Impact**
Major - Frontend changes have a broad blast radius, and future features will keep adding to an already overloaded file.

**Fix**
Split `app.js` into modules such as `dashboard_state`, `charts`, `socket_client`, `controls`, `models`, `screenshots`, `game_stats`, and `nn_visualizer`. Keep a tiny boot module that wires dependencies together.

**Effort**
L

**Grade lift**
C+ -> B

### C2 - Extract chart setup and update logic

**Where**
- `src/web/static/app.js:156` starts `initCharts()`.
- `src/web/static/app.js:244`, `src/web/static/app.js:316`, and `src/web/static/app.js:382` repeat score, loss, and Q-value chart configuration.
- `src/web/static/app.js:1079` starts `updateCharts()`, which preserves viewport state, tracks panning, appends data, and refreshes all charts.

**What's wrong**
Chart setup and updates repeat similar options and lifecycle code across three charts. Small behavior changes must be copied correctly in multiple places.

**Impact**
Moderate - Chart bugs are likely to be fixed inconsistently, especially around pan, zoom, and history reset behavior.

**Fix**
Create a chart registry/factory with shared defaults, per-chart overrides, and a single update path keyed by metric.

**Effort**
M

**Grade lift**
C+ -> B-

### ~~C3 - Replace ad hoc `innerHTML` rendering with safe render helpers~~ ✓ done 2026-06-16

**Where**
- `src/web/static/app.js:1282` renders console log rows with `innerHTML`.
- `src/web/static/app.js:1777` writes model list markup into the modal.
- `src/web/static/app.js:2500` builds game stats markup from API data.
- `src/web/static/app.js:2898` and `src/web/static/app.js:3078` build neural inspection panels from API payloads.
- `src/web/templates/launcher.html:398` renders launcher game cards with `innerHTML`.

**What's wrong**
Some dynamic payloads are escaped through `DashboardCore`, but others are inserted directly into HTML templates. The pattern is inconsistent and easy to misuse.

**Impact**
Major - A malformed or unexpected API payload can break UI rendering and may become an injection path if metadata ever includes untrusted strings.

**Fix**
Move repeated HTML templates into safe render helpers that either create DOM nodes with `textContent` or enforce escaping at every interpolated field. Add tests for unsafe characters in game metadata, stats, logs, and inspection payloads.

**Effort**
M

**Grade lift**
C+ -> B

### C4 - Add real browser or DOM tests for dashboard workflows

**Where**
- `tests/js/dashboard_core.test.mjs:1` tests only the small `dashboard_core.js` helper module.
- `tests/test_frontend_templates.py:13` only checks for inline event handler attributes in templates.
- `src/web/static/app.js` and `src/web/templates/launcher.html` have no direct DOM workflow coverage.

**What's wrong**
Most frontend behavior is validated only indirectly through backend tests or static template checks. Socket events, controls, modals, charts, launcher cards, and inspection panels can regress without a failing frontend test.

**Impact**
Major - The dashboard is the main app surface, but its interactive behavior has thin automated coverage.

**Fix**
Add jsdom or Playwright tests for the main flows: dashboard boot, auth failure display, chart update/reset, pause/save/reset controls, model modal, launcher selection, game stats rendering, and neural inspection panel rendering.

**Effort**
M

**Grade lift**
C+ -> B

### ~~C5 - Move launcher inline CSS and JavaScript into static modules~~ ✓ done 2026-06-16

**Where**
- `src/web/templates/launcher.html:12` starts 270 lines of inline CSS.
- `src/web/templates/launcher.html:313` starts inline JavaScript.
- `src/web/templates/launcher.html:319` duplicates tokenized fetch behavior already present in `src/web/static/dashboard_core.js`.
- `src/web/templates/launcher.html:329` duplicates socket token setup behavior already present in `src/web/static/dashboard_core.js`.

**What's wrong**
The launcher is effectively a second standalone frontend app embedded in a template. It duplicates helpers and is harder to test, cache, and maintain.

**Impact**
Moderate - Launcher changes can drift away from dashboard behavior, especially auth and socket setup.

**Fix**
Move launcher CSS to a static stylesheet and launcher logic to a static JavaScript module that imports/reuses `DashboardCore`.

**Effort**
M

**Grade lift**
C+ -> B-

### ~~C6 - Centralize frontend action dispatch and command names~~ ✓ done 2026-06-16

**Where**
- `src/web/templates/dashboard.html` uses many `data-action` attributes for controls.
- `src/web/static/app.js:1836` dispatches actions through a string switch.
- `src/web/static/app.js:1434`, `src/web/static/app.js:1962`, `src/web/static/app.js:2156`, and related handlers emit stringly typed socket commands.
- `src/web/server.py:1646` and following socket handlers consume related action strings.

**What's wrong**
Command names are distributed across HTML, JavaScript, and Python with no shared contract. Renaming or adding a command can silently break one side.

**Impact**
Moderate - Control regressions are easy to introduce because there is no authoritative action list.

**Fix**
Define one frontend command registry and one backend control schema, then use tests to verify every template `data-action` has a handler and every emitted socket command has a backend handler.

**Effort**
M

**Grade lift**
C+ -> B-

### C7 - Add a frontend state/store layer for dashboard metrics

**Where**
- `src/web/static/app.js:800` directly clears chart datasets, logs, metric fields, and memory UI after reset events.
- `src/web/static/app.js:940` starts a large `updateDashboard()` function that writes many DOM nodes directly.
- `src/web/static/app.js:1079` separately mutates chart state from metrics payloads.

**What's wrong**
The frontend has no canonical state model. Socket payloads, DOM updates, charts, and control states are coupled together inside event handlers.

**Impact**
Moderate - Reset, reconnect, pause, and mode-change edge cases are harder to reason about because state exists both in globals and in the DOM.

**Fix**
Introduce a small dashboard store with reducers for `metrics`, `training_reset`, `control_ack`, `settings`, and `connection` events. Render from the store instead of patching DOM from each event handler.

**Effort**
M

**Grade lift**
C+ -> B

### C8 - Split the neural network visualizer into its own module

**Where**
- `src/web/static/app.js:2732` starts `NeuralNetworkVisualizer`.
- `src/web/static/app.js:2879` and `src/web/static/app.js:3047` fetch and render neuron/layer details.
- `src/web/static/app.js:3226` and later render canvas state and animation.
- The class occupies roughly the last quarter of `app.js`.

**What's wrong**
The visualizer is a substantial subsystem with fetch logic, panel templates, canvas rendering, event handling, and animation lifecycle embedded inside the dashboard script.

**Impact**
Moderate - Visualization changes increase the complexity of the entire dashboard file and make focused testing difficult.

**Fix**
Move the visualizer into `nn_visualizer.js`, split payload adapters from rendering, and test inspection payload rendering separately from canvas drawing.

**Effort**
L

**Grade lift**
C+ -> B

### ~~C9 - Improve accessibility for interactive controls~~ ✓ done 2026-06-16

**Where**
- `src/web/templates/dashboard.html:112`, `src/web/templates/dashboard.html:412`, and `src/web/templates/dashboard.html:482` use clickable card headers as `div` controls.
- `src/web/templates/dashboard.html:299` and `src/web/templates/dashboard.html:511` use icon-only buttons.
- `src/web/templates/dashboard.html:361` has a speed range input whose visible label is not associated with the input.
- `src/web/templates/dashboard.html:403` uses a placeholder as the only visible label for the save-as input.
- `src/web/templates/dashboard.html:507` defines a modal without dialog semantics.

**What's wrong**
Several controls are clickable visually but lack complete keyboard and assistive-technology semantics.

**Impact**
Moderate - Keyboard-only users and screen-reader users will have a degraded dashboard experience.

**Fix**
Use real buttons where possible. Add `aria-label`, `aria-expanded`, `aria-controls`, `role="dialog"`, `aria-modal`, focus management, and explicit labels for inputs.

**Effort**
S/M

**Grade lift**
C+ -> B-

### ~~C10 - Make tooltip content keyboard and touch accessible~~ ✓ done 2026-06-16

**Where**
- `src/web/static/styles.css:1583` defines tooltip behavior using `[data-tooltip]::before`.
- Dashboard template controls rely heavily on `data-tooltip` for explanations.

**What's wrong**
Tooltip content is exposed mainly through CSS hover behavior. It is not consistently available to keyboard focus, touch users, or screen readers.

**Impact**
Moderate - Important operational hints disappear for non-mouse users.

**Fix**
Add focus-visible tooltip behavior, `aria-describedby` support for non-decorative tooltips, and avoid relying on tooltip-only copy for required context.

**Effort**
S

**Grade lift**
C+ -> B-

### ~~C11 - Move overlay markup and styles into reusable components~~ ✓ done 2026-06-16

**Where**
- `src/web/static/app.js:1561` creates the shutdown overlay with inline styles and HTML.
- `src/web/static/app.js:2569` creates the restart banner with inline styles and HTML.
- `src/web/static/app.js:2662` creates the restarting overlay with inline styles and HTML.

**What's wrong**
Transient UI elements are built as large inline strings, mixing styling, markup, control wiring, and escaped data in the same functions.

**Impact**
Minor - This is manageable today, but it makes visual consistency and accessibility improvements harder.

**Fix**
Create reusable overlay/banner helpers with CSS classes, text-node assignment, and tested action bindings.

**Effort**
S/M

**Grade lift**
C+ -> B-

### ~~C12 - Fix performance-mode contract drift~~ ✓ done 2026-06-16

**Where**
- `src/web/templates/dashboard.html:328` describes Ultra mode as "Learn every 16 steps + 4 gradient updates + batch 256".
- `src/web/static/app.js:1997` sets Ultra mode to learn every 32 steps, 2 gradient updates, and batch size 128.
- `main.py:965` and `main.py:2830` apply the same 32/2/128 backend Ultra values.

**What's wrong**
The UI copy describes a different training configuration than the JavaScript and Python backend actually apply.

**Impact**
Moderate - Users choosing a performance mode can make decisions based on incorrect training parameters.

**Fix**
Make mode definitions data-driven from one backend endpoint or shared JSON payload, and render both button copy and applied settings from that source.

**Effort**
S

**Grade lift**
C+ -> B-

## Architecture & Design Expansion

### A1 - Split `main.py` into focused runtime modules

**Where**
- `main.py:136` starts `GameApp`.
- `main.py:2121` starts `HeadlessTrainer`.
- `main.py:3656` starts web-mode orchestration.
- `main.py:4048` starts `main()`.
- `main.py` is 4,227 lines.

**What's wrong**
One entrypoint owns CLI flow, pygame lifecycle, visual training, headless training, vectorized training, web dashboard launch, save/load, and dashboard callbacks.

**Impact**
Major - Architectural changes are risky because unrelated runtime modes are tightly colocated.

**Fix**
Extract modules for CLI orchestration, visual runtime, headless runtime, vectorized runtime, dashboard launch, and checkpoint integration. Keep `main.py` as a thin entrypoint.

**Effort**
L

**Grade lift**
B- -> B

### A2 - Extract shared trainer lifecycle behavior

**Where**
- `main.py:303` and `main.py:2369` wire similar dashboard callbacks.
- `main.py:548` and `main.py:2450` implement similar start-fresh behavior.
- `main.py:761` and `main.py:2599` sync dashboard history in parallel implementations.
- `main.py:833` and `main.py:2716` apply config updates in parallel implementations.
- `main.py:950` and `main.py:2816` apply performance modes in parallel implementations.

**What's wrong**
`GameApp` and `HeadlessTrainer` duplicate lifecycle and dashboard integration logic. Some differences are intentional, but the common behavior is not isolated.

**Impact**
Major - Bug fixes must be applied twice, and subtle divergence is likely across visual, headless, and web modes.

**Fix**
Create a shared trainer controller or mixin for dashboard callbacks, config updates, performance modes, history syncing, reset/start-fresh, and save-stop handling.

**Effort**
L

**Grade lift**
B- -> B

### ~~A3 - Use the game registry for vectorized environment selection~~ ✓ done 2026-06-16

**Where**
- `src/game/__init__.py:38` defines `GAME_REGISTRY` with `vec_class` entries.
- `src/game/__init__.py:112` exposes `get_vec_game()`.
- `main.py:2196` hard-codes vectorized class selection for each game.

**What's wrong**
The registry already knows the vectorized class, but `HeadlessTrainer` still repeats a game-name switch.

**Impact**
Moderate - Adding or renaming a game requires changes in multiple places and increases drift risk.

**Fix**
Replace the hard-coded branch with `get_vec_game(game_name)` and keep game-class metadata in the registry.

**Effort**
S/M

**Grade lift**
B- -> B

### ~~A4 - Formalize optional game capabilities~~ ✓ done 2026-06-16

**Where**
- `src/game/base_game.py:50` defines the core `BaseGame` abstract interface.
- `main.py:1162` handles human mode with game-name-specific branches and `getattr` checks.
- Individual game files implement methods such as `get_human_action`, `step_human`, `show_controls`, and `get_action_labels` outside the base contract.

**What's wrong**
Important runtime capabilities are optional conventions rather than explicit protocols.

**Impact**
Moderate - New games can appear compatible while missing methods required by a runtime mode.

**Fix**
Define capability protocols such as `HumanPlayableGame`, `ActionLabelProvider`, and `VectorizableGame`, then validate registry entries against the modes they advertise.

**Effort**
M

**Grade lift**
B- -> B

### A5 - Break the global `Config` object into scoped config groups

**Where**
- `config.py:19` defines one broad `Config` class for the whole project.
- `config.py:37` and following lines mix game-specific configuration into the same class.
- `config.py:172` keeps legacy state-size mapping for only two games.
- `config.py:197` has a global action-size value despite registry-specific games.
- `config.py:486` and `config.py:508` mix device/path logic into the same class.

**What's wrong**
Configuration mixes game metadata, training hyperparameters, model paths, device selection, rewards, and validation in one mutable object.

**Impact**
Moderate - Runtime code must understand too many config details, and adding games or modes increases config coupling.

**Fix**
Introduce scoped dataclasses such as `GameConfig`, `TrainingConfig`, `ModelConfig`, `DashboardConfig`, and `RuntimeConfig`. Derive game size/action metadata from the game registry.

**Effort**
M/L

**Grade lift**
B- -> B

### A6 - Split web server state, routes, sockets, and metrics publishing

**Where**
- `src/web/server.py:383` starts the large `MetricsPublisher` class.
- `src/web/server.py:1114` starts `WebDashboard`.
- `src/web/server.py:1444` registers HTTP routes as nested functions.
- `src/web/server.py:1626` registers Socket.IO events as nested functions.
- `src/web/server.py` is 2,086 lines.

**What's wrong**
The web layer combines state aggregation, API serialization, Flask route registration, Socket.IO control handling, auth helpers, and neural-inspection sync in one module.

**Impact**
Major - The backend web surface is difficult to change safely and encourages unrelated edits in the same file.

**Fix**
Split into `publisher`, `routes`, `socket_handlers`, `auth`, `schemas`, and `inspection_sync` modules. Keep `WebDashboard` as composition over those modules.

**Effort**
L

**Grade lift**
B- -> B

### A7 - Define typed API and socket payload contracts

**Where**
- `src/web/server.py:141` defines `TrainingState`.
- `src/web/server.py:220` defines `NNVisualizationData`.
- `src/web/server.py:1849` emits metrics and visualization payloads.
- `src/web/server.py:1918` builds phase-2 inspection payloads.
- `src/web/static/app.js:940`, `src/web/static/app.js:2898`, and `src/web/static/app.js:3078` consume payloads by ad hoc property access.

**What's wrong**
Python and JavaScript exchange complex payloads without a versioned schema shared across both sides.

**Impact**
Major - Field renames or partial payloads can break the dashboard at runtime while backend tests still pass.

**Fix**
Add explicit payload schemas for metrics, controls, game stats, model lists, and inspection data. Generate or mirror TypeScript/JSDoc typedefs for frontend validation and add contract tests using representative payloads.

**Effort**
M/L

**Grade lift**
B- -> B+

### A8 - Consolidate checkpoint/model service boundaries

**Where**
- `src/web/model_service.py:19` defines a web-facing `ModelService`.
- `src/app/model_service.py:14` defines another `ModelService`.
- `src/app/training_runtime.py:26` defines model-path resolution helpers.
- `main.py:641`, `main.py:2018`, and `main.py:3428` still own substantial load/save behavior.

**What's wrong**
Model listing, model-path resolution, history construction, retention cleanup, and save/load operations are split across multiple services and still partly embedded in `main.py`.

**Impact**
Moderate - Checkpoint behavior is harder to audit and easy to make inconsistent between visual, headless, and web flows.

**Fix**
Create one app-level checkpoint service with clear methods for resolve, list, save, load, history, retention, and web serialization. Have web code depend on that service instead of a parallel implementation.

**Effort**
M

**Grade lift**
B- -> B

### ~~A9 - Retire or merge duplicate web launcher paths~~ ✓ done 2026-06-16

**Where**
- `main.py:3693` starts the primary web-mode flow.
- `main.py:3886` starts `run_web_launcher()`.
- Both paths manage launcher/dashboard startup and transition into training.

**What's wrong**
The project has more than one web-launch orchestration path with overlapping responsibilities and different behavior.

**Impact**
Moderate - Future launcher fixes may land in one path while the other path remains stale.

**Fix**
Choose one canonical web-launch path. Delete or redirect the older path, and add a test that verifies the selected game/mode transition contract.

**Effort**
M

**Grade lift**
B- -> B

### A10 - Decompose large game implementation files

**Where**
- `src/game/space_invaders.py` is 1,931 lines and contains gameplay, collision handling, state extraction, rendering, human controls, and vectorized environment code.
- `src/game/asteroids.py` is 1,052 lines with similar mixed concerns.
- `src/game/breakout.py` is 965 lines and includes both single-game and vectorized implementations.

**What's wrong**
Individual game modules mix physics, rendering, state encoding, reward shaping, human controls, and vectorized runtime support in one file.

**Impact**
Moderate - Game changes have high local complexity, and shared improvements across games are difficult to reuse.

**Fix**
Split large games into focused modules for entities/physics, state encoding, rendering, human input, and vectorized wrappers. Start with `space_invaders.py` because it is the largest.

**Effort**
L

**Grade lift**
B- -> B

## Validation

Validated after writing the expansion:

- `git diff --check` - passed.
- `node --test tests/js/*.test.mjs` - 11 passed.
- `python -m pytest tests/test_frontend_templates.py tests/test_game_registry_contract.py tests/test_game_contracts.py tests/test_training_runtime.py tests/test_main_lifecycle.py tests/test_web_socket_controls.py -q` - 69 passed.
- `python -m pytest tests/test_web_server.py tests/test_web_routes.py tests/test_web_nn_inspection.py tests/test_phase2_neuron_inspection.py -q` - 76 passed.

Validated after implementation pass:

- `git diff --check` - passed.
- `node --test tests/js/*.test.mjs` - 13 passed.
- `python -m pytest tests/test_game_factory.py tests/test_app_model_service.py tests/test_training_runtime.py tests/test_main_lifecycle.py tests/test_game_registry_contract.py tests/test_frontend_templates.py tests/test_web_routes.py tests/test_web_socket_controls.py -q` - 86 passed.
- `python -m compileall -q main.py src/app src/game src/web tests` - passed.
- Browser smoke check against launcher-mode `WebDashboard` - loaded 5 game cards, Socket.IO connected, selecting Breakout enabled `TRAIN BREAKOUT`.

Validated after C5 CSS extraction:

- `python -m pytest tests/test_frontend_templates.py tests/test_web_routes.py -q` - 33 passed.
- `node --test tests/js/*.test.mjs` - 13 passed.
- `git diff --check` - passed.
- Browser repeat smoke was attempted against launcher-mode `WebDashboard`, but the in-app browser tab crashed before page verification; the temporary server was stopped.

Final validation after C5 continuation:

- `git diff --check` - passed.
- `node --test tests/js/*.test.mjs` - 13 passed.
- `python -m pytest tests/test_game_factory.py tests/test_app_model_service.py tests/test_training_runtime.py tests/test_main_lifecycle.py tests/test_game_registry_contract.py tests/test_frontend_templates.py tests/test_web_routes.py tests/test_web_socket_controls.py -q` - 87 passed.
- `python -m compileall -q main.py src/app src/game src/web tests` - passed.

Validated after stability follow-up:

- `git diff --check` - passed.
- `node --test tests/js/*.test.mjs` - 17 passed.
- `python -m pytest tests/test_web_socket_controls.py tests/test_frontend_templates.py tests/test_web_routes.py tests/test_web_nn_inspection.py tests/test_phase2_neuron_inspection.py -q` - 81 passed.
- `python -m compileall -q main.py src/app src/game src/web tests` - passed.
- Browser smoke against launcher-mode `WebDashboard` - loaded 5 cards, Socket.IO connected, selected Breakout, clicked `TRAIN BREAKOUT`, and reached `Training breakout...` with no console errors.

Final focused validation after stability follow-up:

- `git diff --check` - passed.
- `node --test tests/js/*.test.mjs` - 17 passed.
- `python -m pytest tests/test_game_factory.py tests/test_app_model_service.py tests/test_training_runtime.py tests/test_main_lifecycle.py tests/test_game_registry_contract.py tests/test_frontend_templates.py tests/test_web_routes.py tests/test_web_socket_controls.py tests/test_web_nn_inspection.py tests/test_phase2_neuron_inspection.py -q` - 117 passed.
- `python -m compileall -q main.py src/app src/game src/web tests` - passed.
- Main dashboard browser smoke - first in-app browser tab crashed before DOM inspection; a fresh-tab retry loaded the dashboard, reached `Connected`, found metrics, and had no console errors.

Validated after browser crash fix:

- `git diff --check` - passed.
- `node --check src/web/static/app.js && node --check src/web/static/launcher.js && node --check src/web/static/dashboard_core.js` - passed.
- `node --test tests/js/*.test.mjs` - 20 passed.
- `python -m compileall -q main.py src/app src/game src/web tests` - passed.
- `python -m pytest tests/test_game_factory.py tests/test_app_model_service.py tests/test_training_runtime.py tests/test_main_lifecycle.py tests/test_game_registry_contract.py tests/test_frontend_templates.py tests/test_web_routes.py tests/test_web_socket_controls.py tests/test_web_nn_inspection.py tests/test_phase2_neuron_inspection.py -q` - 117 passed.
- Main dashboard browser smoke - loaded in a fresh in-app browser tab, reached `Connected`, clicked `Save`, returned the button to enabled state, logged `Save requested`, and had no console warnings or errors.
