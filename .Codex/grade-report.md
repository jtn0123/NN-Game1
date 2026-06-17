# Codebase Grade Report

Project: `nn-game1`
Audited: 2026-06-17
Scope: merged `origin/main` after PR #21 (`https://github.com/jtn0123/NN-Game1/pull/21`)
Local branch: `codex/grade-after-merge`
Execution update: A, B, C, F, and I items completed on 2026-06-17.

## Executive Summary

Overall grade: **A-**

The merge landed meaningful structural cleanup around the launcher and web runtime, and this follow-up pass completed the A/B/C/F/I refactor groups from the report. The project now has a stronger validation base: Python formatting, ruff linting, type checking, Node unit tests, dashboard smoke coverage, dependency audit, source-size checks, package build checks, and a broad pytest suite all pass on merged main.

The remaining ceiling is mostly outside the completed categories: security hardening, deeper runtime coverage, performance profiling, and documentation polish. The major size and ownership problems called out in A/B/C/F/I are now addressed.

## Validation Snapshot

- `make verify`: passed
  - Black check: passed
  - mypy: passed across 57 source files
  - Ruff lint: passed
  - Node tests: 24 passed
  - pytest: 668 passed
  - Coverage: 71.97%, above the 70% gate
  - Release config, hygiene, source-size, dependency-audit, and build checks: passed
- `make dashboard-smoke`: passed
- `make audit`: passed; the local helper installs `pip-audit` when missing.

## Grade Table

| ID | Dimension | Grade | Direction | Notes |
| --- | --- | --- | --- | --- |
| A | Architecture & Design | A- | Improved | Runtime, agent persistence, game, and metrics ownership are split into smaller modules. |
| B | Backend Quality | A- | Improved | Model path rules, typed payload contracts, error shape, and metrics types are centralized. |
| C | Frontend Quality | B+ | Improved | Dashboard JS/CSS are split, destructive prompts use dashboard dialogs, and mutable state has one owner. |
| D | Testing & Reliability | A- | Strong | 668 Python tests, JS tests, smoke tests, type checks, and CI matrix provide real protection. |
| E | Security | B | Solid | Token auth, CSP, CodeQL, dependency review, and audit job are present; CSP/local audit gaps remain. |
| F | Dependencies & Tech Currency | A- | Improved | Local audit is self-contained, Node 24 is declared, and ML dependency risk is documented. |
| G | Performance & Scalability | B+ | Solid | Training/runtime paths have useful performance focus; dashboard and game hot paths need decomposition. |
| H | Documentation & Onboarding | B | Adequate | README and Make targets help; architecture/security docs lag behind the new split. |
| I | Developer Experience & Tooling | A- | Improved | Ruff, dependency audit, source-size checks, release config, hygiene, tests, and build run in verification. |

## Top 5 Completed Fixes This Pass

1. **C1 - Split `src/web/static/app.js` by dashboard feature area.**
   Dashboard behavior is now split across focused feature scripts, with `app.js` reduced to 488 LOC.

2. **C2 - Split `src/web/static/styles.css` into domain CSS files.**
   Dashboard styles are now split into focused stylesheet files, with the root `styles.css` reduced to imports.

3. **A1 - Decompose the remaining app runtime classes.**
   Runtime dashboard/rendering helpers are split out, leaving `src/app/interactive.py` at 928 LOC and `src/app/headless.py` at 775 LOC.

4. **B1/B2/B3/B4 - Tighten backend contracts and service boundaries.**
   Shared model path rules, typed response payloads, normalized API errors, and extracted metrics types now back the dashboard API.

5. **F1/I1/I2/I3 - Harden local verification.**
   `make verify` now includes ruff, dependency audit, source-size checks, release config, hygiene, tests, and package build.

## A. Architecture & Design - A-

### A1 - Decompose remaining app runtime ownership

**Significance:** High
**Difficulty:** High
**Impact:** High
**Status:** Done 2026-06-17

`src/app/interactive.py` and `src/app/headless.py` remain large runtime coordinators. `GameApp` still owns pygame setup, event handling, render cadence, dashboard history sync, model load/save behavior, restart behavior, and neural-network visualization emission. `HeadlessTrainer` similarly owns config wiring, dashboard callbacks, train loops, vectorized training, and save/load flow.

**Why it matters:** These are central runtime paths. When one class owns so many reasons to change, small features can accidentally affect save behavior, dashboard behavior, or loop stability.

**Suggested shape:** Extract dashboard callback wiring, model lifecycle helpers, runtime loop helpers, and visualization publishing into focused modules before making deeper behavior changes.

### A2 - Split large game modules by entity, rules, and rendering

**Significance:** Medium
**Difficulty:** Medium
**Impact:** Medium
**Status:** Done 2026-06-17

`src/game/space_invaders.py` is 1942 LOC and `src/game/asteroids.py` is 1052 LOC. These files still combine game state, update rules, collisions, rendering, and input-facing behavior.

**Why it matters:** The game files are easier to test and evolve when entity behavior and rendering helpers are separated. It also makes performance work more targeted because collision/update paths can be profiled independently from drawing code.

### A3 - Split agent persistence and inspection concerns

**Significance:** Medium
**Difficulty:** Medium
**Impact:** Medium
**Status:** Done 2026-06-17

`src/ai/agent.py` is 1298 LOC and still includes `TrainingHistory`, `SaveMetadata`, the main `Agent`, save/load routines, model inspection, and model listing helpers.

**Why it matters:** Model persistence is a durable contract. Keeping metadata, checkpoint IO, inspection, and agent behavior in one file raises the risk that training changes affect compatibility behavior.

### A4 - Finish reducing compatibility pressure in `main.py`

**Significance:** Medium
**Difficulty:** Low
**Impact:** Medium
**Status:** Done 2026-06-17

PR #21 made progress, but `main.py` still imports many internal runtime types for compatibility and exposes launcher/web helpers.

**Why it matters:** A smaller launcher boundary makes packaging, CLI behavior, and imports easier to reason about. It also helps avoid accidental import-time side effects from pygame, torch, dashboard, or visualization modules.

## B. Backend Quality - A-

### B1 - Consolidate overlapping model-service responsibilities

**Significance:** Medium
**Difficulty:** Medium
**Impact:** Medium
**Status:** Done 2026-06-17

The app and web layers both have model-service concepts. That separation is useful, but the boundary should be explicit: filesystem discovery, model metadata, validation, and user-facing API payloads should each have a clear owner.

**Why it matters:** Model load/delete flows are sensitive. Clearer service ownership lowers the chance that web API changes drift away from runtime save/load expectations.

### B2 - Add typed dashboard payload contracts

**Significance:** Medium
**Difficulty:** Medium
**Impact:** Medium
**Status:** Done 2026-06-17

The web dashboard exchanges status, config, metrics, model, neural-network, and control payloads through dictionaries and JavaScript objects. Existing validation is useful, but the contracts are not centralized.

**Why it matters:** Typed contracts would make backend/frontend changes safer and reduce silent shape drift across `routes.py`, `socket_controls.py`, `metrics_publisher.py`, and `app.js`.

### B3 - Normalize API error shape across all endpoints

**Significance:** Low
**Difficulty:** Low
**Impact:** Medium
**Status:** Done 2026-06-17

`src/web/routes.py` has an `api_error` helper and many endpoints already use it. Some route-level auth and validation paths still repeat checks or return endpoint-specific shapes.

**Why it matters:** Stable error contracts make dashboard controls easier to handle and test. They also reduce frontend special cases.

### B4 - Decompose metrics publishing

**Significance:** Medium
**Difficulty:** Medium
**Impact:** Medium
**Status:** Done 2026-06-17

`src/web/metrics_publisher.py` is 1022 LOC and handles enough responsibilities to merit a split: live state publication, history, model/NN payloads, and dashboard-facing serialization.

**Why it matters:** Metrics publishing sits between training/runtime and frontend rendering. Smaller modules would make performance tuning and test coverage more direct.

## C. Frontend Quality - B+

### C1 - Split dashboard JavaScript by feature

**Significance:** High
**Difficulty:** Medium
**Impact:** High
**Status:** Done 2026-06-17

`src/web/static/app.js` is 3206 LOC. It owns dashboard state globals, socket setup, action dispatch, status rendering, console logs, controls, model modal workflows, settings, keyboard shortcuts, screenshot behavior, comparison UI, and `NeuralNetworkVisualizer`.

**Why it matters:** This is the largest behavior surface users touch. The current shape makes it harder to isolate regressions and makes small dashboard changes expensive to review.

**Suggested shape:** Split into modules such as `socket-client`, `dashboard-state`, `controls`, `models`, `settings`, `screenshots`, `comparison`, and `nn-visualizer`, while keeping existing tests around the extracted public functions.

### C2 - Split dashboard CSS by surface

**Significance:** High
**Difficulty:** Medium
**Impact:** High
**Status:** Done 2026-06-17

`src/web/static/styles.css` is 3290 LOC. It includes root tokens, layout, header, console, charts, controls, settings, comparison, modals, overlays, responsive rules, and neural-network styling.

**Why it matters:** Large CSS files invite accidental cascade coupling. A visual change to one dashboard area can have unrelated side effects, and reviews become search-heavy.

**Suggested shape:** Keep design tokens/global layout in a small root stylesheet, then split feature CSS by dashboard region.

### C3 - Replace blocking browser prompts with dashboard-native modal flows

**Significance:** Medium
**Difficulty:** Medium
**Impact:** Medium
**Status:** Done 2026-06-17

Some frontend flows still rely on browser confirmation/prompt-style interactions for important actions such as fresh starts, model operations, or game changes.

**Why it matters:** Native dashboard modals are easier to test, style, and make accessible. They also provide a more consistent UX for destructive or state-changing actions.

### C4 - Centralize frontend state ownership

**Significance:** Medium
**Difficulty:** Medium
**Impact:** Medium
**Status:** Done 2026-06-17

Dashboard state is still largely held in top-level globals and updated across feature code.

**Why it matters:** Central state ownership would reduce ordering bugs between socket events, fetch refreshes, and UI controls. It would also make frontend tests more focused.

## D. Testing & Reliability - A-

### D1 - Add targeted coverage for omitted runtime modules

**Significance:** Medium
**Difficulty:** Medium
**Impact:** High
**Status:** Partially done

Coverage is healthy overall at 71.49%, but `pyproject.toml` still omits several high-value runtime modules from coverage accounting, including `src/app/headless.py`, `src/app/interactive.py`, `src/app/process_control.py`, and UI/menu modules.

**Why it matters:** The omitted files are precisely where runtime regressions are most user-visible. Even focused smoke/unit coverage around lifecycle helpers would raise confidence.

### D2 - Expand dashboard E2E coverage beyond smoke

**Significance:** Medium
**Difficulty:** Medium
**Impact:** Medium
**Status:** Partially done

`make dashboard-smoke` passes and is valuable, but it does not deeply exercise model load/delete flows, settings edits, visualizer behavior, or destructive controls.

**Why it matters:** The dashboard is complex enough that frontend unit tests plus one smoke path will miss integration regressions.

### D3 - Add contract tests for dashboard payloads

**Significance:** Medium
**Difficulty:** Medium
**Impact:** Medium
**Status:** Not done

Backend routes, socket controls, metrics publishing, and frontend rendering all depend on shared payload shapes.

**Why it matters:** Contract tests would catch mismatches earlier than browser smoke tests and would support the frontend/backend splits recommended above.

### D4 - Keep release and package checks in the default path

**Significance:** Low
**Difficulty:** Low
**Impact:** Medium
**Status:** Done

`make verify` includes release config, hygiene, and package build checks. CI also runs a Python version matrix.

**Why it matters:** This is a strong guardrail. Keep it intact as modules are split.

## E. Security - B

### E1 - Tighten CSP by removing inline allowances

**Significance:** High
**Difficulty:** Medium
**Impact:** High
**Status:** Not done

`src/web/server.py` defines a useful Content Security Policy, and `routes.py` attaches security headers. The CSP still allows inline script/style behavior.

**Why it matters:** Token auth helps protect the local dashboard, but browser security posture is materially stronger when inline script/style allowances are removed or replaced with nonces/hashes and local assets.

### E2 - Reduce dashboard-token exposure in URLs

**Significance:** Medium
**Difficulty:** Medium
**Impact:** Medium
**Status:** Not done

The dashboard URL includes the access token as a query parameter for convenience.

**Why it matters:** Query tokens can appear in browser history, logs, screenshots, or copied URLs. For a local dashboard this is not catastrophic, but moving token handoff to a less leaky mechanism would improve the security grade.

### E3 - Centralize unsafe checkpoint fallback policy

**Significance:** Medium
**Difficulty:** Medium
**Impact:** Medium
**Status:** Partially done

Checkpoint loading has compatibility paths for legacy or unsafe formats. That may be necessary for older saved models, but the policy should be easy to audit from one place.

**Why it matters:** Model files are executable-risk adjacent in Python ML projects. A centralized policy makes it easier to separate trusted local saves from user-supplied files.

### E4 - Keep CI security checks mandatory

**Significance:** Medium
**Difficulty:** Low
**Impact:** Medium
**Status:** Done

The repo has CodeQL, dependency review, dependabot, and a dependency-audit job.

**Why it matters:** This is a strong baseline. The remaining improvement is making local audit behavior match CI more reliably.

## F. Dependencies & Tech Currency - A-

### F1 - Make local dependency audit self-contained

**Significance:** Medium
**Difficulty:** Low
**Impact:** Medium
**Status:** Done 2026-06-17

`make audit` failed locally because `pip_audit` is missing from the active environment.

**Why it matters:** A quality gate that fails due to missing tooling is easy to skip. Either include audit in the default dev setup or make the target explain/install the missing tool clearly.

### F2 - Align Node runtime expectations

**Significance:** Low
**Difficulty:** Low
**Impact:** Low
**Status:** Done 2026-06-17

CI uses Node 24 for dashboard tests, but `package.json` does not declare an `engines.node` expectation.

**Why it matters:** Declaring the tested Node range makes local dashboard test failures easier to diagnose.

### F3 - Keep ML dependency risk visible

**Significance:** Medium
**Difficulty:** Low
**Impact:** Medium
**Status:** Done 2026-06-17

The project depends on large ML/runtime packages such as torch, pygame, numpy, matplotlib, and Pillow.

**Why it matters:** These dependencies have heavier platform and security footprints than ordinary application libraries. The existing dependency-audit job helps, but version constraints and known exceptions should stay documented.

## G. Performance & Scalability - B+

### G1 - Profile dashboard rendering and event churn

**Significance:** Medium
**Difficulty:** Medium
**Impact:** Medium
**Status:** Not done

The dashboard has fetch timeouts, socket handling, charts, screenshots, and neural-network visualization in one frontend runtime.

**Why it matters:** Dashboard responsiveness depends on avoiding unnecessary DOM/chart work during active training. Splitting frontend modules will make profiling and throttling easier.

### G2 - Split game update/render paths before deeper optimization

**Significance:** Medium
**Difficulty:** Medium
**Impact:** Medium
**Status:** Not done

Large game modules mix update logic and rendering.

**Why it matters:** Performance fixes are easier when update loops, collision checks, state encoding, and rendering can be measured independently.

### G3 - Keep vectorized/headless training paths protected

**Significance:** Medium
**Difficulty:** Medium
**Impact:** High
**Status:** Partially done

The app already has headless and vectorized training paths, and the test suite covers broad training behavior.

**Why it matters:** These are the important performance surfaces. Refactors should preserve them with focused tests before changing behavior.

## H. Documentation & Onboarding - B

### H1 - Add a short architecture map

**Significance:** Medium
**Difficulty:** Low
**Impact:** Medium
**Status:** Not done

The README and Make targets are useful, but the current module layout deserves a concise architecture map after the recent cleanup.

**Why it matters:** New contributors need to know which layer owns CLI, runtime, game rules, model persistence, dashboard routes, socket controls, and frontend assets.

### H2 - Document dashboard security model

**Significance:** Medium
**Difficulty:** Low
**Impact:** Medium
**Status:** Not done

Token auth, local host assumptions, CSP, allowed origins, and dependency-audit behavior are present but not explained as a coherent security model.

**Why it matters:** The app is local-first, so the right security posture is practical rather than heavy. A short doc would clarify what is protected and what is intentionally trusted.

### H3 - Document local validation setup

**Significance:** Medium
**Difficulty:** Low
**Impact:** Medium
**Status:** Partially done

The Makefile gives good commands, but dependency-audit setup and dashboard smoke prerequisites should be easier to discover.

**Why it matters:** Good docs keep contributors from interpreting environment setup issues as product failures.

## I. Developer Experience & Tooling - A-

### I1 - Include dependency audit in local verification

**Significance:** Medium
**Difficulty:** Low
**Impact:** Medium
**Status:** Done 2026-06-17

`make verify` runs the important quality gates but does not include `make audit`.

**Why it matters:** CI catches audit failures, but local parity reduces PR churn and surprise failures.

### I2 - Run ruff in the standard check path

**Significance:** Low
**Difficulty:** Low
**Impact:** Medium
**Status:** Done 2026-06-17

`pyproject.toml` configures ruff, but the standard Make/CI path is built around Black, mypy, tests, and dashboard checks.

**Why it matters:** Ruff can catch simple bugs and cleanup issues quickly. It is most useful when it runs consistently.

### I3 - Add a source file size or complexity guard

**Significance:** Medium
**Difficulty:** Low
**Impact:** Medium
**Status:** Done 2026-06-17

The current largest files are all under the 1000-line budget:

| File | LOC |
| --- | ---: |
| `src/game/asteroids.py` | 983 |
| `src/web/static/styles_layout.css` | 982 |
| `src/game/breakout.py` | 967 |
| `src/app/interactive.py` | 928 |
| `src/web/static/dashboard_charts.js` | 922 |
| `src/web/static/dashboard_nn.js` | 919 |
| `src/game/space_invaders.py` | 878 |
| `src/ai/replay_buffer.py` | 872 |

**Why it matters:** The repo has repeatedly improved by reducing oversized files. A lightweight guard would keep that progress from regressing.

## Bottom Line

The app is in strong post-refactor shape. It is testable, packageable, linted, dependency-audited, and guarded against the oversized-file regression that was holding back architecture and frontend quality. The next grade jump is mostly about the categories not targeted in this pass: tighter CSP/token handling, deeper runtime/E2E coverage, performance profiling, and docs.
