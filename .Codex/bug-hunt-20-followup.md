# Bug Hunt: 20 More Validated Areas

Branch: `codex/bug-hunt-20`
Scope: second pass after the first 20-finding remediation.
Remediation status: all 20 findings below are fixed on this branch.

Grading key:

- Impact: A = highest user/security/correctness risk, B = meaningful risk, C = moderate risk, D = low risk.
- Difficulty: A = easy, B = moderate, C = hard or design-heavy.

| # | Area | Layman explanation | Evidence before fix | Impact | Difficulty |
|---|------|--------------------|---------------------|--------|------------|
| 1 | Socket speed control accepted non-numeric input | The dashboard could say "set speed to fast" and the control path would report success or let bad data reach runtime code. | Socket repro sent `{"action": "speed", "value": "fast"}` and received success before validation was added. | B | A |
| 2 | Socket config changes accepted non-object payloads | A malformed dashboard client could send a list instead of a settings object, leading to confusing no-op success or runtime callback risk. | Socket repro sent `{"action": "config_change", "config": []}` and received success. | B | A |
| 3 | Socket performance mode accepted unknown modes | The app could acknowledge a mode like `warp` even though the runtime only knows real modes. | Socket repro sent `{"action": "performance_mode", "mode": "warp"}` and received success. | C | A |
| 4 | Socket select-game accepted unknown games | The dashboard could request a game that does not exist and still get a successful response. | Socket repro sent `{"action": "select_game", "game": "not_a_game"}` and received success. | B | A |
| 5 | Socket restart-with-game did not validate game registry | A restart callback could be invoked with an invalid game name, moving the failure deeper into the app. | Repro with a registered restart callback showed the handler only guarded callback presence, not game validity. | B | A |
| 6 | Socket save-as accepted non-string filenames | A bad client could send an object as a filename and rely on downstream callbacks to crash or stringify it oddly. | Socket repro sent `{"action": "save_as", "filename": {"bad": true}}` and received success when no callback was installed. | B | A |
| 7 | Visual runtime speed setter crashed on malformed speed | A bad speed value could raise instead of being ignored safely. | Direct repro called `GameApp._set_speed("fast")`; now it logs a warning and keeps the previous speed. | B | A |
| 8 | Headless config path allowed invalid learning rates | The visual path already guarded bad learning rates, but headless training could receive `NaN` and poison the optimizer. | Direct repro called `HeadlessTrainer._apply_config({"learning_rate": nan})`; before fix it could write the value through. | A | A |
| 9 | Config allowed invalid runtime loop cadences | Values like `LOG_EVERY=0` or `SAVE_EVERY=0` can cause divide-by-zero or disabled persistence behavior. | Config repro set zero cadence values and validation did not reject them. | B | A |
| 10 | Config allowed invalid replay/model sizing | Values like `MEMORY_SIZE=0` or hidden layers containing `0` could create broken neural-net or replay-buffer state. | Config repro set `MEMORY_SIZE=0` and `HIDDEN_LAYERS=[64, 0]`; validation did not reject all of them. | A | A |
| 11 | Config allowed invalid PER, N-step, and evaluation knobs | Bad replay or evaluation values can crash training in less obvious places. | Config repro set `PER_BETA_FRAMES=0`, `N_STEP_SIZE=0`, and `EVAL_MAX_STEPS=0`; validation did not reject them. | A | A |
| 12 | ReplayBuffer.sample accepted zero batch size | Training code could ask for an empty batch and get meaningless arrays instead of a clear error. | Direct repro called `ReplayBuffer.sample(0)` and got an empty batch. | B | A |
| 13 | ReplayBuffer.sample_no_copy accepted zero batch size | The faster sampling path had the same zero-batch hole. | Direct repro called `ReplayBuffer.sample_no_copy(0)` and got empty views. | B | A |
| 14 | ReplayBuffer.is_ready treated zero as ready | Readiness checks could say the buffer is ready for a zero-sized training batch, hiding caller bugs. | Direct repro called `ReplayBuffer.is_ready(0)` and got `True`. | C | A |
| 15 | ReplayBuffer.push_batch accepted empty or mismatched batches | Vectorized training could silently broadcast or partially write bad experience arrays. | Direct repro pushed mismatched action/reward/done lengths and empty arrays without a clear validation error. | A | B |
| 16 | Prioritized replay sampling with zero batch crashed unclearly | PER used NumPy reductions on empty arrays, producing cryptic errors instead of actionable validation. | Direct repro called `PrioritizedReplayBuffer.sample(0)` and hit a NumPy empty-array error. | B | A |
| 17 | Prioritized replay no-copy sampling had the same zero-batch crash | The optimized PER sampling path failed the same way as the copied path. | Direct repro called `PrioritizedReplayBuffer.sample_no_copy(0)` and hit the same class of error. | B | A |
| 18 | Prioritized replay priority updates accepted invalid indices/errors | Negative indices updated the last slot, mismatched arrays were unclear, and non-finite TD errors could corrupt priorities. | Direct repro called `update_priorities([-1], [1.0])` and mutated the wrong entry. | A | A |
| 19 | N-step replay accepted impossible settings | `n_steps=0` or invalid gamma values create nonsensical return calculations. | Direct repro constructed `NStepReplayBuffer(..., n_steps=0)` and invalid gamma values without an immediate error. | B | A |
| 20 | Game actions and vector batches were not consistently validated | Some games accepted impossible actions, one crashed with `KeyError`, and vector envs could step only part of a batch. | Direct repro used invalid actions for Breakout, Pong, Snake, Space Invaders, Asteroids, and short/invalid vector action batches. | A | B |

## Fix Summary

- Added shared game action validators and wired them into all five single-game and vectorized environments.
- Hardened replay buffers for zero-sized samples, malformed vector batches, invalid priority updates, and invalid N-step parameters.
- Hardened socket controls before callbacks run, including speed, config payloads, performance modes, game names, restart targets, and save-as filenames.
- Aligned headless runtime config validation with the visual path and made malformed speed changes safe.
- Expanded config validation for runtime cadences, replay settings, evaluation limits, and model hidden-layer sizes.
- Added regression tests for sockets, game contracts, replay buffers, config validation, and lifecycle/config controls.

## Verification

- Focused regression suite:
  - `python -m pytest tests/test_web_socket_controls.py tests/test_game_contracts.py tests/test_replay_buffer.py tests/test_config.py tests/test_main_lifecycle.py -q`
  - Result: `147 passed`.
- Direct repro script:
  - Verified 36 concrete fixed failure paths across replay buffers, PER, N-step replay, game actions, vector batches, runtime speed/config handling, and config validation.
- Full gate:
  - `make check`
  - Result: passed.
  - Black: passed.
  - Mypy focused targets: passed.
  - Dashboard JavaScript tests: 7 passed.
  - Python tests: 625 passed.
  - Coverage: 51.98%, above the 40% gate.
- Dependency audit:
  - Temporary CI-style venv ran `python -m pip_audit -r requirements.txt`.
  - Result: no known vulnerabilities found; local package `nn-game1` was skipped because it is not published on PyPI.
