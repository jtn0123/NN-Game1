# Codebase Grade Report

**Project:** NN-Game1
**Audited:** 2026-06-02
**Stack:** Python, PyTorch, Pygame, Flask/SocketIO, pytest

## Summary

The historical report files in the repository were validated against current code:
`PROGRESS_REPORT.md`, `BUGS_AND_IMPROVEMENTS.md`, `30_BUGS_LIST.md`,
`30_MORE_BUGS_LIST.md`, and `POTENTIAL_BUGS.md`.

Most old critical replay-buffer and game-state items were already fixed. This
pass fixed all currently validated report items, including the larger follow-ups
that were initially separated for compatibility and lifecycle work.

## Fixed In This PR

#### D1 - Guard prioritized replay sampling
- **Where:** `src/ai/replay_buffer.py`
- **Validated:** `PrioritizedReplayBuffer.sample_no_copy()` already handled empty and uninitialized buffers, but `sample()` did not.
- **Fix:** Added matching clear `RuntimeError` guards and regression coverage.

#### D2 - Fail early on invalid agent state shapes
- **Where:** `src/ai/agent.py`
- **Validated:** Wrong-sized states could reach Torch tensor copies and fail with cryptic shape errors.
- **Fix:** Added single-state and batch-state validation, plus explicit empty-batch handling.

#### D3 - Validate training metrics history length
- **Where:** `src/ai/trainer.py`
- **Validated:** Non-positive `history_length` could make trimming behavior surprising.
- **Fix:** Added constructor validation and tests.

#### C1 - Correct Space Invaders state vector accounting
- **Where:** `src/game/space_invaders.py`
- **Validated:** The state-size formula allocated two unused values. The state writer filled 78 values while the formula expected 80 with the default config.
- **Fix:** Replaced the stale magic-number formula with named component counts and an assertion that writer count matches `state_size`.

#### B1 - Complete Phase 2 layer API
- **Where:** `src/web/server.py`
- **Validated:** `PROGRESS_REPORT.md` documented `GET /api/layers`, but only single-layer and neuron routes existed.
- **Fix:** Added `/api/layers` endpoint and tests.

#### B2 - Make layer analysis robust to empty arrays
- **Where:** `src/web/server.py`
- **Validated:** Empty activation, weight, or gradient arrays could crash statistics calls.
- **Fix:** Added empty-array defaults and histogram coverage.

#### C2 - Fix console timestamp precision
- **Where:** `src/web/server.py`
- **Validated:** Timestamp formatting sliced microseconds to centiseconds while the comment said milliseconds.
- **Fix:** Emits `HH:MM:SS.mmm` timestamps and tests the format.

#### E1 - Safer checkpoint loading
- **Where:** `src/ai/agent.py`, `src/web/server.py`, `main.py`, `src/utils/checkpoint_loader.py`
- **Validated:** Several model inspection/load paths still used `torch.load(..., weights_only=False)`.
- **Fix:** Added a shared checkpoint loader that tries `weights_only=True` first and only falls back to unrestricted loading for explicitly trusted local model directories.

#### A1 - Replace hard process exit in save-and-quit
- **Where:** `main.py`
- **Validated:** The SocketIO save-and-quit path called `os._exit(0)`, bypassing normal shutdown.
- **Fix:** Save-and-quit now requests loop shutdown by setting `running = False`; headless loops now honor that flag, including while paused.

#### C3 - Wire live Phase 2 neuron/layer data into training
- **Where:** `main.py`, `src/web/server.py`
- **Validated:** The frontend could fetch neuron/layer details, but training did not consistently populate live inspection data.
- **Fix:** `emit_nn_visualization()` now syncs Phase 2 layer analysis and neuron details from live NN snapshots, and the training paths pass full activation/weight data for inspection while still sending sampled data to the visualizer.

## Already Fixed Or Invalid

- PER save/load no longer calls a nonexistent superclass method and now persists beta/frame count.
- PER `sample_no_copy()` handles empty buffers, batch sizes larger than buffer size, and zero max-weight normalization.
- N-step early termination tests already cover correct terminal next-state handling.
- Breakout speed clamp, brick state bounds, and relevant division guards are already present.
- Space Invaders alien-bullet top-k selection and bullet x-position clamping are already present.
- Snake and Pong crash items from the older report are invalid or already guarded.
- Web win-rate empty-history division is already guarded.
- Dashboard epsilon/config validation is already materially improved.

## Remaining Follow-Ups

No validated grade-report items remain open from this pass.

## Validation

- `python -m pytest -q`
- Result: 470 passed, 1 existing PyTorch scheduler-order warning.
