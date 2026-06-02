# Codebase Grade Report

**Project:** NN-Game1
**Audited:** 2026-06-02
**Stack:** Python, PyTorch, Pygame, Flask/SocketIO, pytest

## Summary

The historical report files in the repository were validated against current code:
`PROGRESS_REPORT.md`, `BUGS_AND_IMPROVEMENTS.md`, `30_BUGS_LIST.md`,
`30_MORE_BUGS_LIST.md`, and `POTENTIAL_BUGS.md`.

Most old critical replay-buffer and game-state items were already fixed. This
pass fixed the remaining small/medium report items that were current, cohesive,
and safe for one PR.

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

## Already Fixed Or Invalid

- PER save/load no longer calls a nonexistent superclass method and now persists beta/frame count.
- PER `sample_no_copy()` handles empty buffers, batch sizes larger than buffer size, and zero max-weight normalization.
- N-step early termination tests already cover correct terminal next-state handling.
- Breakout speed clamp, brick state bounds, and relevant division guards are already present.
- Space Invaders alien-bullet top-k selection and bullet x-position clamping are already present.
- Snake and Pong crash items from the older report are invalid or already guarded.
- Web win-rate empty-history division is already guarded.
- Dashboard epsilon/config validation is already materially improved.

## Remaining Larger Follow-Ups

These were validated as real but intentionally left out of this PR because they
need broader design and compatibility decisions:

#### E1 - Safer checkpoint loading
- **Where:** `src/ai/agent.py`, `src/web/server.py`, `main.py`
- **What's left:** Several model inspection/load paths still use `torch.load(..., weights_only=False)`.
- **Why separate:** The checkpoints contain metadata and history beyond tensors, so a safe-loader migration needs compatibility fallback and explicit allowlisting.

#### A1 - Replace hard process exit in save-and-quit
- **Where:** `main.py`
- **What's left:** The SocketIO save-and-quit path still calls `os._exit(0)`.
- **Why separate:** Removing it cleanly needs lifecycle work across the pygame loop and web thread shutdown.

#### C3 - Wire live Phase 2 neuron/layer data into training
- **Where:** `main.py`, `src/web/static/app.js`, `src/web/server.py`
- **What's left:** The frontend can fetch neuron/layer details, but the training path still does not consistently publish live layer inspection updates.
- **Why separate:** This is feature wiring with UX/runtime implications, not just a guard fix.

## Validation

- `python -m pytest -q`
- Result: 464 passed, 1 existing PyTorch scheduler-order warning.
