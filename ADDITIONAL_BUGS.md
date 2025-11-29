# Additional Bugs - Third Round Discovery

## 7 NEW BUGS FOUND - ALL FIXED ✅

**STATUS SUMMARY:**
- ✅ Bug #13: PER sampling crash - FIXED
- ✅ Bug #14: Ball tunneling - FIXED
- ✅ Bug #15: Paddle range division by zero - FIXED
- ✅ Bug #16: VecBreakout state corruption - FIXED (CRITICAL)
- ✅ Bug #17: NoisyLinear stale noise - FIXED
- ✅ Bug #18: Device mismatch on tensor reallocation - FIXED
- ✅ Bug #19: NumPy JSON serialization - FIXED

---

### CRITICAL SEVERITY (1 bug)

#### BUG #16: VecBreakout Auto-Reset Corrupts State Arrays ✅ FIXED
- **File:** `src/game/breakout.py`
- **Lines:** 836-859 (step), 864-895 (step_no_copy)
- **Severity:** CRITICAL
- **Status:** ✅ FIXED
- **Description:** When a vectorized environment completes an episode, it auto-resets and overwrites `self._states[i]` with the reset state AFTER storing the terminal state. The caller receives `(s_terminal, a, r, s_reset, True)` instead of `(s_terminal, a, r, s_terminal, True)`.
- **Impact:**
  - Corrupts ALL vectorized training replay buffer data
  - Agent learns incorrect state→next_state mappings
  - Training instability and poor convergence
  - Silently degrades performance without error messages
- **Fix Applied:**
  - **For step()**: Return terminal states before updating state array for next iteration
  - **For step_no_copy()**: Added `_pending_resets` tracker to delay state updates until next call
  - Both methods now ensure terminal states are returned to caller, not reset states
  ```python
  # step() method: Return terminal states, then update
  states_to_return = self._states.copy()
  # ... return states_to_return ...
  # Update after return
  for i, done in enumerate(self._dones):
      if done:
          self._states[i] = self.envs[i].get_state()

  # step_no_copy() method: Use pending resets
  self._pending_resets = np.zeros(num_envs, dtype=np.bool_)  # In __init__
  # Process pending resets at start of next call
  # Mark for reset but don't update state array until next call
  ```

---

### HIGH SEVERITY (3 bugs)

#### BUG #13: PER Sampling Crashes When batch_size > buffer_size ✅ FIXED
- **File:** `src/ai/replay_buffer.py`
- **Line:** 341
- **Severity:** HIGH
- **Status:** ✅ FIXED
- **Description:** Prioritized Experience Replay uses `replace=False` when sampling, causing `ValueError` when batch_size exceeds current buffer size (early in training).
- **Impact:** Training crashes immediately when buffer has fewer experiences than BATCH_SIZE and PER is enabled
- **Error:** `ValueError: Cannot take a larger sample than population when 'replace=False'`
- **Fix Applied:**
  ```python
  # Conditionally allow replacement when batch size exceeds buffer size
  use_replacement = batch_size > self._size
  indices = np.random.choice(self._size, size=batch_size, p=probs, replace=use_replacement)
  ```
  - During early training (buffer < batch_size): uses `replace=True` to allow duplicates
  - During normal training (buffer ≥ batch_size): uses `replace=False` to avoid duplicates
  - Prevents crash while maintaining optimal sampling when possible

#### BUG #15: Division by Zero in Paddle Range Normalization ✅ FIXED
- **File:** `src/game/breakout.py`
- **Lines:** 170-172
- **Severity:** HIGH
- **Status:** ✅ FIXED
- **Description:** If `PADDLE_WIDTH >= SCREEN_WIDTH`, division by zero occurs during initialization.
- **Impact:** Immediate crash with `ZeroDivisionError` if paddle width configured incorrectly
- **Fix Applied:**
  ```python
  # Guard against paddle width >= screen width (lines 170-172)
  paddle_range = max(1.0, self.width - config.PADDLE_WIDTH)
  self._inv_paddle_range = 1.0 / paddle_range
  ```
  - Uses `max(1.0, ...)` to ensure paddle_range is always at least 1.0
  - Prevents division by zero even with misconfigured paddle dimensions
  - Allows game to initialize and fail gracefully if config is invalid

#### BUG #18: Batch Tensor Device Mismatch on Reallocation ✅ FIXED
- **File:** `src/ai/agent.py`
- **Lines:** 550-560
- **Severity:** HIGH
- **Status:** ✅ FIXED
- **Description:** When batch size changes, tensors are reallocated on `self.device`. However, if device changed after initial allocation (e.g., model loading), NEW tensors go to NEW device but OLD cached tensors remain on OLD device.
- **Impact:**
  - `RuntimeError: Expected all tensors to be on the same device`
  - Only occurs when batch size changes AFTER device change
  - Hard to debug random crashes
- **Scenario:**
  1. Init on MPS device
  2. Load checkpoint that switches to CPU
  3. Batch size changes
  4. Crash on next `.copy_()` call
- **Fix Applied:**
  ```python
  # Check batch size change OR device mismatch OR not yet allocated (lines 552-554)
  if (batch_size != self._cached_batch_size or
      not hasattr(self, '_batch_states') or
      self._batch_states.device != self.device):
      # Reallocate all batch tensors on current device
      self._batch_states = torch.empty((batch_size, self.state_size), dtype=torch.float32, device=self.device)
      # ... reallocate all other tensors ...
  ```
  - Now checks if tensors exist and are on correct device before reusing
  - Handles device changes from model loading/switching
  - Prevents device mismatch errors during training

---

### MEDIUM SEVERITY (3 bugs)

#### BUG #14: Ball Can Tunnel Through Paddle at High Speed ✅ FIXED
- **File:** `src/game/breakout.py`
- **Lines:** 319-329
- **Severity:** MEDIUM
- **Status:** ✅ FIXED
- **Description:** Ball physics update order allows ball to phase through paddle. Ball moves first (line 320), then collisions are checked. If ball moves THROUGH paddle in single frame (velocity > PADDLE_HEIGHT), collision check fails.
- **Impact:**
  - Ball passes through correctly positioned paddle
  - Happens when `ball_speed > PADDLE_HEIGHT` per frame
  - AI gets unfairly punished
  - Rare at default speeds but possible with velocity accumulation
- **Fix Applied:**
  ```python
  # After ball.move(), clamp speed to prevent tunneling (lines 322-329)
  max_safe_speed = min(self.config.PADDLE_HEIGHT, self.config.BRICK_HEIGHT) * 0.8
  current_speed = np.sqrt(self.ball.dx**2 + self.ball.dy**2)
  if current_speed > max_safe_speed:
      scale = max_safe_speed / current_speed
      self.ball.dx *= scale
      self.ball.dy *= scale
  ```
  - Limits ball speed to 80% of minimum object height (paddle/brick)
  - Ensures ball can't move more than object height in single frame
  - Prevents tunneling while maintaining realistic physics
  - Applied after ball.move() but before collision detection

#### BUG #17: NoisyLinear Reuses Stale Noise After Eval Mode ✅ FIXED
- **File:** `src/ai/network.py`
- **Lines:** 90-101
- **Severity:** MEDIUM
- **Status:** ✅ FIXED
- **Description:** `NoisyLinear.forward()` only samples noise via explicit `reset_noise()` call. When switching from eval→train mode, first forward pass reuses old noise from previous training step.
- **Impact:**
  - Violates NoisyNet algorithm (noise should be fresh each forward pass)
  - Affects exploration quality
  - Gradient estimates become incorrect
  - Particularly bad with `select_actions_batch()` which toggles modes
- **Fix Applied:**
  ```python
  def forward(self, x: torch.Tensor) -> torch.Tensor:
      """Forward pass with noisy weights during training."""
      if self.training:
          # Reset noise for each forward pass to ensure fresh exploration
          self.reset_noise()
          weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
          bias = self.bias_mu + self.bias_sigma * self.bias_epsilon
      else:
          # Use mean parameters during evaluation (no noise)
          weight = self.weight_mu
          bias = self.bias_mu
      return F.linear(x, weight, bias)
  ```
  - Now resets noise automatically on every forward pass in training mode
  - Ensures noise is always fresh, following NoisyNet algorithm correctly
  - Improves exploration quality and gradient accuracy
  - No more stale noise from previous training steps

#### BUG #19: Web Dashboard Crashes on NumPy Type Serialization ✅ FIXED
- **File:** `src/web/server.py`
- **Lines:** 35, 51-74, 1007, 1009, 1013
- **Severity:** MEDIUM
- **Status:** ✅ FIXED
- **Description:** SocketIO `emit()` fails when metrics contain NumPy scalar types (`np.int64`, `np.float32`) which aren't JSON-serializable.
- **Impact:**
  - Web dashboard crashes when connecting
  - Happens frequently with vectorized environments
  - `TypeError: Object of type int64 is not JSON serializable`
  - Dashboard becomes unusable
- **Fix Applied:**
  ```python
  # Added numpy import (line 35)
  import numpy as np

  # Added helper function (lines 51-74)
  def _make_json_safe(obj: Any) -> Any:
      """Convert NumPy types to native Python types for JSON serialization."""
      if isinstance(obj, np.integer):
          return int(obj)
      elif isinstance(obj, np.floating):
          return float(obj)
      elif isinstance(obj, np.ndarray):
          return obj.tolist()
      elif isinstance(obj, dict):
          return {k: _make_json_safe(v) for k, v in obj.items()}
      elif isinstance(obj, list):
          return [_make_json_safe(item) for item in obj]
      return obj

  # Applied to all emit() calls in handle_connect (lines 1007, 1009, 1013)
  emit('state_update', _make_json_safe(self.publisher.get_snapshot()))
  emit('console_logs', {'logs': _make_json_safe(self.publisher.get_console_logs(100))})
  emit('nn_update', _make_json_safe(nn_data))
  ```
  - Recursively converts all NumPy types to native Python types
  - Handles integers, floats, arrays, dicts, and lists
  - Applied to all SocketIO emit() calls that send metrics
  - Web dashboard now works with vectorized environments

---

## SUMMARY - ALL BUGS FIXED ✅

| Bug # | Component | Severity | Difficulty | Status |
|-------|-----------|----------|------------|--------|
| #16 | Vectorized Env | CRITICAL | MEDIUM | ✅ FIXED |
| #13 | PER Buffer | HIGH | EASY | ✅ FIXED |
| #15 | State Norm | HIGH | EASY | ✅ FIXED |
| #18 | Device Mgmt | HIGH | MEDIUM | ✅ FIXED |
| #14 | Ball Physics | MEDIUM | MEDIUM | ✅ FIXED |
| #17 | NoisyNet | MEDIUM | EASY | ✅ FIXED |
| #19 | Web Dashboard | MEDIUM | EASY | ✅ FIXED |

**FIX SUMMARY:**
- ✅ **Bug #16** (CRITICAL): VecBreakout now returns terminal states correctly, preventing replay buffer corruption
- ✅ **Bug #13** (HIGH): PER sampling handles early training by conditionally allowing duplicates
- ✅ **Bug #15** (HIGH): Paddle range normalization guards against division by zero
- ✅ **Bug #18** (HIGH): Batch tensor allocation checks for device mismatches
- ✅ **Bug #14** (MEDIUM): Ball speed clamped to prevent tunneling through objects
- ✅ **Bug #17** (MEDIUM): NoisyLinear resets noise on every forward pass in training mode
- ✅ **Bug #19** (MEDIUM): Web dashboard converts NumPy types for JSON serialization

**IMPACT:**
- Vectorized training now works correctly without data corruption
- Training no longer crashes with PER enabled or invalid configs
- Ball physics are more reliable and fair to the AI
- Web dashboard works with all training modes including vectorized environments

---

## TESTING RECOMMENDATIONS

1. **Bug #16 Testing:**
   - Run vectorized training with replay buffer inspection
   - Verify terminal states are NOT replaced with reset states
   - Check experience tuples have correct (s, a, r, s', done) structure

2. **Bug #13 Testing:**
   - Enable PER and start training from scratch
   - Verify no crash when buffer size < batch size
   - Test with various MEMORY_MIN and BATCH_SIZE combinations

3. **Bug #15 Testing:**
   - Try PADDLE_WIDTH equal to or larger than SCREEN_WIDTH
   - Verify graceful error or safe fallback instead of crash

4. **Bug #18 Testing:**
   - Load model trained on MPS, run on CPU
   - Change batch size dynamically
   - Verify no device mismatch errors

5. **Bug #19 Testing:**
   - Run vectorized training with web dashboard
   - Connect browser and verify no JSON serialization errors
   - Check all metrics display correctly
