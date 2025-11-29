# Additional Bugs Found - Beyond Original 36

## STATUS SUMMARY

**FIXED (10 bugs):**
- ✅ Bug #1: Epsilon Decay in Vectorized Training - FIXED (uses `if np.any(dones)` not loop)
- ✅ Bug #2: Config Values Division by Zero - FIXED
- ✅ Bug #4: Race Condition in Callback Registration - FIXED
- ✅ Bug #5: Gradient Accumulation During Noise Reset - FIXED
- ✅ Bug #6: Ball Landing Prediction - FIXED (now handles top wall bounces)
- ✅ Bug #7: Loss Array Thread Safety - FIXED
- ✅ Bug #9: Brick State Update Bounds Check - FIXED
- ✅ Bug #10: Symlink Security - FIXED
- ✅ Bug #11: Misleading Comment - FIXED
- ✅ Bug #12: Screenshot Fallback Error Handling - FIXED

**NOT FIXED (Design/Low Priority):**
- ❌ Bug #3: State Array Fragile Design (not a functional bug, design pattern issue)
- ❌ Bug #8: Config Validation Timing (LOW - requires dataclass refactor)

---

## NEW BUGS DISCOVERED (12 total)

### ALL CRITICAL BUGS FIXED ✅

#### BUG #1: Epsilon Decay Called Multiple Times in Vectorized Training ✅ FIXED
- **File:** `main.py:2594-2598`
- **Fix Applied:** Changed from loop-based decay to single decay per step:
  ```python
  # Before: for _ in range(int(np.sum(dones))): decay...
  # After:
  if np.any(dones):
      self.agent.decay_epsilon(self.current_episode)
      self.agent.step_scheduler()
  ```
- **Note:** Epsilon now decays once per step if any episode completes, not once per environment

---

### REMAINING DESIGN ISSUES (2 bugs - LOW PRIORITY)

#### BUG #3: State Array Corruption Risk (Fragile Design Pattern)
- **File:** `src/game/breakout.py`
- **Lines:** 568-570
- **Severity:** HIGH
- **Description:** While the code currently DOES copy the state array (line 570), the design is fragile. The `_state_array` is reused for performance, and the code relies on `.copy()` being called. If a future developer removes this for "optimization", the entire replay buffer would be corrupted with all states pointing to the same memory.
- **Impact:** If `.copy()` is ever removed, catastrophic replay buffer corruption
- **Recommendation:** Add defensive validation or make state array immutable
- **Status:** NOT A FUNCTIONAL BUG - Design pattern concern

#### BUG #6: Ball Landing Prediction Ignores Top Wall ✅ FIXED
- **File:** `src/game/breakout.py:470-545`
- **Fix Applied:** Rewrote prediction to simulate ball trajectory with top wall bounces:
  - Now handles balls moving upward (toward top wall)
  - Simulates bounce off top wall and subsequent descent
  - Still approximates (doesn't simulate brick collisions - documented as limitation)
- **Note:** Brick collision simulation was deemed too complex for minimal benefit

#### BUG #8: Config Validation After Object Creation
- **File:** `config.py`
- **Lines:** 451-458
- **Severity:** LOW
- **Description:** `__post_init__` validation happens after dataclass initialization, so invalid values exist briefly before exception is raised.
- **Impact:** Low - exceptions halt execution, but there's a brief window of invalid state
- **Complexity:** MEDIUM - Requires dataclass refactoring
- **Status:** SKIPPED - Minimal practical impact

---

## DETAILED LIST OF FIXED BUGS

### Bug #2: Config Values Division by Zero ✅ FIXED
- **File:** `config.py:459-461`
- **Fix Applied:** Added validation assertions:
  ```python
  assert self.SCREEN_WIDTH > 0, "Screen width must be positive"
  assert self.SCREEN_HEIGHT > 0, "Screen height must be positive"
  assert self.BALL_SPEED > 0, "Ball speed must be positive"
  ```

### Bug #4: Race Condition in Callback Registration ✅ FIXED
- **File:** `src/web/server.py:200, 304-307, 325-328, 383-384, 388-389, 442-445, 476-477, 523-526, 534-535, 1084-1088`
- **Fix Applied:**
  - Added `self._callback_lock = threading.Lock()` to MetricsPublisher
  - Wrapped all callback list operations (clear, append, iteration) with lock
  - Used lock.copy() pattern to safely iterate callbacks without blocking registration

### Bug #5: Gradient Accumulation During Noise Reset ✅ FIXED
- **File:** `src/ai/agent.py:601-604`
- **Fix Applied:** Moved noise reset to AFTER `optimizer.zero_grad()`:
  ```python
  self.optimizer.zero_grad()
  # Reset noise for NoisyNet exploration (after zero_grad to avoid gradient issues)
  if hasattr(self.policy_net, 'reset_noise'):
      self.policy_net.reset_noise()
      self.target_net.reset_noise()
  loss.backward()
  ```

### Bug #7: Loss Array Thread Safety ✅ FIXED
- **File:** `src/ai/agent.py:290, 626-627, 1072-1080`
- **Fix Applied:**
  - Added `self._losses_lock = threading.Lock()`
  - Wrapped `self.losses.append()` with lock
  - Wrapped `get_average_loss()` iteration with lock

### Bug #9: Brick State Update Bounds Check ✅ FIXED
- **File:** `src/game/breakout.py:434-435`
- **Fix Applied:**
  ```python
  # Update pre-allocated array (with bounds check for safety)
  if brick_idx < len(self._brick_states):
      self._brick_states[brick_idx] = 0.0
  ```

### Bug #10: Symlink Security ✅ FIXED
- **File:** `src/web/server.py:889-897`
- **Fix Applied:** Check ALL path components for symlinks:
  ```python
  path_to_check = os.path.join(legacy_model_dir, filepath)
  current_path = path_to_check
  while current_path and current_path != legacy_model_dir:
      if os.path.islink(current_path):
          return jsonify({'error': 'Cannot delete files with symbolic links in path'}), 403
      current_path = os.path.dirname(current_path)
  ```

### Bug #11: Misleading Comment ✅ FIXED
- **File:** `src/ai/replay_buffer.py:396`
- **Fix Applied:** Updated comment for clarity:
  ```python
  priorities = np.abs(td_errors) + 1e-6  # Small epsilon ensures non-zero priorities when td_error=0
  ```

### Bug #12: Screenshot Fallback Error Handling ✅ FIXED
- **File:** `src/web/server.py:347-358`
- **Fix Applied:** Wrapped fallback in try/except and removed hardcoded filename:
  ```python
  try:
      buffer = io.BytesIO()
      temp_surface = surface.copy()
      pygame.image.save(temp_surface, buffer)  # Removed 'screenshot.png'
      buffer.seek(0)
      self._screenshot_data = base64.b64encode(buffer.read()).decode('utf-8')
  except Exception as fallback_error:
      print(f"Screenshot fallback error: {fallback_error}")
      self._screenshot_data = None
  ```

---

## SUMMARY BY STATUS

- **Total NEW Bugs Found:** 12
- **Fixed:** 10 bugs ✅
- **Remaining (Design/Low Priority):** 2 bugs

## REMAINING ITEMS (LOW PRIORITY)

**Design Pattern Concerns (Not Functional Bugs):**
1. **Bug #3** - State array design pattern (already works, just fragile)
2. **Bug #8** - Config validation timing (minimal impact)

---

## TESTING RECOMMENDATIONS

After these fixes, test the following scenarios:
1. **Thread Safety:** Run vectorized training with web dashboard active
2. **Config Validation:** Try setting SCREEN_WIDTH=0 in config (should fail with clear error)
3. **Symlink Security:** Attempt to create symlink in models directory and delete via API
4. **Screenshot Handling:** Test screenshot capture with and without PIL installed
5. **Brick Collision:** Play game manually to ensure brick hit detection still works

**Most Critical Remaining:** Bug #1 (Epsilon decay) should be top priority for the next fix session.
