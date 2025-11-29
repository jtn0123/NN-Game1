# Potential Bugs - Deep Q-Learning Game AI

**Generated:** 2025-11-29
**Total Issues Found:** 50

This document lists potential bugs, edge cases, and issues discovered during comprehensive code review.

---

## Critical Issues (High Priority)

### 1. Security: Unsafe Model Loading
**File:** `src/ai/agent.py:923`
**Severity:** High
**Issue:** Model loading uses `weights_only=False`, which is a security risk when loading untrusted models. PyTorch can execute arbitrary code during unpickling.
```python
checkpoint = torch.load(filepath, map_location=self.device, weights_only=False)
```
**Impact:** Arbitrary code execution if loading malicious model files.
**Fix:** Use `weights_only=True` or implement validation.

### 2. Race Condition: Process Termination from Thread
**File:** `src/web/server.py:517`
**Severity:** High
**Issue:** `os._exit(0)` called from SocketIO thread in `_save_and_quit()`. This immediately terminates the process without cleanup, potentially leaving resources unclosed.
```python
os._exit(0)  # Terminates process from any thread
```
**Impact:** Corrupted files, unclosed database connections, orphaned processes.
**Fix:** Use proper shutdown signal instead of `os._exit()`.

### 3. Directory Traversal: Symlink Bypass
**File:** `src/web/server.py:848-865`
**Severity:** High
**Issue:** Path traversal check uses `os.path.commonpath()` but doesn't handle symlinks. A symlink in the model directory pointing outside could bypass security.
```python
common = os.path.commonpath([model_dir, full_path])
if common != model_dir:
    return jsonify({'error': 'Invalid path'}), 403
```
**Impact:** Unauthorized file access outside model directory.
**Fix:** Check for symlinks with `os.path.islink()` before validation.

### 4. Buffer Overflow: Negative Step Delta
**File:** `src/web/server.py:305-326`
**Severity:** Medium
**Issue:** Steps per second calculation assumes steps always increase. If step counter is reset, `step_delta` could be negative.
```python
step_delta = newest_steps - oldest_steps
if time_delta > 0.1 and step_delta > 0:  # Checks positive but doesn't handle reset
```
**Impact:** Incorrect performance metrics, potential division errors.
**Fix:** Handle counter resets explicitly.

---

## Game Logic Bugs

### 5. Ball Speed Clamping Doesn't Preserve Direction
**File:** `src/game/breakout.py:323-329`
**Severity:** Medium
**Issue:** Ball speed clamping scales both dx and dy uniformly, but doesn't preserve the original velocity ratio. This could cause unexpected direction changes.
```python
current_speed = np.sqrt(self.ball.dx**2 + self.ball.dy**2)
if current_speed > max_safe_speed:
    scale = max_safe_speed / current_speed
    self.ball.dx *= scale
    self.ball.dy *= scale
```
**Impact:** Ball may change direction unpredictably when speed is clamped.
**Fix:** Verify this is the intended behavior or preserve velocity vector direction.

### 6. Win Condition Off-by-One Error
**File:** `src/game/space_invaders.py:969`
**Severity:** Medium
**Issue:** Win condition uses `>=` comparison. If `SI_WIN_LEVELS=3`, does level 3 mean "completed 3 levels" or "on level 3"?
```python
if self.config.SI_WIN_LEVELS > 0 and self.level >= self.config.SI_WIN_LEVELS:
    self.won = True
```
**Impact:** Agent may win one level early or late depending on interpretation.
**Fix:** Clarify semantics and adjust comparison operator.

### 7. Exponential Alien Speed Increase
**File:** `src/game/space_invaders.py:876`
**Severity:** Medium
**Issue:** Alien speed multiplier is applied to base speed each level. Over many levels, this creates exponential difficulty growth.
```python
self.alien_base_speed = self.config.SI_ALIEN_SPEED_X * (1 + 0.15 * (self.level - 1))
```
**Impact:** Later levels become impossibly fast.
**Fix:** Consider capping speed or using a different scaling function.

### 8. Brick State Array Bounds Check Too Lenient
**File:** `src/game/breakout.py:445`
**Severity:** Low
**Issue:** Bounds check `if brick_idx < len(self._brick_states)` silently fails if arrays are mismatched. Should assert or raise error.
```python
if brick_idx < len(self._brick_states):
    self._brick_states[brick_idx] = 0.0
```
**Impact:** Silent data corruption if brick list and state array length differ.
**Fix:** Add assertion that arrays are same length.

### 9. Landing Prediction May Not Converge
**File:** `src/game/breakout.py:556-560`
**Severity:** Low
**Issue:** `_predict_landing_x()` returns fallback value if simulation doesn't converge in 100 iterations. For complex trajectories, prediction may be inaccurate.
```python
# Fallback: return current x if simulation didn't converge
return float(self.ball.x)
```
**Impact:** Inaccurate reward shaping signals for complex ball physics.
**Fix:** Log when fallback is used; consider increasing iteration limit.

### 10. Terminal State Array Update Timing
**File:** `src/game/breakout.py:857-862`
**Severity:** Low
**Issue:** In `VecBreakout.step()`, state array is updated after returning terminal states. While correct per comment, it's error-prone.
```python
# Return terminal states for done episodes
states_to_return = self._states.copy()
# NOW update state array for next iteration
for i, done in enumerate(self._dones):
    if done:
        self._states[i] = self.envs[i].get_state()
```
**Impact:** Future refactoring could break this subtle timing requirement.
**Fix:** Add defensive assertions to verify correctness.

---

## Neural Network & Training Issues

### 11. Pre-allocated State Tensor Never Resized
**File:** `src/ai/agent.py:299`
**Severity:** Medium
**Issue:** `_state_tensor` allocated once at init. If state_size somehow changes, tensor would be wrong size.
```python
self._state_tensor = torch.empty((1, state_size), dtype=torch.float32, device=self.device)
```
**Impact:** Crash or silent corruption if state size changes.
**Fix:** Add validation that state size matches tensor shape.

### 12. Batch Tensor Device Mismatch
**File:** `src/ai/agent.py:570-578`
**Severity:** Medium
**Issue:** Only checks `self._batch_states.device != self.device`. What if other batch tensors (actions, rewards) are on different devices?
```python
if (batch_size != self._cached_batch_size or
    not hasattr(self, '_batch_states') or
    self._batch_states.device != self.device):
```
**Impact:** Partial device migration could cause runtime errors.
**Fix:** Check all batch tensor devices, not just states.

### 13. Hard Target Update Includes Buffers
**File:** `src/ai/agent.py:691`
**Severity:** Low
**Issue:** `load_state_dict()` may include buffers like running stats. For models with BatchNorm, this might be incorrect.
```python
self.target_net.load_state_dict(self.policy_net.state_dict())
```
**Impact:** Target network may have training mode statistics when it shouldn't.
**Fix:** Only copy parameters, not buffers (or verify no such layers exist).

### 14. Empty Hidden Layers Not Validated
**File:** `config.py:474-482`
**Severity:** Medium
**Issue:** `__post_init__` validation doesn't check if `HIDDEN_LAYERS` is empty. Empty list would cause network creation to fail.
```python
# No check for:
assert len(self.HIDDEN_LAYERS) > 0, "Must have at least one hidden layer"
```
**Impact:** Crash during network initialization with obscure error.
**Fix:** Add validation for empty hidden layers list.

### 15. Random Action on Zero Action Size
**File:** `src/ai/agent.py:332-334`
**Severity:** Medium
**Issue:** `random.randrange(self.action_size)` raises `ValueError` if `action_size` is 0.
```python
return random.randrange(self.action_size)
```
**Impact:** Crash if game has 0 actions (invalid but not validated).
**Fix:** Validate action_size > 0 in agent initialization.

---

## Replay Buffer Issues

### 16. Duplicate Samples in Small Buffers
**File:** `src/ai/replay_buffer.py:176-177`
**Severity:** Low
**Issue:** Sampling with replacement means duplicates are possible. For small buffers with large batches, could get many duplicates.
```python
indices = np.random.choice(self._size, size=batch_size, replace=True)
```
**Impact:** Reduced training diversity; agent overfits to repeated samples.
**Fix:** Consider replace=False or warn when buffer is small.

### 17. PER Sampling Replace Logic Inconsistent
**File:** `src/ai/replay_buffer.py:343`
**Severity:** Medium
**Issue:** `use_replacement = batch_size > self._size` but buffer size could shrink at any time via clear().
```python
use_replacement = batch_size > self._size
indices = np.random.choice(self._size, size=batch_size, p=probs, replace=use_replacement)
```
**Impact:** May sample without replacement when buffer is too small, causing crash.
**Fix:** Always use replacement=True or add size check before sampling.

### 18. Sample No-Copy Fails on Large Batch
**File:** `src/ai/replay_buffer.py:374`
**Severity:** Medium
**Issue:** `sample_no_copy()` uses `replace=False`, which fails if batch_size > buffer size. No error handling.
```python
indices = np.random.choice(self._size, size=batch_size, p=probs, replace=False)
```
**Impact:** Crash with ValueError when batch size exceeds buffer.
**Fix:** Add check or use replace=True.

### 19. N-Step Buffer Unnecessary Copies
**File:** `src/ai/replay_buffer.py:465-466`
**Severity:** Low
**Issue:** `NStepReplayBuffer` stores `.copy()` of states in temporary buffer, but parent `push()` doesn't copy.
```python
self._n_step_buffer.append((state.copy(), action, reward, next_state.copy(), done))
```
**Impact:** Unnecessary memory usage and slowdown.
**Fix:** Remove copies or document why they're needed.

### 20. N-Step Final State May Be Wrong
**File:** `src/ai/replay_buffer.py:494`
**Severity:** Medium
**Issue:** `n_step_next_state` uses `actual_final_idx` which could be the last element even if episode didn't terminate.
```python
_, _, _, n_step_next_state, n_step_done = self._n_step_buffer[actual_final_idx]
```
**Impact:** Incorrect bootstrapping for N-step returns if episode doesn't end.
**Fix:** Verify this is correct for non-terminal N-step trajectories.

---

## Configuration & State Management

### 21. Legacy STATE_SIZE Property Still Exists
**File:** `config.py:168`
**Severity:** Low
**Issue:** `STATE_SIZE` property is marked as legacy and says "use game.state_size instead", but it's still in Config and could cause confusion.
```python
@property
def STATE_SIZE(self) -> int:
    """Calculate input layer size based on game state representation.

    NOTE: This is a legacy property used only for standalone tests.
```
**Impact:** Developers may use wrong state size value.
**Fix:** Remove property or raise DeprecationWarning.

### 22. Space Invaders State Size is Approximation
**File:** `config.py:179`
**Severity:** Medium
**Issue:** Space Invaders state size calculation is marked as "approximation" with comment "use game.state_size in production".
```python
return 1 + max_player_bullets * 2 + num_aliens + 5 + 7  # Approximation
```
**Impact:** If actual state size differs, network won't match game.
**Fix:** Either make it exact or remove the property entirely.

### 23. TARGET_TAU Ignored When Soft Update Disabled
**File:** `config.py:238-242`
**Severity:** Low
**Issue:** Both `TARGET_TAU` and `USE_SOFT_UPDATE` exist, but if `USE_SOFT_UPDATE=False`, `TAU` is ignored. This could confuse users who set TAU but forget to enable soft updates.
```python
TARGET_TAU: float = 0.005
USE_SOFT_UPDATE: bool = True
```
**Impact:** User sets TAU but nothing happens if soft updates are off.
**Fix:** Add validation or warning if TAU is set but soft updates disabled.

### 24. Episode Restore from History Length
**File:** `main.py:357`
**Severity:** Low
**Issue:** Restoring episode counter as `len(training_history.scores)` assumes one score per episode with no gaps.
```python
self.episode = len(self.training_history.scores)
```
**Impact:** Incorrect episode count if history has gaps or duplicates.
**Fix:** Store episode count in metadata instead of inferring.

### 25. Epsilon Not Validated for NaN/Inf
**File:** `main.py:688`
**Severity:** Low
**Issue:** Epsilon is clamped but not validated for NaN or infinity values from user input.
```python
self.agent.epsilon = max(self.config.EPSILON_END, min(self.config.EPSILON_START, config_data['epsilon']))
```
**Impact:** NaN epsilon could cause all actions to be random or crash.
**Fix:** Add `math.isfinite()` check.

---

## Performance & Resource Issues

### 26. Noisy Layer Outer Product Memory
**File:** `src/ai/network.py:81`
**Severity:** Medium
**Issue:** `epsilon_out.outer(epsilon_in)` creates outer product for noise. For large layers, this could create very large temporary tensors.
```python
self.weight_epsilon.copy_(epsilon_out.outer(epsilon_in))
```
**Impact:** High memory usage for large networks.
**Fix:** Profile memory usage; consider factored noise alternatives.

### 27. NoisyLinear Allocates Weights Each Forward
**File:** `src/ai/network.py:94`
**Severity:** Low
**Issue:** NoisyLinear forward pass creates new weight tensors every forward pass.
```python
weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
bias = self.bias_mu + self.bias_sigma * self.bias_epsilon
```
**Impact:** Unnecessary allocations slow down training.
**Fix:** Profile if this is significant; consider in-place operations.

### 28. Inefficient Bullet Sorting
**File:** `src/game/space_invaders.py:1308`
**Severity:** Low
**Issue:** Bullet distance calculation creates list of tuples then sorts. For many bullets, this is inefficient.
```python
alive_bullets = [(b, ship_y - b.y) for b in self.alien_bullets if b.alive and b.y < ship_y]
alive_bullets.sort(key=lambda x: x[1])
```
**Impact:** Minor slowdown when many alien bullets exist.
**Fix:** Use heap for top-K selection or optimize with numpy.

### 29. Scale Factor Zero on Minimized Window
**File:** `main.py:805-821`
**Severity:** Low
**Issue:** `_update_scale()` guards against zero dimensions, but `scale_factor` could still be 0 if window is minimized.
```python
if self.window_width <= 0 or self.window_height <= 0:
    return
```
**Impact:** Division by zero or rendering errors when window is minimized.
**Fix:** Set scale_factor to 1.0 as fallback.

---

## Web Dashboard Issues

### 30. Log Timestamp Format Incorrect
**File:** `src/web/server.py:350`
**Severity:** Low
**Issue:** Timestamp formatting `strftime("%H:%M:%S.%f")[:12]` - `%f` is 6 digits (microseconds), so `[:12]` gives `HH:MM:SS.ff` instead of `HH:MM:SS.fff` (milliseconds).
```python
timestamp=datetime.now().strftime("%H:%M:%S.%f")[:12]
```
**Impact:** Timestamp precision is wrong (centiseconds instead of milliseconds).
**Fix:** Use `[:15]` for microseconds or manually format milliseconds.

### 31. Screenshot Save Without Extension
**File:** `src/web/server.py:385-391`
**Severity:** Low
**Issue:** Fallback tries `pygame.image.save()` without filename extension, which may fail on some pygame versions.
```python
pygame.image.save(temp_surface, buffer)
```
**Impact:** Screenshot capture may fail silently.
**Fix:** Specify format explicitly or add error handling.

### 32. Model Delete Symlink Check Incomplete
**File:** `src/web/server.py:923`
**Severity:** Medium
**Issue:** Symlink check only looks at path components, not the file itself. A symlink file pointing outside the directory could be deleted.
```python
while current_path and current_path != legacy_model_dir:
    if os.path.islink(current_path):
        return jsonify({'error': 'Cannot delete files with symbolic links in path'}), 403
```
**Impact:** Could delete files outside model directory via symlink.
**Fix:** Also check if the final file is a symlink.

### 33. Game Name Change Breaks Save Callback
**File:** `main.py:238`
**Severity:** Low
**Issue:** `on_save_callback` lambda captures `config.GAME_NAME` in f-string. If config is changed, filename could be wrong.
```python
self.web_dashboard.on_save_callback = lambda: self._save_model(f"{config.GAME_NAME}_web_save.pth", save_reason="manual")
```
**Impact:** Saved file has wrong game name prefix if config changes.
**Fix:** Capture game name at lambda creation time, not execution time.

---

## Edge Cases & Boundary Conditions

### 34. Zero Ball Speed Division
**File:** `src/game/breakout.py:168`
**Severity:** Low
**Issue:** `_max_speed = max(config.BALL_SPEED * 1.5, 1.0)` - if `BALL_SPEED=0`, this gives 1.0. Division is safe but semantically wrong.
```python
self._max_speed = max(config.BALL_SPEED * 1.5, 1.0)
self._inv_max_speed = 1.0 / self._max_speed
```
**Impact:** Incorrect velocity normalization if ball speed is zero.
**Fix:** Validate ball speed > 0 in config.

### 35. Wide Paddle Division by One
**File:** `src/game/breakout.py:171-172`
**Severity:** Low
**Issue:** `paddle_range = max(1.0, self.width - config.PADDLE_WIDTH)` - if paddle wider than screen, divides by 1.0 which gives wrong normalization.
```python
paddle_range = max(1.0, self.width - config.PADDLE_WIDTH)
self._inv_paddle_range = 1.0 / paddle_range
```
**Impact:** Paddle position normalization incorrect for oversized paddles.
**Fix:** Validate paddle width < screen width.

### 36. State Value Overflow for Far Bullets
**File:** `src/game/space_invaders.py:1315`
**Severity:** Low
**Issue:** Bullet X relative to ship: `(bullet.x - ship_center_x) * self._inv_width + 0.5` - far-off bullets could overflow expected [0, 1] range.
```python
self._state_array[idx] = (bullet.x - ship_center_x) * self._inv_width + 0.5
```
**Impact:** State values outside expected range could confuse neural network.
**Fix:** Clamp to [0, 1] or use unbounded feature encoding.

### 37. Hardcoded State Size Magic Number
**File:** `src/game/space_invaders.py:681-684`
**Severity:** Medium
**Issue:** State size calculation has `+ 5` with comment explaining what the 5 values are. If implementation changes, this won't match.
```python
self._state_size = (1 + self._max_player_bullets * 2 + self._num_aliens + 5
                  + self._max_tracked_alien_bullets * 2 + 5)  # +5 for cooldown, active, ratio, lives, level
```
**Impact:** State size mismatch causes crash with cryptic error.
**Fix:** Calculate dynamically or add assertion.

---

## Type Safety & Validation Issues

### 38. PER Return Value Inconsistency
**File:** `src/ai/agent.py:567`
**Severity:** Medium
**Issue:** `sample_no_copy()` returns 7 values for PER but 5 for regular buffer. Caller must know which buffer type is in use.
```python
if self._use_per:
    states_np, actions_np, rewards_np, next_states_np, dones_np, indices, weights_np = \
        self.memory.sample_no_copy(batch_size)
```
**Impact:** Wrong unpacking causes crash or silent corruption.
**Fix:** Make return type consistent or use wrapper class.

### 39. Shape Mismatch in Replay Buffer Copy
**File:** `src/ai/replay_buffer.py:108`
**Severity:** Medium
**Issue:** `np.copyto()` could fail if state shape doesn't match pre-allocated array. No try/except.
```python
np.copyto(self.states[self._position], state)
```
**Impact:** Crash with cryptic numpy error.
**Fix:** Add shape validation or wrap in try/except.

### 40. Type Ignore Hides Actual Type Error
**File:** `src/game/space_invaders.py:1104`
**Severity:** Low
**Issue:** Type ignore comment `# type: ignore` suggests actual type issue with `bottom_per_col[col].y`.
```python
if bottom_per_col[col] is None or alien.y > bottom_per_col[col].y:  # type: ignore
```
**Impact:** Type checker can't help find bugs in this code.
**Fix:** Fix actual type issue instead of ignoring.

---

## Training & Metric Issues

### 41. Epsilon Warmup Bypass
**File:** `src/ai/trainer.py:196`
**Severity:** Low
**Issue:** `decay_epsilon()` called without episode parameter, bypassing warmup check.
```python
self.agent.decay_epsilon()
```
**Impact:** Epsilon decays during warmup period when it shouldn't.
**Fix:** Pass episode number to decay_epsilon().

### 42. Model Auto-Load Missing Compatibility Check
**File:** `main.py:311`
**Severity:** Low
**Issue:** Auto-loading most recent save doesn't check compatibility. Just returns the path.
```python
most_recent = model_files[0][0]
print(f"📂 Auto-loading most recent save: {os.path.basename(most_recent)}")
return most_recent
```
**Impact:** Loads incompatible model, then fails during actual load.
**Fix:** Check compatibility before returning path.

### 43. Decay Function Backward Compatibility Hole
**File:** `src/ai/agent.py:729-731`
**Severity:** Low
**Issue:** If `episode=None` and `EPSILON_WARMUP > 0`, warmup is bypassed. Documented as backward compatible but could be unexpected.
```python
if episode is not None and episode < warmup:
    return
```
**Impact:** Warmup doesn't work if caller doesn't pass episode number.
**Fix:** Make episode parameter required or warn.

---

## Documentation & Clarity Issues

### 44. Confusing Terminal State Comment
**File:** `src/game/breakout.py:862`
**Severity:** Low
**Issue:** Comment explains the subtle timing requirement for terminal state updates, but it's still error-prone.
```python
# Return terminal states for done episodes (critical for correct replay buffer data)
states_to_return = self._states.copy()
```
**Impact:** Future maintainers could break this assumption.
**Fix:** Add explicit test to verify this behavior.

### 45. Approximation Warning in Critical Code
**File:** `src/game/breakout.py:486`
**Severity:** Low
**Issue:** `_predict_landing_x()` has comment "Does not simulate brick collisions - this is an approximation" but is used for reward shaping.
```python
Note: Does not simulate brick collisions - this is an approximation.
For balls moving through brick area, prediction accuracy decreases.
```
**Impact:** Inaccurate predictions through brick field could mislead training.
**Fix:** Document impact on training performance.

---

## Concurrency & Threading Issues

### 46. Callback Lock Not Held During Callback Execution
**File:** `src/web/server.py:337-340`
**Severity:** Low
**Issue:** Callback list is copied while holding lock, but then lock is released before calling callbacks. New callbacks registered during execution won't be in the copy.
```python
with self._callback_lock:
    callbacks = self._on_update_callbacks.copy()
for callback in callbacks:
    callback(self.get_snapshot())
```
**Impact:** Callback registered during update won't fire until next update.
**Fix:** Document this behavior or change locking strategy.

### 47. Web Dashboard Publisher State Race
**File:** `src/web/server.py:273-286`
**Severity:** Low
**Issue:** Multiple attributes of `self.state` are updated without locking. Concurrent reads could see inconsistent state.
```python
self.state.episode = episode
self.state.score = score
self.state.best_score = max(self.state.best_score, score)
```
**Impact:** Web dashboard shows inconsistent metrics during updates.
**Fix:** Lock during multi-attribute updates or use atomic state object.

---

## Initialization & Setup Issues

### 48. Font Creation in Non-Headless Mode
**File:** `src/game/breakout.py:187-189`
**Severity:** Low
**Issue:** Fonts are cached in `__init__` but only if not headless. What if headless flag changes?
```python
if not headless:
    pygame.font.init()
    self._hud_font = pygame.font.Font(None, 36)
```
**Impact:** Crash if headless flag is changed after initialization.
**Fix:** Document that headless is immutable or add validation.

### 49. Circular Buffer Size Validation Missing
**File:** `src/ai/replay_buffer.py:54`
**Severity:** Low
**Issue:** ReplayBuffer accepts any capacity value. What if capacity=0 or negative?
```python
def __init__(self, capacity: int, state_size: int = 0):
    self.capacity = capacity
```
**Impact:** Crash or infinite loop with invalid capacity.
**Fix:** Validate capacity > 0.

### 50. Model Directory Creation Race Condition
**File:** `src/ai/agent.py:773-775`
**Severity:** Low
**Issue:** `os.makedirs(dir_path, exist_ok=True)` could race if multiple processes create same directory.
```python
if dir_path:
    os.makedirs(dir_path, exist_ok=True)
```
**Impact:** Rare crash if two processes create directory simultaneously.
**Fix:** Wrap in try/except for concurrent creation.

---

## Summary

**Critical Issues:** 4
**High Severity:** 0
**Medium Severity:** 18
**Low Severity:** 28

**Top Recommendations:**
1. Fix security issues (unsafe model loading, directory traversal)
2. Handle edge cases in game physics and state representation
3. Add validation for configuration values
4. Improve buffer overflow and array bounds checking
5. Document threading assumptions and add proper locking

**Testing Recommendations:**
- Add unit tests for edge cases (zero values, empty arrays, boundary conditions)
- Add integration tests for save/load functionality
- Add stress tests for concurrent access to web dashboard
- Add fuzzing tests for model loading security
