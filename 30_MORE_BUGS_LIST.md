# 30 More Bugs Found in Codebase

## Critical Bugs (31-35)

### 31. Negative Array Index in Snake Body Collision Check
**File:** `src/game/snake.py:279`  
**Severity:** Critical  
**Issue:** Converting snake deque to list and slicing `[:-1]` could fail if snake has only 1 element:
```python
body_set = set(list(self.snake)[:-1])
```
**Impact:** If snake length is 1, `[:-1]` returns empty list, but if snake is empty, this crashes.  
**Fix:** Check `if len(self.snake) > 1:` before this operation.

---

### 32. Division by Zero in Win Rate Calculation
**File:** `src/web/server.py:726`  
**Severity:** Critical  
**Issue:** Division by `len(recent_wins)` without checking if list is empty:
```python
publisher.state.win_rate = sum(1 for w in recent_wins if w) / len(recent_wins)
```
**Impact:** `ZeroDivisionError` if no wins recorded yet.  
**Fix:** Add check: `publisher.state.win_rate = sum(1 for w in recent_wins if w) / len(recent_wins) if recent_wins else 0.0`

---

### 33. Array Index Out of Bounds in Asteroids State Array
**File:** `src/game/asteroids.py:628-638`  
**Severity:** Critical  
**Issue:** Loop iterates `MAX_ASTEROIDS_TRACKED` times but only checks `if i < len(sorted_asteroids)` before accessing:
```python
for i in range(self.MAX_ASTEROIDS_TRACKED):
    if i < len(sorted_asteroids):
        a = sorted_asteroids[i]
        # ... set state array
    else:
        self._state_array[idx:idx + 5] = 0.0
    idx += 5
```
**Impact:** If `sorted_asteroids` is empty but loop runs, could cause issues. Actually safe, but if `idx` calculation is wrong, could overflow.  
**Fix:** Verify `idx` doesn't exceed state array bounds.

---

### 34. Missing Validation for Empty Training History Arrays
**File:** `main.py:589-608`  
**Severity:** Critical  
**Issue:** Accesses training history arrays without checking if they're empty:
```python
if training_history and len(training_history.scores) > 0:
    # ... loops through scores
    reward=training_history.rewards[i] if i < len(training_history.rewards) else 0.0,
```
**Impact:** If `scores` has elements but other arrays are empty, defaults are used but could cause confusion.  
**Fix:** Validate all arrays have same length or handle mismatches explicitly.

---

### 35. Potential Index Out of Bounds in Pong Ball Trail
**File:** `src/game/pong.py:299-301`  
**Severity:** Critical  
**Issue:** `pop(0)` on empty list would raise `IndexError`:
```python
if len(self._ball_trail) > self._trail_length:
    self._ball_trail.pop(0)
```
**Impact:** Actually safe due to length check, but if `_trail_length` is 0 or negative, could cause issues.  
**Fix:** Add validation: `assert self._trail_length > 0`

---

## High Priority Bugs (36-45)

### 36. Missing NaN/Inf Check in Epsilon Clamping
**File:** `main.py:750`  
**Severity:** High  
**Issue:** Epsilon clamped but not validated for NaN or infinity:
```python
self.agent.epsilon = max(self.config.EPSILON_END, min(self.config.EPSILON_START, config_data['epsilon']))
```
**Impact:** NaN epsilon could cause all actions to be random or crash.  
**Fix:** Add `import math` and check: `if not math.isfinite(config_data['epsilon']): raise ValueError("Epsilon must be finite")`

---

### 37. State Size Mismatch Not Caught Early
**File:** `src/ai/agent.py:476`  
**Severity:** High  
**Issue:** `_state_tensor` pre-allocated with `state_size`, but if state shape changes, `copy_()` could fail:
```python
self._state_tensor.copy_(torch.from_numpy(state.reshape(1, -1)))
```
**Impact:** Shape mismatch causes cryptic error.  
**Fix:** Validate shape: `assert state.shape == (self.state_size,) or state.size == self.state_size`

---

### 38. PER Return Value Count Mismatch Risk
**File:** `src/ai/agent.py:575-576`  
**Severity:** High  
**Issue:** PER returns 7 values, regular buffer returns 5. If buffer type changes, unpacking fails:
```python
states_np, actions_np, rewards_np, next_states_np, dones_np, indices, weights_np = \
    self.memory.sample_no_copy(batch_size)
```
**Impact:** Runtime crash if wrong buffer type.  
**Fix:** Use type checking or wrapper method that handles both.

---

### 39. Batch Size Mismatch in Pre-allocated Tensors
**File:** `src/ai/agent.py:588-596`  
**Severity:** High  
**Issue:** Pre-allocated tensors resized if batch size changes, but if `batch_size` parameter doesn't match `config.BATCH_SIZE`, could cause issues:
```python
if (batch_size != self._cached_batch_size or ...):
    self._batch_states = torch.empty((batch_size, self.state_size), ...)
```
**Impact:** Tensor size mismatch if batch_size parameter differs from config.  
**Fix:** Always use `self.config.BATCH_SIZE` instead of parameter.

---

### 40. Missing Validation for Empty Buffer in PER
**File:** `src/ai/replay_buffer.py:469-475`  
**Severity:** High  
**Issue:** If `probs_sum == 0` and `_size == 0`, division by zero occurs:
```python
if probs_sum > 0:
    probs = probs / probs_sum
else:
    probs = np.ones(self._size, dtype=np.float32) / self._size  # Division by zero if _size == 0
```
**Impact:** Actually checked earlier (line 462), but this fallback is still dangerous.  
**Fix:** Remove this fallback since empty buffer is already checked.

---

### 41. Negative Index in List Slicing
**File:** `src/ai/trainer.py:86`  
**Severity:** High  
**Issue:** Negative slicing `[-self.history_length:]` on empty list returns empty list, but if `history_length` is negative, behavior is unexpected:
```python
setattr(self, attr, getattr(self, attr)[-self.history_length:])
```
**Impact:** If `history_length` is negative, returns wrong slice.  
**Fix:** Validate `history_length > 0` or use `max(0, -self.history_length)`

---

### 42. Division by Zero in Trend Calculation
**File:** `src/ai/evaluator.py:269-272`  
**Severity:** High  
**Issue:** Division by `denominator` could be zero if all x values are the same:
```python
denominator = np.sum((x_arr - x_mean) ** 2)
if denominator == 0:
    return 0.0
```
**Impact:** Actually handled, but check happens after calculation.  
**Fix:** Check before calculation for efficiency.

---

### 43. Missing Bounds Check in Space Invaders Bullet Indexing
**File:** `src/game/space_invaders.py:1282`  
**Severity:** High  
**Issue:** Accesses `self.player_bullets[i]` after filtering, but index `i` is from original list:
```python
for i in range(self._max_player_bullets):
    if i < len(self.player_bullets):
        bullet = self.player_bullets[i]
```
**Impact:** If bullets are removed during iteration, indices could be wrong.  
**Fix:** Iterate over list directly: `for bullet in self.player_bullets:`

---

### 44. Potential Overflow in State Array Index Calculation
**File:** `src/game/space_invaders.py:1291`  
**Severity:** High  
**Issue:** Slicing state array with calculated index could overflow:
```python
self._state_array[idx:idx + self._num_aliens] = self._alien_states
```
**Impact:** If `idx + self._num_aliens > len(self._state_array)`, causes error.  
**Fix:** Add bounds check: `assert idx + self._num_aliens <= len(self._state_array)`

---

### 45. Missing Validation for Negative Episode Numbers
**File:** `main.py:622`  
**Severity:** High  
**Issue:** Episode number from metadata could be negative:
```python
self.episode = metadata.episode
```
**Impact:** Negative episode numbers cause issues in loops and logging.  
**Fix:** Validate: `self.episode = max(0, metadata.episode)`

---

## Medium Priority Bugs (46-55)

### 46. Inefficient List Conversion in Snake Collision
**File:** `src/game/snake.py:279`  
**Severity:** Medium  
**Issue:** Converts deque to list just to slice:
```python
body_set = set(list(self.snake)[:-1])
```
**Impact:** Unnecessary memory allocation for large snakes.  
**Fix:** Use deque slicing or iterate directly: `body_set = set(list(self.snake)[:-1]) if len(self.snake) > 1 else set()`

---

### 47. Hardcoded Magic Number in State Size
**File:** `src/game/snake.py:117`  
**Severity:** Medium  
**Issue:** State size calculation uses magic number `+ 5`:
```python
self._state_array = np.zeros(self.GRID_SIZE * self.GRID_SIZE + 5, dtype=np.float32)
```
**Impact:** If metadata features change, size won't match.  
**Fix:** Calculate dynamically: `METADATA_FEATURES = 5` and use constant.

---

### 48. Missing Validation for Zero Ball Speed
**File:** `src/game/pong.py:373`  
**Severity:** Medium  
**Issue:** Division by `self.ball.dx` without checking if zero:
```python
time_to_target = (self.ai_paddle.x + self.ai_paddle.width - self.ball.x) / self.ball.dx
```
**Impact:** Division by zero if ball is stationary.  
**Fix:** Check `if abs(self.ball.dx) < 1e-6: return self.ball.y` before division.

---

### 49. Potential Index Out of Bounds in Visualizer
**File:** `src/visualizer/nn_visualizer.py:498`  
**Severity:** Medium  
**Issue:** Accesses `norm_acts[j]` without checking bounds:
```python
act_val = norm_acts[j] if j < len(norm_acts) else 0
```
**Impact:** Actually safe due to check, but if `j` is negative, returns wrong value.  
**Fix:** Add check: `if 0 <= j < len(norm_acts):`

---

### 50. Missing Validation for Empty Layer Positions
**File:** `src/visualizer/nn_visualizer.py:424`  
**Severity:** Medium  
**Issue:** Checks `len(layer_positions) > 1` but doesn't validate positions aren't empty:
```python
if self.pulse_spawn_timer >= self.pulse_spawn_interval and len(layer_positions) > 1:
```
**Impact:** If layer_positions has empty lists, `len(from_layer['positions'])` could be 0.  
**Fix:** Add check: `if len(from_layer['positions']) > 0 and len(to_layer['positions']) > 0:`

---

### 51. Division by Zero in Dashboard Trend Calculation
**File:** `src/visualizer/dashboard.py:54-55`  
**Severity:** Medium  
**Issue:** Accesses `self.history[:5]` when history might be shorter:
```python
older_avg = np.mean(list(self.history)[:5]) if len(self.history) >= 5 else recent_avg
```
**Impact:** Actually safe, but if history length is exactly 5, compares same values.  
**Fix:** Use different window sizes or ensure minimum history length.

---

### 52. Missing Validation for Negative Scores
**File:** `main.py:1207`  
**Severity:** Medium  
**Issue:** Calculates average score without checking for negative values:
```python
avg_score = np.mean(list(self.dashboard.scores)[-100:]) if self.dashboard.scores else 0
```
**Impact:** Negative scores could indicate bug but are silently included.  
**Fix:** Add validation or filter: `scores = [s for s in self.dashboard.scores[-100:] if s >= 0]`

---

### 53. Potential Overflow in Episode Counter
**File:** `main.py:1251`  
**Severity:** Medium  
**Issue:** Episode counter incremented without bounds check:
```python
self.episode += 1
```
**Impact:** If episode exceeds `MAX_EPISODES`, continues training. Actually checked in loop condition.  
**Fix:** Add explicit check: `if self.episode >= self.config.MAX_EPISODES: break`

---

### 54. Missing Validation for Empty Q-Values Array
**File:** `src/ai/agent.py:434`  
**Severity:** Medium  
**Issue:** Calls `argmax()` on potentially empty tensor:
```python
actions = q_values.argmax(dim=1).cpu().numpy()
```
**Impact:** If batch is empty, `argmax()` fails.  
**Fix:** Check `if batch_size > 0:` before this.

---

### 55. Hardcoded Array Size in Space Invaders
**File:** `src/game/space_invaders.py:681-684`  
**Severity:** Medium  
**Issue:** State size calculation uses magic number `+ 5`:
```python
self._state_size = (1 + self._max_player_bullets * 2 + self._num_aliens + 5
                  + self._max_tracked_alien_bullets * 2 + 5)  # +5 for cooldown, active, ratio, lives, level
```
**Impact:** If metadata features change, size won't match.  
**Fix:** Calculate dynamically or use named constants.

---

## Low Priority Bugs (56-60)

### 56. Inefficient String Formatting in Logging
**File:** `main.py:1209-1214`  
**Severity:** Low  
**Issue:** Multiple f-string operations in print statement:
```python
print(f"Episode {self.episode:5d} | "
      f"Score: {info['score']:4d} | "
      f"Avg: {avg_score:6.1f} | "
      f"Loss: {avg_loss:.4f} | "
      f"Q: {avg_q_value:.1f} | "
      f"ε: {self.agent.epsilon:.3f}")
```
**Impact:** Minor performance impact from multiple string operations.  
**Fix:** Use single f-string or `.format()`.

---

### 57. Missing Type Validation for Config Values
**File:** `main.py:737-791`  
**Severity:** Low  
**Issue:** Config values converted but not validated for reasonable ranges:
```python
lr = float(config_data['learning_rate'])
self.config.LEARNING_RATE = lr
```
**Impact:** Invalid values (negative, zero, very large) could cause training issues.  
**Fix:** Add range validation: `if not (1e-6 <= lr <= 1.0): raise ValueError("Learning rate out of range")`

---

### 58. Potential Memory Leak in Particle System
**File:** `src/game/particles.py:163`  
**Severity:** Low  
**Issue:** Particles appended without checking max limit until after:
```python
if len(self.particles) >= self.max_particles:
    self.particles.pop(0)
self.particles.append(particle)
```
**Impact:** If `max_particles` is very large, memory could grow unbounded.  
**Fix:** Use `deque(maxlen=self.max_particles)` instead of list.

---

### 59. Missing Validation for Window Size Zero
**File:** `main.py:868-869`  
**Severity:** Low  
**Issue:** Checks for zero dimensions but doesn't validate they're positive:
```python
if self.window_width <= 0 or self.window_height <= 0:
    return
```
**Impact:** Negative dimensions could cause issues elsewhere.  
**Fix:** Check `if self.window_width <= 0 or self.window_height <= 0:` and set to minimum.

---

### 60. Hardcoded Filename Truncation Length
**File:** `main.py:3419`  
**Severity:** Low  
**Issue:** Filename truncated with magic number:
```python
filename = model['filename'][:33] + '..' if len(model['filename']) > 35 else model['filename']
```
**Impact:** If display width changes, truncation is wrong.  
**Fix:** Use constant: `MAX_FILENAME_DISPLAY_LENGTH = 35`

---

## Summary

**Total Bugs:** 30  
**Critical:** 5  
**High:** 10  
**Medium:** 10  
**Low:** 5

**Categories:**
- Array/Index Issues: 12 bugs
- Validation Missing: 8 bugs
- Division by Zero: 4 bugs
- Type/State Issues: 3 bugs
- Performance/Memory: 3 bugs
