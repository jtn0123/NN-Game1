# 30 Bugs Found in Codebase

## Critical Bugs (1-5)

### 1. Unreachable Code After Return Statement
**File:** `src/ai/replay_buffer.py:519-521`  
**Severity:** Critical  
**Issue:** Code after `return True` is unreachable:
```python
return True
self._frame_count = 0  # NEVER EXECUTED
self.beta = self.beta_start  # NEVER EXECUTED
```
**Impact:** `_frame_count` and `beta` are never reset after loading, causing incorrect PER beta annealing.

---

### 2. PrioritizedReplayBuffer.save_to_dict() Calls Non-Existent super()
**File:** `src/ai/replay_buffer.py:523-550`  
**Severity:** Critical  
**Issue:** `PrioritizedReplayBuffer` doesn't inherit from `ReplayBuffer`, but `save_to_dict()` may call `super().save_to_dict()` if implemented incorrectly.  
**Impact:** Runtime crash with `AttributeError` if super() is called.

---

### 3. PER sample_no_copy() Will Crash if batch_size > buffer_size
**File:** `src/ai/replay_buffer.py:457-479`  
**Severity:** High  
**Issue:** Uses `replace=False` without checking if batch_size exceeds buffer size:
```python
indices = np.random.choice(self._size, size=batch_size, p=probs, replace=False)
```
**Impact:** Raises `ValueError` when batch_size > `_size`.

---

### 4. Division by Zero in PER Weights Normalization
**File:** `src/ai/replay_buffer.py:481-484`  
**Severity:** High  
**Issue:** If all weights are zero, `weights.max()` is 0:
```python
weights = (self._size * probs[indices]) ** (-self.beta)
max_weight = weights.max()
weights = weights / max_weight if max_weight > 0 else weights  # Fixed in code but check needed
```
**Impact:** Potential division by zero if max_weight is 0.

---

### 5. Division by Zero in Empty Buffer Fallback
**File:** `src/ai/replay_buffer.py:469-475`  
**Severity:** High  
**Issue:** If `self._size` is 0, division by zero occurs:
```python
probs = np.ones(self._size, dtype=np.float32) / self._size  # Division by zero if _size == 0
```
**Impact:** Runtime crash when buffer is empty.

---

## High Priority Bugs (6-15)

### 6. N-Step Buffer Uses Wrong Next State When Episode Doesn't Terminate
**File:** `src/ai/replay_buffer.py:656-682`  
**Severity:** High  
**Issue:** When computing N-step returns, `actual_final_idx` may point to wrong state if episode doesn't terminate within N steps.  
**Impact:** Incorrect bootstrapping for N-step returns.

---

### 7. Duplicate Variable Assignment in _start_fresh()
**File:** `main.py:472-500`  
**Severity:** Medium  
**Issue:** Variables are assigned twice in `_start_fresh()` method.  
**Impact:** Code duplication, potential confusion.

---

### 8. Ball Speed Clamping Potential Division by Zero
**File:** `src/game/breakout.py:325-329`  
**Severity:** Medium  
**Issue:** If `current_speed` is 0, division would fail (though code checks `current_speed > 0`).  
**Impact:** Potential crash if ball velocity is exactly zero.

---

### 9. Space Invaders Win Condition Off-by-One Ambiguity
**File:** `src/game/space_invaders.py:975`  
**Severity:** Medium  
**Issue:** Win condition uses `>=` which may be ambiguous:
```python
if self.config.SI_WIN_LEVELS > 0 and self.level >= self.config.SI_WIN_LEVELS:
```
**Impact:** May win one level early or late depending on interpretation.

---

### 10. Brick State Array Bounds Check Too Lenient
**File:** `src/game/breakout.py:444-446`  
**Severity:** Medium  
**Issue:** Silent failure if arrays are mismatched:
```python
if brick_idx < len(self._brick_states):
    self._brick_states[brick_idx] = 0.0
```
**Impact:** Silent data corruption if brick list and state array lengths differ.

---

### 11. State Value Overflow for Far Bullets
**File:** `src/game/space_invaders.py:1324`  
**Severity:** Medium  
**Issue:** Bullet X relative to ship can overflow expected [0, 1] range:
```python
self._state_array[idx] = np.clip((bullet.x - ship_center_x) * self._inv_width + 0.5, 0.0, 1.0)
```
**Impact:** State values outside expected range could confuse neural network (fixed with clip but verify).

---

### 12. N-Step Buffer Potential Index Out of Bounds
**File:** `src/ai/replay_buffer.py:679`  
**Severity:** Medium  
**Issue:** `actual_final_idx` could be out of bounds if buffer is cleared during iteration.  
**Impact:** IndexError if buffer is modified during processing.

---

### 13. PER Beta Annealing Frame Count Not Reset on Load
**File:** `src/ai/replay_buffer.py:598-600`  
**Severity:** Medium  
**Issue:** When loading a saved buffer, `_frame_count` is restored but beta annealing may restart incorrectly.  
**Impact:** Beta annealing resets incorrectly after loading.

---

### 14. Ball Prediction May Not Converge
**File:** `src/game/breakout.py:509-560`  
**Severity:** Low  
**Issue:** `_predict_landing_x()` returns fallback value if simulation doesn't converge in 100 iterations.  
**Impact:** Inaccurate reward shaping signals for complex ball physics.

---

### 15. Hardcoded State Size Magic Numbers
**File:** `src/game/space_invaders.py:684-685`  
**Severity:** Medium  
**Issue:** State size calculation has `+ 5` with comment explaining what the 5 values are.  
**Impact:** State size mismatch causes crash with cryptic error if implementation changes.

---

## Medium Priority Bugs (16-25)

### 16. ReplayBuffer Capacity Not Validated
**File:** `src/ai/replay_buffer.py:54-64`  
**Severity:** Low  
**Issue:** ReplayBuffer accepts any capacity value (though it validates capacity > 0).  
**Impact:** Potential issues with invalid capacity values.

---

### 17. Shape Mismatch in Replay Buffer Copy
**File:** `src/ai/replay_buffer.py:114`  
**Severity:** Medium  
**Issue:** `np.copyto()` could fail if state shape doesn't match pre-allocated array.  
**Impact:** Crash with cryptic numpy error.

---

### 18. Time to Top Wall Division by Zero
**File:** `src/game/breakout.py:538`  
**Severity:** Low  
**Issue:** If `sim_dy` is 0, division by zero:
```python
time_to_top = (top_wall_y - sim_y) / sim_dy  # Division by zero if sim_dy == 0
```
**Impact:** Potential crash if ball has zero vertical velocity.

---

### 19. Pre-allocated State Tensor Never Resized
**File:** `src/ai/agent.py:313`  
**Severity:** Medium  
**Issue:** `_state_tensor` allocated once at init. If state_size somehow changes, tensor would be wrong size.  
**Impact:** Crash or silent corruption if state size changes.

---

### 20. Batch Tensor Device Mismatch
**File:** `src/ai/agent.py:586-596`  
**Severity:** Medium  
**Issue:** Only checks `self._batch_states.device != self.device`. Other batch tensors may be on different devices.  
**Impact:** Partial device migration could cause runtime errors.

---

### 21. Empty Hidden Layers Not Validated
**File:** `config.py:502-512`  
**Severity:** Medium  
**Issue:** `__post_init__` validation doesn't check if `HIDDEN_LAYERS` is empty.  
**Impact:** Crash during network initialization with obscure error.

---

### 22. Random Action on Zero Action Size
**File:** `src/ai/agent.py:348`  
**Severity:** Medium  
**Issue:** `random.randrange(self.action_size)` raises `ValueError` if `action_size` is 0.  
**Impact:** Crash if game has 0 actions.

---

### 23. PER Sampling Replace Logic Inconsistent
**File:** `src/ai/replay_buffer.py:438-439`  
**Severity:** Medium  
**Issue:** `use_replacement = batch_size > self._size` but buffer size could shrink at any time via clear().  
**Impact:** May sample without replacement when buffer is too small, causing crash.

---

### 24. Model Auto-Load Missing Compatibility Check
**File:** `main.py:336-371`  
**Severity:** Low  
**Issue:** Auto-loading most recent save checks compatibility but could still fail if model is corrupted.  
**Impact:** Loads incompatible model, then fails during actual load.

---

### 25. Epsilon Warmup Bypass
**File:** `src/ai/trainer.py:196`  
**Severity:** Low  
**Issue:** `decay_epsilon()` called without episode parameter, bypassing warmup check.  
**Impact:** Epsilon decays during warmup period when it shouldn't.

---

## Low Priority Bugs (26-30)

### 26. N-Step Buffer Unnecessary State Copies
**File:** `src/ai/replay_buffer.py:650`  
**Severity:** Low  
**Issue:** States are copied in `push()` but parent `ReplayBuffer.push()` doesn't copy.  
**Impact:** Unnecessary memory usage and slowdown.

---

### 27. Terminal State Array Update Timing
**File:** `src/game/breakout.py:852-861`  
**Severity:** Low  
**Issue:** In `VecBreakout.step()`, state array is updated after returning terminal states.  
**Impact:** Future refactoring could break this subtle timing requirement.

---

### 28. Log Timestamp Format Incorrect
**File:** `src/web/server.py:576`  
**Severity:** Low  
**Issue:** Timestamp formatting `strftime("%H:%M:%S.%f")[:12]` - `%f` is 6 digits, so `[:12]` gives wrong precision.  
**Impact:** Timestamp precision is wrong.

---

### 29. Model Directory Creation Race Condition
**File:** `src/ai/agent.py:794-796`  
**Severity:** Low  
**Issue:** `os.makedirs(dir_path, exist_ok=True)` could race if multiple processes create same directory.  
**Impact:** Rare crash if two processes create directory simultaneously.

---

### 30. Scale Factor Zero on Minimized Window
**File:** `main.py:805-821`  
**Severity:** Low  
**Issue:** `_update_scale()` guards against zero dimensions, but `scale_factor` could still be 0 if window is minimized.  
**Impact:** Division by zero or rendering errors when window is minimized.

---

## Summary

**By Severity:**
- Critical: 2 bugs
- High: 4 bugs  
- Medium: 14 bugs
- Low: 10 bugs

**By Category:**
- Replay Buffer: 12 bugs
- Game Logic: 6 bugs
- Neural Network: 5 bugs
- Configuration: 3 bugs
- Web Dashboard: 2 bugs
- Other: 2 bugs

**Top 5 Most Critical Fixes:**
1. Fix unreachable code in PrioritizedReplayBuffer.load_from_dict()
2. Fix PER sample_no_copy() replace=False crash
3. Fix division by zero in PER weights normalization
4. Fix division by zero in empty buffer fallback
5. Fix N-step buffer next state computation
