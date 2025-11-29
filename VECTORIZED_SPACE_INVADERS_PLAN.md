# Vectorized Space Invaders Implementation Plan

## Overview

This document provides detailed implementation instructions for adding vectorized (parallel) environment support to Space Invaders, following the existing pattern used by Breakout. This will enable ~3-4x performance improvement when using `--vec-envs N` flag.

Additionally, this plan covers suppressing verbose HTTP logging that clutters the terminal output.

---

## Part 1: Vectorized Space Invaders Environment

### 1.1 Reference Implementation

The existing `VecBreakout` class in `src/game/breakout.py` (lines 709-804) serves as the template. Study this implementation first:

```python
class VecBreakout:
    """
    Vectorized Breakout environment for parallel game execution.
    """
    def __init__(self, num_envs: int, config: Config, headless: bool = True):
        self.num_envs = num_envs
        self.config = config
        self.headless = headless
        
        # Create N independent game instances
        self.envs = [Breakout(config, headless=headless) for _ in range(num_envs)]
        
        # Pre-allocate numpy arrays for batched operations
        self._states = np.empty((num_envs, self.state_size), dtype=np.float32)
        self._rewards = np.empty(num_envs, dtype=np.float32)
        self._dones = np.empty(num_envs, dtype=np.bool_)
    
    def reset(self) -> np.ndarray:
        """Reset all environments and return batched initial states."""
        for i, env in enumerate(self.envs):
            self._states[i] = env.reset()
        return self._states.copy()
    
    def step(self, actions: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Step all environments with batched actions.
        
        Args:
            actions: Array of actions, shape (num_envs,)
        
        Returns:
            - next_states: shape (num_envs, state_size)
            - rewards: shape (num_envs,)
            - dones: shape (num_envs,)
        """
        for i, (env, action) in enumerate(zip(self.envs, actions)):
            state, reward, done = env.step(int(action))
            self._states[i] = state
            self._rewards[i] = reward
            self._dones[i] = done
            
            # Auto-reset completed environments
            if done:
                self._states[i] = env.reset()
        
        return self._states.copy(), self._rewards.copy(), self._dones.copy()
```

### 1.2 Create VecSpaceInvaders Class

**File:** `src/game/space_invaders.py`  
**Location:** Add after the main `SpaceInvaders` class (around line 1450+)

Create a new class `VecSpaceInvaders` with the following structure:

```python
class VecSpaceInvaders:
    """
    Vectorized Space Invaders environment for parallel game execution.
    
    This enables running multiple Space Invaders games simultaneously,
    significantly improving training throughput when using batch action selection.
    
    Example:
        >>> vec_env = VecSpaceInvaders(num_envs=8, config=config)
        >>> states = vec_env.reset()  # Shape: (8, state_size)
        >>> actions = agent.select_actions_batch(states)
        >>> next_states, rewards, dones = vec_env.step(actions)
    """
    
    def __init__(self, num_envs: int, config: Config, headless: bool = True):
        """
        Initialize vectorized Space Invaders environment.
        
        Args:
            num_envs: Number of parallel environments
            config: Game configuration
            headless: Whether to run in headless mode (no rendering)
        """
        self.num_envs = num_envs
        self.config = config
        self.headless = headless
        
        # Create N independent game instances
        self.envs = [SpaceInvaders(config, headless=headless) for _ in range(num_envs)]
        
        # Copy properties from first environment
        self.state_size = self.envs[0].state_size
        self.action_size = self.envs[0].action_size
        
        # Pre-allocate numpy arrays for batched operations (avoid repeated allocation)
        self._states = np.empty((num_envs, self.state_size), dtype=np.float32)
        self._rewards = np.empty(num_envs, dtype=np.float32)
        self._dones = np.empty(num_envs, dtype=np.bool_)
    
    def reset(self) -> np.ndarray:
        """
        Reset all environments.
        
        Returns:
            Batched initial states of shape (num_envs, state_size)
        """
        for i, env in enumerate(self.envs):
            self._states[i] = env.reset()
        return self._states.copy()
    
    def step(self, actions: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Step all environments with batched actions.
        
        Args:
            actions: Array of actions, shape (num_envs,)
        
        Returns:
            Tuple of:
            - next_states: shape (num_envs, state_size)
            - rewards: shape (num_envs,)
            - dones: shape (num_envs,)
        """
        for i, (env, action) in enumerate(zip(self.envs, actions)):
            state, reward, done = env.step(int(action))
            self._states[i] = state
            self._rewards[i] = reward
            self._dones[i] = done
            
            # Auto-reset completed environments
            if done:
                self._states[i] = env.reset()
        
        return self._states.copy(), self._rewards.copy(), self._dones.copy()
    
    def close(self) -> None:
        """Clean up all environments."""
        for env in self.envs:
            if hasattr(env, 'close'):
                env.close()
```

### 1.3 Export VecSpaceInvaders

**File:** `src/game/__init__.py`

Add `VecSpaceInvaders` to the imports and exports:

```python
from .space_invaders import SpaceInvaders, VecSpaceInvaders
```

Make sure it's included in `__all__` if that list exists.

### 1.4 Update HeadlessTrainer to Support VecSpaceInvaders

**File:** `main.py`  
**Location:** In `HeadlessTrainer.__init__()` (around lines 1598-1614)

Currently the code only creates VecBreakout. Update it to also handle Space Invaders:

Find this section (around line 1598):
```python
if self.num_envs > 1:
    # Create vectorized environment for parallel game execution
    if config.GAME_NAME == 'breakout':
        self.vec_env = VecBreakout(self.num_envs, config, headless=True)
        self.game = self.vec_env.envs[0]  # Reference for state/action size
        print(f"🎮 Vectorized: {self.num_envs} parallel environments")
    else:
        # Fallback: vectorized not supported for this game
        # Concrete game classes accept (config, headless) but BaseGame has no __init__ params
        self.game = GameClass(config, headless=True)  # type: ignore[call-arg]
        self.num_envs = 1
```

Replace with:
```python
if self.num_envs > 1:
    # Create vectorized environment for parallel game execution
    if config.GAME_NAME == 'breakout':
        from src.game.breakout import VecBreakout
        self.vec_env = VecBreakout(self.num_envs, config, headless=True)
        self.game = self.vec_env.envs[0]  # Reference for state/action size
        print(f"🎮 Vectorized: {self.num_envs} parallel environments")
    elif config.GAME_NAME == 'space_invaders':
        from src.game.space_invaders import VecSpaceInvaders
        self.vec_env = VecSpaceInvaders(self.num_envs, config, headless=True)
        self.game = self.vec_env.envs[0]  # Reference for state/action size
        print(f"🎮 Vectorized: {self.num_envs} parallel environments")
    else:
        # Fallback: vectorized not supported for this game
        print(f"⚠️ Vectorized environments not yet supported for {config.GAME_NAME}")
        print(f"   Falling back to single environment")
        self.game = GameClass(config, headless=True)  # type: ignore[call-arg]
        self.num_envs = 1
```

### 1.5 Verify Imports

**File:** `main.py`  
**Location:** Top of file imports (around line 75)

Ensure `VecSpaceInvaders` can be imported. The import is done lazily in the code above, so this should work. But verify that `src/game/space_invaders.py` has the proper numpy import at the top:

```python
import numpy as np
from typing import Tuple, Optional, List
```

---

## Part 2: Suppress Verbose HTTP Logging

### 2.1 Problem

The terminal is flooded with HTTP request logs like:
```
127.0.0.1 - - [29/Nov/2025 03:02:24] "GET /api/screenshot HTTP/1.1" 200 -
127.0.0.1 - - [29/Nov/2025 03:02:24] "GET /api/save-status HTTP/1.1" 200 -
```

These appear every 2 seconds and clutter the training output.

### 2.2 Current State

**File:** `src/web/server.py` (lines 1069-1073)

The logging is set to ERROR level but Flask's default output still shows:
```python
import logging
logging.getLogger('werkzeug').setLevel(logging.ERROR)
logging.getLogger('engineio').setLevel(logging.ERROR)
logging.getLogger('socketio').setLevel(logging.ERROR)
```

### 2.3 Fix: Disable Werkzeug Request Logging

**File:** `src/web/server.py`  
**Location:** Inside `start()` method (around line 1069)

Update the logging suppression to be more aggressive:

```python
def start(self) -> None:
    """Start the web server in a background thread."""
    if self._running:
        return
    
    self._running = True
    self.publisher.set_running(True)
    
    # Suppress Flask/werkzeug logging COMPLETELY
    import logging
    import sys
    
    # Disable werkzeug request logging
    log = logging.getLogger('werkzeug')
    log.setLevel(logging.ERROR)
    log.disabled = True  # ADD THIS LINE
    
    # Also suppress via click (werkzeug uses click for CLI output)
    import click
    def secho_noop(*args, **kwargs):
        pass
    def echo_noop(*args, **kwargs):
        pass
    # Monkey-patch click's output functions used by werkzeug
    # Only if we want complete silence
    
    # Alternative: Use a custom log handler that filters 200 responses
    class RequestFilter(logging.Filter):
        def filter(self, record):
            # Filter out successful requests (200 status)
            message = record.getMessage()
            if '" 200 -' in message or '" 304 -' in message:
                return False
            return True
    
    log.addFilter(RequestFilter())
    
    logging.getLogger('engineio').setLevel(logging.ERROR)
    logging.getLogger('socketio').setLevel(logging.ERROR)
    
    # ... rest of method
```

### 2.4 Alternative: Use Environment Variable

A simpler approach is to set the environment variable before Flask starts:

**File:** `src/web/server.py`  
**Location:** At the top of the file (after imports, around line 45)

Add:
```python
import os
os.environ['WERKZEUG_RUN_MAIN'] = 'true'  # Suppress reloader messages
```

And in `start()` method:
```python
def start(self) -> None:
    """Start the web server in a background thread."""
    if self._running:
        return
    
    self._running = True
    self.publisher.set_running(True)
    
    # Suppress Flask/werkzeug logging
    import logging
    import os
    
    # Completely disable werkzeug logging
    os.environ['WERKZEUG_LOG_LEVEL'] = 'ERROR'
    
    log = logging.getLogger('werkzeug')
    log.setLevel(logging.CRITICAL)  # Use CRITICAL instead of ERROR
    log.disabled = True
    
    logging.getLogger('engineio').setLevel(logging.ERROR)
    logging.getLogger('socketio').setLevel(logging.ERROR)
    
    # ... rest of method
```

### 2.5 Most Reliable Fix: Custom WSGI Wrapper

If the above doesn't work, wrap the app to intercept logging:

**File:** `src/web/server.py`  
**Location:** In `start()` method

```python
def run_server():
    # Redirect stderr for werkzeug silencing
    import io
    import sys
    
    class FilteredStderr:
        def __init__(self, original):
            self.original = original
        
        def write(self, msg):
            # Filter out HTTP 200/304 request logs
            if '"] "GET /' in msg or '"] "POST /' in msg:
                if '" 200 -' in msg or '" 304 -' in msg:
                    return  # Suppress successful requests
            self.original.write(msg)
        
        def flush(self):
            self.original.flush()
    
    # Only filter during server run
    # sys.stderr = FilteredStderr(sys.stderr)
    
    print(f"\n🌐 Web Dashboard running at http://localhost:{self.port}")
    print("   Open in browser to view training progress\n")
    
    self.socketio.run(
        self.app,
        host=self.host,
        port=self.port,
        debug=False,
        use_reloader=False,
        log_output=False  # This should disable request logging
    )
```

The key is `log_output=False` in `socketio.run()`. Verify this is already set. If it is and logging still appears, the issue is werkzeug ignoring this setting.

---

## Part 3: Testing

### 3.1 Test Vectorized Space Invaders

```bash
# Test with 4 parallel environments
python main.py --headless --turbo --web --vec-envs 4 --game space_invaders --port 5001

# Expected output should NOT show the warning:
# "⚠️ Vectorized environments not yet supported for space_invaders"

# Instead should show:
# "🎮 Vectorized: 4 parallel environments"
```

### 3.2 Test Performance Improvement

Compare steps/sec:
- Without vec-envs: ~2,000 steps/sec
- With `--vec-envs 4`: ~6,000-8,000 steps/sec expected
- With `--vec-envs 8`: ~10,000-15,000 steps/sec expected

### 3.3 Test HTTP Log Suppression

After implementing the fix, the terminal should show:
```
Ep   750 | Score: 2010 | Avg: 1655.0 | ε: 0.833 | ⚡ 2,342 steps/s | 📊 2,045 ep/hr
Ep   751 | Score: 4870 | Avg: 2298.0 | ε: 0.832 | ⚡ 2,207 steps/s | 📊 1,339 ep/hr
```

Without the interleaved HTTP request logs like:
```
127.0.0.1 - - [timestamp] "GET /api/screenshot HTTP/1.1" 200 -
```

---

## Summary of Files to Modify

| File | Changes |
|------|---------|
| `src/game/space_invaders.py` | Add `VecSpaceInvaders` class at end of file |
| `src/game/__init__.py` | Export `VecSpaceInvaders` |
| `main.py` | Add space_invaders case to vectorized env creation |
| `src/web/server.py` | Suppress werkzeug HTTP request logging |

---

## Checklist

- [ ] `VecSpaceInvaders` class created with same interface as `VecBreakout`
- [ ] `VecSpaceInvaders` exported from `src/game/__init__.py`
- [ ] `HeadlessTrainer` updated to create `VecSpaceInvaders` when `--vec-envs > 1`
- [ ] No more "Vectorized environments not yet supported" warning for space_invaders
- [ ] HTTP 200/304 request logs suppressed from terminal output
- [ ] Performance improvement verified with `--vec-envs 4` or `--vec-envs 8`

