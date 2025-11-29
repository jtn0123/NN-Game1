# Performance Controls Enhancement - Implementation Plan

## Overview

This plan adds two main features to the web dashboard:
1. **Save & Quit Button** - A 6th control button that saves the model and exits gracefully
2. **Enhanced Performance Scaling** - Ultra mode preset, parallel environments input, and exposed performance parameters

---

## Part 1: Save & Quit Button

### 1.1 Add Button HTML

**File:** `src/web/templates/dashboard.html`  
**Location:** Lines 287-303 (inside `.controls-grid` div)

Add a 6th button after the "Start Fresh" button:

```html
<!-- Current buttons: Pause, Save, Reset Episode, Load, Start Fresh -->
<!-- ADD THIS after the Start Fresh button (around line 302): -->
<button class="control-btn quit" onclick="saveAndQuit()" data-tooltip="Save current progress and exit the training application" data-tooltip-position="bottom">
    🚪 Save & Quit
</button>
```

The full controls-grid section should look like:
```html
<div class="controls-grid">
    <button class="control-btn pause" id="pause-btn" onclick="togglePause()" ...>⏸️ Pause</button>
    <button class="control-btn save" onclick="saveModel()" ...>💾 Save</button>
    <button class="control-btn reset" onclick="resetEpisode()" ...>🔄 Reset Episode</button>
    <button class="control-btn load" onclick="showLoadModal()" ...>📂 Load</button>
    <button class="control-btn fresh" onclick="startFresh()" ...>🆕 Start Fresh</button>
    <button class="control-btn quit" onclick="saveAndQuit()" data-tooltip="Save current progress and exit the training application" data-tooltip-position="bottom">
        🚪 Save & Quit
    </button>
</div>
```

### 1.2 Add Button CSS Styles

**File:** `src/web/static/styles.css`  
**Location:** After `.control-btn.fresh` styles (around line 886)

Add these styles:

```css
.control-btn.quit {
    border-color: rgba(156, 39, 176, 0.5);
    background: linear-gradient(145deg, rgba(156, 39, 176, 0.08) 0%, var(--bg-secondary) 100%);
}

.control-btn.quit:hover {
    background: linear-gradient(145deg, rgba(156, 39, 176, 0.15) 0%, var(--bg-secondary) 100%);
    border-color: #9c27b0;
    box-shadow: 0 4px 16px rgba(156, 39, 176, 0.15);
}

.control-btn.quit.quitting {
    background: linear-gradient(145deg, #9c27b0 0%, #7b1fa2 100%);
    border-color: #9c27b0;
    color: white;
}
```

### 1.3 Add JavaScript Handler

**File:** `src/web/static/app.js`  
**Location:** After `startFresh()` function (around line 871)

Add this function:

```javascript
/**
 * Save model and quit the application
 */
function saveAndQuit() {
    const confirmed = confirm(
        '🚪 Save & Quit\n\n' +
        'This will:\n' +
        '• Save your current training progress\n' +
        '• Shut down the training server\n\n' +
        'Continue?'
    );
    
    if (!confirmed) {
        addConsoleLog('Save & Quit cancelled', 'info');
        return;
    }
    
    // Update button to show saving state
    const btn = document.querySelector('.control-btn.quit');
    if (btn) {
        const originalText = btn.textContent;
        btn.textContent = '⏳ Saving...';
        btn.classList.add('quitting');
        btn.disabled = true;
    }
    
    socket.emit('control', { action: 'save_and_quit' });
    addConsoleLog('💾 Saving and shutting down...', 'warning');
    
    // Show shutdown message after a delay
    setTimeout(() => {
        showShutdownOverlay();
    }, 1000);
}

/**
 * Show shutdown overlay when server is stopping
 */
function showShutdownOverlay() {
    const overlay = document.createElement('div');
    overlay.id = 'shutdown-overlay';
    overlay.style.cssText = `
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background: rgba(10, 10, 15, 0.95);
        z-index: 10000;
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        font-family: 'JetBrains Mono', monospace;
    `;
    
    overlay.innerHTML = `
        <div style="font-size: 4rem; margin-bottom: 20px;">👋</div>
        <h2 style="color: #4caf50; margin: 0 0 15px 0; font-size: 1.5rem;">Training Saved & Stopped</h2>
        <p style="color: #7a7e8c; margin: 0;">Your progress has been saved. You can close this tab.</p>
        <p style="color: #5a5e72; margin: 15px 0 0 0; font-size: 0.85rem;">To resume: python main.py --headless --web</p>
    `;
    
    document.body.appendChild(overlay);
}
```

### 1.4 Add Socket Handler in Server

**File:** `src/web/server.py`  
**Location:** Inside `_register_socket_events()` method, in the `handle_control()` function (around line 970)

Add this case to the if/elif chain:

```python
elif action == 'save_and_quit':
    if self.on_save_and_quit_callback:
        self.on_save_and_quit_callback()
```

Also add the callback attribute in `__init__()` (around line 640):

```python
self.on_save_and_quit_callback: Optional[Callable[[], None]] = None
```

### 1.5 Wire Up Callback in HeadlessTrainer

**File:** `main.py`  
**Location:** Inside `_setup_web_callbacks()` method of `HeadlessTrainer` class (around line 1743)

Add this line:

```python
self.web_dashboard.on_save_and_quit_callback = self._save_and_quit
```

Then add the method to `HeadlessTrainer` class (around line 2116, after `_set_performance_mode`):

```python
def _save_and_quit(self) -> None:
    """Save the model and exit the application gracefully."""
    if self.web_dashboard:
        self.web_dashboard.log("💾 Saving model before shutdown...", "warning")
    
    # Save the model
    self._save_model(f"{self.config.GAME_NAME}_final.pth", save_reason="shutdown")
    
    if self.web_dashboard:
        self.web_dashboard.log("✅ Model saved. Shutting down...", "success")
    
    print("\n👋 Save & Quit requested. Model saved. Exiting...")
    
    # Give time for the save event to propagate to clients
    import time
    time.sleep(0.5)
    
    # Exit gracefully
    self.running = False
    import sys
    sys.exit(0)
```

---

## Part 2: Ultra Performance Mode

### 2.1 Add Ultra Button HTML

**File:** `src/web/templates/dashboard.html`  
**Location:** Inside `.mode-buttons` div (around line 274-284)

Add Ultra button after Turbo:

```html
<div class="mode-buttons">
    <button class="mode-btn active" id="mode-normal" onclick="setPerformanceMode('normal')" data-tooltip="Learn every step. Best for watching learning in detail. ~200-300 steps/sec">
        Normal
    </button>
    <button class="mode-btn" id="mode-fast" onclick="setPerformanceMode('fast')" data-tooltip="Learn every 4 steps. ~4x faster training with minimal learning impact. ~800-1200 steps/sec">
        Fast
    </button>
    <button class="mode-btn turbo" id="mode-turbo" onclick="setPerformanceMode('turbo')" data-tooltip="Learn every 8 steps + 2 gradient updates. Maximum training speed. ~5000 steps/sec on M4">
        🚀 Turbo
    </button>
    <button class="mode-btn ultra" id="mode-ultra" onclick="setPerformanceMode('ultra')" data-tooltip="Learn every 16 steps + 4 gradient updates + batch 256. Extreme throughput. ~8000+ steps/sec">
        ⚡ Ultra
    </button>
</div>
```

### 2.2 Add Ultra Button CSS

**File:** `src/web/static/styles.css`  
**Location:** After `.mode-btn.turbo.active` styles (around line 1641)

Add these styles:

```css
.mode-btn.ultra {
    border-color: rgba(233, 30, 99, 0.4);
}

.mode-btn.ultra:hover {
    background: rgba(233, 30, 99, 0.1);
    border-color: #e91e63;
    color: #e91e63;
}

.mode-btn.ultra.active {
    background: linear-gradient(145deg, #e91e63 0%, #c2185b 100%);
    border-color: #e91e63;
    color: white;
    text-shadow: 0 1px 2px rgba(0, 0, 0, 0.3);
}
```

### 2.3 Update JavaScript for Ultra Mode

**File:** `src/web/static/app.js`  
**Location:** Inside `syncSettingsFromMode()` function (around line 1242-1275)

Update the function to include ultra:

```javascript
function syncSettingsFromMode(mode) {
    let learnEvery, batchSize, gradientSteps;
    
    if (mode === 'normal') {
        learnEvery = 1;
        batchSize = 128;
        gradientSteps = 1;
    } else if (mode === 'fast') {
        learnEvery = 4;
        batchSize = 128;
        gradientSteps = 1;
    } else if (mode === 'turbo') {
        learnEvery = 8;
        batchSize = 128;
        gradientSteps = 2;
    } else if (mode === 'ultra') {
        learnEvery = 16;
        batchSize = 256;
        gradientSteps = 4;
    }
    
    // ... rest of function unchanged
}
```

Also update `setPerformanceMode()` log messages (around line 1231):

```javascript
const modeNames = {
    'normal': 'Normal (learn every step)',
    'fast': 'Fast (learn every 4 steps)',
    'turbo': 'Turbo (learn every 8, batch 128, 2 grad steps)',
    'ultra': 'Ultra (learn every 16, batch 256, 4 grad steps)'
};
```

### 2.4 Add Ultra Mode to Backend

**File:** `main.py`  
**Location:** Inside `_set_performance_mode()` method in both `GameApp` class (around line 692) and `HeadlessTrainer` class (around line 2092)

Update both methods to include ultra:

```python
def _set_performance_mode(self, mode: str) -> None:
    """Set performance mode preset."""
    if mode == 'normal':
        self.config.LEARN_EVERY = 1
        self.config.BATCH_SIZE = 128
        self.config.GRADIENT_STEPS = 1
    elif mode == 'fast':
        self.config.LEARN_EVERY = 4
        self.config.BATCH_SIZE = 128
        self.config.GRADIENT_STEPS = 1
    elif mode == 'turbo':
        self.config.LEARN_EVERY = 8
        self.config.BATCH_SIZE = 128
        self.config.GRADIENT_STEPS = 2
    elif mode == 'ultra':
        self.config.LEARN_EVERY = 16
        self.config.BATCH_SIZE = 256
        self.config.GRADIENT_STEPS = 4
    
    if self.web_dashboard:
        self.web_dashboard.publisher.set_performance_mode(mode)
        self.web_dashboard.publisher.state.learn_every = self.config.LEARN_EVERY
        self.web_dashboard.publisher.state.batch_size = self.config.BATCH_SIZE
        self.web_dashboard.publisher.state.gradient_steps = self.config.GRADIENT_STEPS
        self.web_dashboard.log(
            f"⚡ Performance mode: {mode.upper()} (learn_every={self.config.LEARN_EVERY}, batch={self.config.BATCH_SIZE}, grad_steps={self.config.GRADIENT_STEPS})",
            "action"
        )
    print(f"⚡ Performance mode: {mode.upper()}")
```

---

## Part 3: Parallel Environments Input (Vec-Envs)

### 3.1 Add Vec-Envs Input Field

**File:** `src/web/templates/dashboard.html`  
**Location:** Inside Settings panel, under the Performance section (around line 405-417)

Add after the "Gradient Steps" setting row:

```html
<div class="setting-row vec-envs-row" data-tooltip="Number of parallel game environments. Higher = faster training but requires more memory. Requires restart to apply changes.">
    <label for="setting-vec-envs">Parallel Envs</label>
    <div class="vec-envs-control">
        <input type="number" id="setting-vec-envs" value="1" min="1" max="32" step="1">
        <span class="restart-badge" id="vec-envs-restart-badge">Restart required</span>
    </div>
</div>
```

### 3.2 Add Vec-Envs CSS Styles

**File:** `src/web/static/styles.css`  
**Location:** After the settings-related styles (around line 1480)

Add these styles:

```css
/* Vec-Envs Control */
.vec-envs-control {
    display: flex;
    align-items: center;
    gap: 8px;
}

.vec-envs-control input {
    width: 70px;
}

.restart-badge {
    font-size: 0.7rem;
    padding: 2px 6px;
    background: rgba(255, 152, 0, 0.15);
    border: 1px solid rgba(255, 152, 0, 0.3);
    border-radius: 4px;
    color: #ff9800;
    white-space: nowrap;
    opacity: 0;
    transition: opacity 0.2s;
}

.restart-badge.visible {
    opacity: 1;
}

.vec-envs-row.changed .restart-badge {
    opacity: 1;
}
```

### 3.3 Add Vec-Envs JavaScript Handler

**File:** `src/web/static/app.js`  
**Location:** After `applySettings()` function (around line 1413)

Add this code:

```javascript
// Track original vec-envs value
let originalVecEnvs = 1;

/**
 * Initialize vec-envs input handler
 */
function initVecEnvsHandler() {
    const vecEnvsInput = document.getElementById('setting-vec-envs');
    const restartBadge = document.getElementById('vec-envs-restart-badge');
    const settingRow = vecEnvsInput?.closest('.setting-row');
    
    if (!vecEnvsInput) return;
    
    vecEnvsInput.addEventListener('input', () => {
        const newValue = parseInt(vecEnvsInput.value, 10);
        const hasChanged = newValue !== originalVecEnvs;
        
        if (settingRow) {
            settingRow.classList.toggle('changed', hasChanged);
        }
        
        if (hasChanged && restartBadge) {
            restartBadge.classList.add('visible');
        } else if (restartBadge) {
            restartBadge.classList.remove('visible');
        }
    });
}

/**
 * Show restart command for vec-envs change
 */
function showVecEnvsRestartCommand() {
    const vecEnvsInput = document.getElementById('setting-vec-envs');
    const newValue = parseInt(vecEnvsInput?.value || '1', 10);
    
    if (newValue === originalVecEnvs) {
        addConsoleLog('No change to parallel environments', 'info');
        return;
    }
    
    const command = `python main.py --headless --turbo --web --vec-envs ${newValue}`;
    
    // Show modal with restart command
    showRestartBanner('parallel environments change', command);
    addConsoleLog(`⚡ To use ${newValue} parallel environments, restart with: ${command}`, 'warning');
}

// Call this in loadConfig() to set original value
// Add to loadConfig() after fetching config:
// originalVecEnvs = data.vec_envs || 1;
// document.getElementById('setting-vec-envs').value = originalVecEnvs;
```

Also update `document.addEventListener('DOMContentLoaded', ...)` (around line 63-73) to include:

```javascript
document.addEventListener('DOMContentLoaded', () => {
    initCharts();
    initNNVisualizer();
    connectSocket();
    startScreenshotPolling();
    updateFooterTime();
    setInterval(updateFooterTime, 1000);
    loadConfig();
    loadGames();
    loadGameStats();
    initVecEnvsHandler();  // ADD THIS LINE
});
```

### 3.4 Update Settings Apply Button

**File:** `src/web/static/app.js`  
**Location:** Inside `applySettings()` function (around line 1392-1413)

Add vec-envs check:

```javascript
function applySettings() {
    // Check if vec-envs changed (requires restart)
    const vecEnvsInput = document.getElementById('setting-vec-envs');
    const newVecEnvs = parseInt(vecEnvsInput?.value || '1', 10);
    
    if (newVecEnvs !== originalVecEnvs) {
        showVecEnvsRestartCommand();
        // Don't return - still apply other settings
    }
    
    const config = {
        learning_rate: parseFloat(document.getElementById('setting-lr').value),
        epsilon: parseFloat(document.getElementById('setting-epsilon').value),
        epsilon_decay: parseFloat(document.getElementById('setting-decay').value),
        gamma: parseFloat(document.getElementById('setting-gamma').value),
        batch_size: parseInt(document.getElementById('setting-batch').value),
        learn_every: parseInt(document.getElementById('setting-learn-every').value),
        gradient_steps: parseInt(document.getElementById('setting-grad-steps').value)
    };

    socket.emit('control', { action: 'config_change', config: config });
    addConsoleLog('Settings updated', 'action', null, config);

    // ... rest of function
}
```

### 3.5 Track Vec-Envs in Server State

**File:** `src/web/server.py`  
**Location:** In `TrainingState` dataclass (around line 79-113)

Add this field:

```python
@dataclass
class TrainingState:
    """Current training state for API."""
    # ... existing fields ...
    
    # Add this field:
    num_envs: int = 1  # Number of parallel environments
```

**File:** `src/web/server.py`  
**Location:** In `/api/config` route (around line 657-676)

Add num_envs to the returned config:

```python
@self.app.route('/api/config')
def api_config():
    return jsonify({
        # ... existing fields ...
        'vec_envs': self.publisher.state.num_envs,  # ADD THIS
    })
```

### 3.6 Set Vec-Envs Count on Startup

**File:** `main.py`  
**Location:** In `HeadlessTrainer._send_system_info()` (around line 1745-1763)

Add after setting system info:

```python
def _send_system_info(self) -> None:
    """Send system information to web dashboard."""
    if not self.web_dashboard:
        return
    
    torch_compiled = getattr(self.agent, '_compiled', False)
    device_str = str(self.config.DEVICE)
    
    self.web_dashboard.publisher.set_system_info(
        device=device_str,
        torch_compiled=torch_compiled,
        target_episodes=self.config.MAX_EPISODES
    )
    
    # ADD: Set number of parallel environments
    self.web_dashboard.publisher.state.num_envs = self.num_envs
    
    # Set performance mode based on turbo flag
    if self.args.turbo:
        self.web_dashboard.publisher.set_performance_mode('turbo')
    else:
        self.web_dashboard.publisher.set_performance_mode('normal')
```

---

## Part 4: Display Current Vec-Envs in Training Stats

### 4.1 Add Vec-Envs Display to Stats Panel

**File:** `src/web/templates/dashboard.html`  
**Location:** In the Training Stats info-grid section (around line 225-234)

Add a new info-row to show current parallel environments:

```html
<!-- After the "Explore / Exploit" row, add: -->
<div class="info-row" data-tooltip="Number of parallel game environments running. More environments = faster training throughput.">
    <span class="info-label">🎮 Parallel Envs</span>
    <span class="info-value" id="info-vec-envs">1</span>
</div>
```

### 4.2 Update JavaScript to Display Vec-Envs

**File:** `src/web/static/app.js`  
**Location:** In `updateDashboard()` function (around line 455-562)

Add this line to update the display:

```javascript
// Inside updateDashboard(), add after other info updates:
const vecEnvsEl = document.getElementById('info-vec-envs');
if (vecEnvsEl && state.num_envs) {
    vecEnvsEl.textContent = state.num_envs;
}
```

---

## Summary of Files to Modify

| File | Changes |
|------|---------|
| `src/web/templates/dashboard.html` | Add Save & Quit button, Ultra mode button, vec-envs input, vec-envs display |
| `src/web/static/styles.css` | Add `.control-btn.quit`, `.mode-btn.ultra`, `.vec-envs-control`, `.restart-badge` styles |
| `src/web/static/app.js` | Add `saveAndQuit()`, `showShutdownOverlay()`, `initVecEnvsHandler()`, update `syncSettingsFromMode()` for ultra |
| `src/web/server.py` | Add `on_save_and_quit_callback`, add `num_envs` to TrainingState, add `save_and_quit` handler |
| `main.py` | Add `_save_and_quit()` method, update `_set_performance_mode()` for ultra, set `num_envs` in system info |

---

## Bug Fixes Applied

### Bug 1: `save_and_quit` Handler NameError (FIXED)

**File:** `src/web/server.py`  
**Issue:** Lines 997-1001 contained extraneous code that referenced `game_name`, which was not defined in the `save_and_quit` context. This would cause a `NameError` at runtime.

**Fix:** Removed the erroneous `game_switched` emit that was mistakenly copied from the `switch_game` handler.

### Bug 2: Missing `gradient_steps` in GameApp (FIXED)

**File:** `main.py`  
**Issue:** `GameApp._set_performance_mode()` only updated `learn_every` and `batch_size` on the publisher state, but omitted `gradient_steps`. The log message also didn't include grad_steps. This caused stale values in the dashboard when Ultra mode was activated.

**Fix:** Added `self.web_dashboard.publisher.state.gradient_steps = self.config.GRADIENT_STEPS` and updated the log message to include `grad_steps`.

---

## Improvements Added

### Keyboard Shortcuts

**File:** `src/web/static/app.js`

Added keyboard shortcuts for new features:
- `Ctrl+Q` / `Cmd+Q` - Save & Quit
- `4` - Ultra performance mode (complements existing 1/2/3 for Normal/Fast/Turbo)

Updated tooltip on Save & Quit button to document the shortcut.

---

## Testing Checklist

- [ ] Save & Quit button appears in controls grid
- [ ] Save & Quit shows confirmation dialog
- [ ] Save & Quit saves model and exits gracefully (Ctrl+Q works)
- [ ] Ultra mode button appears and activates (key 4 works)
- [ ] Ultra mode sets correct parameters (learn_every=16, batch=256, grad_steps=4)
- [ ] Ultra mode updates all three values in dashboard display
- [ ] Vec-envs input shows in settings
- [ ] Changing vec-envs shows restart warning
- [ ] Current vec-envs count displays in training stats
- [ ] All styles render correctly in dark theme
- [ ] No NameError when using Save & Quit

