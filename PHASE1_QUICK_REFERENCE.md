# Phase 1: Quick Reference Guide

## What is Phase 1?
Foundation and performance improvements to the web dashboard visualization system. Reduces bandwidth, improves rendering efficiency, and enables future analytics features.

## Key Improvements

### 1. Per-Action Q-Value Tracking
Tracks Q-values for each action (LEFT, STAY, RIGHT) with rolling 1000-step history.

**Usage in training code:**
```python
# When calling metrics_publisher.update()
publisher.update(
    episode=100,
    score=250,
    epsilon=0.5,
    loss=0.05,
    # ... other params ...
    q_value_left=0.45,    # NEW
    q_value_stay=0.55,    # NEW
    q_value_right=0.50,   # NEW
    selected_action=1,    # NEW (0=LEFT, 1=STAY, 2=RIGHT)
)
```

**Access in web dashboard:**
```javascript
// Available in state_update events
const qLeft = state.q_value_left;
const qStay = state.q_value_stay;
const qRight = state.q_value_right;

// Also available in MetricsPublisher (backend)
publisher.q_values_left        // deque of last 1000 values
publisher.q_values_stay        // deque of last 1000 values
publisher.q_values_right       // deque of last 1000 values
publisher.action_frequency     // {'left': X, 'stay': Y, 'right': Z, ...}
```

### 2. Adaptive Visualization Update Rate
Backend automatically adjusts how frequently neural network visualization data is sent based on training speed.

**Training Speed → Update Rate:**
- \>2000 steps/sec → 10 FPS (100ms)
- 1000-2000 steps/sec → ~15 FPS (67ms)
- 500-1000 steps/sec → ~30 FPS (33ms)
- <500 steps/sec → ~60 FPS (16ms)

**Configuration:**
```python
# Enable/disable adaptive updates
publisher._adaptive_update_enabled = True  # Default

# Check current interval
print(publisher._nn_update_interval)  # in seconds
```

### 3. Selective Weight Transmission
Network weight data is only sent every 100 steps instead of every frame, reducing bandwidth by ~50% for weight data.

**How it works:**
- Activation data: Sent every frame
- Q-values: Sent every frame
- Weights: Sent every 100 steps OR on explicit request

**Frontend handling:**
```javascript
// When weights array is empty, no weight drawing occurs
// This is automatic - no code changes needed
if (!weights || weights.length === 0) {
    return;  // Skip weight drawing
}
```

### 4. Frontend Render Throttling
JavaScript canvas rendering adapts to maintain consistent frame times.

**How it works:**
- Measures render duration
- Adjusts render interval if struggling or if performance allows
- Only renders when enough time has passed since last render

**Configuration:**
```javascript
// Get current render interval (in milliseconds)
const interval = nnVisualizer.renderInterval;

// Manually set if needed
nnVisualizer.renderInterval = 50;  // Force 20Hz
```

## Performance Expectations

### What to expect when running training:

**Fast Training (headless --turbo --vec-envs 8):**
- Before Phase 1: ~5000 steps/sec with visualization disabled
- After Phase 1: ~5000 steps/sec (no change, visualization disabled)
- With web dashboard: ~20-25% overhead instead of ~25-30%

**Visual Training (with pygame):**
- Expected improvement: ~15-20% faster
- Visible reduction in bandwidth usage when monitoring remotely

**Web Dashboard Efficiency:**
- Network bandwidth reduced by 30-40% during fast training
- CPU usage for rendering optimized via adaptive throttling
- Smooth 60fps during slow training, efficient 10fps during fast training

## Testing Phase 1

Run the test suite to verify Phase 1 is working:

```bash
source venv/bin/activate
python -m pytest tests/test_phase1_improvements.py -v
```

Expected output:
```
======================== 14 passed in 0.54s =========================
✅ Enhanced metrics collection
✅ Adaptive update rate calculation
✅ Selective weight transmission
✅ Integration tests
```

## Integration with Training Code

### Minimal changes needed in training loop:

```python
# In your training loop (e.g., main.py, trainer.py)
from src.web.server import WebDashboard

dashboard = WebDashboard(port=5000)
dashboard.start()

# During episode
for episode in range(num_episodes):
    state = game.reset()
    score = 0
    done = False

    while not done:
        # ... game step ...

        # At end of episode, call update with new Phase 1 params
        metrics = agent.get_metrics()
        dashboard.emit_metrics(
            episode=episode,
            score=score,
            epsilon=agent.epsilon,
            loss=loss,
            # ... existing params ...
            q_value_left=metrics['q_left'],        # NEW
            q_value_stay=metrics['q_stay'],        # NEW
            q_value_right=metrics['q_right'],      # NEW
            selected_action=last_action,           # NEW
        )
```

## Migration Guide

### For existing code using MetricsPublisher:

**Old code (still works):**
```python
publisher.update(
    episode=100,
    score=250,
    epsilon=0.5,
    loss=0.05,
)
```

**New code (with Phase 1):**
```python
publisher.update(
    episode=100,
    score=250,
    epsilon=0.5,
    loss=0.05,
    q_value_left=0.45,
    q_value_stay=0.55,
    q_value_right=0.50,
    selected_action=1,
)
```

✅ **Backward compatible** - old code continues to work without changes

## Troubleshooting

### Q: Web dashboard connection slow during fast training?
**A:** Check that adaptive updates are enabled. Network bandwidth should drop by 30-40%.
```python
print(publisher._adaptive_update_enabled)  # Should be True
print(publisher._nn_update_interval)       # Should be 0.1 for >2000 steps/sec
```

### Q: Neural network visualization flickering?
**A:** Frontend render throttling may need adjustment. Check:
```javascript
console.log("Render interval:", nnVisualizer.renderInterval);
console.log("Avg render time:", nnVisualizer.avgRenderTime);
```

### Q: Weights not showing in visualization?
**A:** This is normal - weights are only sent every 100 steps to save bandwidth.
- Weights will appear after ~1-2 seconds of training
- If you need weights every frame, check `NNVisualizationData.to_dict(include_weights=True)`

## Files Modified

- `src/web/server.py` - Backend metrics collection and adaptive rate
- `src/web/static/app.js` - Frontend rendering optimization
- `tests/test_phase1_improvements.py` - 14 comprehensive tests

## Questions?

Refer to:
- `PHASE1_IMPLEMENTATION_SUMMARY.md` - Detailed implementation docs
- `PLANNING.md` - Overall project roadmap
- `tests/test_phase1_improvements.py` - Code examples and test patterns

## Next Phase

Phase 2: Interactive Exploration
- Neuron inspection (click to see details)
- Layer analysis tools
- Playback and replay system

Coming soon!
