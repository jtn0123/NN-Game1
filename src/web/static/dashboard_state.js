// ============================================================
// DASHBOARD STATE
// ============================================================

(function(global) {
    const state = {
        isPaused: false,
        socket: null,
        currentLogFilter: 'all',
        consoleLogs: [],
        lastRenderedLogCount: 0,
        currentPerformanceMode: 'normal',
        trainingStartTime: 0,
        lastSpeedChangeTime: 0,
    };

    function exposeStateProperty(name) {
        Object.defineProperty(global, name, {
            configurable: true,
            get() {
                return state[name];
            },
            set(value) {
                state[name] = value;
            },
        });
    }

    Object.keys(state).forEach(exposeStateProperty);

    global.DashboardState = state;
})(typeof window !== 'undefined' ? window : globalThis);
