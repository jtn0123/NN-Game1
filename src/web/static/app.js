/**
 * Neural Network AI - Training Dashboard
 * Real-time visualization with Chart.js and SocketIO
 *
 * Features:
 * - Live training metrics charts
 * - Console log with filtering
 * - Full training controls
 * - Model management
 */

// State
let isPaused = false;
let socket = null;
let currentLogFilter = 'all';
let consoleLogs = [];
const MAX_CONSOLE_LOGS = 500;
let lastRenderedLogCount = 0;  // Track for incremental updates
let currentPerformanceMode = 'normal';
let trainingStartTime = 0;
const DASHBOARD_TOKEN = (
    typeof DashboardCore !== 'undefined' && typeof document !== 'undefined'
) ? DashboardCore.readToken(document) : '';

// Speed slider state - prevent server updates from fighting with user input
let lastSpeedChangeTime = 0;
const SPEED_UPDATE_DEBOUNCE = 2000; // Ignore server speed updates for 2s after user change

// Fetch timeout configuration
const FETCH_TIMEOUT_MS = 10000; // 10 second timeout for API calls

const PERFORMANCE_MODES = Object.freeze({
    normal: { label: 'Normal', learnEvery: 1, batchSize: 128, gradientSteps: 1, log: 'Normal (learn every step)' },
    fast: { label: 'Fast', learnEvery: 4, batchSize: 128, gradientSteps: 1, log: 'Fast (learn every 4 steps)' },
    turbo: { label: 'Turbo', learnEvery: 8, batchSize: 128, gradientSteps: 2, log: 'Turbo (learn every 8, batch 128, 2 grad steps)' },
    ultra: { label: 'Ultra', learnEvery: 32, batchSize: 128, gradientSteps: 2, log: 'Ultra (learn every 32, batch 128, 2 grad steps)' }
});

const DASHBOARD_ACTIONS = Object.freeze({
    'switch-game': (target, event) => event.type === 'change' && switchGame(target.value),
    'go-to-launcher': () => goToLauncher(),
    'reset-chart': (target) => resetChartView(target.dataset.chart || 'all'),
    'toggle-nn-panel': () => toggleNNPanel(),
    'toggle-nn-visualization': () => toggleNNVisualization(),
    'set-log-filter': (target) => setLogFilter(target.dataset.filter || 'all'),
    'copy-logs': () => copyLogsToClipboard(),
    'clear-logs': () => clearLogs(),
    'refresh-screenshot': () => fetchScreenshot(),
    'set-performance-mode': (target) => setPerformanceMode(target.dataset.mode),
    'toggle-pause': () => togglePause(),
    'save-model': () => saveModel(),
    'reset-episode': () => resetEpisode(),
    'show-load-modal': () => showLoadModal(),
    'start-fresh': () => startFresh(),
    'save-and-quit': () => saveAndQuit(),
    'update-speed': (target, event) => event.type !== 'click' && updateSpeed(target.value),
    'save-model-as': () => saveModelAs(),
    'toggle-settings': () => toggleSettings(),
    'update-learn-every-label': (target, event) => event.type !== 'click' && updateLearnEveryLabel(target.value),
    'apply-settings': () => applySettings(),
    'toggle-comparison': () => toggleComparison(),
    'refresh-game-stats': () => loadGameStats(),
    'hide-load-modal': () => hideLoadModal(),
    'load-model': (target) => loadModel(target.dataset.modelId),
    'delete-model': (target, event) => {
        event.stopPropagation();
        deleteModel(target.dataset.modelId, target.dataset.modelName);
    },
    'copy-restart-command': (_target, event) => copyRestartCommand(event),
    'close-restart-banner': () => closeRestartBanner(),
    'close-neuron-inspection': () => closeNeuronInspection(),
    'close-layer-analysis': () => closeLayerAnalysis()
});

/**
 * Fetch with timeout wrapper
 */
function fetchWithTimeout(url, options = {}, timeout = FETCH_TIMEOUT_MS) {
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), timeout);
    const authorizedOptions = DashboardCore.withDashboardToken(options, DASHBOARD_TOKEN);

    return fetch(url, { ...authorizedOptions, signal: controller.signal })
        .finally(() => clearTimeout(timeoutId));
}

function emitDashboardControl(payload, failureContext = 'Command failed') {
    return DashboardCore.emitControl(socket, payload).then((response) => {
        if (!response || !response.success) {
            addConsoleLog(
                `❌ ${failureContext}: ${DashboardCore.controlErrorMessage(response)}`,
                'error'
            );
        }
        return response;
    });
}

function setPendingButton(button, pendingText, className = '') {
    if (!button) {
        return () => {};
    }

    const originalText = button.textContent;
    button.textContent = pendingText;
    button.disabled = true;
    if (className) {
        button.classList.add(className);
    }

    return () => {
        button.textContent = originalText;
        button.disabled = false;
        if (className) {
            button.classList.remove(className);
        }
    };
}

function showTemporaryButtonText(button, text, restore, delay = 1500) {
    if (!button) {
        restore();
        return;
    }

    button.textContent = text;
    setTimeout(restore, delay);
}

/**
 * Throttle function - limits how often a function can be called
 */
function throttle(func, limit) {
    let inThrottle;
    let lastResult;
    return function(...args) {
        if (!inThrottle) {
            lastResult = func.apply(this, args);
            inThrottle = true;
            setTimeout(() => inThrottle = false, limit);
        }
        return lastResult;
    };
}

function initDashboard() {
    registerDashboardActions();
    if (typeof DashboardCore === 'undefined') {
        console.error('Dashboard core failed to load; dashboard startup stopped.');
        addConsoleLog('Dashboard core failed to load; refresh the page', 'error');
        updateConnectionStatus(false);
        return;
    }

    initCharts();
    initNNVisualizer();
    connectSocket();
    startScreenshotPolling();
    updateFooterTime();
    setInterval(updateFooterTime, 1000);
    loadConfig();  // This will call initVecEnvsHandler() after config is loaded
    loadGames();
    loadGameStats();
    fetchInitialData();
}

// Initialize on DOM load
if (typeof document !== 'undefined') {
    document.addEventListener('DOMContentLoaded', initDashboard);
}

function initCharts() {
    if (typeof DashboardCharts === 'undefined') {
        console.error('Dashboard charts failed to load; charts are disabled.');
        addConsoleLog('Charts unavailable: dashboard chart module failed to load', 'error');
        return false;
    }
    return DashboardCharts.initCharts({ addConsoleLog });
}

function updateCharts(history, currentEpisode) {
    if (typeof DashboardCharts === 'undefined') {
        return;
    }
    DashboardCharts.updateCharts(history, currentEpisode);
}

function resetChartView(chartName = 'all') {
    if (typeof DashboardCharts !== 'undefined') {
        DashboardCharts.resetChartView(chartName);
    }
}

function clearCharts() {
    if (typeof DashboardCharts !== 'undefined') {
        DashboardCharts.clearCharts();
    }
}

/**
 * Connect to SocketIO server
 */
function connectSocket() {
    if (typeof io !== 'function') {
        console.error('Socket.IO failed to load; live updates are disabled.');
        updateConnectionStatus(false);
        addConsoleLog('Live connection unavailable: Socket.IO failed to load', 'error');
        return false;
    }

    socket = DashboardCore.createAuthorizedSocket(io, DASHBOARD_TOKEN);

    // Throttle dashboard updates to 60fps max (prevent excessive DOM manipulation)
    const throttledUpdateDashboard = throttle(updateDashboard, 16);  // ~60fps

    socket.on('connect', () => {
        console.log('Connected to server');
        updateConnectionStatus(true);
        addConsoleLog('Connected to training server', 'success');
    });

    socket.on('disconnect', () => {
        console.log('Disconnected from server');
        updateConnectionStatus(false);
        addConsoleLog('Disconnected from server', 'error');
    });

    socket.on('state_update', (data) => {
        try {
            throttledUpdateDashboard(data);
        } catch (err) {
            console.error('Error processing state update:', err);
        }
    });

    socket.on('console_log', (log) => {
        try {
            addConsoleLog(log.message, log.level, log.timestamp, log.data);
        } catch (err) {
            console.error('Error processing console log:', err);
        }
    });

    socket.on('training_reset', (data) => {
        try {
            clearCharts();

            // Clear console logs
            consoleLogs = [];
            renderConsoleLogs();

            // Reset metrics display
            document.getElementById('metric-episode').textContent = '0';
            document.getElementById('metric-score').textContent = '0';
            document.getElementById('metric-best').textContent = '0';
            document.getElementById('metric-winrate').textContent = '0%';

            // Reset epsilon gauge
            document.getElementById('epsilon-value').textContent = '1.000';
            document.getElementById('epsilon-fill').style.width = '100%';

            // Reset extended info
            document.getElementById('info-loss').textContent = '0.0000';
            document.getElementById('info-steps').textContent = '0';
            document.getElementById('info-eps').textContent = '0.00';
            document.getElementById('info-qvalue').textContent = '0.00';
            document.getElementById('info-target').textContent = '0';
            document.getElementById('info-actions').textContent = '0 / 0';
            // Memory will be updated from state when connected
            document.getElementById('info-memory').textContent = '0 / 0';
            document.getElementById('info-steps-sec').textContent = '0';

            // Reset memory bar
            const memoryBar = document.getElementById('memory-bar-fill');
            if (memoryBar) {
                memoryBar.style.width = '0%';
                memoryBar.style.background = 'var(--accent-warning)';
            }

            console.log('Training reset - charts and UI cleared');
        } catch (err) {
            console.error('Error handling training reset:', err);
        }
    });

    socket.on('console_logs', (data) => {
        try {
            // If empty array, clear logs
            if (data && data.logs && data.logs.length === 0) {
                consoleLogs = [];
                renderConsoleLogs();
            } else if (data && data.logs) {
                // Initial batch of logs on connect
                data.logs.forEach(log => {
                    addConsoleLog(log.message, log.level, log.timestamp, log.data, false);
                });
                renderConsoleLogs();
            }
        } catch (err) {
            console.error('Error processing console logs:', err);
        }
    });

    socket.on('save_event', (data) => {
        try {
            updateSaveStatus(data);
            flashSaveIndicator();
        } catch (err) {
            console.error('Error processing save event:', err);
        }
    });

    socket.on('restarting', (data) => {
        try {
            addConsoleLog(`🔄 ${data.message}`, 'warning');
            // Show restarting overlay
            showRestartingOverlay(data.game);
        } catch (err) {
            console.error('Error processing restart:', err);
        }
    });

    socket.on('redirect_to_launcher', (data) => {
        try {
            addConsoleLog(`🎮 ${data.message}`, 'warning');
            // Short delay then redirect to launcher
            setTimeout(() => {
                window.location.href = '/';
            }, 500);
        } catch (err) {
            console.error('Error processing redirect:', err);
        }
    });

    socket.on('nn_update', (data) => {
        try {
            if (nnVisualizer) {
                nnVisualizer.update(data);
            }
        } catch (err) {
            console.error('Error processing NN update:', err);
        }
    });

    return true;
}

/**
 * Update connection status indicator
 */
function updateConnectionStatus(connected) {
    const dot = document.querySelector('.status-dot');
    const text = document.querySelector('.status-text');
    if (!dot || !text) return;

    if (connected) {
        dot.classList.add('connected');
        dot.classList.remove('disconnected');
        text.textContent = 'Connected';
    } else {
        dot.classList.add('disconnected');
        dot.classList.remove('connected');
        text.textContent = 'Disconnected';
    }
}

/**
 * Update dashboard with new data
 */
function updateDashboard(data) {
    const state = data.state;
    const history = data.history;

    // Update metrics
    document.getElementById('metric-episode').textContent = state.episode.toLocaleString();
    document.getElementById('metric-score').textContent = state.score;
    document.getElementById('metric-best').textContent = state.best_score;
    document.getElementById('metric-winrate').textContent = (state.win_rate * 100).toFixed(1) + '%';

    // Update epsilon gauge
    document.getElementById('epsilon-value').textContent = state.epsilon.toFixed(3);
    document.getElementById('epsilon-fill').style.width = (state.epsilon * 100) + '%';

    // Update extended info
    document.getElementById('info-loss').textContent = state.loss.toFixed(4);
    document.getElementById('info-steps').textContent = state.total_steps.toLocaleString();
    document.getElementById('info-eps').textContent = state.episodes_per_second.toFixed(2);
    // Update memory with visual indicator
    const memoryPct = state.memory_capacity > 0 ? (state.memory_size / state.memory_capacity * 100) : 0;
    const memoryEl = document.getElementById('info-memory');
    const memoryText = `${(state.memory_size / 1000).toFixed(0)}k / ${(state.memory_capacity / 1000).toFixed(0)}k`;
    memoryEl.textContent = memoryText;

    // Update memory progress bar if it exists
    const memoryBar = document.getElementById('memory-bar-fill');
    if (memoryBar) {
        memoryBar.style.width = `${memoryPct}%`;
        // Change color based on fill level
        if (memoryPct >= 100) {
            memoryBar.style.background = 'var(--accent-success)';
        } else if (memoryPct >= 50) {
            memoryBar.style.background = 'var(--accent-primary)';
        } else {
            memoryBar.style.background = 'var(--accent-warning)';
        }
    }
    document.getElementById('info-qvalue').textContent = state.avg_q_value.toFixed(2);
    document.getElementById('info-target').textContent = state.target_updates.toLocaleString();
    document.getElementById('info-actions').textContent =
        `${state.exploration_actions.toLocaleString()} / ${state.exploitation_actions.toLocaleString()}`;

    // Update vec-envs display
    const vecEnvsEl = document.getElementById('info-vec-envs');
    if (vecEnvsEl && state.num_envs) {
        vecEnvsEl.textContent = state.num_envs;
    }

    // Update steps/sec (new performance metric)
    const stepsPerSec = state.steps_per_second || 0;
    document.getElementById('info-steps-sec').textContent = stepsPerSec.toLocaleString(undefined, {maximumFractionDigits: 0});

    // Phase 1.2: Update neural network visualizer render rate based on training speed
    if (nnVisualizer) {
        updateNNVisualizerRenderRate(stepsPerSec);
    }

    // Update ETA
    updateETA(state);

    // Update system status badges
    updateSystemStatus(state);

    // Update performance mode buttons and sync settings
    if (state.performance_mode) {
        updatePerformanceModeUI(state.performance_mode);
        // Sync settings inputs to match current mode
        syncSettingsFromMode(state.performance_mode);
    }

    // Update learn_every in settings if changed from server
    if (state.learn_every) {
        document.getElementById('setting-learn-every').value = state.learn_every;
        updateLearnEveryLabel(state.learn_every);
    }
    if (state.gradient_steps) {
        document.getElementById('setting-grad-steps').value = state.gradient_steps;
    }

    // Update status badge - check for actual training activity, not just is_running flag
    const statusBadge = document.getElementById('info-status');
    const isActivelyTraining = state.total_steps > 0 || state.episode > 0 || state.steps_per_second > 0;

    if (state.is_paused) {
        statusBadge.textContent = 'Paused';
        statusBadge.className = 'info-value status-badge paused';
    } else if (isActivelyTraining) {
        statusBadge.textContent = 'Training';
        statusBadge.className = 'info-value status-badge training';
    } else if (state.is_running) {
        statusBadge.textContent = 'Starting...';
        statusBadge.className = 'info-value status-badge starting';
    } else {
        statusBadge.textContent = 'Idle';
        statusBadge.className = 'info-value status-badge idle';
    }

    // Update pause button
    const pauseBtn = document.getElementById('pause-btn');
    isPaused = state.is_paused;
    pauseBtn.textContent = isPaused ? '▶️ Resume' : '⏸️ Pause';

    // Update speed slider if changed externally (but not if user recently changed it)
    const speedSlider = document.getElementById('speed-slider');
    const timeSinceLastChange = Date.now() - lastSpeedChangeTime;
    if (timeSinceLastChange > SPEED_UPDATE_DEBOUNCE) {
        // Only sync from server if user hasn't touched it recently
        const sliderValue = parseInt(speedSlider.value, 10);
        const serverValue = Math.round(state.game_speed);
        // Use tolerance for comparison
        if (Math.abs(sliderValue - serverValue) > 2) {
            speedSlider.value = serverValue;
            document.getElementById('speed-value').textContent = serverValue + 'x';
        }
    }

    // Update charts - pass current episode for accurate labels
    updateCharts(history, state.episode);
}
