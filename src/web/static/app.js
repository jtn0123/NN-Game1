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

// Charts
let scoreChart = null;
let lossChart = null;
let qvalueChart = null;

// State
let isPaused = false;
let socket = null;
let currentLogFilter = 'all';
let consoleLogs = [];
const MAX_CONSOLE_LOGS = 500;
let currentPerformanceMode = 'normal';
let trainingStartTime = 0;
let targetEpisodes = 2000;

// Speed slider state - prevent server updates from fighting with user input
let lastSpeedChangeTime = 0;
const SPEED_UPDATE_DEBOUNCE = 2000; // Ignore server speed updates for 2s after user change

// Fetch timeout configuration
const FETCH_TIMEOUT_MS = 10000; // 10 second timeout for API calls

/**
 * Fetch with timeout wrapper
 */
function fetchWithTimeout(url, options = {}, timeout = FETCH_TIMEOUT_MS) {
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), timeout);
    
    return fetch(url, { ...options, signal: controller.signal })
        .finally(() => clearTimeout(timeoutId));
}

// Initialize on DOM load
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
});

/**
 * Initialize Chart.js charts
 */
function initCharts() {
    const chartOptions = {
        responsive: true,
        maintainAspectRatio: false,
        animation: {
            duration: 300
        },
        interaction: {
            mode: 'index',
            intersect: false
        },
        plugins: {
            legend: {
                display: false
            },
            tooltip: {
                enabled: true,
                backgroundColor: 'rgba(18, 20, 28, 0.95)',
                titleColor: '#e4e6f0',
                bodyColor: '#e4e6f0',
                borderColor: '#64b5f6',
                borderWidth: 1,
                padding: 12,
                cornerRadius: 8,
                titleFont: {
                    family: "'Plus Jakarta Sans', sans-serif",
                    size: 13,
                    weight: '600'
                },
                bodyFont: {
                    family: "'JetBrains Mono', monospace",
                    size: 12
                },
                displayColors: true,
                boxPadding: 4,
                callbacks: {
                    title: function(context) {
                        return 'Episode ' + context[0].label;
                    }
                }
            }
        },
        scales: {
            x: {
                grid: {
                    color: 'rgba(255, 255, 255, 0.05)'
                },
                ticks: {
                    color: '#5a5e72',
                    maxTicksLimit: 8
                }
            },
            y: {
                grid: {
                    color: 'rgba(255, 255, 255, 0.05)'
                },
                ticks: {
                    color: '#5a5e72'
                },
                beginAtZero: true
            }
        }
    };

    // Score Chart
    const scoreCtx = document.getElementById('scoreChart').getContext('2d');
    scoreChart = new Chart(scoreCtx, {
        type: 'line',
        data: {
            labels: [],
            datasets: [
                {
                    label: 'Score',
                    data: [],
                    borderColor: '#4caf50',
                    backgroundColor: 'rgba(76, 175, 80, 0.1)',
                    fill: true,
                    tension: 0.4,
                    pointRadius: 0,
                    pointHoverRadius: 6,
                    pointHoverBackgroundColor: '#4caf50',
                    pointHoverBorderColor: '#fff',
                    pointHoverBorderWidth: 2,
                    borderWidth: 2
                },
                {
                    label: 'Avg (20ep)',
                    data: [],
                    borderColor: '#ffc107',
                    backgroundColor: 'transparent',
                    tension: 0.4,
                    pointRadius: 0,
                    pointHoverRadius: 6,
                    pointHoverBackgroundColor: '#ffc107',
                    pointHoverBorderColor: '#fff',
                    pointHoverBorderWidth: 2,
                    borderWidth: 3,
                    borderDash: [5, 5]
                }
            ]
        },
        options: chartOptions
    });

    // Loss Chart
    const lossCtx = document.getElementById('lossChart').getContext('2d');
    lossChart = new Chart(lossCtx, {
        type: 'line',
        data: {
            labels: [],
            datasets: [{
                label: 'Loss',
                data: [],
                borderColor: '#ef5350',
                backgroundColor: 'rgba(239, 83, 80, 0.1)',
                fill: true,
                tension: 0.4,
                pointRadius: 0,
                pointHoverRadius: 5,
                pointHoverBackgroundColor: '#ef5350',
                pointHoverBorderColor: '#fff',
                pointHoverBorderWidth: 2,
                borderWidth: 2
            }]
        },
        options: {
            ...chartOptions,
            plugins: {
                ...chartOptions.plugins,
                tooltip: {
                    ...chartOptions.plugins.tooltip,
                    callbacks: {
                        title: function(context) {
                            return 'Episode ' + context[0].label;
                        },
                        label: function(context) {
                            return 'Loss: ' + context.parsed.y.toFixed(6);
                        }
                    }
                }
            },
            scales: {
                ...chartOptions.scales,
                y: {
                    ...chartOptions.scales.y,
                    type: 'logarithmic',
                    min: 0.0001
                }
            }
        }
    });

    // Q-Value Chart
    const qvalueCtx = document.getElementById('qvalueChart').getContext('2d');
    qvalueChart = new Chart(qvalueCtx, {
        type: 'line',
        data: {
            labels: [],
            datasets: [{
                label: 'Avg Q-Value',
                data: [],
                borderColor: '#64b5f6',
                backgroundColor: 'rgba(100, 181, 246, 0.1)',
                fill: true,
                tension: 0.4,
                pointRadius: 0,
                pointHoverRadius: 5,
                pointHoverBackgroundColor: '#64b5f6',
                pointHoverBorderColor: '#fff',
                pointHoverBorderWidth: 2,
                borderWidth: 2
            }]
        },
        options: {
            ...chartOptions,
            plugins: {
                ...chartOptions.plugins,
                tooltip: {
                    ...chartOptions.plugins.tooltip,
                    callbacks: {
                        title: function(context) {
                            return 'Episode ' + context[0].label;
                        },
                        label: function(context) {
                            return 'Q-Value: ' + context.parsed.y.toFixed(4);
                        }
                    }
                }
            }
        }
    });
}

/**
 * Connect to SocketIO server
 */
function connectSocket() {
    socket = io();

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
            updateDashboard(data);
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

    socket.on('console_logs', (data) => {
        try {
            // Initial batch of logs on connect
            if (data && data.logs) {
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
    
    socket.on('game_switched', (data) => {
        try {
            addConsoleLog(data.message, 'success');
            // Reload games to update UI
            loadGames();
            loadGameStats();
        } catch (err) {
            console.error('Error processing game switch:', err);
        }
    });
    
    socket.on('stop_for_switch', (data) => {
        try {
            addConsoleLog(`✅ Progress saved. Restart with: ${data.command}`, 'success');
            // Show prominent message
            showRestartBanner(data.game, data.command);
        } catch (err) {
            console.error('Error processing stop for switch:', err);
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
    
    socket.on('nn_update', (data) => {
        try {
            if (nnVisualizer) {
                nnVisualizer.update(data);
            }
        } catch (err) {
            console.error('Error processing NN update:', err);
        }
    });
}

/**
 * Update connection status indicator
 */
function updateConnectionStatus(connected) {
    const dot = document.querySelector('.status-dot');
    const text = document.querySelector('.status-text');
    
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
    
    // Update steps/sec (new performance metric)
    const stepsPerSec = state.steps_per_second || 0;
    document.getElementById('info-steps-sec').textContent = stepsPerSec.toLocaleString(undefined, {maximumFractionDigits: 0});
    
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

/**
 * Update charts with history data
 * @param {Object} history - History data with scores, losses, q_values arrays
 * @param {number} currentEpisode - The actual current episode number
 */
function updateCharts(history, currentEpisode) {
    const maxPoints = 200;
    const scores = history.scores.slice(-maxPoints);
    const losses = history.losses.slice(-maxPoints);
    const qvalues = history.q_values ? history.q_values.slice(-maxPoints) : [];
    
    // Calculate running average
    const avgScores = calculateRunningAverage(scores, 20);
    
    // Generate labels based on actual current episode, not history length
    // This correctly handles the case where history.scores.length is capped at 500
    // but the actual episode number continues beyond that
    const startEp = Math.max(0, currentEpisode - scores.length);
    const labels = scores.map((_, i) => startEp + i + 1);

    // Update score chart
    scoreChart.data.labels = labels;
    scoreChart.data.datasets[0].data = scores;
    scoreChart.data.datasets[1].data = avgScores;
    scoreChart.update('none');

    // Update loss chart
    const validLosses = losses.map(l => Math.max(l, 0.0001));
    lossChart.data.labels = labels;
    lossChart.data.datasets[0].data = validLosses;
    lossChart.update('none');

    // Update Q-value chart
    if (qvalues.length > 0) {
        qvalueChart.data.labels = labels;
        qvalueChart.data.datasets[0].data = qvalues;
        qvalueChart.update('none');
    }
}

/**
 * Calculate running average
 */
function calculateRunningAverage(data, window) {
    const result = [];
    for (let i = 0; i < data.length; i++) {
        const start = Math.max(0, i - window + 1);
        const slice = data.slice(start, i + 1);
        const avg = slice.reduce((a, b) => a + b, 0) / slice.length;
        result.push(avg);
    }
    return result;
}

// ============================================================
// CONSOLE LOG FUNCTIONS
// ============================================================

/**
 * Add a log entry to the console
 */
function addConsoleLog(message, level = 'info', timestamp = null, data = null, render = true) {
    const time = timestamp || new Date().toLocaleTimeString('en-US', { hour12: false });
    
    consoleLogs.push({
        time: time,
        level: level,
        message: message,
        data: data
    });

    // Keep console size limited
    if (consoleLogs.length > MAX_CONSOLE_LOGS) {
        consoleLogs = consoleLogs.slice(-MAX_CONSOLE_LOGS);
    }

    if (render) {
        renderConsoleLogs();
    }
}

/**
 * Render console logs based on current filter
 */
function renderConsoleLogs() {
    const container = document.getElementById('console-output');
    
    let filteredLogs = consoleLogs;
    if (currentLogFilter !== 'all') {
        filteredLogs = consoleLogs.filter(log => log.level === currentLogFilter);
    }

    // Keep only last 100 visible logs for performance
    const visibleLogs = filteredLogs.slice(-100);

    container.innerHTML = visibleLogs.map(log => {
        let dataStr = '';
        if (log.data) {
            dataStr = `<span class="log-data">${JSON.stringify(log.data)}</span>`;
        }
        return `
            <div class="console-line ${log.level}">
                <span class="log-time">${log.time}</span>
                <span class="log-level">${log.level.toUpperCase()}</span>
                <span class="log-message">${escapeHtml(log.message)}</span>
                ${dataStr}
            </div>
        `;
    }).join('');

    // Auto-scroll to bottom
    const consoleContainer = document.getElementById('console-container');
    consoleContainer.scrollTop = consoleContainer.scrollHeight;
}

/**
 * Set log filter
 */
function setLogFilter(filter) {
    currentLogFilter = filter;
    
    // Update button states
    document.querySelectorAll('.filter-btn').forEach(btn => {
        btn.classList.toggle('active', btn.dataset.filter === filter);
    });

    renderConsoleLogs();
}

/**
 * Clear console logs
 */
function clearLogs() {
    consoleLogs = [];
    socket.emit('clear_logs');
    renderConsoleLogs();
    addConsoleLog('Console cleared', 'info');
}

/**
 * Copy all console logs to clipboard
 */
function copyLogsToClipboard() {
    let text = '';

    // Get filtered logs
    let logsToExport = consoleLogs;
    if (currentLogFilter !== 'all') {
        logsToExport = consoleLogs.filter(log => log.level === currentLogFilter);
    }

    // Format logs as text
    text = logsToExport.map(log => {
        let line = `[${log.time}] ${log.level.toUpperCase().padEnd(8)} ${log.message}`;
        if (log.data) {
            line += ` ${JSON.stringify(log.data)}`;
        }
        return line;
    }).join('\n');

    // Copy to clipboard
    navigator.clipboard.writeText(text).then(() => {
        // Visual feedback
        const btn = document.querySelector('.copy-btn');
        const originalText = btn.textContent;
        btn.textContent = '✓ Copied!';
        btn.classList.add('copied');
        setTimeout(() => {
            btn.textContent = originalText;
            btn.classList.remove('copied');
        }, 1500);
    }).catch(err => {
        console.error('Failed to copy logs:', err);
    });
}

/**
 * Escape HTML entities
 */
function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

// ============================================================
// CONTROL FUNCTIONS
// ============================================================

/**
 * Toggle pause state
 */
function togglePause() {
    socket.emit('control', { action: 'pause' });
}

/**
 * Save model
 */
function saveModel() {
    socket.emit('control', { action: 'save' });
    
    // Visual feedback
    const btn = document.querySelector('.control-btn.save');
    const originalText = btn.textContent;
    btn.textContent = '✓ Saving...';
    btn.classList.add('saving');
    setTimeout(() => {
        btn.textContent = originalText;
        btn.classList.remove('saving');
    }, 1500);
}

/**
 * Reset current episode
 */
function resetEpisode() {
    socket.emit('control', { action: 'reset' });
    addConsoleLog('Episode reset requested', 'action');
}

/**
 * Update game speed with snapping to preset values
 */
function updateSpeed(value) {
    let speed = Math.round(parseFloat(value));
    
    // Mark that user is actively changing speed (prevents server from overwriting)
    lastSpeedChangeTime = Date.now();
    
    // Snap to preset values: 1, 5, 10, 25, 50, 100, 250, 500, 1000
    const snapValues = [1, 5, 10, 25, 50, 100, 250, 500, 1000];
    
    // Find closest snap value with proportional threshold
    for (const snap of snapValues) {
        const snapThreshold = Math.max(2, snap * 0.15); // 15% threshold, min 2
        if (Math.abs(speed - snap) <= snapThreshold) {
            speed = snap;
            document.getElementById('speed-slider').value = snap;
            break;
        }
    }
    
    // Ensure minimum of 1
    speed = Math.max(1, speed);
    
    // Display as integer
    document.getElementById('speed-value').textContent = speed + 'x';
    socket.emit('control', { action: 'speed', value: speed });
}

/**
 * Refresh screenshot
 */
function refreshScreenshot() {
    fetchWithTimeout('/api/screenshot')
        .then(response => response.json())
        .then(data => {
            if (data.image && data.image.length > 0) {
                const img = document.getElementById('game-preview');
                const placeholder = document.getElementById('preview-placeholder');

                // Create a new image to verify it loads properly
                const tempImg = new Image();
                tempImg.onload = () => {
                    // Image loaded successfully
                    img.src = 'data:image/png;base64,' + data.image;
                    img.classList.add('visible');
                    placeholder.classList.add('hidden');
                };
                tempImg.onerror = () => {
                    console.error('Failed to load screenshot image');
                    placeholder.classList.remove('hidden');
                    img.classList.remove('visible');
                };
                tempImg.src = 'data:image/png;base64,' + data.image;
            } else {
                // No image data
                const placeholder = document.getElementById('preview-placeholder');
                const img = document.getElementById('game-preview');
                placeholder.classList.remove('hidden');
                img.classList.remove('visible');
            }
        })
        .catch(err => {
            if (err.name !== 'AbortError') {
                console.error('Screenshot fetch error:', err);
            }
            const placeholder = document.getElementById('preview-placeholder');
            if (placeholder) placeholder.classList.remove('hidden');
        });
}

/**
 * Start polling for screenshots
 */
function startScreenshotPolling() {
    // Fetch immediately on page load, then poll every 2 seconds
    refreshScreenshot();
    setInterval(refreshScreenshot, 2000);
}

// ============================================================
// SETTINGS FUNCTIONS
// ============================================================

/**
 * Toggle settings panel
 */
function toggleSettings() {
    const card = document.getElementById('settings-card');
    const icon = document.getElementById('settings-icon');
    
    card.classList.toggle('collapsed');
    icon.textContent = card.classList.contains('collapsed') ? '▼' : '▲';
}

// loadConfig and applySettings moved to end of file with enhancements

// ============================================================
// MODEL LOADING
// ============================================================

/**
 * Show load model modal with enhanced metadata
 */
function showLoadModal() {
    const modal = document.getElementById('load-modal');
    modal.classList.add('visible');
    
    // Fetch available models
    fetchWithTimeout('/api/models')
        .then(response => response.json())
        .then(data => {
            const list = document.getElementById('model-list');
            
            if (data.models.length === 0) {
                list.innerHTML = '<div class="no-models">No saved models found</div>';
                return;
            }

            list.innerHTML = data.models.map(model => {
                const size = (model.size / (1024 * 1024)).toFixed(2) + ' MB';
                const meta = model.metadata || {};
                const hasMeta = model.has_metadata;
                
                // Get values from metadata or fallback
                const episode = hasMeta ? (meta.episode || '?') : '?';
                const bestScore = hasMeta ? (meta.best_score || '?') : '?';
                const avgScore = hasMeta ? (meta.avg_score_last_100?.toFixed(1) || '?') : '?';
                const epsilon = model.epsilon ? model.epsilon.toFixed(3) : '?';
                const reason = hasMeta ? (meta.save_reason || '') : '';
                
                // Format episode and best score
                const episodeStr = typeof episode === 'number' ? episode.toLocaleString() : episode;
                
                // Escape model name and path to prevent XSS
                const safeName = escapeHtml(model.name);
                // For onclick, escape backslashes and single quotes for JS string context
                const safePathForJs = model.path.replace(/\\/g, '\\\\').replace(/'/g, "\\'");
                const safeModifiedStr = escapeHtml(model.modified_str || '');
                
                // Reason badge (reason is from our own metadata, but escape anyway)
                const safeReason = escapeHtml(reason);
                const reasonBadge = reason ? `<span class="reason-badge ${safeReason}">${safeReason}</span>` : '';
                
                return `
                    <div class="model-item" onclick="loadModel('${safePathForJs}')">
                        <div class="model-header">
                            <div class="model-name">
                                📁 ${safeName}
                                ${reasonBadge}
                            </div>
                            <span class="model-size">${size}</span>
                        </div>
                        <div class="model-stats">
                            <div class="model-stat">
                                <span class="model-stat-label">Episode</span>
                                <span class="model-stat-value">${episodeStr}</span>
                            </div>
                            <div class="model-stat">
                                <span class="model-stat-label">Best</span>
                                <span class="model-stat-value">${bestScore}</span>
                            </div>
                            <div class="model-stat">
                                <span class="model-stat-label">Avg(100)</span>
                                <span class="model-stat-value">${avgScore}</span>
                            </div>
                            <div class="model-stat">
                                <span class="model-stat-label">Epsilon</span>
                                <span class="model-stat-value">${epsilon}</span>
                            </div>
                        </div>
                        <div class="model-date">${safeModifiedStr}</div>
                    </div>
                `;
            }).join('');
        })
        .catch(err => {
            document.getElementById('model-list').innerHTML = 
                '<div class="error">Failed to load models</div>';
        });
}

/**
 * Hide load model modal
 */
function hideLoadModal() {
    const modal = document.getElementById('load-modal');
    modal.classList.remove('visible');
}

/**
 * Load a specific model
 */
function loadModel(path) {
    socket.emit('control', { action: 'load_model', path: path });
    hideLoadModal();
    addConsoleLog(`Loading model: ${path.split('/').pop()}`, 'action');
}

// Close modal on outside click
document.addEventListener('click', (e) => {
    const modal = document.getElementById('load-modal');
    if (e.target === modal) {
        hideLoadModal();
    }
});

// ============================================================
// UTILITY FUNCTIONS
// ============================================================

/**
 * Update footer time
 */
function updateFooterTime() {
    const now = new Date();
    document.getElementById('footer-time').textContent = now.toLocaleTimeString();
}

/**
 * Initial data fetch
 */
function fetchInitialData() {
    fetchWithTimeout('/api/status')
        .then(response => response.json())
        .then(data => {
            if (data && data.state && data.history) {
                updateDashboard(data);
            }
        })
        .catch(err => {
            if (err.name !== 'AbortError') {
                console.error('Initial fetch error:', err);
            }
        });
}

// Fetch initial data immediately (socket will also send state on connect)
fetchInitialData();

// Keyboard shortcuts
document.addEventListener('keydown', (e) => {
    // Don't trigger if typing in input
    if (e.target.tagName === 'INPUT') return;
    
    switch(e.key.toLowerCase()) {
        case 'p':
            togglePause();
            break;
        case 's':
            if (e.ctrlKey || e.metaKey) {
                e.preventDefault();
                saveModel();
            }
            break;
        case 'r':
            // Require Ctrl/Cmd+R to prevent accidental resets
            if (e.ctrlKey || e.metaKey) {
                e.preventDefault(); // Prevent browser refresh
                resetEpisode();
            }
            break;
        case '1':
            setPerformanceMode('normal');
            break;
        case '2':
            setPerformanceMode('fast');
            break;
        case '3':
            setPerformanceMode('turbo');
            break;
    }
});

// ============================================================
// PERFORMANCE MODE FUNCTIONS
// ============================================================

/**
 * Set performance mode preset
 */
function setPerformanceMode(mode) {
    currentPerformanceMode = mode;
    socket.emit('control', { action: 'performance_mode', mode: mode });
    updatePerformanceModeUI(mode);
    
    // Update settings inputs to match the mode
    syncSettingsFromMode(mode);
    
    // Log the change
    const modeNames = {
        'normal': 'Normal (learn every step)',
        'fast': 'Fast (learn every 4 steps)',
        'turbo': 'Turbo (learn every 8, batch 128, 2 grad steps)'
    };
    addConsoleLog(`Performance mode: ${modeNames[mode]}`, 'action');
}

/**
 * Sync settings inputs when performance mode changes
 */
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
        // Match backend turbo preset - optimized for M4 CPU based on benchmarks
        learnEvery = 8;
        batchSize = 128;
        gradientSteps = 2;
    }
    
    // Update the settings inputs
    const learnEveryInput = document.getElementById('setting-learn-every');
    const batchInput = document.getElementById('setting-batch');
    const gradStepsInput = document.getElementById('setting-grad-steps');
    
    if (learnEveryInput) {
        learnEveryInput.value = learnEvery;
        updateLearnEveryLabel(learnEvery);
    }
    if (batchInput) {
        batchInput.value = batchSize;
    }
    if (gradStepsInput) {
        gradStepsInput.value = gradientSteps;
    }
}

/**
 * Update performance mode button states
 */
function updatePerformanceModeUI(mode) {
    document.querySelectorAll('.mode-btn').forEach(btn => {
        btn.classList.remove('active');
    });
    const activeBtn = document.getElementById(`mode-${mode}`);
    if (activeBtn) {
        activeBtn.classList.add('active');
    }
}

// ============================================================
// SYSTEM STATUS FUNCTIONS
// ============================================================

/**
 * Update system status badges (device, compile)
 */
function updateSystemStatus(state) {
    // Update device badge
    const deviceBadge = document.getElementById('device-badge');
    if (deviceBadge && state.device) {
        const device = state.device.toLowerCase();
        deviceBadge.classList.remove('mps', 'cuda', 'cpu');
        
        if (device.includes('mps')) {
            deviceBadge.textContent = '🍎 MPS';
            deviceBadge.classList.add('mps');
        } else if (device.includes('cuda')) {
            deviceBadge.textContent = '🎮 CUDA';
            deviceBadge.classList.add('cuda');
        } else {
            deviceBadge.textContent = '🖥️ CPU';
        }
    }
    
    // Update compile badge
    const compileBadge = document.getElementById('compile-badge');
    if (compileBadge) {
        if (state.torch_compiled) {
            compileBadge.textContent = '⚡ Compiled';
            compileBadge.classList.add('active');
        } else {
            compileBadge.textContent = '📦 Eager';
            compileBadge.classList.remove('active');
        }
    }
}

// ============================================================
// ETA CALCULATION
// ============================================================

/**
 * Update estimated time remaining
 */
function updateETA(state) {
    const etaElement = document.getElementById('info-eta');
    if (!etaElement) return;
    
    const currentEpisode = state.episode || 0;
    const targetEps = state.target_episodes || 0;
    const epsPerSec = state.episodes_per_second || 0;
    
    // Unlimited mode (target_episodes == 0)
    if (targetEps === 0) {
        etaElement.textContent = '∞ Unlimited';
        return;
    }
    
    if (currentEpisode >= targetEps) {
        etaElement.textContent = 'Complete!';
        return;
    }
    
    if (epsPerSec <= 0) {
        etaElement.textContent = 'Calculating...';
        return;
    }
    
    const remainingEps = targetEps - currentEpisode;
    const remainingSeconds = remainingEps / epsPerSec;
    
    if (remainingSeconds < 60) {
        etaElement.textContent = `${Math.ceil(remainingSeconds)}s`;
    } else if (remainingSeconds < 3600) {
        const mins = Math.floor(remainingSeconds / 60);
        const secs = Math.ceil(remainingSeconds % 60);
        etaElement.textContent = `${mins}m ${secs}s`;
    } else {
        const hours = Math.floor(remainingSeconds / 3600);
        const mins = Math.ceil((remainingSeconds % 3600) / 60);
        etaElement.textContent = `${hours}h ${mins}m`;
    }
}

// ============================================================
// SETTINGS ENHANCEMENTS
// ============================================================

/**
 * Update learn every label
 */
function updateLearnEveryLabel(value) {
    const label = document.getElementById('learn-every-value');
    if (label) {
        label.textContent = value === '1' || value === 1 ? '1 step' : `${value} steps`;
    }
}

/**
 * Apply settings changes (enhanced)
 */
function applySettings() {
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

    // Visual feedback
    const btn = document.querySelector('.apply-btn');
    const originalText = btn.textContent;
    btn.textContent = '✓ Applied!';
    setTimeout(() => {
        btn.textContent = originalText;
    }, 1500);
}

/**
 * Load config from server (enhanced)
 */
function loadConfig() {
    fetchWithTimeout('/api/config')
        .then(response => response.json())
        .then(data => {
            document.getElementById('setting-lr').value = data.learning_rate;
            document.getElementById('setting-epsilon').value = data.epsilon_start;
            document.getElementById('setting-decay').value = data.epsilon_decay;
            document.getElementById('setting-gamma').value = data.gamma;
            document.getElementById('setting-batch').value = data.batch_size;
            
            // Performance settings
            if (data.learn_every) {
                document.getElementById('setting-learn-every').value = data.learn_every;
                updateLearnEveryLabel(data.learn_every);
            }
            if (data.gradient_steps) {
                document.getElementById('setting-grad-steps').value = data.gradient_steps;
            }
            
            // Update system status from config
            if (data.device) {
                updateSystemStatus({ device: data.device, torch_compiled: false });
            }
        })
        .catch(err => {
            if (err.name !== 'AbortError') {
                console.error('Config load error:', err);
            }
        });
    
    // Also fetch save status
    fetchSaveStatus();
}

// ============================================================
// SAVE MANAGEMENT FUNCTIONS
// ============================================================

/**
 * Fetch current save status from server
 */
function fetchSaveStatus() {
    fetchWithTimeout('/api/save-status')
        .then(response => response.json())
        .then(data => {
            updateSaveStatus(data);
        })
        .catch(err => {
            if (err.name !== 'AbortError') {
                console.error('Save status fetch error:', err);
            }
        });
}

/**
 * Update save status display
 */
function updateSaveStatus(data) {
    const timeEl = document.getElementById('last-save-time');
    const fileEl = document.getElementById('last-save-file');
    const reasonEl = document.getElementById('last-save-reason');
    const countEl = document.getElementById('saves-count');
    
    if (timeEl) {
        timeEl.textContent = data.time_since_save_str || 'Never';
    }
    if (fileEl) {
        fileEl.textContent = data.last_save_filename || '-';
    }
    if (reasonEl) {
        reasonEl.textContent = data.last_save_reason || '-';
        reasonEl.className = 'save-value save-reason ' + (data.last_save_reason || '');
    }
    if (countEl) {
        countEl.textContent = data.saves_this_session || 0;
    }
}

/**
 * Flash save indicator when save occurs
 */
function flashSaveIndicator() {
    const indicator = document.getElementById('save-indicator');
    if (indicator) {
        indicator.classList.remove('active');
        // Trigger reflow
        void indicator.offsetWidth;
        indicator.classList.add('active');
        
        // Remove after animation
        setTimeout(() => {
            indicator.classList.remove('active');
        }, 2000);
    }
}

/**
 * Save model with custom name
 */
function saveModelAs() {
    const input = document.getElementById('save-as-name');
    let filename = input.value.trim();
    
    if (!filename) {
        filename = 'custom_save';
    }
    
    // Clean filename - only allow alphanumeric, underscore, hyphen
    // (dots not allowed to match Python backend sanitization)
    filename = filename.replace(/[^a-zA-Z0-9_-]/g, '_');
    
    // Remove leading/trailing underscores that may result from sanitization
    filename = filename.replace(/^_+|_+$/g, '');
    
    // Ensure we have a valid filename after sanitization
    if (!filename) {
        filename = 'custom_save';
    }
    
    socket.emit('control', { action: 'save_as', filename: filename });
    addConsoleLog(`Saving as: ${filename}.pth`, 'action');
    
    // Clear input and show feedback
    input.value = '';
    const btn = document.querySelector('.save-as-btn');
    const originalText = btn.textContent;
    btn.textContent = '✓ Saved!';
    setTimeout(() => {
        btn.textContent = originalText;
    }, 1500);
}

// Periodically update save status time
setInterval(() => {
    fetchWithTimeout('/api/save-status', {}, 5000)  // 5s timeout for periodic update
        .then(response => response.json())
        .then(data => {
            const timeEl = document.getElementById('last-save-time');
            if (timeEl && data.time_since_save_str) {
                timeEl.textContent = data.time_since_save_str;
            }
        })
        .catch(() => {});  // Silently ignore errors for periodic updates
}, 10000);  // Update every 10 seconds

// ============================================================
// GAME SELECTION FUNCTIONS
// ============================================================

/**
 * Load available games and populate the dropdown
 */
function loadGames() {
    fetchWithTimeout('/api/games')
        .then(response => response.json())
        .then(data => {
            const select = document.getElementById('game-select');
            if (!select) return;
            
            // Store current game for switch detection
            select.dataset.currentGame = data.current_game;
            
            // Clear existing options
            select.innerHTML = '';
            
            // Add games
            data.games.forEach(game => {
                const option = document.createElement('option');
                option.value = game.id;
                option.textContent = `${game.icon} ${game.name}`;
                if (game.is_current) {
                    option.selected = true;
                }
                select.appendChild(option);
            });
            
            // Update subtitle
            const subtitle = document.getElementById('game-subtitle');
            if (subtitle && data.current_game) {
                const currentGame = data.games.find(g => g.id === data.current_game);
                if (currentGame) {
                    subtitle.textContent = `${currentGame.name} Training Dashboard`;
                }
            }
        })
        .catch(err => {
            console.error('Failed to load games:', err);
        });
}

/**
 * Switch to a different game
 */
function switchGame(gameId) {
    if (!gameId) return;
    
    // Get current game from dropdown's data
    const select = document.getElementById('game-select');
    const currentGame = select ? select.dataset.currentGame : 'breakout';
    
    // If same game, do nothing
    if (gameId === currentGame) {
        return;
    }
    
    // Confirm switch
    const confirmed = confirm(
        `Switch to ${gameId.replace('_', ' ').toUpperCase()}?\n\n` +
        `This will save your current progress and restart with the new game.`
    );
    
    if (confirmed) {
        addConsoleLog(`🔄 Switching to ${gameId}...`, 'warning');
        addConsoleLog(`💾 Saving current progress...`, 'info');
        
        // Request game switch - server will save and restart
        socket.emit('control', { action: 'restart_with_game', game: gameId });
    } else {
        // Reset dropdown to current game
        loadGames();
    }
}

// ============================================================
// GAME COMPARISON FUNCTIONS
// ============================================================

/**
 * Toggle comparison panel
 */
function toggleComparison() {
    const card = document.getElementById('comparison-card');
    const icon = document.getElementById('comparison-icon');
    
    if (!card) return;
    
    card.classList.toggle('collapsed');
    if (icon) {
        icon.textContent = card.classList.contains('collapsed') ? '▼' : '▲';
    }
    
    // Load stats when opening
    if (!card.classList.contains('collapsed')) {
        loadGameStats();
    }
}

/**
 * Load game statistics for comparison
 */
function loadGameStats() {
    const grid = document.getElementById('comparison-grid');
    if (!grid) return;
    
    fetchWithTimeout('/api/game-stats')
        .then(response => response.json())
        .then(data => {
            const stats = data.stats;
            const currentGame = data.current_game;
            
            // Find max score for bar scaling
            let maxScore = 0;
            for (const gameId in stats) {
                if (stats[gameId].best_score > maxScore) {
                    maxScore = stats[gameId].best_score;
                }
            }
            maxScore = maxScore || 1; // Avoid division by zero
            
            // Build comparison items
            let html = '';
            for (const gameId in stats) {
                const game = stats[gameId];
                const isCurrent = gameId === currentGame;
                const barWidth = (game.best_score / maxScore) * 100;
                const colorRgb = `rgb(${game.color[0]}, ${game.color[1]}, ${game.color[2]})`;
                
                // Format training time
                let timeStr = 'No training';
                if (game.total_training_time > 0) {
                    const hours = Math.floor(game.total_training_time / 3600);
                    const mins = Math.floor((game.total_training_time % 3600) / 60);
                    timeStr = hours > 0 ? `${hours}h ${mins}m` : `${mins}m`;
                }
                
                html += `
                    <div class="comparison-item ${isCurrent ? 'current' : ''}">
                        <div class="comparison-icon">${game.icon}</div>
                        <div class="comparison-info">
                            <div class="comparison-name">${game.name} ${isCurrent ? '(current)' : ''}</div>
                            <div class="comparison-stats">
                                Best: ${game.best_score} | Episodes: ${game.total_episodes.toLocaleString()} | Time: ${timeStr}
                            </div>
                            <div class="comparison-bar">
                                <div class="comparison-bar-fill" style="width: ${barWidth}%; background: ${colorRgb};"></div>
                            </div>
                        </div>
                    </div>
                `;
            }
            
            grid.innerHTML = html || '<div class="no-data">No game data available</div>';
        })
        .catch(err => {
            console.error('Failed to load game stats:', err);
            grid.innerHTML = '<div class="error">Failed to load game statistics</div>';
        });
}

/**
 * Refresh game stats
 */
function refreshGameStats() {
    loadGameStats();
}

/**
 * Show a restart banner for game switching
 */
function showRestartBanner(game, command) {
    // Create banner element
    const banner = document.createElement('div');
    banner.id = 'restart-banner';
    banner.style.cssText = `
        position: fixed;
        top: 50%;
        left: 50%;
        transform: translate(-50%, -50%);
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border: 2px solid #4caf50;
        border-radius: 16px;
        padding: 30px 40px;
        z-index: 10000;
        text-align: center;
        box-shadow: 0 10px 40px rgba(0,0,0,0.5);
        max-width: 500px;
    `;
    
    banner.innerHTML = `
        <h2 style="color: #4caf50; margin: 0 0 15px 0; font-size: 1.5rem;">🔄 Ready to Switch Games</h2>
        <p style="color: #e4e6f0; margin: 0 0 20px 0;">Progress has been saved. Restart with the new game:</p>
        <div style="background: #0d0e12; padding: 12px 16px; border-radius: 8px; margin-bottom: 20px;">
            <code id="restart-command" style="color: #64b5f6; font-family: 'JetBrains Mono', monospace; font-size: 0.9rem; word-break: break-all;">${escapeHtml(command)}</code>
        </div>
        <div style="display: flex; gap: 12px; justify-content: center;">
            <button onclick="copyRestartCommand()" style="
                background: #4caf50;
                border: none;
                color: white;
                padding: 10px 20px;
                border-radius: 8px;
                cursor: pointer;
                font-weight: 500;
            ">📋 Copy Command</button>
            <button onclick="closeRestartBanner()" style="
                background: #2a2e3d;
                border: 1px solid #3a3e4d;
                color: #e4e6f0;
                padding: 10px 20px;
                border-radius: 8px;
                cursor: pointer;
            ">Close</button>
        </div>
    `;
    
    // Add overlay
    const overlay = document.createElement('div');
    overlay.id = 'restart-overlay';
    overlay.style.cssText = `
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background: rgba(0,0,0,0.7);
        z-index: 9999;
    `;
    overlay.onclick = closeRestartBanner;
    
    document.body.appendChild(overlay);
    document.body.appendChild(banner);
}

/**
 * Copy restart command to clipboard
 */
function copyRestartCommand() {
    const command = document.getElementById('restart-command');
    if (command) {
        navigator.clipboard.writeText(command.textContent).then(() => {
            const btn = event.target;
            btn.textContent = '✓ Copied!';
            setTimeout(() => {
                btn.textContent = '📋 Copy Command';
            }, 2000);
        });
    }
}

/**
 * Close restart banner
 */
function closeRestartBanner() {
    const banner = document.getElementById('restart-banner');
    const overlay = document.getElementById('restart-overlay');
    if (banner) banner.remove();
    if (overlay) overlay.remove();
}

/**
 * Show restarting overlay while process restarts
 */
function showRestartingOverlay(game) {
    const overlay = document.createElement('div');
    overlay.id = 'restarting-overlay';
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
        <div style="font-size: 4rem; margin-bottom: 20px; animation: pulse 1s ease-in-out infinite;">🔄</div>
        <h2 style="color: #00d4ff; margin: 0 0 15px 0; font-size: 1.5rem;">Restarting with ${game.replace('_', ' ').toUpperCase()}</h2>
        <p style="color: #7a7e8c; margin: 0;">Please wait while the server restarts...</p>
        <p style="color: #5a5e72; margin: 15px 0 0 0; font-size: 0.85rem;">Page will auto-refresh when ready</p>
        <style>
            @keyframes pulse {
                0%, 100% { transform: scale(1); opacity: 1; }
                50% { transform: scale(1.1); opacity: 0.8; }
            }
        </style>
    `;
    
    document.body.appendChild(overlay);
    
    // Start checking if server is back up
    setTimeout(checkServerAndReload, 2000);
}

/**
 * Check if server is back up and reload
 */
function checkServerAndReload() {
    fetch('/api/status')
        .then(response => {
            if (response.ok) {
                location.reload();
            } else {
                setTimeout(checkServerAndReload, 1000);
            }
        })
        .catch(() => {
            setTimeout(checkServerAndReload, 1000);
        });
}

// ============================================================
// NEURAL NETWORK VISUALIZATION
// ============================================================

/**
 * Neural Network Visualizer - Canvas-based real-time visualization
 * 
 * Features:
 * - Network architecture (layers, neurons, connections)
 * - Live activations with diverging color palette
 * - Animated data flow pulses
 * - Q-values bar chart with action selection
 * - Layer labels with neuron counts
 */
class NeuralNetworkVisualizer {
    constructor(canvasId) {
        this.canvas = document.getElementById(canvasId);
        if (!this.canvas) {
            console.warn('NN Visualizer: Canvas not found:', canvasId);
            return;
        }
        this.ctx = this.canvas.getContext('2d');
        
        // State
        this.data = null;
        this.isEnabled = true;
        this.animationId = null;
        
        // Animation state
        this.pulsePhase = 0;
        this.pulses = [];
        this.pulseSpawnTimer = 0;
        this.prevActivations = {};
        this.interpolationSpeed = 0.3;
        
        // Colors - matches pygame visualizer
        this.colors = {
            bg: '#0c0c18',
            panel: '#121220',
            text: '#c8c8dc',
            inactive: '#282837',
            negative: '#4287f5',     // Blue for negative
            neutral: '#c8c8d2',      // White-ish for zero
            positive: '#f5426c',     // Red/pink for positive
            weightNegative: '#2962ff',  // Blue
            weightNeutral: '#646478',   // Gray
            weightPositive: '#ff6229',  // Orange/red
            border: '#283c64',
            live: '#64c896'
        };
        
        // Layout
        this.margin = 20;
        this.headerHeight = 50;
        this.qvalueHeight = 85;
        this.layerLabelHeight = 25;
        this.neuronRadius = 6;
        this.maxNeurons = 15;
        
        // Start animation loop
        this.startAnimation();
        
        // Handle resize
        this.setupResize();
    }
    
    setupResize() {
        const resizeObserver = new ResizeObserver(() => {
            if (this.canvas && this.canvas.parentElement) {
                const rect = this.canvas.parentElement.getBoundingClientRect();
                const dpr = window.devicePixelRatio || 1;
                this.canvas.width = rect.width * dpr;
                this.canvas.height = rect.height * dpr;
                this.canvas.style.width = rect.width + 'px';
                this.canvas.style.height = rect.height + 'px';
                this.ctx.scale(dpr, dpr);
                this.width = rect.width;
                this.height = rect.height;
            }
        });
        
        if (this.canvas.parentElement) {
            resizeObserver.observe(this.canvas.parentElement);
        }
    }
    
    startAnimation() {
        const animate = () => {
            if (this.isEnabled && this.data) {
                this.render();
            }
            this.animationId = requestAnimationFrame(animate);
        };
        animate();
    }
    
    stopAnimation() {
        if (this.animationId) {
            cancelAnimationFrame(this.animationId);
            this.animationId = null;
        }
    }
    
    update(data) {
        if (!data || !data.layer_info || data.layer_info.length === 0) {
            return;
        }
        this.data = data;
        
        // Hide placeholder when we receive valid data
        const placeholder = document.getElementById('nn-viz-placeholder');
        if (placeholder) {
            placeholder.style.display = 'none';
        }
    }
    
    toggle(enabled) {
        this.isEnabled = enabled;
        const indicator = document.getElementById('nn-viz-status');
        if (indicator) {
            indicator.textContent = enabled ? '● LIVE' : '○ Paused';
            indicator.style.color = enabled ? this.colors.live : '#666';
        }
    }
    
    render() {
        if (!this.ctx || !this.data || !this.width || !this.height) return;
        
        const ctx = this.ctx;
        const dpr = window.devicePixelRatio || 1;
        
        // Clear with scale reset
        ctx.setTransform(1, 0, 0, 1, 0, 0);
        ctx.scale(dpr, dpr);
        
        // Draw background
        this.drawBackground();
        
        // Calculate layer positions
        const layerPositions = this.calculateLayerPositions();
        if (!layerPositions || layerPositions.length === 0) return;
        
        // Smooth activations
        const smoothedActivations = this.smoothActivations();
        
        // Draw connections
        this.drawConnections(layerPositions, smoothedActivations);
        
        // Update and draw pulses
        this.updatePulses(layerPositions, smoothedActivations);
        this.drawPulses();
        
        // Draw neurons
        this.drawNeurons(layerPositions, smoothedActivations);
        
        // Draw layer labels
        this.drawLayerLabels(layerPositions);
        
        // Draw Q-values
        this.drawQValues();
        
        // Draw title
        this.drawTitle();
        
        // Update animation phase
        this.pulsePhase = (this.pulsePhase + 0.08) % (Math.PI * 2);
    }
    
    drawBackground() {
        const ctx = this.ctx;
        
        // Gradient background
        const gradient = ctx.createLinearGradient(0, 0, 0, this.height);
        gradient.addColorStop(0, this.colors.bg);
        gradient.addColorStop(1, this.colors.panel);
        ctx.fillStyle = gradient;
        ctx.fillRect(0, 0, this.width, this.height);
        
        // Animated border glow
        const glowIntensity = 20 + 10 * Math.sin(this.pulsePhase);
        ctx.strokeStyle = `rgb(${40 + glowIntensity}, ${60 + glowIntensity}, ${100 + glowIntensity})`;
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.roundRect(1, 1, this.width - 2, this.height - 2, 8);
        ctx.stroke();
    }
    
    drawTitle() {
        const ctx = this.ctx;
        
        // Main title
        ctx.font = 'bold 16px "Plus Jakarta Sans", sans-serif';
        ctx.fillStyle = '#64b5f6';
        ctx.textAlign = 'center';
        ctx.fillText('Neural Network', this.width / 2, 22);
        
        // LIVE indicator with pulse
        const pulse = 0.7 + 0.3 * Math.sin(this.pulsePhase * 0.5);
        ctx.font = '11px "JetBrains Mono", monospace';
        ctx.fillStyle = `rgba(${100 * pulse}, ${200 * pulse}, ${150 * pulse}, 1)`;
        ctx.textAlign = 'left';
        ctx.fillText('● LIVE', 10, 38);
        
        // Step counter
        ctx.textAlign = 'right';
        ctx.fillStyle = '#787890';
        ctx.fillText(`Step: ${(this.data.step || 0).toLocaleString()}`, this.width - 10, 38);
    }
    
    calculateLayerPositions() {
        const layerInfo = this.data.layer_info;
        if (!layerInfo || layerInfo.length === 0) return [];
        
        const numLayers = layerInfo.length;
        const hMargin = 35;
        const availableWidth = this.width - (hMargin * 2);
        const layerSpacing = availableWidth / Math.max(numLayers - 1, 1);
        
        const networkTop = this.headerHeight + this.layerLabelHeight;
        const networkBottom = this.height - this.qvalueHeight - 10;
        const availableHeight = networkBottom - networkTop;
        
        const positions = [];
        
        for (let i = 0; i < layerInfo.length; i++) {
            const info = layerInfo[i];
            const layerX = hMargin + i * layerSpacing;
            const numNeurons = Math.min(info.neurons, this.maxNeurons);
            
            const neuronSpacing = Math.min(22, availableHeight / Math.max(numNeurons + 1, 1));
            const totalHeight = numNeurons * neuronSpacing;
            const startY = networkTop + (availableHeight - totalHeight) / 2;
            
            const neuronPositions = [];
            for (let j = 0; j < numNeurons; j++) {
                const ny = startY + j * neuronSpacing + neuronSpacing / 2;
                neuronPositions.push({ x: layerX, y: ny });
            }
            
            positions.push({
                x: layerX,
                neurons: numNeurons,
                actualNeurons: info.neurons,
                positions: neuronPositions,
                type: info.type,
                name: info.name
            });
        }
        
        return positions;
    }
    
    smoothActivations() {
        const activations = this.data.activations || {};
        const smoothed = {};
        
        for (const key in activations) {
            const newAct = activations[key];
            if (!newAct || newAct.length === 0) continue;
            
            if (this.prevActivations[key] && this.prevActivations[key].length === newAct.length) {
                smoothed[key] = this.prevActivations[key].map((prev, i) => 
                    prev + (newAct[i] - prev) * this.interpolationSpeed
                );
            } else {
                smoothed[key] = [...newAct];
            }
            
            this.prevActivations[key] = [...smoothed[key]];
        }
        
        return smoothed;
    }
    
    drawConnections(layerPositions, activations) {
        const ctx = this.ctx;
        const weights = this.data.weights || [];
        
        for (let i = 0; i < layerPositions.length - 1; i++) {
            const fromLayer = layerPositions[i];
            const toLayer = layerPositions[i + 1];
            
            if (i < weights.length && weights[i]) {
                const weightMatrix = weights[i];
                let maxWeight = 0;
                
                // Find max weight for normalization
                for (const row of weightMatrix) {
                    for (const w of row) {
                        if (Math.abs(w) > maxWeight) maxWeight = Math.abs(w);
                    }
                }
                maxWeight = maxWeight || 1;
                
                // Sample connections
                const maxConnections = 60;
                const fromSample = this.sampleIndices(fromLayer.positions.length, 6);
                const toSample = this.sampleIndices(toLayer.positions.length, 10);
                
                for (const fi of fromSample) {
                    for (const ti of toSample) {
                        if (ti < weightMatrix.length && fi < (weightMatrix[ti]?.length || 0)) {
                            const weight = weightMatrix[ti][fi];
                            const normWeight = weight / maxWeight;
                            
                            // Diverging color
                            let color;
                            if (normWeight > 0) {
                                color = this.interpolateColor(
                                    this.hexToRgb(this.colors.weightNeutral),
                                    this.hexToRgb(this.colors.weightPositive),
                                    Math.abs(normWeight)
                                );
                            } else {
                                color = this.interpolateColor(
                                    this.hexToRgb(this.colors.weightNeutral),
                                    this.hexToRgb(this.colors.weightNegative),
                                    Math.abs(normWeight)
                                );
                            }
                            
                            const fromPos = fromLayer.positions[fi];
                            const toPos = toLayer.positions[ti];
                            
                            ctx.strokeStyle = `rgb(${color.r}, ${color.g}, ${color.b})`;
                            ctx.lineWidth = Math.max(0.5, Math.abs(normWeight) * 2);
                            ctx.globalAlpha = 0.4 + Math.abs(normWeight) * 0.4;
                            
                            ctx.beginPath();
                            ctx.moveTo(fromPos.x, fromPos.y);
                            ctx.lineTo(toPos.x, toPos.y);
                            ctx.stroke();
                        }
                    }
                }
                
                ctx.globalAlpha = 1;
            }
        }
    }
    
    sampleIndices(length, max) {
        if (length <= max) {
            return Array.from({ length }, (_, i) => i);
        }
        const step = length / max;
        return Array.from({ length: max }, (_, i) => Math.floor(i * step));
    }
    
    updatePulses(layerPositions, activations) {
        // Update existing pulses
        this.pulses = this.pulses.filter(p => {
            p.progress += p.speed;
            return p.progress < 1;
        });
        
        // Spawn new pulses
        this.pulseSpawnTimer++;
        if (this.pulseSpawnTimer >= 5 && layerPositions.length > 1) {
            this.pulseSpawnTimer = 0;
            
            for (let i = 0; i < layerPositions.length - 1; i++) {
                const fromLayer = layerPositions[i];
                const toLayer = layerPositions[i + 1];
                
                if (fromLayer.positions.length > 0 && toLayer.positions.length > 0) {
                    for (let j = 0; j < 2; j++) {
                        const fromIdx = Math.floor(Math.random() * fromLayer.positions.length);
                        const toIdx = Math.floor(Math.random() * toLayer.positions.length);
                        
                        const layerKey = `layer_${i}`;
                        let actLevel = 0.5;
                        if (activations[layerKey]) {
                            const acts = activations[layerKey];
                            actLevel = acts.reduce((a, b) => a + Math.abs(b), 0) / acts.length;
                        }
                        
                        const intensity = Math.min(1, actLevel * 2);
                        const color = this.interpolateColor(
                            { r: 60, g: 80, b: 120 },
                            { r: 100, g: 255, b: 180 },
                            intensity
                        );
                        
                        this.pulses.push({
                            start: fromLayer.positions[fromIdx],
                            end: toLayer.positions[toIdx],
                            color: color,
                            progress: 0,
                            speed: 0.08
                        });
                    }
                }
            }
        }
    }
    
    drawPulses() {
        const ctx = this.ctx;
        
        for (const pulse of this.pulses) {
            const x = pulse.start.x + (pulse.end.x - pulse.start.x) * pulse.progress;
            const y = pulse.start.y + (pulse.end.y - pulse.start.y) * pulse.progress;
            
            const alpha = 1 - Math.abs(pulse.progress - 0.5) * 2;
            const size = 3 + 2 * alpha;
            
            // Glow
            ctx.beginPath();
            ctx.arc(x, y, size + 2, 0, Math.PI * 2);
            ctx.fillStyle = `rgba(${pulse.color.r}, ${pulse.color.g}, ${pulse.color.b}, ${alpha * 0.5})`;
            ctx.fill();
            
            // Core
            ctx.beginPath();
            ctx.arc(x, y, size, 0, Math.PI * 2);
            ctx.fillStyle = `rgba(${Math.min(255, pulse.color.r * 1.5)}, ${Math.min(255, pulse.color.g * 1.5)}, ${Math.min(255, pulse.color.b * 1.5)}, ${alpha})`;
            ctx.fill();
        }
    }
    
    drawNeurons(layerPositions, activations) {
        const ctx = this.ctx;
        const layerInfo = this.data.layer_info;
        
        for (let i = 0; i < layerPositions.length; i++) {
            const layerPos = layerPositions[i];
            const info = layerInfo[i];
            
            // Get activations for this layer
            let layerActs = [];
            if (info.type === 'input') {
                // For input layer, use first activation values if available
                const inputKey = 'input' in activations ? 'input' : 'layer_-1';
                layerActs = activations[inputKey] || [];
            } else {
                const layerKey = `layer_${i - 1}`;
                layerActs = activations[layerKey] || [];
            }
            
            // Normalize activations
            let maxAct = 1;
            if (layerActs.length > 0) {
                maxAct = Math.max(...layerActs.map(Math.abs), 0.001);
            }
            
            for (let j = 0; j < layerPos.positions.length; j++) {
                const pos = layerPos.positions[j];
                const actVal = j < layerActs.length ? layerActs[j] / maxAct : 0;
                
                // Diverging color based on activation sign
                let color;
                if (actVal > 0) {
                    color = this.interpolateColor(
                        this.hexToRgb(this.colors.inactive),
                        this.hexToRgb(this.colors.positive),
                        Math.min(Math.abs(actVal), 1)
                    );
                } else {
                    color = this.interpolateColor(
                        this.hexToRgb(this.colors.inactive),
                        this.hexToRgb(this.colors.negative),
                        Math.min(Math.abs(actVal), 1)
                    );
                }
                
                // Radius with pulse for active neurons
                let radius = this.neuronRadius;
                if (Math.abs(actVal) > 0.7) {
                    radius *= 1 + 0.2 * Math.sin(this.pulsePhase + j * 0.5);
                }
                
                // Outer glow for active neurons
                if (Math.abs(actVal) > 0.4) {
                    ctx.beginPath();
                    ctx.arc(pos.x, pos.y, radius + 4, 0, Math.PI * 2);
                    ctx.fillStyle = `rgba(${color.r}, ${color.g}, ${color.b}, ${Math.abs(actVal) * 0.4})`;
                    ctx.fill();
                }
                
                // Neuron body
                ctx.beginPath();
                ctx.arc(pos.x, pos.y, radius, 0, Math.PI * 2);
                ctx.fillStyle = `rgb(${color.r}, ${color.g}, ${color.b})`;
                ctx.fill();
                
                // Border
                ctx.strokeStyle = '#505a6e';
                ctx.lineWidth = 1;
                ctx.stroke();
                
                // Highlight
                ctx.beginPath();
                ctx.arc(pos.x - radius * 0.3, pos.y - radius * 0.3, radius / 3, 0, Math.PI * 2);
                ctx.fillStyle = `rgba(${Math.min(255, color.r + 50)}, ${Math.min(255, color.g + 50)}, ${Math.min(255, color.b + 50)}, 0.6)`;
                ctx.fill();
            }
            
            // Ellipsis for hidden neurons
            if (layerPos.neurons < layerPos.actualNeurons) {
                const lastPos = layerPos.positions[layerPos.positions.length - 1];
                ctx.font = '11px "JetBrains Mono", monospace';
                ctx.fillStyle = '#646478';
                ctx.textAlign = 'center';
                ctx.fillText(`+${layerPos.actualNeurons - layerPos.neurons}`, layerPos.x, lastPos.y + 20);
            }
        }
    }
    
    drawLayerLabels(layerPositions) {
        const ctx = this.ctx;
        const layerInfo = this.data.layer_info;
        const labelY = this.headerHeight + 5;
        
        ctx.font = '10px "JetBrains Mono", monospace';
        ctx.textAlign = 'center';
        
        for (let i = 0; i < layerPositions.length; i++) {
            const layerPos = layerPositions[i];
            const info = layerInfo[i];
            
            let name, color;
            if (info.type === 'input') {
                name = 'IN';
                color = '#64b5f6';
            } else if (info.type === 'output') {
                name = 'OUT';
                color = '#ff9664';
            } else {
                name = `H${i}`;
                color = '#96c896';
            }
            
            // Layer name
            ctx.fillStyle = color;
            ctx.fillText(name, layerPos.x, labelY);
            
            // Neuron count
            ctx.fillStyle = '#5a5a6e';
            ctx.fillText(`(${layerPos.actualNeurons})`, layerPos.x, labelY + 12);
        }
    }
    
    drawQValues() {
        const ctx = this.ctx;
        const qValues = this.data.q_values || [];
        const selectedAction = this.data.selected_action || 0;
        const actionLabels = this.data.action_labels || ['LEFT', 'STAY', 'RIGHT'];
        const actionIcons = ['◀', '●', '▶'];
        
        if (qValues.length === 0) return;
        
        const qvY = this.height - this.qvalueHeight;
        const qvHeight = this.qvalueHeight - 5;
        
        // Background panel
        ctx.fillStyle = '#14161e';
        ctx.beginPath();
        ctx.roundRect(8, qvY, this.width - 16, qvHeight, 8);
        ctx.fill();
        ctx.strokeStyle = '#32374b';
        ctx.lineWidth = 1;
        ctx.stroke();
        
        // Title
        ctx.font = '10px "JetBrains Mono", monospace';
        ctx.fillStyle = '#8c8ca0';
        ctx.textAlign = 'left';
        ctx.fillText('Q-Values', 15, qvY + 14);
        
        // Normalize Q-values
        const qMin = Math.min(...qValues);
        const qMax = Math.max(...qValues);
        const qRange = qMax - qMin + 0.001;
        
        // Bar layout
        const contentWidth = this.width - 50;
        const barWidth = contentWidth / qValues.length;
        const barMaxHeight = 35;
        const bestAction = qValues.indexOf(Math.max(...qValues));
        
        for (let i = 0; i < qValues.length; i++) {
            const qVal = qValues[i];
            const barX = 20 + i * barWidth;
            
            const normQ = (qVal - qMin) / qRange;
            const barHeight = Math.max(8, normQ * barMaxHeight);
            
            const isSelected = i === selectedAction || (selectedAction === undefined && i === bestAction);
            
            if (isSelected) {
                // Animated selected action
                const pulse = 0.8 + 0.2 * Math.sin(this.pulsePhase * 2);
                ctx.fillStyle = `rgb(${Math.floor(50 * pulse)}, ${Math.floor(220 * pulse)}, ${Math.floor(120 * pulse)})`;
                
                // Glow
                ctx.beginPath();
                ctx.roundRect(barX - 2, qvY + 45 - barHeight - 2, barWidth - 8 + 4, barHeight + 4, 4);
                ctx.fillStyle = 'rgba(30, 100, 60, 0.5)';
                ctx.fill();
                
                ctx.fillStyle = `rgb(${Math.floor(50 * pulse)}, ${Math.floor(220 * pulse)}, ${Math.floor(120 * pulse)})`;
            } else {
                ctx.fillStyle = '#373c50';
            }
            
            // Draw bar
            ctx.beginPath();
            ctx.roundRect(barX, qvY + 45 - barHeight, barWidth - 10, barHeight, 4);
            ctx.fill();
            ctx.strokeStyle = isSelected ? '#64ffa0' : '#505564';
            ctx.lineWidth = 1;
            ctx.stroke();
            
            // Draw icon
            ctx.font = '14px sans-serif';
            ctx.fillStyle = isSelected ? '#dcdcf0' : '#787890';
            ctx.textAlign = 'center';
            ctx.fillText(actionIcons[i] || '?', barX + barWidth / 2 - 5, qvY + 60);
            
            // Q-value number
            ctx.font = '9px "JetBrains Mono", monospace';
            ctx.fillStyle = '#646478';
            ctx.fillText(qVal.toFixed(2), barX + barWidth / 2 - 5, qvY + 45 - barHeight - 4);
        }
    }
    
    // Utility functions
    hexToRgb(hex) {
        const result = /^#?([a-f\d]{2})([a-f\d]{2})([a-f\d]{2})$/i.exec(hex);
        return result ? {
            r: parseInt(result[1], 16),
            g: parseInt(result[2], 16),
            b: parseInt(result[3], 16)
        } : { r: 0, g: 0, b: 0 };
    }
    
    interpolateColor(color1, color2, t) {
        t = Math.max(0, Math.min(1, t));
        // Ease-out for smoother transitions
        t = 1 - Math.pow(1 - t, 2);
        return {
            r: Math.round(color1.r + (color2.r - color1.r) * t),
            g: Math.round(color1.g + (color2.g - color1.g) * t),
            b: Math.round(color1.b + (color2.b - color1.b) * t)
        };
    }
}

// Global NN visualizer instance
let nnVisualizer = null;

/**
 * Initialize the neural network visualizer
 */
function initNNVisualizer() {
    const canvas = document.getElementById('nn-canvas');
    if (canvas) {
        nnVisualizer = new NeuralNetworkVisualizer('nn-canvas');
        console.log('Neural Network Visualizer initialized');
    }
}

/**
 * Toggle neural network visualization
 */
function toggleNNVisualization() {
    if (nnVisualizer) {
        nnVisualizer.toggle(!nnVisualizer.isEnabled);
    }
}

/**
 * Collapse/expand NN visualization panel
 */
function toggleNNPanel() {
    const card = document.getElementById('nn-viz-card');
    const icon = document.getElementById('nn-viz-icon');
    
    if (!card) return;
    
    card.classList.toggle('collapsed');
    if (icon) {
        icon.textContent = card.classList.contains('collapsed') ? '▼' : '▲';
    }
}
