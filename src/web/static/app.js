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

// Initialize on DOM load
document.addEventListener('DOMContentLoaded', () => {
    initCharts();
    connectSocket();
    startScreenshotPolling();
    updateFooterTime();
    setInterval(updateFooterTime, 1000);
    loadConfig();
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
        updateDashboard(data);
    });

    socket.on('console_log', (log) => {
        addConsoleLog(log.message, log.level, log.timestamp, log.data);
    });

    socket.on('console_logs', (data) => {
        // Initial batch of logs on connect
        data.logs.forEach(log => {
            addConsoleLog(log.message, log.level, log.timestamp, log.data, false);
        });
        renderConsoleLogs();
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
    document.getElementById('info-memory').textContent = 
        `${(state.memory_size / 1000).toFixed(0)}k / ${(state.memory_capacity / 1000).toFixed(0)}k`;
    document.getElementById('info-qvalue').textContent = state.avg_q_value.toFixed(2);
    document.getElementById('info-target').textContent = state.target_updates.toLocaleString();
    document.getElementById('info-actions').textContent = 
        `${state.exploration_actions.toLocaleString()} / ${state.exploitation_actions.toLocaleString()}`;
    
    // Update status badge
    const statusBadge = document.getElementById('info-status');
    if (state.is_paused) {
        statusBadge.textContent = 'Paused';
        statusBadge.className = 'info-value status-badge paused';
    } else if (state.is_running) {
        statusBadge.textContent = 'Training';
        statusBadge.className = 'info-value status-badge training';
    } else {
        statusBadge.textContent = 'Idle';
        statusBadge.className = 'info-value status-badge idle';
    }

    // Update pause button
    const pauseBtn = document.getElementById('pause-btn');
    isPaused = state.is_paused;
    pauseBtn.textContent = isPaused ? '▶️ Resume' : '⏸️ Pause';

    // Update speed slider if changed externally
    const speedSlider = document.getElementById('speed-slider');
    if (parseFloat(speedSlider.value) !== state.game_speed) {
        speedSlider.value = state.game_speed;
        // Format display nicely
        let displayText;
        if (state.game_speed >= 10 || Number.isInteger(state.game_speed)) {
            displayText = state.game_speed.toFixed(0) + 'x';
        } else if (state.game_speed >= 1) {
            displayText = state.game_speed.toFixed(1) + 'x';
        } else {
            displayText = state.game_speed.toFixed(2) + 'x';
        }
        document.getElementById('speed-value').textContent = displayText;
    }

    // Update charts
    updateCharts(history);
}

/**
 * Update charts with history data
 */
function updateCharts(history) {
    const maxPoints = 200;
    const scores = history.scores.slice(-maxPoints);
    const losses = history.losses.slice(-maxPoints);
    const qvalues = history.q_values ? history.q_values.slice(-maxPoints) : [];
    
    // Calculate running average
    const avgScores = calculateRunningAverage(scores, 20);
    
    // Generate labels
    const startEp = Math.max(0, history.scores.length - maxPoints);
    const labels = scores.map((_, i) => startEp + i);

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
 * Update game speed with snapping to nice values
 */
function updateSpeed(value) {
    let speed = parseFloat(value);
    
    // Snap to common values when close
    const snapValues = [0.25, 0.5, 1, 2, 5, 10, 25, 50, 100, 150, 200];
    const snapThreshold = 1.5; // snap if within this range
    
    for (const snap of snapValues) {
        if (Math.abs(speed - snap) < snapThreshold && Math.abs(speed - snap) > 0.01) {
            speed = snap;
            document.getElementById('speed-slider').value = snap;
            break;
        }
    }
    
    // Format display nicely
    let displayText;
    if (speed >= 10 || Number.isInteger(speed)) {
        displayText = speed.toFixed(0) + 'x';
    } else if (speed >= 1) {
        displayText = speed.toFixed(1) + 'x';
    } else {
        displayText = speed.toFixed(2) + 'x';
    }
    
    document.getElementById('speed-value').textContent = displayText;
    socket.emit('control', { action: 'speed', value: speed });
}

/**
 * Refresh screenshot
 */
function refreshScreenshot() {
    fetch('/api/screenshot')
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
            console.error('Screenshot fetch error:', err);
            const placeholder = document.getElementById('preview-placeholder');
            placeholder.classList.remove('hidden');
        });
}

/**
 * Start polling for screenshots
 */
function startScreenshotPolling() {
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

/**
 * Load config from server
 */
function loadConfig() {
    fetch('/api/config')
        .then(response => response.json())
        .then(data => {
            document.getElementById('setting-lr').value = data.learning_rate;
            document.getElementById('setting-epsilon').value = data.epsilon_start;
            document.getElementById('setting-decay').value = data.epsilon_decay;
            document.getElementById('setting-gamma').value = data.gamma;
            document.getElementById('setting-batch').value = data.batch_size;
        })
        .catch(err => console.error('Config load error:', err));
}

/**
 * Apply settings changes
 */
function applySettings() {
    const config = {
        learning_rate: parseFloat(document.getElementById('setting-lr').value),
        epsilon: parseFloat(document.getElementById('setting-epsilon').value),
        epsilon_decay: parseFloat(document.getElementById('setting-decay').value),
        gamma: parseFloat(document.getElementById('setting-gamma').value),
        batch_size: parseInt(document.getElementById('setting-batch').value)
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

// ============================================================
// MODEL LOADING
// ============================================================

/**
 * Show load model modal
 */
function showLoadModal() {
    const modal = document.getElementById('load-modal');
    modal.classList.add('visible');
    
    // Fetch available models
    fetch('/api/models')
        .then(response => response.json())
        .then(data => {
            const list = document.getElementById('model-list');
            
            if (data.models.length === 0) {
                list.innerHTML = '<div class="no-models">No saved models found</div>';
                return;
            }

            list.innerHTML = data.models.map(model => {
                const date = new Date(model.modified * 1000).toLocaleString();
                const size = (model.size / 1024).toFixed(1) + ' KB';
                return `
                    <div class="model-item" onclick="loadModel('${model.path}')">
                        <div class="model-name">📁 ${model.name}</div>
                        <div class="model-info">
                            <span>${size}</span>
                            <span>${date}</span>
                        </div>
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
    fetch('/api/status')
        .then(response => response.json())
        .then(data => {
            updateDashboard(data);
        })
        .catch(err => console.error('Initial fetch error:', err));
}

// Fetch initial data after connection
setTimeout(fetchInitialData, 500);

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
            resetEpisode();
            break;
    }
});
