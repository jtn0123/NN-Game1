/**
 * Neural Network AI - Training Dashboard
 * Real-time visualization with Chart.js and SocketIO
 */

// Charts
let scoreChart = null;
let lossChart = null;

// State
let isPaused = false;
let socket = null;

// Initialize on DOM load
document.addEventListener('DOMContentLoaded', () => {
    initCharts();
    connectSocket();
    startScreenshotPolling();
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
        plugins: {
            legend: {
                display: false
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
                    borderWidth: 2
                },
                {
                    label: 'Average',
                    data: [],
                    borderColor: '#ffc107',
                    backgroundColor: 'transparent',
                    tension: 0.4,
                    pointRadius: 0,
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
                borderWidth: 2
            }]
        },
        options: {
            ...chartOptions,
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
}

/**
 * Connect to SocketIO server
 */
function connectSocket() {
    socket = io();

    socket.on('connect', () => {
        console.log('Connected to server');
        updateConnectionStatus(true);
    });

    socket.on('disconnect', () => {
        console.log('Disconnected from server');
        updateConnectionStatus(false);
    });

    socket.on('state_update', (data) => {
        updateDashboard(data);
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

    // Update info
    document.getElementById('info-loss').textContent = state.loss.toFixed(4);
    document.getElementById('info-steps').textContent = state.total_steps.toLocaleString();
    
    const statusBadge = document.getElementById('info-status');
    if (state.is_paused) {
        statusBadge.textContent = 'Paused';
        statusBadge.classList.add('paused');
    } else {
        statusBadge.textContent = 'Training';
        statusBadge.classList.remove('paused');
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

/**
 * Toggle pause state
 */
function togglePause() {
    isPaused = !isPaused;
    socket.emit('control', { action: 'pause' });
    
    const btn = document.getElementById('pause-btn');
    btn.textContent = isPaused ? '▶️ Resume' : '⏸️ Pause';
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
    setTimeout(() => {
        btn.textContent = originalText;
    }, 1500);
}

/**
 * Refresh screenshot
 */
function refreshScreenshot() {
    fetch('/api/screenshot')
        .then(response => response.json())
        .then(data => {
            if (data.image) {
                const img = document.getElementById('game-preview');
                const placeholder = document.getElementById('preview-placeholder');
                img.src = 'data:image/png;base64,' + data.image;
                img.classList.add('visible');
                placeholder.style.display = 'none';
            }
        })
        .catch(err => console.error('Screenshot error:', err));
}

/**
 * Start polling for screenshots
 */
function startScreenshotPolling() {
    // Refresh screenshot every 2 seconds
    setInterval(refreshScreenshot, 2000);
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

