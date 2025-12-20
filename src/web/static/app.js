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

// Chart history storage - keep all data for scrolling
let fullHistory = {
    scores: [],
    losses: [],
    q_values: [],
    epsilons: [],
    labels: []
};

// State
let isPaused = false;
let socket = null;
let currentLogFilter = 'all';
let consoleLogs = [];
const MAX_CONSOLE_LOGS = 500;
let lastRenderedLogCount = 0;  // Track for incremental updates
let currentPerformanceMode = 'normal';
let trainingStartTime = 0;

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

/**
 * Downsample array data using LTTB algorithm for chart performance
 * Largest-Triangle-Three-Buckets preserves visual shape of data
 * @param {Array} data - Data array to downsample
 * @param {number} targetPoints - Target number of points
 * @returns {Object} - {data: downsampled data, indices: original indices}
 */
function downsampleLTTB(data, targetPoints) {
    if (data.length <= targetPoints) {
        return { data: data, indices: data.map((_, i) => i) };
    }

    const sampled = [];
    const indices = [];
    const bucketSize = (data.length - 2) / (targetPoints - 2);

    // Always keep first point
    sampled.push(data[0]);
    indices.push(0);

    for (let i = 0; i < targetPoints - 2; i++) {
        const bucketStart = Math.floor((i) * bucketSize) + 1;
        const bucketEnd = Math.floor((i + 1) * bucketSize) + 1;
        const nextBucketStart = Math.floor((i + 1) * bucketSize) + 1;
        const nextBucketEnd = Math.min(Math.floor((i + 2) * bucketSize) + 1, data.length);

        // Calculate average of next bucket
        let avgX = 0, avgY = 0, count = 0;
        for (let j = nextBucketStart; j < nextBucketEnd; j++) {
            avgX += j;
            avgY += data[j];
            count++;
        }
        avgX /= count;
        avgY /= count;

        // Find point in current bucket with largest triangle area
        let maxArea = -1;
        let maxIndex = bucketStart;
        const prevX = indices[indices.length - 1];
        const prevY = sampled[sampled.length - 1];

        for (let j = bucketStart; j < bucketEnd && j < data.length; j++) {
            const area = Math.abs(
                (prevX - avgX) * (data[j] - prevY) -
                (prevX - j) * (avgY - prevY)
            );
            if (area > maxArea) {
                maxArea = area;
                maxIndex = j;
            }
        }

        sampled.push(data[maxIndex]);
        indices.push(maxIndex);
    }

    // Always keep last point
    sampled.push(data[data.length - 1]);
    indices.push(data.length - 1);

    return { data: sampled, indices: indices };
}

// Chart downsampling threshold - only downsample above this many points
const CHART_DOWNSAMPLE_THRESHOLD = 2000;
const CHART_DOWNSAMPLE_TARGET = 500;

// Initialize on DOM load
document.addEventListener('DOMContentLoaded', () => {
    initCharts();
    initNNVisualizer();
    connectSocket();
    startScreenshotPolling();
    updateFooterTime();
    setInterval(updateFooterTime, 1000);
    loadConfig();  // This will call initVecEnvsHandler() after config is loaded
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
            },
            zoom: {
                pan: {
                    enabled: true,
                    mode: 'x',
                    modifierKey: null, // Allow panning without modifier key
                    threshold: 10
                },
                zoom: {
                    wheel: {
                        enabled: true,
                        modifierKey: 'ctrl',
                    },
                    pinch: {
                        enabled: true
                    },
                    mode: 'x'
                },
                limits: {
                    x: { min: 'original', max: 'original', minRange: 10 }
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
                    maxTicksLimit: 20
                },
                type: 'linear',
                min: 1
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
                },
                {
                    label: 'Best Avg',
                    data: [],
                    borderColor: '#e91e63',
                    backgroundColor: 'transparent',
                    tension: 0,
                    pointRadius: 0,
                    pointHoverRadius: 4,
                    pointHoverBackgroundColor: '#e91e63',
                    pointHoverBorderColor: '#fff',
                    pointHoverBorderWidth: 2,
                    borderWidth: 2,
                    stepped: 'before'
                }
            ]
        },
        options: chartOptions
    });
    
    // Track pan/zoom events for score chart
    scoreChart.canvas.addEventListener('mousemove', (e) => {
        if (e.buttons === 1) { // Left mouse button held down
            userHasPanned.score = true;
            updateScrollbarPosition('score', scoreChart);
        }
    });
    scoreChart.canvas.addEventListener('wheel', (e) => {
        if (e.ctrlKey || e.metaKey) {
            userHasPanned.score = true;
            setTimeout(() => updateScrollbarPosition('score', scoreChart), 10);
        }
    });
    // Update scrollbar after chart updates (from pan/zoom)
    scoreChart.options.onResize = function() {
        updateScrollbarPosition('score', scoreChart);
    };

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
    
    // Track pan/zoom events for loss chart
    lossChart.canvas.addEventListener('mousemove', (e) => {
        if (e.buttons === 1) { // Left mouse button held down
            userHasPanned.loss = true;
            updateScrollbarPosition('loss', lossChart);
        }
    });
    lossChart.canvas.addEventListener('wheel', (e) => {
        if (e.ctrlKey || e.metaKey) {
            userHasPanned.loss = true;
            setTimeout(() => updateScrollbarPosition('loss', lossChart), 10);
        }
    });
    // Update scrollbar after chart updates (from pan/zoom)
    lossChart.options.onResize = function() {
        updateScrollbarPosition('loss', lossChart);
    };

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
    
    // Track pan/zoom events for Q-value chart
    qvalueChart.canvas.addEventListener('mousemove', (e) => {
        if (e.buttons === 1) { // Left mouse button held down
            userHasPanned.qvalue = true;
            updateScrollbarPosition('qvalue', qvalueChart);
        }
    });
    qvalueChart.canvas.addEventListener('wheel', (e) => {
        if (e.ctrlKey || e.metaKey) {
            userHasPanned.qvalue = true;
            setTimeout(() => updateScrollbarPosition('qvalue', qvalueChart), 10);
        }
    });
    // Update scrollbar after chart updates (from pan/zoom)
    qvalueChart.options.onResize = function() {
        updateScrollbarPosition('qvalue', qvalueChart);
    };
    
    // Initialize scrollbars
    initializeChartScrollbars();
}

/**
 * Initialize scrollbar functionality for all charts
 */
function initializeChartScrollbars() {
    // Setup score chart scrollbar
    setupChartScrollbar('score', scoreChart);
    
    // Setup loss chart scrollbar
    setupChartScrollbar('loss', lossChart);
    
    // Setup Q-value chart scrollbar
    setupChartScrollbar('qvalue', qvalueChart);
}

/**
 * Setup scrollbar for a specific chart
 */
function setupChartScrollbar(chartName, chart) {
    const thumbId = `${chartName}-scrollbar-thumb`;
    const trackId = `${chartName}-scrollbar-track`;
    const thumb = document.getElementById(thumbId);
    const track = document.getElementById(trackId);
    
    if (!thumb || !track || !chart) return;
    
    let dragStartX = 0;
    let dragStartLeft = 0;
    
    // Mouse down on thumb
    thumb.addEventListener('mousedown', (e) => {
        scrollbarDragState[chartName] = true;
        dragStartX = e.clientX;
        dragStartLeft = thumb.offsetLeft;
        e.preventDefault();
        e.stopPropagation();
    });
    
    // Mouse down on track (click to jump)
    track.addEventListener('mousedown', (e) => {
        if (e.target === thumb) return; // Ignore if clicking on thumb
        
        const trackRect = track.getBoundingClientRect();
        const clickX = e.clientX - trackRect.left;
        const trackWidth = trackRect.width;
        const thumbWidth = thumb.offsetWidth || 30;
        const availableWidth = Math.max(1, trackWidth - thumbWidth);
        
        const labels = fullHistory.labels || [];
        if (labels.length === 0) return;
        
        const dataStart = labels[0];
        const dataEnd = labels[labels.length - 1];
        const totalRange = dataEnd - dataStart;
        const visibleWindow = getChartVisibleWindow(chart);
        if (!visibleWindow || totalRange === 0) return;
        
        // Calculate the scrollable range
        const scrollableRange = Math.max(0, totalRange - visibleWindow);
        
        // Convert click position to scroll percent (centering the thumb at click point)
        const targetThumbCenter = clickX - (thumbWidth / 2);
        const scrollPercent = Math.max(0, Math.min(1, targetThumbCenter / availableWidth));
        
        // Calculate viewport bounds
        const viewportStart = dataStart + (scrollableRange * scrollPercent);
        const minX = Math.max(dataStart, viewportStart);
        const maxX = Math.min(dataEnd, viewportStart + visibleWindow);
        
        updateChartViewport(chart, minX, maxX);
        userHasPanned[chartName] = true;
        updateScrollbarPosition(chartName, chart);
        
        e.preventDefault();
    });
    
    // Mouse move (dragging) - use a single document listener per chart
    const handleMouseMove = (e) => {
        if (!scrollbarDragState[chartName]) return;
        
        const trackRect = track.getBoundingClientRect();
        const trackWidth = trackRect.width;
        const thumbWidth = thumb.offsetWidth || 30;
        const availableWidth = Math.max(1, trackWidth - thumbWidth);
        const deltaX = e.clientX - dragStartX;
        const newLeft = Math.max(0, Math.min(availableWidth, dragStartLeft + deltaX));
        
        const labels = fullHistory.labels || [];
        if (labels.length === 0) return;
        
        const dataStart = labels[0];
        const dataEnd = labels[labels.length - 1];
        const totalRange = dataEnd - dataStart;
        const visibleWindow = getChartVisibleWindow(chart);
        if (!visibleWindow || totalRange === 0) return;
        
        // Calculate the scrollable range (total data range minus visible window)
        const scrollableRange = Math.max(0, totalRange - visibleWindow);
        
        // Calculate the scroll position (0 to 1) based on thumb position
        const scrollPercent = availableWidth > 0 ? newLeft / availableWidth : 0;
        
        // Calculate viewport bounds - start position based on scroll percent
        const viewportStart = dataStart + (scrollableRange * scrollPercent);
        const minX = Math.max(dataStart, viewportStart);
        const maxX = Math.min(dataEnd, viewportStart + visibleWindow);
        
        // Update thumb position directly during drag (don't wait for chart update)
        thumb.style.left = `${newLeft}px`;
        
        updateChartViewport(chart, minX, maxX);
        userHasPanned[chartName] = true;
    };
    
    const handleMouseUp = () => {
        if (scrollbarDragState[chartName]) {
            scrollbarDragState[chartName] = false;
            // Sync scrollbar position with actual chart viewport after drag
            updateScrollbarPosition(chartName, chart, true);
        }
    };
    
    document.addEventListener('mousemove', handleMouseMove);
    document.addEventListener('mouseup', handleMouseUp);
    
    // Store cleanup function (for potential future use)
    thumb._cleanup = () => {
        document.removeEventListener('mousemove', handleMouseMove);
        document.removeEventListener('mouseup', handleMouseUp);
    };
    
    // Update scrollbar when chart is panned/zoomed
    chart.canvas.addEventListener('mousemove', () => {
        if (scrollbarDragState[chartName]) return; // Don't update while dragging
        updateScrollbarPosition(chartName, chart);
    });
    
    chart.canvas.addEventListener('wheel', () => {
        if (scrollbarDragState[chartName]) return;
        setTimeout(() => updateScrollbarPosition(chartName, chart), 10);
    });
}

/**
 * Get the visible window size for a chart
 */
function getChartVisibleWindow(chart) {
    if (!chart || !chart.scales || !chart.scales.x) return null;
    
    const labels = fullHistory.labels || [];
    if (labels.length === 0) return null;
    
    const minX = chart.scales.x.min;
    const maxX = chart.scales.x.max;
    
    if (minX === undefined || maxX === undefined) {
        // Showing all data
        return labels[labels.length - 1] - labels[0];
    }
    
    return maxX - minX;
}

/**
 * Update chart viewport with bounds clamping
 */
function updateChartViewport(chart, minX, maxX) {
    if (!chart || !chart.options || !chart.options.scales) return;
    
    const labels = fullHistory.labels || [];
    if (labels.length === 0) return;
    
    const dataStart = labels[0];
    const dataEnd = labels[labels.length - 1];
    
    // Clamp viewport to data bounds, ensuring minimum of 1
    let clampedMinX = Math.max(1, Math.max(dataStart, minX));
    let clampedMaxX = Math.min(dataEnd, maxX);
    
    // Ensure min is not greater than max
    if (clampedMinX > clampedMaxX) {
        clampedMinX = Math.max(1, dataStart);
        clampedMaxX = dataEnd;
    }
    
    // Ensure viewport has some width (at least 1 unit or 1% of data range)
    const minWidth = Math.max(1, (dataEnd - dataStart) * 0.01);
    if (clampedMaxX - clampedMinX < minWidth) {
        const center = (clampedMinX + clampedMaxX) / 2;
        clampedMinX = Math.max(1, Math.max(dataStart, center - minWidth / 2));
        clampedMaxX = Math.min(dataEnd, center + minWidth / 2);
    }
    
    chart.options.scales.x.min = clampedMinX;
    chart.options.scales.x.max = clampedMaxX;
    chart.update('none');
}

/**
 * Update scrollbar position based on chart viewport
 */
function updateScrollbarPosition(chartName, chart, forceUpdate = false) {
    // Don't update scrollbar while user is dragging it (unless forced)
    if (scrollbarDragState[chartName] && !forceUpdate) return;
    
    const thumbId = `${chartName}-scrollbar-thumb`;
    const trackId = `${chartName}-scrollbar-track`;
    const containerId = trackId.replace('-track', '-container');
    const thumb = document.getElementById(thumbId);
    const track = document.getElementById(trackId);
    const container = track?.parentElement;
    
    if (!thumb || !track || !chart) return;
    
    const labels = fullHistory.labels || [];
    if (labels.length === 0) {
        // No data - hide scrollbar
        if (container) container.style.display = 'none';
        return;
    }
    
    const totalRange = labels[labels.length - 1] - labels[0];
    if (totalRange === 0) {
        // Only one data point - hide scrollbar
        if (container) container.style.display = 'none';
        return;
    }
    
    const minX = chart.scales?.x?.min;
    const maxX = chart.scales?.x?.max;
    
    // Get actual visible window
    let visibleWindow;
    if (minX === undefined || maxX === undefined) {
        // Showing all data
        visibleWindow = totalRange;
    } else {
        visibleWindow = maxX - minX;
    }
    
    // Hide scrollbar if visible window covers all data (or more)
    if (visibleWindow >= totalRange * 0.98) {
        if (container) container.style.display = 'none';
        return;
    }
    
    // Show scrollbar
    if (container) container.style.display = 'block';
    thumb.style.display = 'block';
    
    // Calculate thumb position and size
    const trackWidth = track.offsetWidth;
    const visibleRatio = Math.min(1, Math.max(0, visibleWindow / totalRange));
    const thumbWidth = Math.max(30, visibleRatio * trackWidth);
    const availableWidth = Math.max(1, trackWidth - thumbWidth);
    
    // Calculate position based on viewport start position relative to data range
    const dataStart = labels[0];
    const dataEnd = labels[labels.length - 1];
    const viewportStart = (minX !== undefined) ? Math.max(dataStart, Math.min(dataEnd, minX)) : dataStart;
    const scrollableRange = totalRange - visibleWindow;
    
    let positionRatio = 0;
    if (scrollableRange > 0) {
        positionRatio = Math.max(0, Math.min(1, (viewportStart - dataStart) / scrollableRange));
    }
    
    const thumbPosition = positionRatio * availableWidth;
    
    thumb.style.width = `${thumbWidth}px`;
    thumb.style.left = `${Math.max(0, Math.min(trackWidth - thumbWidth, thumbPosition))}px`;
}

/**
 * Reset chart view to show latest data
 */
function resetChartView(chartName = 'all') {
    const charts = {
        score: scoreChart,
        loss: lossChart,
        qvalue: qvalueChart
    };
    
    const visibleWindow = 200;
    const labels = fullHistory.labels || [];
    
    if (labels.length === 0) return;
    
    const minX = Math.max(0, labels.length - visibleWindow);
    const maxX = labels.length;
    
    const resetChart = (chart) => {
        if (!chart || !chart.options || !chart.options.scales) return;
        chart.options.scales.x.min = Math.max(1, labels[minX]);
        chart.options.scales.x.max = labels[maxX - 1];
        chart.update('none');
    };
    
    if (chartName === 'all') {
        resetChart(scoreChart);
        resetChart(lossChart);
        resetChart(qvalueChart);
        userHasPanned.score = false;
        userHasPanned.loss = false;
        userHasPanned.qvalue = false;
        updateScrollbarPosition('score', scoreChart);
        updateScrollbarPosition('loss', lossChart);
        updateScrollbarPosition('qvalue', qvalueChart);
    } else if (charts[chartName]) {
        resetChart(charts[chartName]);
        userHasPanned[chartName] = false;
        if (chartName === 'score') {
            updateScrollbarPosition('score', scoreChart);
        } else if (chartName === 'loss') {
            updateScrollbarPosition('loss', lossChart);
        } else if (chartName === 'qvalue') {
            updateScrollbarPosition('qvalue', qvalueChart);
        }
    }
}

/**
 * Connect to SocketIO server
 */
function connectSocket() {
    socket = io();

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
            // Clear all charts
            if (scoreChart) {
                scoreChart.data.labels = [];
                scoreChart.data.datasets[0].data = [];
                scoreChart.data.datasets[1].data = [];
                scoreChart.data.datasets[2].data = [];
                scoreChart.update('none');
            }
            if (lossChart) {
                lossChart.data.labels = [];
                lossChart.data.datasets[0].data = [];
                lossChart.update('none');
            }
            if (qvalueChart) {
                qvalueChart.data.labels = [];
                qvalueChart.data.datasets[0].data = [];
                qvalueChart.update('none');
            }
            
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

// Track if user has manually panned charts (don't auto-scroll if they have)
let userHasPanned = {
    score: false,
    loss: false,
    qvalue: false
};

// Track scrollbar drag state
let scrollbarDragState = {
    score: false,
    loss: false,
    qvalue: false
};

/**
 * Update charts with history data
 * @param {Object} history - History data with scores, losses, q_values arrays
 * @param {number} currentEpisode - The actual current episode number
 */
function updateCharts(history, currentEpisode) {
    // Store all available history data (up to 500 from backend)
    const allScores = history.scores || [];
    const allLosses = history.losses || [];
    const allQValues = history.q_values || [];
    
    // Update full history storage
    fullHistory.scores = allScores;
    fullHistory.losses = allLosses;
    fullHistory.q_values = allQValues;
    
    // Generate labels for all episodes
    const startEp = Math.max(0, currentEpisode - allScores.length);
    const labels = allScores.map((_, i) => startEp + i + 1);
    fullHistory.labels = labels;
    
    // Calculate running average for all scores
    const avgScores = calculateRunningAverage(allScores, 20);
    
    // Calculate running maximum of averages (peak tracker - can only go UP)
    const bestAvgScores = calculateRunningMax(avgScores);
    
    // Get current viewport ranges for each chart (to maintain user's pan position)
    let scoreXRange = null;
    let lossXRange = null;
    let qvalueXRange = null;
    
    try {
        if (scoreChart && scoreChart.scales && scoreChart.scales.x && scoreChart.scales.x.min !== undefined) {
            scoreXRange = {
                min: scoreChart.scales.x.min,
                max: scoreChart.scales.x.max
            };
        }
        if (lossChart && lossChart.scales && lossChart.scales.x && lossChart.scales.x.min !== undefined) {
            lossXRange = {
                min: lossChart.scales.x.min,
                max: lossChart.scales.x.max
            };
        }
        if (qvalueChart && qvalueChart.scales && qvalueChart.scales.x && qvalueChart.scales.x.min !== undefined) {
            qvalueXRange = {
                min: qvalueChart.scales.x.min,
                max: qvalueChart.scales.x.max
            };
        }
    } catch (e) {
        // Charts not fully initialized yet, ignore
    }
    
    // Update score chart with all data
    scoreChart.data.labels = labels;
    scoreChart.data.datasets[0].data = allScores;
    scoreChart.data.datasets[1].data = avgScores;
    scoreChart.data.datasets[2].data = bestAvgScores;
    
    // Update loss chart with all data
    const validLosses = allLosses.map(l => Math.max(l, 0.0001));
    lossChart.data.labels = labels;
    lossChart.data.datasets[0].data = validLosses;
    
    // Update Q-value chart with all data
    if (allQValues.length > 0) {
        qvalueChart.data.labels = labels;
        qvalueChart.data.datasets[0].data = allQValues;
    }
    
    // Determine if we should auto-scroll to latest (only if user hasn't panned)
    const shouldAutoScroll = allScores.length > 0 && !userHasPanned.score;
    const visibleWindow = 200; // Show last 200 episodes by default
    
    // Update charts with viewport settings
    if (shouldAutoScroll && allScores.length > visibleWindow) {
        // Auto-scroll to show latest data
        const minX = Math.max(0, allScores.length - visibleWindow);
        const maxX = allScores.length;
        
        if (scoreChart && scoreChart.options && scoreChart.options.scales) {
            scoreChart.options.scales.x.min = Math.max(1, labels[minX]);
            scoreChart.options.scales.x.max = labels[maxX - 1];
        }
        if (lossChart && lossChart.options && lossChart.options.scales) {
            lossChart.options.scales.x.min = Math.max(1, labels[minX]);
            lossChart.options.scales.x.max = labels[maxX - 1];
        }
        if (qvalueChart && qvalueChart.options && qvalueChart.options.scales && allQValues.length > 0) {
            qvalueChart.options.scales.x.min = Math.max(1, labels[minX]);
            qvalueChart.options.scales.x.max = labels[maxX - 1];
        }
    } else if (scoreXRange && userHasPanned.score && scoreChart && scoreChart.options && scoreChart.options.scales) {
        // Maintain user's pan position for score chart (clamped to data bounds)
        const dataStart = labels.length > 0 ? labels[0] : 1;
        const dataEnd = labels.length > 0 ? labels[labels.length - 1] : 1;
        scoreChart.options.scales.x.min = Math.max(1, Math.max(dataStart, Math.min(dataEnd, scoreXRange.min)));
        scoreChart.options.scales.x.max = Math.max(dataStart, Math.min(dataEnd, scoreXRange.max));
    }
    
    if (lossXRange && userHasPanned.loss && lossChart && lossChart.options && lossChart.options.scales) {
        // Maintain user's pan position for loss chart (clamped to data bounds)
        const dataStart = labels.length > 0 ? labels[0] : 1;
        const dataEnd = labels.length > 0 ? labels[labels.length - 1] : 1;
        lossChart.options.scales.x.min = Math.max(1, Math.max(dataStart, Math.min(dataEnd, lossXRange.min)));
        lossChart.options.scales.x.max = Math.max(dataStart, Math.min(dataEnd, lossXRange.max));
    }
    
    if (qvalueXRange && userHasPanned.qvalue && qvalueChart && qvalueChart.options && qvalueChart.options.scales) {
        // Maintain user's pan position for Q-value chart (clamped to data bounds)
        const dataStart = labels.length > 0 ? labels[0] : 1;
        const dataEnd = labels.length > 0 ? labels[labels.length - 1] : 1;
        qvalueChart.options.scales.x.min = Math.max(1, Math.max(dataStart, Math.min(dataEnd, qvalueXRange.min)));
        qvalueChart.options.scales.x.max = Math.max(dataStart, Math.min(dataEnd, qvalueXRange.max));
    }
    
    // If no panning and small dataset, show all (but still enforce min of 1)
    if (allScores.length > 0 && allScores.length <= visibleWindow) {
        if (scoreChart && scoreChart.options && scoreChart.options.scales) {
            scoreChart.options.scales.x.min = labels.length > 0 ? Math.max(1, labels[0]) : 1;
            scoreChart.options.scales.x.max = undefined;
        }
        if (lossChart && lossChart.options && lossChart.options.scales) {
            lossChart.options.scales.x.min = labels.length > 0 ? Math.max(1, labels[0]) : 1;
            lossChart.options.scales.x.max = undefined;
        }
        if (qvalueChart && qvalueChart.options && qvalueChart.options.scales && allQValues.length > 0) {
            qvalueChart.options.scales.x.min = labels.length > 0 ? Math.max(1, labels[0]) : 1;
            qvalueChart.options.scales.x.max = undefined;
        }
    }
    
    // Update charts
    scoreChart.update('none');
    lossChart.update('none');
    if (allQValues.length > 0) {
        qvalueChart.update('none');
    }
    
    // Update scrollbar positions
    updateScrollbarPosition('score', scoreChart);
    updateScrollbarPosition('loss', lossChart);
    updateScrollbarPosition('qvalue', qvalueChart);
    
    // Detect if user pans (reset auto-scroll flag)
    // We'll add event listeners for this
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
 * Calculate running maximum (peak tracker)
 * This line can only go UP - shows if we're hitting new peaks
 */
function calculateRunningMax(data) {
    const result = [];
    let runningMax = -Infinity;
    for (let i = 0; i < data.length; i++) {
        runningMax = Math.max(runningMax, data[i]);
        result.push(runningMax);
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

    // Always do full rebuild when filtering (simpler and more reliable)
    // Incremental updates only work well for 'all' filter
    const shouldDoIncremental = currentLogFilter === 'all' && lastRenderedLogCount > 0;
    const newLogsCount = visibleLogs.length - lastRenderedLogCount;
    
    if (shouldDoIncremental && newLogsCount > 0) {
        // Append only new logs (only when showing all logs)
        const newLogs = visibleLogs.slice(-newLogsCount);
        const fragment = document.createDocumentFragment();
        newLogs.forEach(log => {
            const div = document.createElement('div');
            div.className = `console-line ${log.level}`;
            let dataStr = log.data ? `<span class="log-data">${JSON.stringify(log.data)}</span>` : '';
            div.innerHTML = `
                <span class="log-time">${log.time}</span>
                <span class="log-level">${log.level.toUpperCase()}</span>
                <span class="log-message">${escapeHtml(log.message)}</span>
                ${dataStr}
            `;
            fragment.appendChild(div);
        });
        container.appendChild(fragment);

        // Limit DOM children to 100
        while (container.children.length > 100) {
            container.removeChild(container.firstChild);
        }
        lastRenderedLogCount = visibleLogs.length;
    } else {
        // Full rebuild (first render, filter change, or log trimmed)
        container.innerHTML = visibleLogs.map(log => {
            let dataStr = '';
            if (log.data) {
                dataStr = `<span class="log-data">${escapeHtml(JSON.stringify(log.data))}</span>`;
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
        lastRenderedLogCount = visibleLogs.length;
    }

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

    // Reset render count when filter changes to force full rebuild
    lastRenderedLogCount = 0;
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
    // Always use simple format for now - just copy the filtered logs
    copyLogsSimple();
}

/**
 * Copy logs in simple format (original behavior)
 */
function copyLogsSimple() {
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
 * Start fresh training - reset agent and clear all training state
 */
function startFresh() {
    // First, ask if they want to save current progress
    const saveFirst = confirm(
        '⚠️ Start Fresh Training\n\n' +
        'Would you like to SAVE your current progress first?\n\n' +
        'Click OK to save before resetting\n' +
        'Click Cancel to skip saving'
    );
    
    // Now confirm the fresh start
    const confirmReset = confirm(
        '🔄 Confirm Fresh Start\n\n' +
        'This will:\n' +
        '• Reset the neural network to random weights\n' +
        '• Clear all training memory (replay buffer)\n' +
        '• Reset episode count, scores, and charts to 0\n' +
        '• Clear console logs\n\n' +
        '✓ Saved models on disk will NOT be deleted\n' +
        '✓ You can load them later from the Load menu\n\n' +
        'Continue with fresh start?'
    );
    
    if (!confirmReset) {
        addConsoleLog('Fresh start cancelled', 'info');
        return;
    }
    
    // Update button to show loading state
    const btn = document.querySelector('.control-btn.fresh');
    if (btn) {
        const originalText = btn.textContent;
        btn.textContent = '⏳ Resetting...';
        btn.disabled = true;
        setTimeout(() => {
            btn.textContent = originalText;
            btn.disabled = false;
        }, 2000);
    }
    
    if (saveFirst) {
        // Save current model first, then reset after a short delay
        socket.emit('control', { action: 'save' });
        addConsoleLog('💾 Saving current progress before reset...', 'action');
        
        // Wait for save to complete before resetting
        setTimeout(() => {
            socket.emit('control', { action: 'start_fresh' });
            addConsoleLog('🔄 Starting fresh training...', 'warning');
        }, 500);
    } else {
        socket.emit('control', { action: 'start_fresh' });
        addConsoleLog('🔄 Starting fresh training...', 'warning');
    }
}

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
 * Returns true if polling should continue, false if headless mode detected
 */
function refreshScreenshot() {
    return fetchWithTimeout('/api/screenshot')
        .then(response => response.json())
        .then(data => {
            const img = document.getElementById('game-preview');
            const placeholder = document.getElementById('preview-placeholder');
            
            // Check if headless mode - stop polling and show headless placeholder
            if (data.headless) {
                if (placeholder) {
                    placeholder.innerHTML = '🚀 Headless Mode<br><span style="font-size: 0.8em; opacity: 0.7;">No preview available</span>';
                    placeholder.classList.remove('hidden');
                }
                if (img) img.classList.remove('visible');
                return false; // Signal to stop polling
            }
            
            if (data.image && data.image.length > 0) {
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
                if (placeholder) {
                    placeholder.innerHTML = '🎮 Game Preview';
                    placeholder.classList.remove('hidden');
                }
                if (img) img.classList.remove('visible');
            }
            return true; // Continue polling
        })
        .catch(err => {
            if (err.name !== 'AbortError') {
                console.error('Screenshot fetch error:', err);
            }
            const placeholder = document.getElementById('preview-placeholder');
            if (placeholder) placeholder.classList.remove('hidden');
            return true; // Continue polling on error (might recover)
        });
}

/**
 * Start polling for screenshots
 * Uses a singleton pattern to prevent duplicate listeners on reconnection
 */
let screenshotPollingInitialized = false;
let screenshotIntervalId = null;
let screenshotIsHeadless = false;

function startScreenshotPolling() {
    // Prevent duplicate initialization on reconnection
    if (screenshotPollingInitialized) {
        // Just restart polling if not headless
        if (!screenshotIsHeadless && !screenshotIntervalId) {
            startScreenshotPollingInternal();
        }
        return;
    }
    screenshotPollingInitialized = true;

    // Function to check screenshot and handle headless mode
    const checkScreenshot = async () => {
        const shouldContinue = await refreshScreenshot();
        if (!shouldContinue) {
            // Headless mode detected - stop polling
            screenshotIsHeadless = true;
            stopScreenshotPollingInternal();
            console.log('Headless mode detected - screenshot polling disabled');
        }
    };

    // Use Page Visibility API to pause polling when tab is hidden (only add once)
    document.addEventListener('visibilitychange', () => {
        if (document.hidden) {
            stopScreenshotPollingInternal();
        } else if (!screenshotIsHeadless) {
            startScreenshotPollingInternal();
        }
    });

    // Internal start function
    function startScreenshotPollingInternal() {
        if (!screenshotIntervalId && !screenshotIsHeadless) {
            checkScreenshot();
            screenshotIntervalId = setInterval(checkScreenshot, 2000);
        }
    }

    // Internal stop function
    function stopScreenshotPollingInternal() {
        if (screenshotIntervalId) {
            clearInterval(screenshotIntervalId);
            screenshotIntervalId = null;
        }
    }

    // Export internal functions to module scope
    window.startScreenshotPollingInternal = startScreenshotPollingInternal;
    window.stopScreenshotPollingInternal = stopScreenshotPollingInternal;

    // Start polling initially
    startScreenshotPollingInternal();
}

function stopScreenshotPollingInternal() {
    if (screenshotIntervalId) {
        clearInterval(screenshotIntervalId);
        screenshotIntervalId = null;
    }
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
                    <div class="model-item">
                        <div class="model-item-content" onclick="loadModel('${safePathForJs}')">
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
                        <button class="model-delete-btn" onclick="event.stopPropagation(); deleteModel('${safePathForJs}', '${safeName}')" title="Delete this model">
                            🗑️
                        </button>
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

/**
 * Delete a model file
 */
function deleteModel(path, name) {
    if (!confirm(`Are you sure you want to delete "${name}"?\n\nThis action cannot be undone.`)) {
        return;
    }
    
    // Encode path for URL (handle special characters)
    const encodedPath = encodeURIComponent(path);
    
    fetchWithTimeout(`/api/models/${encodedPath}`, {
        method: 'DELETE',
        headers: {
            'Content-Type': 'application/json'
        }
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            addConsoleLog(`🗑️ Deleted model: ${name}`, 'action');
            // Refresh the model list
            showLoadModal();
        } else {
            addConsoleLog(`❌ Failed to delete model: ${data.error || 'Unknown error'}`, 'error');
        }
    })
    .catch(err => {
        addConsoleLog(`❌ Error deleting model: ${err.message}`, 'error');
    });
}

// Close modal on outside click (click on backdrop, not content)
document.addEventListener('click', (e) => {
    // Check if click was directly on an element with 'modal' class
    if (e.target.classList && e.target.classList.contains('modal')) {
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
            if ((e.ctrlKey || e.metaKey) && e.shiftKey) {
                // Ctrl/Cmd+Shift+R = Start Fresh (destructive, needs extra modifier)
                e.preventDefault();
                startFresh();
            } else if (e.ctrlKey || e.metaKey) {
                // Ctrl/Cmd+R = Reset Episode
                e.preventDefault(); // Prevent browser refresh
                resetEpisode();
            }
            break;
        case 'q':
            if (e.ctrlKey || e.metaKey) {
                e.preventDefault();
                saveAndQuit();
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
        case '4':
            setPerformanceMode('ultra');
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
        'turbo': 'Turbo (learn every 8, batch 128, 2 grad steps)',
        'ultra': 'Ultra (learn every 32, batch 128, 2 grad steps)'
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
    } else if (mode === 'ultra') {
        // Maximum throughput: 4x less learning than turbo
        learnEvery = 32;
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

    // Visual feedback
    const btn = document.querySelector('.apply-btn');
    const originalText = btn.textContent;
    btn.textContent = '✓ Applied!';
    setTimeout(() => {
        btn.textContent = originalText;
    }, 1500);
}

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

            // Vec-envs setting
            if (data.vec_envs) {
                originalVecEnvs = data.vec_envs;
                const vecEnvsInput = document.getElementById('setting-vec-envs');
                if (vecEnvsInput) {
                    vecEnvsInput.value = originalVecEnvs;
                }
            }
            
            // Initialize vec-envs handler AFTER originalVecEnvs is set from server
            initVecEnvsHandler();

            // Update system status from config
            if (data.device) {
                updateSystemStatus({ device: data.device, torch_compiled: false });
            }
        })
        .catch(err => {
            if (err.name !== 'AbortError') {
                console.error('Config load error:', err);
            }
            // Initialize vec-envs handler even on failure (uses default originalVecEnvs=1)
            initVecEnvsHandler();
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
        // Set base classes only (don't inject unsanitized data into className)
        reasonEl.className = 'save-value save-reason';
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

/**
 * Go back to game launcher to select a different game/mode
 */
function goToLauncher() {
    const confirmed = confirm(
        'Go back to Game Launcher?\n\n' +
        'This will stop the current session. Your progress has been auto-saved.'
    );

    if (confirmed) {
        addConsoleLog('🎮 Returning to launcher...', 'warning');
        // Tell server to switch back to launcher mode
        socket.emit('control', { action: 'go_to_launcher' });
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
            <button onclick="copyRestartCommand(event)" style="
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
function copyRestartCommand(event) {
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
    
    // Sanitize game name to prevent XSS
    const safeGameName = escapeHtml(game.replace('_', ' ').toUpperCase());
    overlay.innerHTML = `
        <div style="font-size: 4rem; margin-bottom: 20px; animation: pulse 1s ease-in-out infinite;">🔄</div>
        <h2 style="color: #00d4ff; margin: 0 0 15px 0; font-size: 1.5rem;">Restarting with ${safeGameName}</h2>
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

        // Cleanup tracking for memory leak prevention
        this.resizeObserver = null;
        this.clickHandler = null;

        // Phase 1.2 & 1.4: Adaptive rendering
        this.renderInterval = 33;  // Default ~30Hz, will adapt based on training speed
        this.lastRenderTime = 0;
        this.avgRenderTime = 0;
        this.renderQueue = [];
        this.skipNextWeightDraw = false;  // Phase 1.3: Skip weights when empty
        this.cachedWeights = null;  // Phase 1.3: Cache for weight data to handle selective transmission

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

        // Phase 2: Setup mouse handling for neuron inspection
        this.setupMouseHandling();
    }
    
    setupResize() {
        // Clean up old observer to prevent memory leak
        if (this.resizeObserver) {
            this.resizeObserver.disconnect();
        }

        this.resizeObserver = new ResizeObserver(() => {
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
            this.resizeObserver.observe(this.canvas.parentElement);
        }
    }

    // ===== Phase 2: Neuron Inspection =====

    setupMouseHandling() {
        // Phase 2: Set up click handlers for neuron and layer inspection
        if (!this.canvas) return;

        // Remove old handler to prevent memory leak
        if (this.clickHandler) {
            this.canvas.removeEventListener('click', this.clickHandler);
        }

        this.clickHandler = (e) => {
            const rect = this.canvas.getBoundingClientRect();
            const x = e.clientX - rect.left;
            const y = e.clientY - rect.top;

            // Check if click is on a layer label
            const layerClicked = this.findLayerLabelAtPosition(x, y);
            if (layerClicked !== null) {
                this.selectLayer(layerClicked);
                return;
            }

            // Check if click is on a neuron
            const neuronClicked = this.findNeuronAtPosition(x, y);
            if (neuronClicked) {
                this.selectNeuron(neuronClicked.layerIdx, neuronClicked.neuronIdx);
            }
        };

        this.canvas.addEventListener('click', this.clickHandler);
    }

    findNeuronAtPosition(clickX, clickY) {
        // Phase 2: Find neuron at mouse position
        if (!this.data || !this.width || !this.height) return null;

        const layerPositions = this.calculateLayerPositions();
        if (!layerPositions) return null;

        const clickRadius = 12; // Larger than neuron for easier clicking

        for (let layerIdx = 0; layerIdx < layerPositions.length; layerIdx++) {
            const layer = layerPositions[layerIdx];
            for (let neuronIdx = 0; neuronIdx < layer.positions.length; neuronIdx++) {
                const pos = layer.positions[neuronIdx];
                const dist = Math.sqrt((clickX - pos.x) ** 2 + (clickY - pos.y) ** 2);
                if (dist < clickRadius) {
                    return { layerIdx, neuronIdx };
                }
            }
        }
        return null;
    }

    selectNeuron(layerIdx, neuronIdx) {
        // Phase 2: Load and display neuron inspection panel
        // Fetch neuron details from backend
        fetch(`/api/neuron/${layerIdx}/${neuronIdx}`)
            .then(response => response.json())
            .then(data => {
                this.displayNeuronInspection(data);
            })
            .catch(err => console.error('Failed to load neuron details:', err));
    }

    displayNeuronInspection(neuronData) {
        // Phase 2: Display neuron inspection panel
        const panel = document.getElementById('neuron-inspection-panel');
        if (!panel) {
            console.warn('Neuron inspection panel not found');
            return;
        }

        // Build HTML for neuron details
        const html = `
            <div class="neuron-header">
                <h3>${neuronData.layer_name} - Neuron #${neuronData.neuron_idx}</h3>
                <button class="close-btn" onclick="closeNeuronInspection()">×</button>
            </div>

            <div class="neuron-content">
                <div class="stat-group">
                    <h4>Activation</h4>
                    <div class="stat-value">${neuronData.current_activation.toFixed(4)}</div>
                    <div class="stat-bar">
                        <div class="stat-fill" style="width: ${Math.abs(neuronData.current_activation) * 100}%"></div>
                    </div>
                </div>

                <div class="stat-group">
                    <h4>Incoming Weights (from layer ${neuronData.layer_idx - 1})</h4>
                    ${neuronData.incoming_weight_stats ? `
                        <div class="weight-stats">
                            <div>Mean: ${neuronData.incoming_weight_stats.mean?.toFixed(4) || 'N/A'}</div>
                            <div>Range: [${neuronData.incoming_weight_stats.min?.toFixed(4) || 'N/A'},
                                         ${neuronData.incoming_weight_stats.max?.toFixed(4) || 'N/A'}]</div>
                        </div>
                    ` : '<div>No data</div>'}
                </div>

                <div class="stat-group">
                    <h4>Outgoing Weights (to next layer)</h4>
                    ${neuronData.outgoing_weight_stats ? `
                        <div class="weight-stats">
                            <div>Mean: ${neuronData.outgoing_weight_stats.mean?.toFixed(4) || 'N/A'}</div>
                            <div>Range: [${neuronData.outgoing_weight_stats.min?.toFixed(4) || 'N/A'},
                                         ${neuronData.outgoing_weight_stats.max?.toFixed(4) || 'N/A'}]</div>
                        </div>
                    ` : '<div>No data</div>'}
                </div>

                <div class="stat-group">
                    <h4>Q-Value Contributions</h4>
                    ${Object.entries(neuronData.q_value_contributions || {}).map(([action, contrib]) => `
                        <div class="q-contrib">
                            <span>${action}:</span>
                            <span>${contrib.toFixed(4)}</span>
                        </div>
                    `).join('')}
                </div>

                ${neuronData.activation_history?.length > 0 ? `
                    <div class="stat-group">
                        <h4>Recent Activation History (${neuronData.activation_history.length} samples)</h4>
                        <div class="sparkline-container">
                            <canvas id="neuron-sparkline" width="300" height="40"></canvas>
                        </div>
                    </div>
                ` : ''}
            </div>
        `;

        panel.innerHTML = html;
        panel.style.display = 'block';

        // Draw sparkline if data available
        if (neuronData.activation_history && neuronData.activation_history.length > 0) {
            this.drawSparkline(document.getElementById('neuron-sparkline'), neuronData.activation_history);
        }
    }

    drawSparkline(canvas, data) {
        // Phase 2: Draw activation history sparkline
        if (!canvas) return;

        const ctx = canvas.getContext('2d');
        const width = canvas.width;
        const height = canvas.height;
        const padding = 5;

        ctx.clearRect(0, 0, width, height);

        if (data.length < 2) {
            ctx.fillStyle = '#999';
            ctx.fillText('Insufficient data', 10, height / 2);
            return;
        }

        // Find min/max for scaling
        const min = Math.min(...data);
        const max = Math.max(...data);
        const range = max - min || 1;

        // Draw area under curve
        ctx.fillStyle = 'rgba(100, 150, 255, 0.3)';
        ctx.beginPath();
        ctx.moveTo(padding, height - padding);

        for (let i = 0; i < data.length; i++) {
            const x = padding + (i / (data.length - 1)) * (width - 2 * padding);
            const y = height - padding - ((data[i] - min) / range) * (height - 2 * padding);
            if (i === 0) ctx.lineTo(x, y);
            else ctx.lineTo(x, y);
        }

        ctx.lineTo(width - padding, height - padding);
        ctx.closePath();
        ctx.fill();

        // Draw line
        ctx.strokeStyle = '#4287f5';
        ctx.lineWidth = 2;
        ctx.beginPath();

        for (let i = 0; i < data.length; i++) {
            const x = padding + (i / (data.length - 1)) * (width - 2 * padding);
            const y = height - padding - ((data[i] - min) / range) * (height - 2 * padding);
            if (i === 0) ctx.moveTo(x, y);
            else ctx.lineTo(x, y);
        }

        ctx.stroke();
    }

    // ===== Phase 2: Layer Analysis =====

    findLayerLabelAtPosition(clickX, clickY) {
        // Phase 2: Find layer label at mouse position
        if (!this.data || !this.data.layer_info) return null;

        const layerPositions = this.calculateLayerPositions();
        if (!layerPositions) return null;

        const labelY = this.headerHeight + 5;
        const clickRadius = 20; // Larger click area for labels

        for (let i = 0; i < layerPositions.length; i++) {
            const layerPos = layerPositions[i];
            const distX = Math.abs(clickX - layerPos.x);
            const distY = Math.abs(clickY - labelY);

            // Check if click is within bounds of layer label
            if (distX < clickRadius && distY < clickRadius) {
                return i;
            }
        }
        return null;
    }

    selectLayer(layerIdx) {
        // Phase 2: Load and display layer analysis panel
        // Fetch layer analysis from backend
        fetch(`/api/layer/${layerIdx}`)
            .then(response => response.json())
            .then(data => {
                this.displayLayerAnalysis(data);
            })
            .catch(err => console.error('Failed to load layer analysis:', err));
    }

    displayLayerAnalysis(layerData) {
        // Phase 2: Display layer analysis panel
        const panel = document.getElementById('layer-analysis-panel');
        if (!panel) {
            console.warn('Layer analysis panel not found');
            return;
        }

        // Determine health status based on dead/saturated neurons
        const deadPercent = layerData.dead_neuron_percent || 0;
        const saturatedPercent = layerData.saturated_percent || 0;
        let healthStatus = '✓ Healthy';
        let healthColor = 'var(--accent-success)';

        if (deadPercent > 10 || saturatedPercent > 50) {
            healthStatus = '⚠ Warning';
            healthColor = 'var(--accent-warning)';
        }
        if (deadPercent > 30 || saturatedPercent > 80) {
            healthStatus = '✗ Critical';
            healthColor = 'var(--accent-danger)';
        }

        // Build HTML for layer details
        const html = `
            <div class="layer-header">
                <h3>${layerData.layer_name || `Layer ${layerData.layer_idx}`}</h3>
                <button class="close-btn" onclick="closeLayerAnalysis()">×</button>
            </div>

            <div class="layer-content">
                <!-- Health Status -->
                <div class="layer-stat-group">
                    <span class="layer-stat-group-title">Network Health</span>
                    <div class="layer-stat-row">
                        <span class="layer-stat-label">Status:</span>
                        <span class="layer-stat-value" style="color: ${healthColor}">${healthStatus}</span>
                    </div>
                    <div class="layer-stat-row">
                        <span class="layer-stat-label">Neurons:</span>
                        <span class="layer-stat-value">${layerData.neuron_count || 0}</span>
                    </div>
                </div>

                <!-- Activation Statistics -->
                <div class="layer-stat-group">
                    <span class="layer-stat-group-title">Activation Stats</span>
                    <div class="layer-stat-row">
                        <span class="layer-stat-label">Mean:</span>
                        <span class="layer-stat-value">${(layerData.avg_activation || 0).toFixed(4)}</span>
                    </div>
                    <div class="layer-stat-row">
                        <span class="layer-stat-label">Std Dev:</span>
                        <span class="layer-stat-value">${(layerData.activation_std || 0).toFixed(4)}</span>
                    </div>
                    <div class="layer-stat-row">
                        <span class="layer-stat-label">Dead Neurons:</span>
                        <span class="layer-stat-value" style="color: ${deadPercent > 10 ? 'var(--accent-warning)' : 'inherit'}">
                            ${layerData.dead_neuron_count || 0} (${deadPercent.toFixed(1)}%)
                        </span>
                    </div>
                    <div class="layer-stat-row">
                        <span class="layer-stat-label">Saturated:</span>
                        <span class="layer-stat-value" style="color: ${saturatedPercent > 50 ? 'var(--accent-warning)' : 'inherit'}">
                            ${layerData.saturated_neuron_count || 0} (${saturatedPercent.toFixed(1)}%)
                        </span>
                    </div>
                </div>

                <!-- Weight Statistics -->
                <div class="layer-stat-group">
                    <span class="layer-stat-group-title">Weight Distribution</span>
                    <div class="weight-stats-container">
                        <div class="weight-stat">
                            <div class="weight-stat-label">Mean</div>
                            <div class="weight-stat-value">${(layerData.weight_mean || 0).toFixed(4)}</div>
                        </div>
                        <div class="weight-stat">
                            <div class="weight-stat-label">Std</div>
                            <div class="weight-stat-value">${(layerData.weight_std || 0).toFixed(4)}</div>
                        </div>
                    </div>
                </div>

                <!-- Gradient Statistics -->
                <div class="layer-stat-group">
                    <span class="layer-stat-group-title">Gradient Flow</span>
                    <div class="layer-stat-row">
                        <span class="layer-stat-label">Mean Magnitude:</span>
                        <span class="layer-stat-value">${(layerData.gradient_mean || 0).toFixed(6)}</span>
                    </div>
                    <div class="layer-stat-row">
                        <span class="layer-stat-label">Std Dev:</span>
                        <span class="layer-stat-value">${(layerData.gradient_std || 0).toFixed(6)}</span>
                    </div>
                    <div class="layer-stat-row">
                        <span class="layer-stat-label">Max Magnitude:</span>
                        <span class="layer-stat-value">${(layerData.gradient_max_magnitude || 0).toFixed(6)}</span>
                    </div>
                </div>
            </div>
        `;

        panel.innerHTML = html;
        panel.style.display = 'block';
    }

    startAnimation() {
        // Stop any existing animation to prevent duplicate loops
        this.stopAnimation();

        const animate = (currentTime) => {
            // Phase 1.4: Adaptive render throttling
            // Only render if enough time has passed since last render
            if (this.isEnabled && this.data) {
                const timeSinceLastRender = currentTime - this.lastRenderTime;
                if (timeSinceLastRender >= this.renderInterval) {
                    const renderStart = performance.now();
                    this.render();
                    const renderDuration = performance.now() - renderStart;

                    // Track render performance and adjust interval if struggling
                    this.avgRenderTime = this.avgRenderTime * 0.8 + renderDuration * 0.2;
                    if (this.avgRenderTime > this.renderInterval * 0.8) {
                        // Struggling to keep up - increase render interval (reduce FPS)
                        this.renderInterval = Math.min(100, this.renderInterval + 5);
                    } else if (this.avgRenderTime < this.renderInterval * 0.3 && this.renderInterval > 16) {
                        // Running smoothly - can afford to render more frequently
                        this.renderInterval = Math.max(16, this.renderInterval - 2);
                    }

                    this.lastRenderTime = currentTime;
                }
            }
            this.animationId = requestAnimationFrame(animate);
        };
        animate(performance.now());
    }
    
    stopAnimation() {
        if (this.animationId) {
            cancelAnimationFrame(this.animationId);
            this.animationId = null;
        }
    }

    /**
     * Clean up all resources to prevent memory leaks.
     * Call this before creating a new visualizer or when disabling.
     */
    destroy() {
        // Stop animation loop
        this.stopAnimation();

        // Clean up ResizeObserver
        if (this.resizeObserver) {
            this.resizeObserver.disconnect();
            this.resizeObserver = null;
        }

        // Clean up click handler
        if (this.clickHandler && this.canvas) {
            this.canvas.removeEventListener('click', this.clickHandler);
            this.clickHandler = null;
        }

        // Clear data references
        this.data = null;
        this.cachedWeights = null;
    }

    update(data) {
        if (!data || !data.layer_info || data.layer_info.length === 0) {
            return;
        }
        this.data = data;

        // Phase 1.3: Cache weights if provided (selective transmission)
        // This ensures we have valid weight data even when backend sends empty arrays
        if (data.weights && data.weights.length > 0) {
            this.cachedWeights = data.weights;
        }

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

        // Phase 1.3: Use cached weights if current data doesn't have them
        // This handles selective transmission where backend sends empty arrays every 100 steps
        const weights = (this.data.weights && this.data.weights.length > 0)
            ? this.data.weights
            : this.cachedWeights;

        // Only skip if we've never received weights
        if (!weights || weights.length === 0) {
            return;
        }

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
 * Phase 1.2: Update NN visualizer render rate based on training speed.
 *
 * High-speed training: Reduce render FPS to save resources
 * Slow training: Increase render FPS for smooth visuals
 */
function updateNNVisualizerRenderRate(stepsPerSec) {
    if (!nnVisualizer) return;

    if (stepsPerSec > 2000) {
        // Very high speed training - render at 10Hz
        nnVisualizer.renderInterval = 100;
    } else if (stepsPerSec > 1000) {
        // High speed - render at ~15Hz
        nnVisualizer.renderInterval = 67;
    } else if (stepsPerSec > 500) {
        // Medium speed - render at ~30Hz
        nnVisualizer.renderInterval = 33;
    } else {
        // Slow training or visual mode - render at up to 60Hz for smoothness
        nnVisualizer.renderInterval = 16;
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

/**
 * Phase 2: Close neuron inspection panel
 */
function closeNeuronInspection() {
    const panel = document.getElementById('neuron-inspection-panel');
    if (panel) {
        panel.style.display = 'none';
        panel.innerHTML = '';
    }
}

function closeLayerAnalysis() {
    const panel = document.getElementById('layer-analysis-panel');
    if (panel) {
        panel.style.display = 'none';
        panel.innerHTML = '';
    }
}
