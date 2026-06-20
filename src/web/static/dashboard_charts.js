/**
 * Dashboard chart management.
 */
(function(root) {
    let addConsoleLogCallback = null;

    function addChartLog(message, level) {
        if (typeof addConsoleLogCallback === 'function') {
            addConsoleLogCallback(message, level);
        }
    }

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

    /**
     * Initialize Chart.js charts
     */
    function initCharts() {
        if (typeof Chart === 'undefined') {
            console.error('Chart.js failed to load; charts are disabled.');
            addChartLog('Charts unavailable: Chart.js failed to load', 'error');
            return false;
        }

        const scoreCanvas = document.getElementById('scoreChart');
        const lossCanvas = document.getElementById('lossChart');
        const qvalueCanvas = document.getElementById('qvalueChart');
        if (!scoreCanvas || !lossCanvas || !qvalueCanvas) {
            console.error('Chart canvases are missing; charts are disabled.');
            addChartLog('Charts unavailable: chart canvases are missing', 'error');
            return false;
        }

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
        const scoreCtx = scoreCanvas.getContext('2d');
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
        const lossCtx = lossCanvas.getContext('2d');
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
        const qvalueCtx = qvalueCanvas.getContext('2d');
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
        return true;
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
        if (!scoreChart || !lossChart || !qvalueChart) {
            return;
        }

        const chartModel = DashboardCore.buildChartUpdateModel(history, currentEpisode);
        const allScores = chartModel.scores;
        const allQValues = chartModel.qValues;
        const labels = chartModel.labels;

        // Update full history storage
        fullHistory.scores = allScores;
        fullHistory.losses = chartModel.losses;
        fullHistory.q_values = allQValues;
        fullHistory.labels = labels;

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
        scoreChart.data.datasets[1].data = chartModel.averageScores;
        scoreChart.data.datasets[2].data = chartModel.bestAverageScores;

        // Update loss chart with all data
        lossChart.data.labels = labels;
        lossChart.data.datasets[0].data = chartModel.losses;

        // Update Q-value chart with all data
        if (allQValues.length > 0) {
            qvalueChart.data.labels = labels;
            qvalueChart.data.datasets[0].data = allQValues;
        }

        // Determine if we should auto-scroll to latest (only if user hasn't panned)
        const shouldAutoScroll = allScores.length > 0 && !userHasPanned.score;

        // Update charts with viewport settings
        if (shouldAutoScroll && chartModel.autoScrollRange) {
            // Auto-scroll to show latest data
            if (scoreChart && scoreChart.options && scoreChart.options.scales) {
                scoreChart.options.scales.x.min = chartModel.autoScrollRange.min;
                scoreChart.options.scales.x.max = chartModel.autoScrollRange.max;
            }
            if (lossChart && lossChart.options && lossChart.options.scales) {
                lossChart.options.scales.x.min = chartModel.autoScrollRange.min;
                lossChart.options.scales.x.max = chartModel.autoScrollRange.max;
            }
            if (qvalueChart && qvalueChart.options && qvalueChart.options.scales && allQValues.length > 0) {
                qvalueChart.options.scales.x.min = chartModel.autoScrollRange.min;
                qvalueChart.options.scales.x.max = chartModel.autoScrollRange.max;
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
        if (chartModel.showAllRange) {
            if (scoreChart && scoreChart.options && scoreChart.options.scales) {
                scoreChart.options.scales.x.min = chartModel.showAllRange.min;
                scoreChart.options.scales.x.max = chartModel.showAllRange.max;
            }
            if (lossChart && lossChart.options && lossChart.options.scales) {
                lossChart.options.scales.x.min = chartModel.showAllRange.min;
                lossChart.options.scales.x.max = chartModel.showAllRange.max;
            }
            if (qvalueChart && qvalueChart.options && qvalueChart.options.scales && allQValues.length > 0) {
                qvalueChart.options.scales.x.min = chartModel.showAllRange.min;
                qvalueChart.options.scales.x.max = chartModel.showAllRange.max;
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

    function clearCharts() {
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
        fullHistory = {
            scores: [],
            losses: [],
            q_values: [],
            epsilons: [],
            labels: []
        };
        userHasPanned = { score: false, loss: false, qvalue: false };
    }

    function resizeCharts() {
        [scoreChart, lossChart, qvalueChart].forEach((chart) => {
            if (chart && typeof chart.resize === 'function') {
                chart.resize();
            }
        });
        updateScrollbarPosition('score', scoreChart);
        updateScrollbarPosition('loss', lossChart);
        updateScrollbarPosition('qvalue', qvalueChart);
    }

    if (typeof window !== 'undefined') {
        let resizeTimer = null;
        window.addEventListener('resize', () => {
            clearTimeout(resizeTimer);
            resizeTimer = setTimeout(resizeCharts, 100);
        });
    }

    function configure(options = {}) {
        addConsoleLogCallback = options.addConsoleLog || addConsoleLogCallback;
    }

    const api = {
        configure,
        initCharts(options = {}) {
            configure(options);
            return initCharts();
        },
        updateCharts,
        resetChartView,
        clearCharts,
        resizeCharts,
        downsampleLTTB,
    };

    root.DashboardCharts = api;
    if (typeof module !== 'undefined' && module.exports) {
        module.exports = api;
    }
})(typeof window !== 'undefined' ? window : globalThis);
