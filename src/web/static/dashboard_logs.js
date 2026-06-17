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
 * Create a console log row without interpolating untrusted HTML.
 */
function createConsoleLogElement(log) {
    const line = document.createElement('div');
    const level = String(log.level || 'info').replace(/[^a-z-]/gi, '').toLowerCase() || 'info';
    line.className = `console-line ${level}`;

    const time = document.createElement('span');
    time.className = 'log-time';
    time.textContent = log.time || '';
    line.appendChild(time);

    const levelLabel = document.createElement('span');
    levelLabel.className = 'log-level';
    levelLabel.textContent = level.toUpperCase();
    line.appendChild(levelLabel);

    const message = document.createElement('span');
    message.className = 'log-message';
    message.textContent = log.message || '';
    line.appendChild(message);

    if (log.data) {
        const data = document.createElement('span');
        data.className = 'log-data';
        data.textContent = JSON.stringify(log.data);
        line.appendChild(data);
    }

    return line;
}

/**
 * Render console logs based on current filter
 */
function renderConsoleLogs() {
    const container = document.getElementById('console-output');
    if (!container) return;

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
            fragment.appendChild(createConsoleLogElement(log));
        });
        container.appendChild(fragment);

        // Limit DOM children to 100
        while (container.children.length > 100) {
            container.removeChild(container.firstChild);
        }
        lastRenderedLogCount = visibleLogs.length;
    } else {
        // Full rebuild (first render, filter change, or log trimmed)
        container.replaceChildren(...visibleLogs.map(createConsoleLogElement));
        lastRenderedLogCount = visibleLogs.length;
    }

    // Auto-scroll to bottom
    const consoleContainer = document.getElementById('console-container');
    if (consoleContainer) {
        consoleContainer.scrollTop = consoleContainer.scrollHeight;
    }
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
    if (socket && typeof socket.emit === 'function') {
        socket.emit('clear_logs', {});
    }
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
    return DashboardCore.escapeHtml(text);
}

function escapeHtmlAttribute(text) {
    return DashboardCore.escapeHtmlAttribute(text);
}
