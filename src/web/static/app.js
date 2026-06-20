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

const MAX_CONSOLE_LOGS = 500;
const DASHBOARD_TOKEN = (
    typeof DashboardCore !== 'undefined' && typeof document !== 'undefined'
) ? DashboardCore.readToken(document) : '';

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
    'toggle-nn-connections': () => toggleNNConnections(),
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

    reportAssetFailures();
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

function reportAssetFailures() {
    const root = typeof window !== 'undefined' ? window : globalThis;
    const failures = Array.isArray(root.DASHBOARD_ASSET_FAILURES)
        ? root.DASHBOARD_ASSET_FAILURES
        : [];
    const uniqueFailures = [...new Set(failures)];
    uniqueFailures.forEach((assetName) => {
        addConsoleLog(`${assetName} failed to load; dashboard will run in reduced mode`, 'warning');
    });
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

// Crystal Caves end-reason → plain-English display (icon, label, sentiment).
const CC_OUTCOMES = {
    won: { icon: '🏆', label: 'won', cls: 'good' },
    killed: { icon: '☠️', label: 'killed', cls: 'bad' },
    timeout: { icon: '⏱️', label: 'timeout', cls: 'warn' },
    stalled: { icon: '🛑', label: 'stalled', cls: 'warn' },
    ended: { icon: '⛳', label: 'ended', cls: '' },
};

function updateHeadlessUi(isHeadless) {
    if (document.body) {
        document.body.classList.toggle('is-headless-mode', isHeadless);
    }
    ['.preview-card', '.speed-control'].forEach((selector) => {
        const element = document.querySelector(selector);
        if (!element) return;
        element.hidden = isHeadless;
        if (isHeadless) {
            element.setAttribute('inert', '');
            element.setAttribute('aria-hidden', 'true');
        } else {
            element.removeAttribute('inert');
            element.removeAttribute('aria-hidden');
        }
    });
}

/**
 * Update the held-out Evaluation panel — the trustworthy generalization measure.
 * Hidden until the first periodic eval runs; draws a sparkline of the eval-mean
 * trajectory so the climb (or plateau) is visible at a glance.
 */
function updateEval(state) {
    const panel = document.getElementById('eval-panel');
    if (!panel) return;
    if (!state.eval_ran) {
        panel.style.display = 'none';
        return;
    }
    panel.style.display = '';

    const setText = (id, text) => {
        const el = document.getElementById(id);
        if (el) el.textContent = text;
    };
    const round0 = (v) => Math.round(Number(v) || 0);
    const isBaseline = Boolean(state.eval_is_baseline);

    setText('eval-mean', round0(state.eval_mean_score).toLocaleString());
    setText('eval-std', isBaseline ? 'saved best' : `± ${round0(state.eval_std_score)}`);
    setText('eval-median', isBaseline ? '—' : round0(state.eval_median_score).toLocaleString());
    setText('eval-best', round0(state.eval_best_mean).toLocaleString());
    setText('eval-games', `${Number(state.eval_num_games) || 0} levels`);
    setText(
        'eval-last-ep',
        isBaseline
            ? `saved best @ ep ${(Number(state.eval_episode) || 0).toLocaleString()}`
            : `last @ ep ${(Number(state.eval_episode) || 0).toLocaleString()}`
    );

    const verdict = document.getElementById('eval-verdict');
    const verdictLabel = document.getElementById('eval-verdict-label');
    const verdictDetail = document.getElementById('eval-verdict-detail');
    if (verdict && verdictLabel && verdictDetail) {
        const mean = Number(state.eval_mean_score) || 0;
        const best = Number(state.eval_best_mean) || 0;
        const delta = Number.isFinite(Number(state.eval_delta_from_best))
            ? Number(state.eval_delta_from_best)
            : mean - best;
        const isNewBest = !isBaseline && typeof state.eval_is_new_best === 'boolean'
            ? state.eval_is_new_best
            : mean > best;
        const isTiedBest = !isNewBest && delta === 0;
        verdict.classList.toggle('best', isNewBest);
        verdict.classList.toggle('tied', isTiedBest || isBaseline);
        verdict.classList.toggle('regressed', !isBaseline && !isNewBest && !isTiedBest);
        if (isBaseline) {
            verdictLabel.textContent = 'Saved held-out best';
            verdictDetail.textContent = 'training wins update this after eval';
        } else if (isNewBest) {
            verdictLabel.textContent = 'New held-out best';
            verdictDetail.textContent = 'save eval-best checkpoint';
        } else if (isTiedBest) {
            verdictLabel.textContent = 'Matches held-out best';
            verdictDetail.textContent = 'no weaker than best checkpoint';
        } else {
            verdictLabel.textContent = 'Below held-out best';
            verdictDetail.textContent = `${round0(Math.abs(delta)).toLocaleString()} below best checkpoint`;
        }
    }

    const winrateEl = document.getElementById('eval-winrate');
    if (winrateEl) {
        if (isBaseline) {
            winrateEl.textContent = '—';
            winrateEl.style.color = '';
        } else {
            const wr = Number(state.eval_win_rate) || 0;
            winrateEl.textContent = `${(wr * 100).toFixed(1)}%`;
            winrateEl.style.color = wr > 0 ? 'var(--accent-success)' : '';
        }
    }

    setText('eval-crystals', `${((Number(state.eval_crystal_frac) || 0) * 100).toFixed(0)}%`);
    setText('eval-switch', `${((Number(state.eval_switch_rate) || 0) * 100).toFixed(0)}%`);
    setText('eval-depth', `${((Number(state.eval_depth_frac) || 0) * 100).toFixed(0)}%`);
    renderOutcomeChips('eval-outcomes', state.eval_end_reason_counts || {});

    // Sparkline of the current stage eval-mean trajectory when available. Fall
    // back to global history for non-curriculum runs and saved baselines.
    const line = document.getElementById('eval-spark-line');
    const dot = document.getElementById('eval-spark-dot');
    if (line) {
        const stageHist = Array.isArray(state.eval_stage_history)
            ? state.eval_stage_history.map(Number).filter(Number.isFinite)
            : [];
        const globalHist = Array.isArray(state.eval_history)
            ? state.eval_history.map(Number).filter(Number.isFinite)
            : [];
        const hist = stageHist.length ? stageHist : globalHist;
        if (hist.length < 2) {
            line.setAttribute('points', '');
            if (dot) {
                dot.style.display = hist.length === 1 ? '' : 'none';
                dot.setAttribute('cx', '150');
                dot.setAttribute('cy', '22');
            }
        } else {
            if (dot) dot.style.display = 'none';
            const W = 300;
            const H = 44;
            const pad = 3;
            const min = Math.min(...hist);
            const max = Math.max(...hist);
            const span = max - min || 1;
            const points = hist
                .map((v, i) => {
                    const x = (i / (hist.length - 1)) * W;
                    const y = H - pad - ((v - min) / span) * (H - 2 * pad);
                    return `${x.toFixed(1)},${y.toFixed(1)}`;
                })
                .join(' ');
            line.setAttribute('points', points);
        }
    }
}

/**
 * Update the staged Crystal Caves curriculum panel.
 */
function updateCurriculum(state) {
    const panel = document.getElementById('curriculum-panel');
    if (!panel) return;

    const active = Boolean(state.curriculum_active);
    panel.style.display = active ? '' : 'none';
    if (!active) return;

    const setText = (id, text) => {
        const el = document.getElementById(id);
        if (el) el.textContent = text;
    };
    const stageIndex = Number(state.curriculum_stage_index) || 0;
    const stageTotal = Number(state.curriculum_stage_total) || 0;
    const start = Number(state.curriculum_stage_start_episode) || 0;
    const target = Number(state.curriculum_stage_target_episode) || 0;
    const episode = Number(state.episode) || start;
    const span = Math.max(1, target - start);
    const completed = Math.max(0, Math.min(span, episode - start));
    const pct = Math.max(0, Math.min(100, (completed / span) * 100));

    setText('curriculum-stage-count', `${stageIndex} / ${stageTotal}`);
    setText('curriculum-stage-name', state.curriculum_stage_name || '—');
    const families = state.curriculum_stage_families || 'all';
    const difficulty = state.curriculum_stage_difficulty || '—';
    setText('curriculum-stage-meta', `${difficulty} · ${families}`);
    setText('curriculum-status', state.curriculum_stage_status || 'running');
    setText(
        'curriculum-episode-text',
        `${completed.toLocaleString()} / ${span.toLocaleString()} ep`
    );
    setText('curriculum-gate', state.curriculum_stage_gate || '—');
    const gateReady = Boolean(state.curriculum_gate_ready);
    const gateStatus = state.curriculum_gate_status || (gateReady ? 'ready' : 'checking');
    const gateDetail = state.curriculum_gate_detail || 'waiting for held-out eval';
    setText('curriculum-gate-readiness', `${gateReady ? '✓' : '•'} ${gateStatus}: ${gateDetail}`);
    setText('curriculum-checkpoint', state.curriculum_checkpoint_mode || '—');
    setText('curriculum-next', state.curriculum_next_stage_name || '—');

    const fill = document.getElementById('curriculum-stage-fill');
    if (fill) fill.style.width = `${pct}%`;
}

/**
 * Update the Crystal Caves progress panel from training state.
 * Hidden entirely for other games; populated only when cc_active is set.
 */
function updateCrystalCaves(state) {
    const panel = document.getElementById('crystal-caves-panel');
    if (!panel) return;

    const isCC = Boolean(state.cc_active) && state.game_name === 'crystal_caves';
    panel.style.display = isCC ? '' : 'none';
    if (!isCC) return;

    const pctText = (v) => `${Math.round((Number(v) || 0) * 100)}%`;
    const widthPct = (v) => `${Math.max(0, Math.min(100, (Number(v) || 0) * 100))}%`;

    // Level completion (Φ) with a "best ever" marker.
    const setWidth = (id, v) => {
        const el = document.getElementById(id);
        if (el) el.style.width = widthPct(v);
    };
    const setText = (id, text) => {
        const el = document.getElementById(id);
        if (el) el.textContent = text;
    };

    setWidth('cc-progress-fill', state.cc_progress);
    setText('cc-progress-text', pctText(state.cc_progress));
    const bestMarker = document.getElementById('cc-progress-best');
    if (bestMarker) bestMarker.style.left = widthPct(state.cc_best_progress);

    // Crystals — the key sub-goal.
    setWidth('cc-crystal-fill', state.cc_crystal_frac);
    const initial = Number(state.cc_initial_crystals) || 0;
    const remaining = Number(state.cc_crystals_remaining) || 0;
    const collected = Math.max(0, initial - remaining);
    setText('cc-crystals-text', `${collected} / ${initial}`);
    setWidth('cc-crystal-trend-fill', state.cc_recent_crystal_frac);
    setText('cc-crystal-trend-text', pctText(state.cc_recent_crystal_frac));

    // Switch: show thrown/total, or "none" when the level has no switch.
    const swTotal = Number(state.cc_switches_total) || 0;
    const swUsed = Number(state.cc_switches_used) || 0;
    let swText;
    if (swTotal === 0) {
        swText = 'none on this level';
    } else if (swUsed >= swTotal) {
        swText = `✓ ${swUsed} / ${swTotal} thrown`;
    } else {
        swText = `${swUsed} / ${swTotal} thrown`;
    }
    setText('cc-switch', swText);

    setText('cc-depth', pctText(state.cc_depth_frac));
    setText('cc-difficulty', state.cc_difficulty || '—');

    // Last outcome (colour-coded).
    const outcomeEl = document.getElementById('cc-outcome');
    if (outcomeEl) {
        const o = CC_OUTCOMES[state.cc_end_reason];
        outcomeEl.textContent = o ? `${o.icon} ${o.label}` : '—';
        outcomeEl.className = 'info-value cc-outcome' + (o ? ` ${o.cls}` : '');
    }

    // Recent-outcome breakdown: where episodes are ending.
    renderOutcomeChips('cc-outcomes', state.cc_end_reason_counts || {});
}

function renderOutcomeChips(containerId, counts) {
    const outcomesEl = document.getElementById(containerId);
    if (!outcomesEl) return;
    const total = Object.values(counts || {}).reduce((sum, n) => sum + (Number(n) || 0), 0);
    if (total <= 0) {
        const empty = document.createElement('span');
        empty.className = 'cc-outcome-empty';
        empty.textContent = containerId === 'eval-outcomes'
            ? 'waiting for eval outcomes…'
            : 'waiting for episodes…';
        outcomesEl.replaceChildren(empty);
        return;
    }

    const order = ['won', 'killed', 'timeout', 'stalled', 'ended'];
    const keys = order
        .filter((k) => counts[k])
        .concat(Object.keys(counts).filter((k) => !order.includes(k)));
    const chips = keys.map((k) => {
        const o = CC_OUTCOMES[k] || { icon: '•', label: k, cls: '' };
        const n = Number(counts[k]) || 0;
        const share = Math.round((n / total) * 100);
        const chip = document.createElement('span');
        chip.className = `cc-outcome-chip ${o.cls || ''}`.trim();
        chip.title = `${o.label}: ${n} of ${total} (${share}%)`;
        chip.textContent = `${o.icon} ${n} (${share}%)`;
        return chip;
    });
    outcomesEl.replaceChildren(...chips);
}

/**
 * Update dashboard with new data
 */
function updateDashboard(data) {
    const state = data.state;
    const history = data.history;
    updateHeadlessUi(Boolean(state.headless));

    // Update metrics
    document.getElementById('metric-episode').textContent = state.episode.toLocaleString();
    document.getElementById('metric-score').textContent = state.score;
    document.getElementById('metric-best').textContent = state.best_score;
    const winrateEl = document.getElementById('metric-winrate');
    winrateEl.textContent = (state.win_rate * 100).toFixed(1) + '%';
    // Green once the agent is actually winning some — easy signal for non-experts.
    winrateEl.style.color = state.win_rate > 0 ? 'var(--accent-success)' : '';

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

    // Update Crystal Caves progress panel (no-op for other games)
    updateCrystalCaves(state);

    // Update staged curriculum panel (no-op for single-stage runs)
    updateCurriculum(state);

    // Update held-out evaluation panel (hidden until the first eval runs)
    updateEval(state);

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
    if (typeof DashboardCharts !== 'undefined' && DashboardCharts.resizeCharts) {
        DashboardCharts.resizeCharts();
    }
}
