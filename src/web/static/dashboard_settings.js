// ============================================================
// PERFORMANCE MODE FUNCTIONS
// ============================================================

/**
 * Set performance mode preset
 */
function setPerformanceMode(mode) {
    const preset = PERFORMANCE_MODES[mode];
    if (!preset) {
        addConsoleLog(`Unknown performance mode: ${mode}`, 'warning');
        return;
    }

    emitDashboardControl({ action: 'performance_mode', mode: mode }, 'Performance mode failed')
        .then((response) => {
            if (response && response.success) {
                currentPerformanceMode = mode;
                updatePerformanceModeUI(mode);

                // Update settings inputs to match the mode
                syncSettingsFromMode(mode);

                addConsoleLog(`Performance mode: ${preset.log}`, 'action');
            }
        });
}

/**
 * Sync settings inputs when performance mode changes
 */
function syncSettingsFromMode(mode) {
    const preset = PERFORMANCE_MODES[mode];
    if (!preset) {
        return;
    }

    // Update the settings inputs
    const learnEveryInput = document.getElementById('setting-learn-every');
    const batchInput = document.getElementById('setting-batch');
    const gradStepsInput = document.getElementById('setting-grad-steps');

    if (learnEveryInput) {
        learnEveryInput.value = preset.learnEvery;
        updateLearnEveryLabel(preset.learnEvery);
    }
    if (batchInput) {
        batchInput.value = preset.batchSize;
    }
    if (gradStepsInput) {
        gradStepsInput.value = preset.gradientSteps;
    }
}

/**
 * Update performance mode button states
 */
function updatePerformanceModeUI(mode) {
    document.querySelectorAll('.mode-btn').forEach(btn => {
        btn.classList.remove('active');
        btn.setAttribute('aria-pressed', 'false');
    });
    const activeBtn = document.getElementById(`mode-${mode}`);
    if (activeBtn) {
        activeBtn.classList.add('active');
        activeBtn.setAttribute('aria-pressed', 'true');
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

    // Visual feedback
    const btn = document.querySelector('.apply-btn');
    const restoreButton = setPendingButton(btn, 'Applying...');

    emitDashboardControl({ action: 'config_change', config: config }, 'Settings update failed')
        .then((response) => {
            if (response && response.success) {
                addConsoleLog('Settings updated', 'action', null, config);
                showTemporaryButtonText(btn, '✓ Applied!', restoreButton);
            } else {
                restoreButton();
            }
        });
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

    // Clear input and show feedback
    const btn = document.querySelector('.save-as-btn');
    const restoreButton = setPendingButton(btn, 'Saving...');

    emitDashboardControl({ action: 'save_as', filename: filename }, 'Save As failed')
        .then((response) => {
            if (response && response.success) {
                addConsoleLog(`Saving as: ${filename}.pth`, 'action');
                input.value = '';
                showTemporaryButtonText(btn, '✓ Saved!', restoreButton);
            } else {
                restoreButton();
            }
        });
}

// Periodically update save status time
if (typeof document !== 'undefined') {
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
}
