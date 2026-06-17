// ============================================================
// CONTROL FUNCTIONS
// ============================================================

/**
 * Toggle pause state
 */
function togglePause() {
    emitDashboardControl({ action: 'pause' }, 'Pause failed');
}

/**
 * Save model
 */
function saveModel() {
    // Visual feedback
    const btn = document.querySelector('.control-btn.save');
    const restoreButton = setPendingButton(btn, '✓ Saving...', 'saving');

    emitDashboardControl({ action: 'save' }, 'Save failed')
        .then((response) => {
            if (response && response.success) {
                addConsoleLog('Save requested', 'action');
            }
        })
        .finally(restoreButton);
}

/**
 * Reset current episode
 */
function resetEpisode() {
    emitDashboardControl({ action: 'reset' }, 'Episode reset failed').then((response) => {
        if (response && response.success) {
            addConsoleLog('Episode reset requested', 'action');
        }
    });
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
    const restoreFreshButton = setPendingButton(btn, '⏳ Resetting...');

    const requestFreshStart = () => {
        return emitDashboardControl({ action: 'start_fresh' }, 'Start fresh failed')
            .then((response) => {
                if (response && response.success) {
                    addConsoleLog('🔄 Starting fresh training...', 'warning');
                    showTemporaryButtonText(btn, '✓ Resetting...', restoreFreshButton);
                } else {
                    restoreFreshButton();
                }
                return response;
            });
    };

    if (saveFirst) {
        // Save current model first, then reset after the server confirms handling it.
        addConsoleLog('💾 Saving current progress before reset...', 'action');

        emitDashboardControl({ action: 'save' }, 'Save before reset failed').then((response) => {
            if (response && response.success) {
                requestFreshStart();
            } else {
                restoreFreshButton();
            }
        });
    } else {
        requestFreshStart();
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
    const restoreButton = setPendingButton(btn, '⏳ Saving...', 'quitting');

    addConsoleLog('💾 Saving and shutting down...', 'warning');

    emitDashboardControl({ action: 'save_and_quit' }, 'Save & Quit failed')
        .then((response) => {
            if (response && response.success) {
                showShutdownOverlay();
            } else {
                restoreButton();
            }
        });
}

/**
 * Append a text element to a parent and return it.
 */
function appendTextElement(parent, tagName, className, text) {
    const element = document.createElement(tagName);
    if (className) {
        element.className = className;
    }
    element.textContent = text;
    parent.appendChild(element);
    return element;
}

function createAppOverlay(id, icon, title, text, subtext, titleClass = '') {
    const overlay = document.createElement('div');
    overlay.id = id;
    overlay.className = 'app-overlay';
    appendTextElement(overlay, 'div', 'app-overlay-icon', icon);
    appendTextElement(overlay, 'h2', `app-overlay-title ${titleClass}`.trim(), title);
    appendTextElement(overlay, 'p', 'app-overlay-text', text);
    if (subtext) {
        appendTextElement(overlay, 'p', 'app-overlay-subtext', subtext);
    }
    return overlay;
}

/**
 * Show shutdown overlay when server is stopping
 */
function showShutdownOverlay() {
    document.body.appendChild(createAppOverlay(
        'shutdown-overlay',
        '👋',
        'Training Saved & Stopped',
        'Your progress has been saved. You can close this tab.',
        'To resume: python main.py --headless --web'
    ));
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
    emitDashboardControl({ action: 'speed', value: speed }, 'Speed update failed');
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
    const trigger = document.querySelector('[data-action="toggle-settings"]');

    card.classList.toggle('collapsed');
    const isExpanded = !card.classList.contains('collapsed');
    icon.textContent = isExpanded ? '▲' : '▼';
    if (trigger) {
        trigger.setAttribute('aria-expanded', String(isExpanded));
    }
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
            list.innerHTML = DashboardCore.modelListHtml(data.models);
        })
        .catch(err => {
            const list = document.getElementById('model-list');
            const error = document.createElement('div');
            error.className = 'error';
            error.textContent = 'Failed to load models';
            list.replaceChildren(error);
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
function loadModel(modelId) {
    emitDashboardControl({ action: 'load_model', id: modelId }, 'Load model failed')
        .then((response) => {
            if (response && response.success) {
                hideLoadModal();
                addConsoleLog(`Loading model: ${DashboardCore.modelDisplayName(modelId)}`, 'action');
            }
        });
}

/**
 * Delete a model file
 */
function deleteModel(modelId, name) {
    if (!confirm(`Are you sure you want to delete "${name}"?\n\nThis action cannot be undone.`)) {
        return;
    }

    // Encode path for URL (handle special characters)
    const encodedPath = encodeURIComponent(modelId);

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

function dispatchDashboardAction(actionTarget, event) {
    const action = actionTarget?.dataset?.action;
    const handler = DASHBOARD_ACTIONS[action];
    if (!handler) {
        console.warn(`No dashboard action handler registered for "${action}"`);
        return false;
    }
    const result = handler(actionTarget, event);
    return result !== false;
}

function registerDashboardActions() {
    document.addEventListener('click', (event) => {
        const actionTarget = event.target.closest('[data-action]');
        if (actionTarget && dispatchDashboardAction(actionTarget, event)) {
            return;
        }

        // Close modal on outside click (click on backdrop, not content)
        if (event.target.classList && event.target.classList.contains('modal')) {
            hideLoadModal();
        }
    });

    document.addEventListener('change', (event) => {
        const actionTarget = event.target.closest('[data-action]');
        if (actionTarget) {
            dispatchDashboardAction(actionTarget, event);
        }
    });

    document.addEventListener('input', (event) => {
        const actionTarget = event.target.closest('[data-action]');
        if (actionTarget) {
            dispatchDashboardAction(actionTarget, event);
        }
    });

    document.addEventListener('keydown', (event) => {
        const actionTarget = event.target.closest('[role="button"][data-action]');
        if (!actionTarget || (event.key !== 'Enter' && event.key !== ' ')) {
            return;
        }
        event.preventDefault();
        dispatchDashboardAction(actionTarget, event);
    });
}

function isInteractiveKeyboardTarget(target) {
    if (!target || !target.closest) {
        return false;
    }
    return Boolean(target.closest('input, textarea, select, button, [role="button"]'));
}

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

// Keyboard shortcuts
if (typeof document !== 'undefined') {
document.addEventListener('keydown', (e) => {
    // Don't trigger if typing in input
    if (isInteractiveKeyboardTarget(e.target)) return;

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
}
