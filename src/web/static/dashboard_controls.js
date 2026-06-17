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
async function startFresh() {
    const saveChoice = await DashboardDialogs.choose({
        title: 'Start Fresh Training',
        message: 'Choose whether to save the current model before resetting training.',
        choices: [
            { value: 'save', label: 'Save first', variant: 'primary' },
            { value: 'skip', label: 'Skip save', variant: 'danger' },
        ],
        cancelText: 'Cancel',
    });

    if (!saveChoice) {
        addConsoleLog('Fresh start cancelled', 'info');
        return;
    }

    const shouldReset = await DashboardDialogs.ask({
        title: 'Confirm Fresh Start',
        message: 'This resets the neural network, replay memory, episode count, charts, and console logs. Saved model files on disk are preserved.',
        details: [
            'Neural-network weights return to random initialization.',
            'Replay memory and training charts are cleared.',
            'Saved models stay available from the Load menu.',
        ],
        confirmText: 'Start fresh',
        danger: true,
    });

    if (!shouldReset) {
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

    if (saveChoice === 'save') {
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
async function saveAndQuit() {
    const confirmed = await DashboardDialogs.ask({
        title: 'Save & Quit',
        message: 'This saves the current training progress and shuts down the training server.',
        confirmText: 'Save & quit',
    });

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
function setPreviewPlaceholder(kind) {
    const placeholder = document.getElementById('preview-placeholder');
    if (!placeholder) return;

    if (kind === 'headless') {
        const label = document.createTextNode('🚀 Headless Mode');
        const subtext = document.createElement('span');
        subtext.style.fontSize = '0.8em';
        subtext.style.opacity = '0.7';
        subtext.textContent = 'No preview available';
        placeholder.replaceChildren(label, document.createElement('br'), subtext);
    } else {
        placeholder.textContent = '🎮 Game Preview';
    }
    placeholder.classList.remove('hidden');
}

function refreshScreenshot() {
    return fetchWithTimeout('/api/screenshot')
        .then(response => response.json())
        .then(data => {
            const img = document.getElementById('game-preview');
            const placeholder = document.getElementById('preview-placeholder');

            // Check if headless mode - stop polling and show headless placeholder
            if (data.headless) {
                setPreviewPlaceholder('headless');
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
                setPreviewPlaceholder('default');
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

let loadModalFocusReturnTarget = null;
const loadModalFocusableSelector = [
    'a[href]',
    'button:not([disabled])',
    'input:not([disabled])',
    'select:not([disabled])',
    'textarea:not([disabled])',
    '[tabindex]:not([tabindex="-1"])',
].join(',');

function loadModalFocusableElements(modal) {
    return Array.from(modal.querySelectorAll(loadModalFocusableSelector))
        .filter(element => {
            const style = window.getComputedStyle(element);
            return style.display !== 'none'
                && style.visibility !== 'hidden'
                && element.getClientRects().length > 0;
        });
}

function handleLoadModalKeydown(event) {
    const modal = document.getElementById('load-modal');
    if (!modal || !modal.classList.contains('visible')) return;

    if (event.key === 'Escape') {
        event.preventDefault();
        hideLoadModal();
        return;
    }

    if (event.key !== 'Tab') return;

    const focusable = loadModalFocusableElements(modal);
    if (focusable.length === 0) {
        event.preventDefault();
        return;
    }

    const first = focusable[0];
    const last = focusable[focusable.length - 1];
    if (!modal.contains(document.activeElement)) {
        event.preventDefault();
        first.focus();
        return;
    }
    if (event.shiftKey && document.activeElement === first) {
        event.preventDefault();
        last.focus();
    } else if (!event.shiftKey && document.activeElement === last) {
        event.preventDefault();
        first.focus();
    }
}

/**
 * Show load model modal with enhanced metadata
 */
function showLoadModal() {
    const modal = document.getElementById('load-modal');
    if (!modal) return;

    loadModalFocusReturnTarget = document.activeElement instanceof HTMLElement
        ? document.activeElement
        : null;
    modal.classList.add('visible');
    document.addEventListener('keydown', handleLoadModalKeydown);

    const closeButton = modal.querySelector('[data-action="hide-load-modal"]');
    if (closeButton && typeof closeButton.focus === 'function') {
        const focusCloseButton = () => closeButton.focus({ preventScroll: true });
        focusCloseButton();
        requestAnimationFrame(focusCloseButton);
        setTimeout(focusCloseButton, 0);
    }

    // Fetch available models
    fetchWithTimeout('/api/models')
        .then(response => response.json())
        .then(data => {
            const list = document.getElementById('model-list');
            if (list) {
                DashboardModelList.renderModelList(list, data.models);
            }
        })
        .catch(err => {
            const list = document.getElementById('model-list');
            if (!list) return;
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
    if (!modal) return;
    modal.classList.remove('visible');
    document.removeEventListener('keydown', handleLoadModalKeydown);
    if (
        loadModalFocusReturnTarget
        && document.contains(loadModalFocusReturnTarget)
        && typeof loadModalFocusReturnTarget.focus === 'function'
    ) {
        loadModalFocusReturnTarget.focus();
    }
    loadModalFocusReturnTarget = null;
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
async function deleteModel(modelId, name) {
    const confirmed = await DashboardDialogs.ask({
        title: 'Delete Model',
        message: `Delete "${name}"? This action cannot be undone.`,
        confirmText: 'Delete',
        danger: true,
    });
    if (!confirmed) {
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
