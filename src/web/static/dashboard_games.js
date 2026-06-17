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
            select.replaceChildren();

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
async function switchGame(gameId) {
    if (!gameId) return;

    // Get current game from dropdown's data
    const select = document.getElementById('game-select');
    const currentGame = select ? select.dataset.currentGame : 'breakout';

    // If same game, do nothing
    if (gameId === currentGame) {
        return;
    }

    const confirmed = await DashboardDialogs.ask({
        title: `Switch to ${gameId.replace('_', ' ').toUpperCase()}?`,
        message: 'This saves the current progress and restarts training with the selected game.',
        confirmText: 'Switch game',
    });

    if (confirmed) {
        addConsoleLog(`🔄 Switching to ${gameId}...`, 'warning');
        addConsoleLog(`💾 Saving current progress...`, 'info');

        // Request game switch - server will save and restart
        emitDashboardControl(
            { action: 'restart_with_game', game: gameId },
            'Game switch failed'
        ).then((response) => {
            if (!response || !response.success) {
                loadGames();
            }
        });
    } else {
        // Reset dropdown to current game
        loadGames();
    }
}

/**
 * Go back to game launcher to select a different game/mode
 */
async function goToLauncher() {
    const confirmed = await DashboardDialogs.ask({
        title: 'Go Back To Game Launcher?',
        message: 'This stops the current session and returns to game selection. Current progress is saved first.',
        confirmText: 'Open launcher',
    });

    if (confirmed) {
        addConsoleLog('🎮 Returning to launcher...', 'warning');
        // Tell server to switch back to launcher mode
        emitDashboardControl({ action: 'go_to_launcher' }, 'Return to launcher failed');
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
    const trigger = document.querySelector('[data-action="toggle-comparison"]');

    if (!card) return;

    card.classList.toggle('collapsed');
    const isExpanded = !card.classList.contains('collapsed');
    if (icon) {
        icon.textContent = isExpanded ? '▲' : '▼';
    }
    if (trigger) {
        trigger.setAttribute('aria-expanded', String(isExpanded));
    }

    // Load stats when opening
    if (!card.classList.contains('collapsed')) {
        loadGameStats();
    }
}

/**
 * Clamp a numeric percentage for inline width styles.
 */
function clampPercent(value) {
    const numberValue = Number(value);
    if (!Number.isFinite(numberValue)) {
        return 0;
    }
    return Math.max(0, Math.min(100, numberValue));
}

function safeRgb(color) {
    const channels = Array.isArray(color) ? color : [100, 181, 246];
    const normalized = channels.slice(0, 3).map((channel) => {
        const value = Number(channel);
        return Number.isFinite(value) ? Math.max(0, Math.min(255, Math.round(value))) : 0;
    });
    while (normalized.length < 3) {
        normalized.push(0);
    }
    return `rgb(${normalized[0]}, ${normalized[1]}, ${normalized[2]})`;
}

function formatFixedValue(value, digits, fallback = 'N/A') {
    const numberValue = Number(value);
    return Number.isFinite(numberValue) ? numberValue.toFixed(digits) : fallback;
}

function createGameStatsElement(gameId, game, currentGame, maxScore) {
    const isCurrent = gameId === currentGame;
    const item = document.createElement('div');
    item.className = `comparison-item${isCurrent ? ' current' : ''}`;

    const icon = document.createElement('div');
    icon.className = 'comparison-icon';
    icon.textContent = game.icon || '';
    item.appendChild(icon);

    const info = document.createElement('div');
    info.className = 'comparison-info';
    item.appendChild(info);

    const name = document.createElement('div');
    name.className = 'comparison-name';
    name.textContent = `${game.name || gameId}${isCurrent ? ' (current)' : ''}`;
    info.appendChild(name);

    const trainingTime = Number(game.total_training_time) || 0;
    let timeStr = 'No training';
    if (trainingTime > 0) {
        const hours = Math.floor(trainingTime / 3600);
        const mins = Math.floor((trainingTime % 3600) / 60);
        timeStr = hours > 0 ? `${hours}h ${mins}m` : `${mins}m`;
    }

    const stats = document.createElement('div');
    stats.className = 'comparison-stats';
    const bestScore = Number(game.best_score) || 0;
    const totalEpisodes = Number(game.total_episodes) || 0;
    stats.textContent = `Best: ${bestScore} | Episodes: ${totalEpisodes.toLocaleString()} | Time: ${timeStr}`;
    info.appendChild(stats);

    const bar = document.createElement('div');
    bar.className = 'comparison-bar';
    const fill = document.createElement('div');
    fill.className = 'comparison-bar-fill';
    fill.style.width = `${clampPercent((bestScore / maxScore) * 100)}%`;
    fill.style.background = safeRgb(game.color);
    bar.appendChild(fill);
    info.appendChild(bar);

    return item;
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
                const bestScore = Number(stats[gameId].best_score) || 0;
                if (bestScore > maxScore) {
                    maxScore = bestScore;
                }
            }
            maxScore = maxScore || 1; // Avoid division by zero

            const fragment = document.createDocumentFragment();
            for (const gameId in stats) {
                const game = stats[gameId];
                fragment.appendChild(createGameStatsElement(gameId, game, currentGame, maxScore));
            }

            if (fragment.childNodes.length > 0) {
                grid.replaceChildren(fragment);
            } else {
                const noData = document.createElement('div');
                noData.className = 'no-data';
                noData.textContent = 'No game data available';
                grid.replaceChildren(noData);
            }
        })
        .catch(err => {
            console.error('Failed to load game stats:', err);
            const error = document.createElement('div');
            error.className = 'error';
            error.textContent = 'Failed to load game statistics';
            grid.replaceChildren(error);
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
    const banner = document.createElement('div');
    banner.id = 'restart-banner';
    banner.className = 'restart-banner';

    appendTextElement(banner, 'h2', 'restart-title', '🔄 Ready to Switch Games');
    appendTextElement(
        banner,
        'p',
        'restart-text',
        'Progress has been saved. Restart with the new game:'
    );

    const commandBox = document.createElement('div');
    commandBox.className = 'restart-command-box';
    const code = document.createElement('code');
    code.id = 'restart-command';
    code.textContent = command;
    commandBox.appendChild(code);
    banner.appendChild(commandBox);

    const actions = document.createElement('div');
    actions.className = 'restart-actions';
    const copyButton = appendTextElement(actions, 'button', 'restart-copy-btn', '📋 Copy Command');
    copyButton.dataset.action = 'copy-restart-command';
    const closeButton = appendTextElement(actions, 'button', 'restart-close-btn', 'Close');
    closeButton.dataset.action = 'close-restart-banner';
    banner.appendChild(actions);

    const overlay = document.createElement('div');
    overlay.id = 'restart-overlay';
    overlay.className = 'restart-overlay';
    overlay.addEventListener('click', closeRestartBanner);

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
    const gameName = String(game || '').replace('_', ' ').toUpperCase();
    const overlay = createAppOverlay(
        'restarting-overlay',
        '🔄',
        `Restarting with ${gameName}`,
        'Please wait while the server restarts...',
        'Page will auto-refresh when ready',
        'info'
    );
    const icon = overlay.querySelector('.app-overlay-icon');
    if (icon) {
        icon.classList.add('pulse');
    }
    document.body.appendChild(overlay);

    // Start checking if server is back up
    setTimeout(checkServerAndReload, 2000);
}

/**
 * Check if server is back up and reload
 */
function checkServerAndReload() {
    fetchWithTimeout('/api/status')
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
