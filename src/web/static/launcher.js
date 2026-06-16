(function (root) {
    const state = {
        socket: null,
        selectedGame: null,
        selectedMode: 'ai',
    };

    function difficultyClass(difficulty = '') {
        const value = String(difficulty).toLowerCase();
        if (value.includes('easy')) return 'easy';
        if (value.includes('hard')) return 'hard';
        return 'medium';
    }

    function gameLabel(gameId) {
        return String(gameId || '').toUpperCase().replace(/_/g, ' ');
    }

    function startLabel(mode, gameId) {
        const modeText = mode === 'human' ? 'PLAY' : 'TRAIN';
        return `${modeText} ${gameLabel(gameId)}`;
    }

    function setStatus(documentRef, message, color = '') {
        const status = documentRef.getElementById('status-msg');
        if (!status) return;
        status.textContent = message;
        status.style.color = color;
    }

    function createSpan(documentRef, className, text) {
        const span = documentRef.createElement('span');
        span.className = className;
        span.textContent = text;
        return span;
    }

    function createGameCard(documentRef, game, onSelect) {
        const card = documentRef.createElement('div');
        const actions = Array.isArray(game.actions) ? game.actions : [];
        card.className = 'game-card';
        card.setAttribute('role', 'button');
        card.setAttribute('tabindex', '0');
        card.setAttribute('aria-label', `Select ${game.name || game.id}`);

        const select = () => onSelect(game.id, card);
        card.addEventListener('click', select);
        card.addEventListener('keydown', (event) => {
            if (event.key === 'Enter' || event.key === ' ') {
                event.preventDefault();
                select();
            }
        });

        card.appendChild(createSpan(documentRef, 'game-icon', game.icon || ''));

        const name = documentRef.createElement('div');
        name.className = 'game-name';
        name.textContent = game.name || game.id;
        card.appendChild(name);

        const description = documentRef.createElement('div');
        description.className = 'game-desc';
        description.textContent = game.description || '';
        card.appendChild(description);

        const difficulty = createSpan(
            documentRef,
            `difficulty ${difficultyClass(game.difficulty)}`,
            game.difficulty || 'Medium'
        );
        card.appendChild(difficulty);

        const stats = documentRef.createElement('div');
        stats.className = 'game-stats';

        const actionCount = documentRef.createElement('span');
        actionCount.appendChild(createSpan(documentRef, 'stat-label', 'Actions:'));
        actionCount.append(` ${actions.length}`);
        stats.appendChild(actionCount);

        const controls = documentRef.createElement('span');
        controls.appendChild(createSpan(documentRef, 'stat-label', 'Controls:'));
        controls.append(` ${actions.join(', ')}`);
        stats.appendChild(controls);

        card.appendChild(stats);
        return card;
    }

    function selectMode(documentRef, mode) {
        state.selectedMode = mode;
        documentRef.getElementById('mode-ai')?.classList.toggle('selected', mode === 'ai');
        documentRef.getElementById('mode-human')?.classList.toggle('selected', mode === 'human');

        const desc = documentRef.getElementById('mode-desc');
        if (desc) {
            desc.textContent = mode === 'ai'
                ? 'Train a neural network to master the game'
                : 'Play the game yourself using keyboard controls';
        }

        if (state.selectedGame) {
            const btn = documentRef.getElementById('start-btn');
            if (btn) {
                btn.textContent = startLabel(mode, state.selectedGame);
            }
        }
    }

    function selectGame(documentRef, gameId, card) {
        documentRef.querySelectorAll('.game-card').forEach((candidate) => {
            candidate.classList.remove('selected');
        });

        card.classList.add('selected');
        state.selectedGame = gameId;

        const btn = documentRef.getElementById('start-btn');
        if (btn) {
            btn.textContent = startLabel(state.selectedMode, gameId);
            btn.classList.add('active');
            btn.disabled = false;
            btn.setAttribute('aria-disabled', 'false');
        }

        setStatus(documentRef, `Selected: ${gameId}`);
    }

    function startGame(documentRef) {
        if (!state.selectedGame) {
            setStatus(documentRef, 'Please select a game first', '#ff4444');
            return;
        }

        const btn = documentRef.getElementById('start-btn');
        if (btn) {
            btn.textContent = 'LAUNCHING...';
            btn.classList.add('loading');
        }

        const modeText = state.selectedMode === 'human' ? 'Playing' : 'Training';
        setStatus(documentRef, `${modeText} ${state.selectedGame}...`, '#00d4ff');

        if (!state.socket || !state.socket.connected) {
            setStatus(documentRef, 'Not connected to server. Refresh the page.', '#ff4444');
            if (btn) {
                btn.textContent = 'CONNECTION ERROR';
                btn.classList.remove('loading');
            }
            return;
        }

        DashboardCore.emitControl(state.socket, {
            action: 'select_game',
            game: state.selectedGame,
            mode: state.selectedMode,
        }).then((response) => {
            if (response && response.success) {
                return;
            }
            setStatus(
                documentRef,
                `Launch failed: ${DashboardCore.controlErrorMessage(response)}`,
                '#ff4444'
            );
            if (btn) {
                btn.textContent = startLabel(state.selectedMode, state.selectedGame);
                btn.classList.remove('loading');
            }
        });
    }

    function loadGames(documentRef, fetchImpl, token) {
        return fetchImpl('/api/games', DashboardCore.withDashboardToken({}, token))
            .then((response) => response.json())
            .then((data) => {
                const grid = documentRef.getElementById('games-grid');
                if (!grid) return;
                const fragment = documentRef.createDocumentFragment();
                (data.games || []).forEach((game) => {
                    fragment.appendChild(createGameCard(
                        documentRef,
                        game,
                        (gameId, card) => selectGame(documentRef, gameId, card)
                    ));
                });
                grid.replaceChildren(fragment);
            });
    }

    function initLauncher(documentRef = root.document, fetchImpl = root.fetch, ioFactory = root.io) {
        const token = DashboardCore.readToken(documentRef);
        const startButton = documentRef.getElementById('start-btn');
        if (startButton) {
            startButton.disabled = true;
            startButton.setAttribute('aria-disabled', 'true');
            startButton.addEventListener('click', () => startGame(documentRef));
        }

        documentRef.querySelectorAll('.mode-btn[data-mode]').forEach((button) => {
            button.addEventListener('click', () => selectMode(documentRef, button.dataset.mode));
        });

        try {
            state.socket = DashboardCore.createAuthorizedSocket(ioFactory, token);
            setStatus(documentRef, 'Connecting...');
        } catch (error) {
            setStatus(documentRef, 'Socket.IO failed to load');
            console.error('Socket.IO initialization error:', error);
        }

        loadGames(documentRef, fetchImpl, token).catch((error) => {
            setStatus(documentRef, 'Failed to load games', '#ff4444');
            console.error('Failed to load games:', error);
        });

        if (state.socket) {
            state.socket.on('game_starting', (data) => {
                setStatus(documentRef, data.message || 'Game starting...');
            });

            state.socket.on('disconnect', () => {
                const status = documentRef.getElementById('status-msg');
                if (state.selectedGame && status && !status.textContent.includes('Redirecting')) {
                    setStatus(documentRef, 'Connection lost. Reconnecting...');
                }
            });

            state.socket.on('connect', () => {
                setStatus(documentRef, 'Connected - Select a game', '#00ff88');
            });

            state.socket.on('connect_error', (error) => {
                setStatus(documentRef, `Connection error: ${error.message}`, '#ff4444');
            });
        }
    }

    const api = {
        difficultyClass,
        gameLabel,
        startLabel,
        createGameCard,
        initLauncher,
    };

    root.LauncherApp = api;
    if (typeof module !== 'undefined' && module.exports) {
        module.exports = api;
    }

    if (typeof document !== 'undefined') {
        document.addEventListener('DOMContentLoaded', () => initLauncher());
    }
})(typeof window !== 'undefined' ? window : globalThis);
