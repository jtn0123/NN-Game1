(function (root) {
    function readToken(documentRef) {
        return documentRef?.querySelector('meta[name="dashboard-token"]')?.content || '';
    }

    function withDashboardToken(options = {}, token = '') {
        return {
            ...options,
            headers: {
                ...(options.headers || {}),
                'X-Dashboard-Token': token,
            },
        };
    }

    function authorizedControlPayload(payload = {}, token = '') {
        if (!payload || typeof payload !== 'object') {
            return payload;
        }
        return { ...payload, token };
    }

    function createAuthorizedSocket(ioFactory, token = '') {
        const socket = ioFactory({ auth: { token } });
        const originalEmit = socket.emit.bind(socket);
        socket.emit = (event, payload = {}, ...args) => {
            if (event === 'control' || event === 'clear_logs') {
                payload = authorizedControlPayload(payload, token);
            }
            return originalEmit(event, payload, ...args);
        };
        return socket;
    }

    function controlErrorMessage(response, fallback = 'Command failed') {
        if (response && response.success) {
            return '';
        }
        return response?.error || fallback;
    }

    function emitControl(socket, payload, options = {}) {
        const timeoutMs = Number.isFinite(options.timeoutMs) ? options.timeoutMs : 5000;
        const timeoutMessage = options.timeoutMessage || 'No response from server';

        return new Promise((resolve) => {
            if (!socket || typeof socket.emit !== 'function' || socket.connected === false) {
                resolve({ success: false, error: 'Not connected to server' });
                return;
            }

            let settled = false;
            let timer = null;
            const finish = (response) => {
                if (settled) return;
                settled = true;
                if (timer) {
                    clearTimeout(timer);
                }
                if (!response || typeof response !== 'object') {
                    resolve({ success: false, error: timeoutMessage });
                    return;
                }
                resolve(response);
            };

            timer = setTimeout(() => {
                finish({ success: false, error: timeoutMessage });
            }, timeoutMs);

            try {
                socket.emit('control', payload, finish);
            } catch (error) {
                finish({ success: false, error: error?.message || 'Control request failed' });
            }
        });
    }

    function escapeHtml(value) {
        return String(value ?? '').replace(/[&<>"']/g, (char) => ({
            '&': '&amp;',
            '<': '&lt;',
            '>': '&gt;',
            '"': '&quot;',
            "'": '&#39;',
        })[char]);
    }

    function escapeHtmlAttribute(value) {
        return escapeHtml(value);
    }

    function formatMegabytes(sizeBytes) {
        const size = Number(sizeBytes);
        if (!Number.isFinite(size) || size < 0) {
            return '? MB';
        }
        return `${(size / (1024 * 1024)).toFixed(2)} MB`;
    }

    function formatFixed(value, digits, fallback = '?') {
        const numberValue = Number(value);
        if (!Number.isFinite(numberValue)) {
            return fallback;
        }
        return numberValue.toFixed(digits);
    }

    function formatNumber(value, fallback = '?') {
        return Number.isFinite(value) ? value.toLocaleString() : fallback;
    }

    function calculateRunningAverage(data = [], windowSize = 20) {
        if (!Array.isArray(data) || data.length === 0) {
            return [];
        }
        const safeWindow = Math.max(1, Number.isFinite(windowSize) ? Math.floor(windowSize) : 20);
        const result = [];
        let runningSum = 0;

        for (let i = 0; i < data.length; i += 1) {
            const value = Number(data[i]);
            runningSum += Number.isFinite(value) ? value : 0;
            if (i >= safeWindow) {
                const expired = Number(data[i - safeWindow]);
                runningSum -= Number.isFinite(expired) ? expired : 0;
            }
            result.push(runningSum / Math.min(i + 1, safeWindow));
        }
        return result;
    }

    function calculateRunningMax(data = []) {
        if (!Array.isArray(data) || data.length === 0) {
            return [];
        }
        const result = [];
        let runningMax = -Infinity;
        for (const value of data) {
            const numberValue = Number(value);
            runningMax = Math.max(runningMax, Number.isFinite(numberValue) ? numberValue : 0);
            result.push(runningMax);
        }
        return result;
    }

    function buildChartUpdateModel(history = {}, currentEpisode = 0, options = {}) {
        const scores = Array.isArray(history?.scores) ? history.scores : [];
        const losses = Array.isArray(history?.losses) ? history.losses : [];
        const qValues = Array.isArray(history?.q_values) ? history.q_values : [];
        const visibleWindow = Math.max(
            1,
            Number.isFinite(options.visibleWindow) ? Math.floor(options.visibleWindow) : 200,
        );
        const averageWindow = Math.max(
            1,
            Number.isFinite(options.averageWindow) ? Math.floor(options.averageWindow) : 20,
        );
        const parsedEpisode = Number(currentEpisode);
        const episodeNumber = Number.isFinite(parsedEpisode) ? parsedEpisode : 0;
        const startEpisode = Math.max(0, episodeNumber - scores.length);
        const labels = scores.map((_, index) => startEpisode + index + 1);
        const averageScores = calculateRunningAverage(scores, averageWindow);
        const bestAverageScores = calculateRunningMax(averageScores);
        const validLosses = losses.map((loss) => Math.max(Number(loss) || 0, 0.0001));
        const autoScrollRange = scores.length > visibleWindow
            ? {
                min: Math.max(1, labels[Math.max(0, scores.length - visibleWindow)]),
                max: labels[scores.length - 1],
            }
            : null;
        const showAllRange = scores.length > 0 && scores.length <= visibleWindow
            ? { min: Math.max(1, labels[0]), max: undefined }
            : null;

        return {
            labels,
            scores,
            losses: validLosses,
            qValues,
            averageScores,
            bestAverageScores,
            visibleWindow,
            autoScrollRange,
            showAllRange,
        };
    }

    function modelId(model) {
        return model?.id || model?.path || '';
    }

    function modelDisplayName(modelRef) {
        return String(modelRef || '').split(':').pop();
    }

    function modelListHtml(models = []) {
        if (!Array.isArray(models) || models.length === 0) {
            return '<div class="no-models">No saved models found</div>';
        }

        return models.map((model) => {
            const meta = model?.metadata || {};
            const hasMeta = Boolean(model?.has_metadata);
            const isLoadable = model?.is_loadable !== false;
            const modelRef = modelId(model);
            const episode = hasMeta ? meta.episode : undefined;
            const bestScore = hasMeta ? meta.best_score : undefined;
            const avgScore = hasMeta ? meta.avg_score_last_100 : undefined;
            const epsilon = model?.epsilon;
            const reason = hasMeta ? meta.save_reason || '' : '';
            const loadAttrs = isLoadable
                ? `data-action="load-model" data-model-id="${escapeHtmlAttribute(modelRef)}" role="button" tabindex="0"`
                : '';
            const loadWarning = isLoadable
                ? ''
                : `<span class="reason-badge error" title="${escapeHtmlAttribute(model?.load_error || 'Unreadable checkpoint')}">unreadable</span>`;
            const metadataNote = isLoadable && !hasMeta
                ? '<span class="reason-badge legacy" title="This checkpoint can be loaded, but it was saved before rich dashboard metadata existed.">legacy</span>'
                : '';
            const reasonBadge = reason
                ? `<span class="reason-badge ${escapeHtmlAttribute(reason)}">${escapeHtml(reason)}</span>`
                : '';

            return `
                    <div class="model-item${isLoadable ? '' : ' model-item-invalid'}">
                        <div class="model-item-content" ${loadAttrs}>
                            <div class="model-header">
                                <div class="model-name">
                                    📁 ${escapeHtml(model?.name || modelDisplayName(modelRef))}
                                    ${reasonBadge}
                                    ${metadataNote}
                                    ${loadWarning}
                                </div>
                                <span class="model-size">${formatMegabytes(model?.size)}</span>
                            </div>
                            <div class="model-stats">
                                <div class="model-stat">
                                    <span class="model-stat-label">Episode</span>
                                    <span class="model-stat-value">${formatNumber(episode)}</span>
                                </div>
                                <div class="model-stat">
                                    <span class="model-stat-label">Best</span>
                                    <span class="model-stat-value">${Number.isFinite(bestScore) ? bestScore : '?'}</span>
                                </div>
                                <div class="model-stat">
                                    <span class="model-stat-label">Avg(100)</span>
                                    <span class="model-stat-value">${formatFixed(avgScore, 1)}</span>
                                </div>
                                <div class="model-stat">
                                    <span class="model-stat-label">Epsilon</span>
                                    <span class="model-stat-value">${formatFixed(epsilon, 3)}</span>
                                </div>
                            </div>
                            <div class="model-date">${escapeHtml(model?.modified_str || '')}</div>
                        </div>
                        <button class="model-delete-btn" data-action="delete-model" data-model-id="${escapeHtmlAttribute(modelRef)}" data-model-name="${escapeHtmlAttribute(model?.name || modelDisplayName(modelRef))}" title="Delete this model">
                            <span aria-hidden="true">🗑️</span>
                            <span>Delete</span>
                        </button>
                    </div>
                `;
        }).join('');
    }

    const api = {
        readToken,
        withDashboardToken,
        authorizedControlPayload,
        createAuthorizedSocket,
        controlErrorMessage,
        emitControl,
        escapeHtml,
        escapeHtmlAttribute,
        formatMegabytes,
        formatFixed,
        formatNumber,
        calculateRunningAverage,
        calculateRunningMax,
        buildChartUpdateModel,
        modelId,
        modelDisplayName,
        modelListHtml,
    };

    root.DashboardCore = api;
    if (typeof module !== 'undefined' && module.exports) {
        module.exports = api;
    }
})(typeof window !== 'undefined' ? window : globalThis);
