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
                ? `data-action="load-model" data-model-id="${escapeHtmlAttribute(modelRef)}"`
                : '';
            const loadWarning = isLoadable
                ? ''
                : `<span class="reason-badge error" title="${escapeHtmlAttribute(model?.load_error || 'Unreadable checkpoint')}">unreadable</span>`;
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
                            🗑️
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
        modelId,
        modelDisplayName,
        modelListHtml,
    };

    root.DashboardCore = api;
    if (typeof module !== 'undefined' && module.exports) {
        module.exports = api;
    }
})(typeof window !== 'undefined' ? window : globalThis);
