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

    function modelId(model) {
        return model?.id || model?.path || '';
    }

    function modelDisplayName(modelRef) {
        return String(modelRef || '').split(':').pop();
    }

    const api = {
        readToken,
        withDashboardToken,
        authorizedControlPayload,
        createAuthorizedSocket,
        modelId,
        modelDisplayName,
    };

    root.DashboardCore = api;
    if (typeof module !== 'undefined' && module.exports) {
        module.exports = api;
    }
})(typeof window !== 'undefined' ? window : globalThis);
