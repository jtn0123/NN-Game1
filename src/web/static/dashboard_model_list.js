/**
 * DOM renderer for the dashboard model browser.
 */
(function(root) {
    function createModelStat(documentRef, label, value) {
        const stat = documentRef.createElement('div');
        stat.className = 'model-stat';
        const labelEl = documentRef.createElement('span');
        labelEl.className = 'model-stat-label';
        labelEl.textContent = label;
        const valueEl = documentRef.createElement('span');
        valueEl.className = 'model-stat-value';
        valueEl.textContent = value;
        stat.append(labelEl, valueEl);
        return stat;
    }

    function createBadge(documentRef, text, className, title = '') {
        const badge = documentRef.createElement('span');
        badge.className = className;
        badge.textContent = text;
        if (title) {
            badge.title = title;
        }
        return badge;
    }

    function createModelItem(documentRef, model) {
        const meta = model?.metadata || {};
        const hasMeta = Boolean(model?.has_metadata);
        const isLoadable = model?.is_loadable !== false;
        const modelRef = root.DashboardCore.modelId(model);
        const displayName = model?.name || root.DashboardCore.modelDisplayName(modelRef);

        const item = documentRef.createElement('div');
        item.className = `model-item${isLoadable ? '' : ' model-item-invalid'}`;

        const content = documentRef.createElement('div');
        content.className = 'model-item-content';
        if (isLoadable) {
            content.dataset.action = 'load-model';
            content.dataset.modelId = modelRef;
        }

        const header = documentRef.createElement('div');
        header.className = 'model-header';
        const name = documentRef.createElement('div');
        name.className = 'model-name';
        name.append(documentRef.createTextNode(`📁 ${displayName}`));
        if (hasMeta && meta.save_reason) {
            const reasonClass = String(meta.save_reason).replace(/[^a-z0-9_-]/gi, '') || 'manual';
            name.append(
                documentRef.createTextNode(' '),
                createBadge(documentRef, meta.save_reason, `reason-badge ${reasonClass}`),
            );
        }
        if (!isLoadable) {
            name.append(
                documentRef.createTextNode(' '),
                createBadge(
                    documentRef,
                    'unreadable',
                    'reason-badge error',
                    model?.load_error || 'Unreadable checkpoint',
                ),
            );
        }
        if (model?.requires_unsafe_load) {
            name.append(
                documentRef.createTextNode(' '),
                createBadge(
                    documentRef,
                    'legacy',
                    'reason-badge warning',
                    model?.security_warning || 'Legacy checkpoint requires compatibility fallback',
                ),
            );
        }
        const size = documentRef.createElement('span');
        size.className = 'model-size';
        size.textContent = root.DashboardCore.formatMegabytes(model?.size);
        header.append(name, size);

        const stats = documentRef.createElement('div');
        stats.className = 'model-stats';
        stats.append(
            createModelStat(
                documentRef,
                'Episode',
                root.DashboardCore.formatNumber(hasMeta ? meta.episode : undefined),
            ),
            createModelStat(
                documentRef,
                'Best',
                Number.isFinite(hasMeta ? meta.best_score : undefined) ? String(meta.best_score) : '?',
            ),
            createModelStat(
                documentRef,
                'Avg(100)',
                root.DashboardCore.formatFixed(hasMeta ? meta.avg_score_last_100 : undefined, 1),
            ),
            createModelStat(documentRef, 'Epsilon', root.DashboardCore.formatFixed(model?.epsilon, 3)),
        );

        const date = documentRef.createElement('div');
        date.className = 'model-date';
        date.textContent = model?.modified_str || '';

        const deleteButton = documentRef.createElement('button');
        deleteButton.className = 'model-delete-btn';
        deleteButton.dataset.action = 'delete-model';
        deleteButton.dataset.modelId = modelRef;
        deleteButton.dataset.modelName = displayName;
        deleteButton.title = 'Delete this model';
        deleteButton.textContent = '🗑️';

        content.append(header, stats, date);
        item.append(content, deleteButton);
        return item;
    }

    function renderModelList(list, models) {
        const documentRef = list.ownerDocument || root.document;
        if (!Array.isArray(models) || models.length === 0) {
            const empty = documentRef.createElement('div');
            empty.className = 'no-models';
            empty.textContent = 'No saved models found';
            list.replaceChildren(empty);
            return;
        }

        list.replaceChildren(...models.map((model) => createModelItem(documentRef, model)));
    }

    const api = { createModelItem, renderModelList };
    root.DashboardModelList = Object.freeze(api);
    if (typeof module !== 'undefined' && module.exports) {
        module.exports = api;
    }
})(typeof window !== 'undefined' ? window : globalThis);
