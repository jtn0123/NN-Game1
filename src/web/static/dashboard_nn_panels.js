// Neural network inspection panel renderers.

function panelElement(tagName, className, text) {
    const element = document.createElement(tagName);
    if (className) {
        element.className = className;
    }
    if (text !== undefined) {
        element.textContent = String(text);
    }
    return element;
}

function closePanelButton(action, label) {
    const button = panelElement('button', 'close-btn', '×');
    button.type = 'button';
    button.dataset.action = action;
    button.setAttribute('aria-label', label);
    return button;
}

function appendStatGroup(parent, title) {
    const group = panelElement('div', 'stat-group');
    group.appendChild(panelElement('h4', '', title));
    parent.appendChild(group);
    return group;
}

function appendWeightStats(parent, stats) {
    if (!stats) {
        parent.appendChild(panelElement('div', '', 'No data'));
        return;
    }
    const container = panelElement('div', 'weight-stats');
    container.appendChild(panelElement('div', '', `Mean: ${formatFixedValue(stats.mean, 4)}`));
    container.appendChild(
        panelElement(
            'div',
            '',
            `Range: [${formatFixedValue(stats.min, 4)}, ${formatFixedValue(stats.max, 4)}]`,
        ),
    );
    parent.appendChild(container);
}

function appendLayerStatRow(parent, label, value, color) {
    const row = panelElement('div', 'layer-stat-row');
    row.appendChild(panelElement('span', 'layer-stat-label', label));
    const valueElement = panelElement('span', 'layer-stat-value', value);
    if (color) {
        valueElement.style.color = color;
    }
    row.appendChild(valueElement);
    parent.appendChild(row);
}

function appendLayerStatGroup(parent, title) {
    const group = panelElement('div', 'layer-stat-group');
    group.appendChild(panelElement('span', 'layer-stat-group-title', title));
    parent.appendChild(group);
    return group;
}

NeuralNetworkVisualizer.prototype.displayNeuronInspection = function(neuronData) {
    const panel = document.getElementById('neuron-inspection-panel');
    if (!panel) {
        console.warn('Neuron inspection panel not found');
        return;
    }

    const layerName = neuronData.layer_name || `Layer ${neuronData.layer_idx ?? '?'}`;
    const neuronIdx = Number.isFinite(Number(neuronData.neuron_idx))
        ? Number(neuronData.neuron_idx)
        : '?';
    const previousLayerIdx = Number.isFinite(Number(neuronData.layer_idx))
        ? Number(neuronData.layer_idx) - 1
        : '?';
    const activation = Number(neuronData.current_activation) || 0;
    const activationWidth = clampPercent(Math.abs(activation) * 100);
    const activationHistory = Array.isArray(neuronData.activation_history)
        ? neuronData.activation_history
        : [];

    const header = panelElement('div', 'neuron-header');
    header.appendChild(panelElement('h3', '', `${layerName} - Neuron #${neuronIdx}`));
    header.appendChild(closePanelButton('close-neuron-inspection', 'Close neuron inspection'));

    const content = panelElement('div', 'neuron-content');

    const activationGroup = appendStatGroup(content, 'Activation');
    activationGroup.appendChild(
        panelElement('div', 'stat-value', formatFixedValue(activation, 4, '0.0000')),
    );
    const statBar = panelElement('div', 'stat-bar');
    const statFill = panelElement('div', 'stat-fill');
    statFill.style.width = `${activationWidth}%`;
    statBar.appendChild(statFill);
    activationGroup.appendChild(statBar);

    const incomingGroup = appendStatGroup(
        content,
        `Incoming Weights (from layer ${previousLayerIdx})`,
    );
    appendWeightStats(incomingGroup, neuronData.incoming_weight_stats);

    const outgoingGroup = appendStatGroup(content, 'Outgoing Weights (to next layer)');
    appendWeightStats(outgoingGroup, neuronData.outgoing_weight_stats);

    const qValueGroup = appendStatGroup(content, 'Q-Value Contributions');
    Object.entries(neuronData.q_value_contributions || {}).forEach(([action, contribution]) => {
        const row = panelElement('div', 'q-contrib');
        row.appendChild(panelElement('span', '', `${action}:`));
        row.appendChild(panelElement('span', '', formatFixedValue(contribution, 4, '0.0000')));
        qValueGroup.appendChild(row);
    });

    let sparklineCanvas = null;
    if (activationHistory.length > 0) {
        const historyGroup = appendStatGroup(
            content,
            `Recent Activation History (${activationHistory.length} samples)`,
        );
        const sparklineContainer = panelElement('div', 'sparkline-container');
        sparklineCanvas = panelElement('canvas');
        sparklineCanvas.id = 'neuron-sparkline';
        sparklineCanvas.width = 300;
        sparklineCanvas.height = 40;
        sparklineContainer.appendChild(sparklineCanvas);
        historyGroup.appendChild(sparklineContainer);
    }

    panel.replaceChildren(header, content);
    panel.classList.remove('is-hidden');

    if (sparklineCanvas) {
        this.drawSparkline(sparklineCanvas, activationHistory);
    }
};

NeuralNetworkVisualizer.prototype.drawSparkline = function(canvas, data) {
    // Phase 2: Draw activation history sparkline
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    const width = canvas.width;
    const height = canvas.height;
    const padding = 5;

    ctx.clearRect(0, 0, width, height);

    if (data.length < 2) {
        ctx.fillStyle = '#999';
        ctx.fillText('Insufficient data', 10, height / 2);
        return;
    }

    // Find min/max for scaling
    const min = Math.min(...data);
    const max = Math.max(...data);
    const range = max - min || 1;

    // Draw area under curve
    ctx.fillStyle = 'rgba(100, 150, 255, 0.3)';
    ctx.beginPath();
    ctx.moveTo(padding, height - padding);

    for (let i = 0; i < data.length; i++) {
        const x = padding + (i / (data.length - 1)) * (width - 2 * padding);
        const y = height - padding - ((data[i] - min) / range) * (height - 2 * padding);
        if (i === 0) ctx.lineTo(x, y);
        else ctx.lineTo(x, y);
    }

    ctx.lineTo(width - padding, height - padding);
    ctx.closePath();
    ctx.fill();

    // Draw line
    ctx.strokeStyle = '#4287f5';
    ctx.lineWidth = 2;
    ctx.beginPath();

    for (let i = 0; i < data.length; i++) {
        const x = padding + (i / (data.length - 1)) * (width - 2 * padding);
        const y = height - padding - ((data[i] - min) / range) * (height - 2 * padding);
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
    }

    ctx.stroke();
};

NeuralNetworkVisualizer.prototype.displayLayerAnalysis = function(layerData) {
    const panel = document.getElementById('layer-analysis-panel');
    if (!panel) {
        console.warn('Layer analysis panel not found');
        return;
    }

    const deadPercent = layerData.dead_neuron_percent || 0;
    const saturatedPercent = layerData.saturated_percent || 0;
    let healthStatus = '✓ Healthy';
    let healthColor = 'var(--accent-success)';

    if (deadPercent > 10 || saturatedPercent > 50) {
        healthStatus = '⚠ Warning';
        healthColor = 'var(--accent-warning)';
    }
    if (deadPercent > 30 || saturatedPercent > 80) {
        healthStatus = '✗ Critical';
        healthColor = 'var(--accent-danger)';
    }

    const layerName = layerData.layer_name || `Layer ${layerData.layer_idx ?? '?'}`;
    const neuronCount = Number(layerData.neuron_count) || 0;
    const deadCount = Number(layerData.dead_neuron_count) || 0;
    const saturatedCount = Number(layerData.saturated_neuron_count) || 0;

    const header = panelElement('div', 'layer-header');
    header.appendChild(panelElement('h3', '', layerName));
    header.appendChild(closePanelButton('close-layer-analysis', 'Close layer analysis'));

    const content = panelElement('div', 'layer-content');

    const healthGroup = appendLayerStatGroup(content, 'Network Health');
    appendLayerStatRow(healthGroup, 'Status:', healthStatus, healthColor);
    appendLayerStatRow(healthGroup, 'Neurons:', neuronCount);

    const activationGroup = appendLayerStatGroup(content, 'Activation Stats');
    appendLayerStatRow(
        activationGroup,
        'Mean:',
        formatFixedValue(layerData.avg_activation, 4, '0.0000'),
    );
    appendLayerStatRow(
        activationGroup,
        'Std Dev:',
        formatFixedValue(layerData.activation_std, 4, '0.0000'),
    );
    appendLayerStatRow(
        activationGroup,
        'Dead Neurons:',
        `${deadCount} (${formatFixedValue(deadPercent, 1, '0.0')}%)`,
        deadPercent > 10 ? 'var(--accent-warning)' : '',
    );
    appendLayerStatRow(
        activationGroup,
        'Saturated:',
        `${saturatedCount} (${formatFixedValue(saturatedPercent, 1, '0.0')}%)`,
        saturatedPercent > 50 ? 'var(--accent-warning)' : '',
    );

    const weightGroup = appendLayerStatGroup(content, 'Weight Distribution');
    const weightStats = panelElement('div', 'weight-stats-container');
    [
        ['Mean', formatFixedValue(layerData.weight_mean, 4, '0.0000')],
        ['Std', formatFixedValue(layerData.weight_std, 4, '0.0000')],
    ].forEach(([label, value]) => {
        const item = panelElement('div', 'weight-stat');
        item.appendChild(panelElement('div', 'weight-stat-label', label));
        item.appendChild(panelElement('div', 'weight-stat-value', value));
        weightStats.appendChild(item);
    });
    weightGroup.appendChild(weightStats);

    const gradientGroup = appendLayerStatGroup(content, 'Gradient Flow');
    appendLayerStatRow(
        gradientGroup,
        'Mean Magnitude:',
        formatFixedValue(layerData.gradient_mean, 6, '0.000000'),
    );
    appendLayerStatRow(
        gradientGroup,
        'Std Dev:',
        formatFixedValue(layerData.gradient_std, 6, '0.000000'),
    );
    appendLayerStatRow(
        gradientGroup,
        'Max Magnitude:',
        formatFixedValue(layerData.gradient_max_magnitude, 6, '0.000000'),
    );

    panel.replaceChildren(header, content);
    panel.classList.remove('is-hidden');
};
