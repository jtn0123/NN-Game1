// Neural network inspection panel renderers.

NeuralNetworkVisualizer.prototype.displayNeuronInspection = function(neuronData) {
    // Phase 2: Display neuron inspection panel
    const panel = document.getElementById('neuron-inspection-panel');
    if (!panel) {
        console.warn('Neuron inspection panel not found');
        return;
    }

    const layerName = escapeHtml(neuronData.layer_name || `Layer ${neuronData.layer_idx ?? '?'}`);
    const neuronIdx = Number.isFinite(Number(neuronData.neuron_idx)) ? Number(neuronData.neuron_idx) : '?';
    const previousLayerIdx = Number.isFinite(Number(neuronData.layer_idx))
        ? Number(neuronData.layer_idx) - 1
        : '?';
    const activation = Number(neuronData.current_activation) || 0;
    const activationWidth = clampPercent(Math.abs(activation) * 100);
    const incoming = neuronData.incoming_weight_stats;
    const outgoing = neuronData.outgoing_weight_stats;
    const qContribHtml = Object.entries(neuronData.q_value_contributions || {}).map(([action, contrib]) => `
        <div class="q-contrib">
            <span>${escapeHtml(action)}:</span>
            <span>${formatFixedValue(contrib, 4, '0.0000')}</span>
        </div>
    `).join('');
    const activationHistory = Array.isArray(neuronData.activation_history)
        ? neuronData.activation_history
        : [];

    // Build HTML for neuron details
    const html = `
        <div class="neuron-header">
            <h3>${layerName} - Neuron #${neuronIdx}</h3>
            <button class="close-btn" data-action="close-neuron-inspection" aria-label="Close neuron inspection">×</button>
        </div>

        <div class="neuron-content">
            <div class="stat-group">
                <h4>Activation</h4>
                <div class="stat-value">${formatFixedValue(activation, 4, '0.0000')}</div>
                <div class="stat-bar">
                    <div class="stat-fill" style="width: ${activationWidth}%"></div>
                </div>
            </div>

            <div class="stat-group">
                <h4>Incoming Weights (from layer ${previousLayerIdx})</h4>
                ${incoming ? `
                    <div class="weight-stats">
                        <div>Mean: ${formatFixedValue(incoming.mean, 4)}</div>
                        <div>Range: [${formatFixedValue(incoming.min, 4)},
                                     ${formatFixedValue(incoming.max, 4)}]</div>
                    </div>
                ` : '<div>No data</div>'}
            </div>

            <div class="stat-group">
                <h4>Outgoing Weights (to next layer)</h4>
                ${outgoing ? `
                    <div class="weight-stats">
                        <div>Mean: ${formatFixedValue(outgoing.mean, 4)}</div>
                        <div>Range: [${formatFixedValue(outgoing.min, 4)},
                                     ${formatFixedValue(outgoing.max, 4)}]</div>
                    </div>
                ` : '<div>No data</div>'}
            </div>

            <div class="stat-group">
                <h4>Q-Value Contributions</h4>
                ${qContribHtml}
            </div>

            ${activationHistory.length > 0 ? `
                <div class="stat-group">
                    <h4>Recent Activation History (${activationHistory.length} samples)</h4>
                    <div class="sparkline-container">
                        <canvas id="neuron-sparkline" width="300" height="40"></canvas>
                    </div>
                </div>
            ` : ''}
        </div>
    `;

    panel.innerHTML = html;
    panel.style.display = 'block';

    // Draw sparkline if data available
    if (activationHistory.length > 0) {
        this.drawSparkline(document.getElementById('neuron-sparkline'), activationHistory);
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
    // Phase 2: Display layer analysis panel
    const panel = document.getElementById('layer-analysis-panel');
    if (!panel) {
        console.warn('Layer analysis panel not found');
        return;
    }

    // Determine health status based on dead/saturated neurons
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
    const layerName = escapeHtml(layerData.layer_name || `Layer ${layerData.layer_idx ?? '?'}`);
    const neuronCount = Number(layerData.neuron_count) || 0;
    const deadCount = Number(layerData.dead_neuron_count) || 0;
    const saturatedCount = Number(layerData.saturated_neuron_count) || 0;

    // Build HTML for layer details
    const html = `
        <div class="layer-header">
            <h3>${layerName}</h3>
            <button class="close-btn" data-action="close-layer-analysis" aria-label="Close layer analysis">×</button>
        </div>

        <div class="layer-content">
            <!-- Health Status -->
            <div class="layer-stat-group">
                <span class="layer-stat-group-title">Network Health</span>
                <div class="layer-stat-row">
                    <span class="layer-stat-label">Status:</span>
                    <span class="layer-stat-value" style="color: ${healthColor}">${healthStatus}</span>
                </div>
                <div class="layer-stat-row">
                    <span class="layer-stat-label">Neurons:</span>
                    <span class="layer-stat-value">${neuronCount}</span>
                </div>
            </div>

            <!-- Activation Statistics -->
            <div class="layer-stat-group">
                <span class="layer-stat-group-title">Activation Stats</span>
                <div class="layer-stat-row">
                    <span class="layer-stat-label">Mean:</span>
                    <span class="layer-stat-value">${formatFixedValue(layerData.avg_activation, 4, '0.0000')}</span>
                </div>
                <div class="layer-stat-row">
                    <span class="layer-stat-label">Std Dev:</span>
                    <span class="layer-stat-value">${formatFixedValue(layerData.activation_std, 4, '0.0000')}</span>
                </div>
                <div class="layer-stat-row">
                    <span class="layer-stat-label">Dead Neurons:</span>
                    <span class="layer-stat-value" style="color: ${deadPercent > 10 ? 'var(--accent-warning)' : 'inherit'}">
                        ${deadCount} (${formatFixedValue(deadPercent, 1, '0.0')}%)
                    </span>
                </div>
                <div class="layer-stat-row">
                    <span class="layer-stat-label">Saturated:</span>
                    <span class="layer-stat-value" style="color: ${saturatedPercent > 50 ? 'var(--accent-warning)' : 'inherit'}">
                        ${saturatedCount} (${formatFixedValue(saturatedPercent, 1, '0.0')}%)
                    </span>
                </div>
            </div>

            <!-- Weight Statistics -->
            <div class="layer-stat-group">
                <span class="layer-stat-group-title">Weight Distribution</span>
                <div class="weight-stats-container">
                    <div class="weight-stat">
                        <div class="weight-stat-label">Mean</div>
                        <div class="weight-stat-value">${formatFixedValue(layerData.weight_mean, 4, '0.0000')}</div>
                    </div>
                    <div class="weight-stat">
                        <div class="weight-stat-label">Std</div>
                        <div class="weight-stat-value">${formatFixedValue(layerData.weight_std, 4, '0.0000')}</div>
                    </div>
                </div>
            </div>

            <!-- Gradient Statistics -->
            <div class="layer-stat-group">
                <span class="layer-stat-group-title">Gradient Flow</span>
                <div class="layer-stat-row">
                    <span class="layer-stat-label">Mean Magnitude:</span>
                    <span class="layer-stat-value">${formatFixedValue(layerData.gradient_mean, 6, '0.000000')}</span>
                </div>
                <div class="layer-stat-row">
                    <span class="layer-stat-label">Std Dev:</span>
                    <span class="layer-stat-value">${formatFixedValue(layerData.gradient_std, 6, '0.000000')}</span>
                </div>
                <div class="layer-stat-row">
                    <span class="layer-stat-label">Max Magnitude:</span>
                    <span class="layer-stat-value">${formatFixedValue(layerData.gradient_max_magnitude, 6, '0.000000')}</span>
                </div>
            </div>
        </div>
    `;

    panel.innerHTML = html;
    panel.style.display = 'block';
};
