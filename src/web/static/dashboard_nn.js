// ============================================================
// NEURAL NETWORK VISUALIZATION
// ============================================================

/**
 * Neural Network Visualizer - Canvas-based real-time visualization
 *
 * Features:
 * - Network architecture (layers, neurons, connections)
 * - Live activations with diverging color palette
 * - Animated data flow pulses
 * - Q-values bar chart with action selection
 * - Layer labels with neuron counts
 */
class NeuralNetworkVisualizer {
    constructor(canvasId) {
        this.canvas = document.getElementById(canvasId);
        if (!this.canvas) {
            console.warn('NN Visualizer: Canvas not found:', canvasId);
            return;
        }
        this.ctx = this.canvas.getContext('2d');

        // State
        this.data = null;
        this.isEnabled = true;
        this.animationId = null;

        // Cleanup tracking for memory leak prevention
        this.resizeObserver = null;
        this.clickHandler = null;

        // Phase 1.2 & 1.4: Adaptive rendering
        this.renderInterval = 33;  // Default ~30Hz, will adapt based on training speed
        this.lastRenderTime = 0;
        this.avgRenderTime = 0;
        this.renderQueue = [];
        this.skipNextWeightDraw = false;  // Phase 1.3: Skip weights when empty
        this.cachedWeights = null;  // Phase 1.3: Cache for weight data to handle selective transmission

        // Animation state
        this.pulsePhase = 0;
        this.pulses = [];
        this.pulseSpawnTimer = 0;
        this.prevActivations = {};
        this.interpolationSpeed = 0.3;

        // Colors - matches pygame visualizer
        this.colors = {
            bg: '#0c0c18',
            panel: '#121220',
            text: '#c8c8dc',
            inactive: '#282837',
            negative: '#4287f5',     // Blue for negative
            neutral: '#c8c8d2',      // White-ish for zero
            positive: '#f5426c',     // Red/pink for positive
            weightNegative: '#2962ff',  // Blue
            weightNeutral: '#646478',   // Gray
            weightPositive: '#ff6229',  // Orange/red
            border: '#283c64',
            live: '#64c896'
        };

        // Layout
        this.margin = 20;
        this.headerHeight = 50;
        this.qvalueHeight = 85;
        this.layerLabelHeight = 25;
        this.neuronRadius = 6;
        this.maxNeurons = 15;

        // Start animation loop
        this.startAnimation();

        // Handle resize
        this.setupResize();

        // Phase 2: Setup mouse handling for neuron inspection
        this.setupMouseHandling();
    }

    setupResize() {
        // Clean up old observer to prevent memory leak
        if (this.resizeObserver) {
            this.resizeObserver.disconnect();
        }

        this.resizeObserver = new ResizeObserver(() => {
            if (this.canvas && this.canvas.parentElement) {
                const rect = this.canvas.parentElement.getBoundingClientRect();
                const dpr = window.devicePixelRatio || 1;
                this.canvas.width = rect.width * dpr;
                this.canvas.height = rect.height * dpr;
                this.canvas.style.width = rect.width + 'px';
                this.canvas.style.height = rect.height + 'px';
                this.ctx.scale(dpr, dpr);
                this.width = rect.width;
                this.height = rect.height;
            }
        });

        if (this.canvas.parentElement) {
            this.resizeObserver.observe(this.canvas.parentElement);
        }
    }

    // ===== Phase 2: Neuron Inspection =====

    setupMouseHandling() {
        // Phase 2: Set up click handlers for neuron and layer inspection
        if (!this.canvas) return;

        // Remove old handler to prevent memory leak
        if (this.clickHandler) {
            this.canvas.removeEventListener('click', this.clickHandler);
        }

        this.clickHandler = (e) => {
            const rect = this.canvas.getBoundingClientRect();
            const x = e.clientX - rect.left;
            const y = e.clientY - rect.top;

            // Check if click is on a layer label
            const layerClicked = this.findLayerLabelAtPosition(x, y);
            if (layerClicked !== null) {
                this.selectLayer(layerClicked);
                return;
            }

            // Check if click is on a neuron
            const neuronClicked = this.findNeuronAtPosition(x, y);
            if (neuronClicked) {
                this.selectNeuron(neuronClicked.layerIdx, neuronClicked.neuronIdx);
            }
        };

        this.canvas.addEventListener('click', this.clickHandler);
    }

    findNeuronAtPosition(clickX, clickY) {
        // Phase 2: Find neuron at mouse position
        if (!this.data || !this.width || !this.height) return null;

        const layerPositions = this.calculateLayerPositions();
        if (!layerPositions) return null;

        const clickRadius = 12; // Larger than neuron for easier clicking

        for (let layerIdx = 0; layerIdx < layerPositions.length; layerIdx++) {
            const layer = layerPositions[layerIdx];
            for (let neuronIdx = 0; neuronIdx < layer.positions.length; neuronIdx++) {
                const pos = layer.positions[neuronIdx];
                const dist = Math.sqrt((clickX - pos.x) ** 2 + (clickY - pos.y) ** 2);
                if (dist < clickRadius) {
                    return { layerIdx, neuronIdx };
                }
            }
        }
        return null;
    }

    selectNeuron(layerIdx, neuronIdx) {
        // Phase 2: Load and display neuron inspection panel
        // Fetch neuron details from backend
        fetchWithTimeout(`/api/neuron/${layerIdx}/${neuronIdx}`)
            .then(response => response.json())
            .then(data => {
                this.displayNeuronInspection(data);
            })
            .catch(err => console.error('Failed to load neuron details:', err));
    }

    // ===== Phase 2: Layer Analysis =====

    findLayerLabelAtPosition(clickX, clickY) {
        // Phase 2: Find layer label at mouse position
        if (!this.data || !this.data.layer_info) return null;

        const layerPositions = this.calculateLayerPositions();
        if (!layerPositions) return null;

        const labelY = this.headerHeight + 5;
        const clickRadius = 20; // Larger click area for labels

        for (let i = 0; i < layerPositions.length; i++) {
            const layerPos = layerPositions[i];
            const distX = Math.abs(clickX - layerPos.x);
            const distY = Math.abs(clickY - labelY);

            // Check if click is within bounds of layer label
            if (distX < clickRadius && distY < clickRadius) {
                return i;
            }
        }
        return null;
    }

    selectLayer(layerIdx) {
        // Phase 2: Load and display layer analysis panel
        // Fetch layer analysis from backend
        fetchWithTimeout(`/api/layer/${layerIdx}`)
            .then(response => response.json())
            .then(data => {
                this.displayLayerAnalysis(data);
            })
            .catch(err => console.error('Failed to load layer analysis:', err));
    }

    startAnimation() {
        // Stop any existing animation to prevent duplicate loops
        this.stopAnimation();

        const animate = (currentTime) => {
            // Phase 1.4: Adaptive render throttling
            // Only render if enough time has passed since last render
            if (this.isEnabled && this.data) {
                const timeSinceLastRender = currentTime - this.lastRenderTime;
                if (timeSinceLastRender >= this.renderInterval) {
                    const renderStart = performance.now();
                    this.render();
                    const renderDuration = performance.now() - renderStart;

                    // Track render performance and adjust interval if struggling
                    this.avgRenderTime = this.avgRenderTime * 0.8 + renderDuration * 0.2;
                    if (this.avgRenderTime > this.renderInterval * 0.8) {
                        // Struggling to keep up - increase render interval (reduce FPS)
                        this.renderInterval = Math.min(100, this.renderInterval + 5);
                    } else if (this.avgRenderTime < this.renderInterval * 0.3 && this.renderInterval > 16) {
                        // Running smoothly - can afford to render more frequently
                        this.renderInterval = Math.max(16, this.renderInterval - 2);
                    }

                    this.lastRenderTime = currentTime;
                }
            }
            this.animationId = requestAnimationFrame(animate);
        };
        animate(performance.now());
    }

    stopAnimation() {
        if (this.animationId) {
            cancelAnimationFrame(this.animationId);
            this.animationId = null;
        }
    }

    /**
     * Clean up all resources to prevent memory leaks.
     * Call this before creating a new visualizer or when disabling.
     */
    destroy() {
        // Stop animation loop
        this.stopAnimation();

        // Clean up ResizeObserver
        if (this.resizeObserver) {
            this.resizeObserver.disconnect();
            this.resizeObserver = null;
        }

        // Clean up click handler
        if (this.clickHandler && this.canvas) {
            this.canvas.removeEventListener('click', this.clickHandler);
            this.clickHandler = null;
        }

        // Clear data references
        this.data = null;
        this.cachedWeights = null;
    }

    update(data) {
        if (!data || !data.layer_info || data.layer_info.length === 0) {
            return;
        }
        this.data = data;

        // Phase 1.3: Cache weights if provided (selective transmission)
        // This ensures we have valid weight data even when backend sends empty arrays
        if (data.weights && data.weights.length > 0) {
            this.cachedWeights = data.weights;
        }

        // Hide placeholder when we receive valid data
        const placeholder = document.getElementById('nn-viz-placeholder');
        if (placeholder) {
            placeholder.style.display = 'none';
        }
    }

    toggle(enabled) {
        this.isEnabled = enabled;
        const indicator = document.getElementById('nn-viz-status');
        if (indicator) {
            indicator.textContent = enabled ? '● LIVE' : '○ Paused';
            indicator.style.color = enabled ? this.colors.live : '#666';
        }
    }

    render() {
        if (!this.ctx || !this.data || !this.width || !this.height) return;

        const ctx = this.ctx;
        const dpr = window.devicePixelRatio || 1;

        // Clear with scale reset
        ctx.setTransform(1, 0, 0, 1, 0, 0);
        ctx.scale(dpr, dpr);

        // Draw background
        this.drawBackground();

        // Calculate layer positions
        const layerPositions = this.calculateLayerPositions();
        if (!layerPositions || layerPositions.length === 0) return;

        // Smooth activations
        const smoothedActivations = this.smoothActivations();

        // Draw connections
        this.drawConnections(layerPositions, smoothedActivations);

        // Update and draw pulses
        this.updatePulses(layerPositions, smoothedActivations);
        this.drawPulses();

        // Draw neurons
        this.drawNeurons(layerPositions, smoothedActivations);

        // Draw layer labels
        this.drawLayerLabels(layerPositions);

        // Draw Q-values
        this.drawQValues();

        // Draw title
        this.drawTitle();

        // Update animation phase
        this.pulsePhase = (this.pulsePhase + 0.08) % (Math.PI * 2);
    }

    drawBackground() {
        const ctx = this.ctx;

        // Gradient background
        const gradient = ctx.createLinearGradient(0, 0, 0, this.height);
        gradient.addColorStop(0, this.colors.bg);
        gradient.addColorStop(1, this.colors.panel);
        ctx.fillStyle = gradient;
        ctx.fillRect(0, 0, this.width, this.height);

        // Animated border glow
        const glowIntensity = 20 + 10 * Math.sin(this.pulsePhase);
        ctx.strokeStyle = `rgb(${40 + glowIntensity}, ${60 + glowIntensity}, ${100 + glowIntensity})`;
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.roundRect(1, 1, this.width - 2, this.height - 2, 8);
        ctx.stroke();
    }

    drawTitle() {
        const ctx = this.ctx;

        // Main title
        ctx.font = 'bold 16px "Plus Jakarta Sans", sans-serif';
        ctx.fillStyle = '#64b5f6';
        ctx.textAlign = 'center';
        ctx.fillText('Neural Network', this.width / 2, 22);

        // LIVE indicator with pulse
        const pulse = 0.7 + 0.3 * Math.sin(this.pulsePhase * 0.5);
        ctx.font = '11px "JetBrains Mono", monospace';
        ctx.fillStyle = `rgba(${100 * pulse}, ${200 * pulse}, ${150 * pulse}, 1)`;
        ctx.textAlign = 'left';
        ctx.fillText('● LIVE', 10, 38);

        // Step counter
        ctx.textAlign = 'right';
        ctx.fillStyle = '#787890';
        ctx.fillText(`Step: ${(this.data.step || 0).toLocaleString()}`, this.width - 10, 38);
    }

    calculateLayerPositions() {
        const layerInfo = this.data.layer_info;
        if (!layerInfo || layerInfo.length === 0) return [];

        const numLayers = layerInfo.length;
        const hMargin = 35;
        const availableWidth = this.width - (hMargin * 2);
        const layerSpacing = availableWidth / Math.max(numLayers - 1, 1);

        const networkTop = this.headerHeight + this.layerLabelHeight;
        const networkBottom = this.height - this.qvalueHeight - 10;
        const availableHeight = networkBottom - networkTop;

        const positions = [];

        for (let i = 0; i < layerInfo.length; i++) {
            const info = layerInfo[i];
            const layerX = hMargin + i * layerSpacing;
            const numNeurons = Math.min(info.neurons, this.maxNeurons);

            const neuronSpacing = Math.min(22, availableHeight / Math.max(numNeurons + 1, 1));
            const totalHeight = numNeurons * neuronSpacing;
            const startY = networkTop + (availableHeight - totalHeight) / 2;

            const neuronPositions = [];
            for (let j = 0; j < numNeurons; j++) {
                const ny = startY + j * neuronSpacing + neuronSpacing / 2;
                neuronPositions.push({ x: layerX, y: ny });
            }

            positions.push({
                x: layerX,
                neurons: numNeurons,
                actualNeurons: info.neurons,
                positions: neuronPositions,
                type: info.type,
                name: info.name
            });
        }

        return positions;
    }

    smoothActivations() {
        const activations = this.data.activations || {};
        const smoothed = {};

        for (const key in activations) {
            const newAct = activations[key];
            if (!newAct || newAct.length === 0) continue;

            if (this.prevActivations[key] && this.prevActivations[key].length === newAct.length) {
                smoothed[key] = this.prevActivations[key].map((prev, i) =>
                    prev + (newAct[i] - prev) * this.interpolationSpeed
                );
            } else {
                smoothed[key] = [...newAct];
            }

            this.prevActivations[key] = [...smoothed[key]];
        }

        return smoothed;
    }

    drawConnections(layerPositions, activations) {
        const ctx = this.ctx;

        // Phase 1.3: Use cached weights if current data doesn't have them
        // This handles selective transmission where backend sends empty arrays every 100 steps
        const weights = (this.data.weights && this.data.weights.length > 0)
            ? this.data.weights
            : this.cachedWeights;

        // Only skip if we've never received weights
        if (!weights || weights.length === 0) {
            return;
        }

        for (let i = 0; i < layerPositions.length - 1; i++) {
            const fromLayer = layerPositions[i];
            const toLayer = layerPositions[i + 1];

            if (i < weights.length && weights[i]) {
                const weightMatrix = weights[i];
                let maxWeight = 0;

                // Find max weight for normalization
                for (const row of weightMatrix) {
                    for (const w of row) {
                        if (Math.abs(w) > maxWeight) maxWeight = Math.abs(w);
                    }
                }
                maxWeight = maxWeight || 1;

                // Sample connections
                const maxConnections = 60;
                const fromSample = this.sampleIndices(fromLayer.positions.length, 6);
                const toSample = this.sampleIndices(toLayer.positions.length, 10);

                for (const fi of fromSample) {
                    for (const ti of toSample) {
                        if (ti < weightMatrix.length && fi < (weightMatrix[ti]?.length || 0)) {
                            const weight = weightMatrix[ti][fi];
                            const normWeight = weight / maxWeight;

                            // Diverging color
                            let color;
                            if (normWeight > 0) {
                                color = this.interpolateColor(
                                    this.hexToRgb(this.colors.weightNeutral),
                                    this.hexToRgb(this.colors.weightPositive),
                                    Math.abs(normWeight)
                                );
                            } else {
                                color = this.interpolateColor(
                                    this.hexToRgb(this.colors.weightNeutral),
                                    this.hexToRgb(this.colors.weightNegative),
                                    Math.abs(normWeight)
                                );
                            }

                            const fromPos = fromLayer.positions[fi];
                            const toPos = toLayer.positions[ti];

                            ctx.strokeStyle = `rgb(${color.r}, ${color.g}, ${color.b})`;
                            ctx.lineWidth = Math.max(0.5, Math.abs(normWeight) * 2);
                            ctx.globalAlpha = 0.4 + Math.abs(normWeight) * 0.4;

                            ctx.beginPath();
                            ctx.moveTo(fromPos.x, fromPos.y);
                            ctx.lineTo(toPos.x, toPos.y);
                            ctx.stroke();
                        }
                    }
                }

                ctx.globalAlpha = 1;
            }
        }
    }

    sampleIndices(length, max) {
        if (length <= max) {
            return Array.from({ length }, (_, i) => i);
        }
        const step = length / max;
        return Array.from({ length: max }, (_, i) => Math.floor(i * step));
    }

    updatePulses(layerPositions, activations) {
        // Update existing pulses
        this.pulses = this.pulses.filter(p => {
            p.progress += p.speed;
            return p.progress < 1;
        });

        // Spawn new pulses
        this.pulseSpawnTimer++;
        if (this.pulseSpawnTimer >= 5 && layerPositions.length > 1) {
            this.pulseSpawnTimer = 0;

            for (let i = 0; i < layerPositions.length - 1; i++) {
                const fromLayer = layerPositions[i];
                const toLayer = layerPositions[i + 1];

                if (fromLayer.positions.length > 0 && toLayer.positions.length > 0) {
                    for (let j = 0; j < 2; j++) {
                        const fromIdx = Math.floor(Math.random() * fromLayer.positions.length);
                        const toIdx = Math.floor(Math.random() * toLayer.positions.length);

                        const layerKey = `layer_${i}`;
                        let actLevel = 0.5;
                        if (activations[layerKey]) {
                            const acts = activations[layerKey];
                            actLevel = acts.reduce((a, b) => a + Math.abs(b), 0) / acts.length;
                        }

                        const intensity = Math.min(1, actLevel * 2);
                        const color = this.interpolateColor(
                            { r: 60, g: 80, b: 120 },
                            { r: 100, g: 255, b: 180 },
                            intensity
                        );

                        this.pulses.push({
                            start: fromLayer.positions[fromIdx],
                            end: toLayer.positions[toIdx],
                            color: color,
                            progress: 0,
                            speed: 0.08
                        });
                    }
                }
            }
        }
    }

    drawPulses() {
        const ctx = this.ctx;

        for (const pulse of this.pulses) {
            const x = pulse.start.x + (pulse.end.x - pulse.start.x) * pulse.progress;
            const y = pulse.start.y + (pulse.end.y - pulse.start.y) * pulse.progress;

            const alpha = 1 - Math.abs(pulse.progress - 0.5) * 2;
            const size = 3 + 2 * alpha;

            // Glow
            ctx.beginPath();
            ctx.arc(x, y, size + 2, 0, Math.PI * 2);
            ctx.fillStyle = `rgba(${pulse.color.r}, ${pulse.color.g}, ${pulse.color.b}, ${alpha * 0.5})`;
            ctx.fill();

            // Core
            ctx.beginPath();
            ctx.arc(x, y, size, 0, Math.PI * 2);
            ctx.fillStyle = `rgba(${Math.min(255, pulse.color.r * 1.5)}, ${Math.min(255, pulse.color.g * 1.5)}, ${Math.min(255, pulse.color.b * 1.5)}, ${alpha})`;
            ctx.fill();
        }
    }

    drawNeurons(layerPositions, activations) {
        const ctx = this.ctx;
        const layerInfo = this.data.layer_info;

        for (let i = 0; i < layerPositions.length; i++) {
            const layerPos = layerPositions[i];
            const info = layerInfo[i];

            // Get activations for this layer
            let layerActs = [];
            if (info.type === 'input') {
                // For input layer, use first activation values if available
                const inputKey = 'input' in activations ? 'input' : 'layer_-1';
                layerActs = activations[inputKey] || [];
            } else {
                const layerKey = `layer_${i - 1}`;
                layerActs = activations[layerKey] || [];
            }

            // Normalize activations
            let maxAct = 1;
            if (layerActs.length > 0) {
                maxAct = Math.max(...layerActs.map(Math.abs), 0.001);
            }

            for (let j = 0; j < layerPos.positions.length; j++) {
                const pos = layerPos.positions[j];
                const actVal = j < layerActs.length ? layerActs[j] / maxAct : 0;

                // Diverging color based on activation sign
                let color;
                if (actVal > 0) {
                    color = this.interpolateColor(
                        this.hexToRgb(this.colors.inactive),
                        this.hexToRgb(this.colors.positive),
                        Math.min(Math.abs(actVal), 1)
                    );
                } else {
                    color = this.interpolateColor(
                        this.hexToRgb(this.colors.inactive),
                        this.hexToRgb(this.colors.negative),
                        Math.min(Math.abs(actVal), 1)
                    );
                }

                // Radius with pulse for active neurons
                let radius = this.neuronRadius;
                if (Math.abs(actVal) > 0.7) {
                    radius *= 1 + 0.2 * Math.sin(this.pulsePhase + j * 0.5);
                }

                // Outer glow for active neurons
                if (Math.abs(actVal) > 0.4) {
                    ctx.beginPath();
                    ctx.arc(pos.x, pos.y, radius + 4, 0, Math.PI * 2);
                    ctx.fillStyle = `rgba(${color.r}, ${color.g}, ${color.b}, ${Math.abs(actVal) * 0.4})`;
                    ctx.fill();
                }

                // Neuron body
                ctx.beginPath();
                ctx.arc(pos.x, pos.y, radius, 0, Math.PI * 2);
                ctx.fillStyle = `rgb(${color.r}, ${color.g}, ${color.b})`;
                ctx.fill();

                // Border
                ctx.strokeStyle = '#505a6e';
                ctx.lineWidth = 1;
                ctx.stroke();

                // Highlight
                ctx.beginPath();
                ctx.arc(pos.x - radius * 0.3, pos.y - radius * 0.3, radius / 3, 0, Math.PI * 2);
                ctx.fillStyle = `rgba(${Math.min(255, color.r + 50)}, ${Math.min(255, color.g + 50)}, ${Math.min(255, color.b + 50)}, 0.6)`;
                ctx.fill();
            }

            // Ellipsis for hidden neurons
            if (layerPos.neurons < layerPos.actualNeurons) {
                const lastPos = layerPos.positions[layerPos.positions.length - 1];
                ctx.font = '11px "JetBrains Mono", monospace';
                ctx.fillStyle = '#646478';
                ctx.textAlign = 'center';
                ctx.fillText(`+${layerPos.actualNeurons - layerPos.neurons}`, layerPos.x, lastPos.y + 20);
            }
        }
    }

    drawLayerLabels(layerPositions) {
        const ctx = this.ctx;
        const layerInfo = this.data.layer_info;
        const labelY = this.headerHeight + 5;

        ctx.font = '10px "JetBrains Mono", monospace';
        ctx.textAlign = 'center';

        for (let i = 0; i < layerPositions.length; i++) {
            const layerPos = layerPositions[i];
            const info = layerInfo[i];

            let name, color;
            if (info.type === 'input') {
                name = 'IN';
                color = '#64b5f6';
            } else if (info.type === 'output') {
                name = 'OUT';
                color = '#ff9664';
            } else {
                name = `H${i}`;
                color = '#96c896';
            }

            // Layer name
            ctx.fillStyle = color;
            ctx.fillText(name, layerPos.x, labelY);

            // Neuron count
            ctx.fillStyle = '#5a5a6e';
            ctx.fillText(`(${layerPos.actualNeurons})`, layerPos.x, labelY + 12);
        }
    }

    drawQValues() {
        const ctx = this.ctx;
        const qValues = this.data.q_values || [];
        const selectedAction = this.data.selected_action || 0;
        const actionLabels = this.data.action_labels || ['LEFT', 'STAY', 'RIGHT'];
        const actionIcons = ['◀', '●', '▶'];

        if (qValues.length === 0) return;

        const qvY = this.height - this.qvalueHeight;
        const qvHeight = this.qvalueHeight - 5;

        // Background panel
        ctx.fillStyle = '#14161e';
        ctx.beginPath();
        ctx.roundRect(8, qvY, this.width - 16, qvHeight, 8);
        ctx.fill();
        ctx.strokeStyle = '#32374b';
        ctx.lineWidth = 1;
        ctx.stroke();

        // Title
        ctx.font = '10px "JetBrains Mono", monospace';
        ctx.fillStyle = '#8c8ca0';
        ctx.textAlign = 'left';
        ctx.fillText('Q-Values', 15, qvY + 14);

        // Normalize Q-values
        const qMin = Math.min(...qValues);
        const qMax = Math.max(...qValues);
        const qRange = qMax - qMin + 0.001;

        // Bar layout
        const contentWidth = this.width - 50;
        const barWidth = contentWidth / qValues.length;
        const barMaxHeight = 35;
        const bestAction = qValues.indexOf(Math.max(...qValues));

        for (let i = 0; i < qValues.length; i++) {
            const qVal = qValues[i];
            const barX = 20 + i * barWidth;

            const normQ = (qVal - qMin) / qRange;
            const barHeight = Math.max(8, normQ * barMaxHeight);

            const isSelected = i === selectedAction || (selectedAction === undefined && i === bestAction);

            if (isSelected) {
                // Animated selected action
                const pulse = 0.8 + 0.2 * Math.sin(this.pulsePhase * 2);
                ctx.fillStyle = `rgb(${Math.floor(50 * pulse)}, ${Math.floor(220 * pulse)}, ${Math.floor(120 * pulse)})`;

                // Glow
                ctx.beginPath();
                ctx.roundRect(barX - 2, qvY + 45 - barHeight - 2, barWidth - 8 + 4, barHeight + 4, 4);
                ctx.fillStyle = 'rgba(30, 100, 60, 0.5)';
                ctx.fill();

                ctx.fillStyle = `rgb(${Math.floor(50 * pulse)}, ${Math.floor(220 * pulse)}, ${Math.floor(120 * pulse)})`;
            } else {
                ctx.fillStyle = '#373c50';
            }

            // Draw bar
            ctx.beginPath();
            ctx.roundRect(barX, qvY + 45 - barHeight, barWidth - 10, barHeight, 4);
            ctx.fill();
            ctx.strokeStyle = isSelected ? '#64ffa0' : '#505564';
            ctx.lineWidth = 1;
            ctx.stroke();

            // Draw icon
            ctx.font = '14px sans-serif';
            ctx.fillStyle = isSelected ? '#dcdcf0' : '#787890';
            ctx.textAlign = 'center';
            ctx.fillText(actionIcons[i] || '?', barX + barWidth / 2 - 5, qvY + 60);

            // Q-value number
            ctx.font = '9px "JetBrains Mono", monospace';
            ctx.fillStyle = '#646478';
            ctx.fillText(qVal.toFixed(2), barX + barWidth / 2 - 5, qvY + 45 - barHeight - 4);
        }
    }

    // Utility functions
    hexToRgb(hex) {
        const result = /^#?([a-f\d]{2})([a-f\d]{2})([a-f\d]{2})$/i.exec(hex);
        return result ? {
            r: parseInt(result[1], 16),
            g: parseInt(result[2], 16),
            b: parseInt(result[3], 16)
        } : { r: 0, g: 0, b: 0 };
    }

    interpolateColor(color1, color2, t) {
        t = Math.max(0, Math.min(1, t));
        // Ease-out for smoother transitions
        t = 1 - Math.pow(1 - t, 2);
        return {
            r: Math.round(color1.r + (color2.r - color1.r) * t),
            g: Math.round(color1.g + (color2.g - color1.g) * t),
            b: Math.round(color1.b + (color2.b - color1.b) * t)
        };
    }
}

// Global NN visualizer instance
let nnVisualizer = null;

/**
 * Initialize the neural network visualizer
 */
function initNNVisualizer() {
    const canvas = document.getElementById('nn-canvas');
    if (canvas) {
        nnVisualizer = new NeuralNetworkVisualizer('nn-canvas');
        console.log('Neural Network Visualizer initialized');
    }
}

/**
 * Toggle neural network visualization
 */
function toggleNNVisualization() {
    if (nnVisualizer) {
        nnVisualizer.toggle(!nnVisualizer.isEnabled);
    }
}

/**
 * Phase 1.2: Update NN visualizer render rate based on training speed.
 *
 * High-speed training: Reduce render FPS to save resources
 * Slow training: Increase render FPS for smooth visuals
 */
function updateNNVisualizerRenderRate(stepsPerSec) {
    if (!nnVisualizer) return;

    if (stepsPerSec > 2000) {
        // Very high speed training - render at 10Hz
        nnVisualizer.renderInterval = 100;
    } else if (stepsPerSec > 1000) {
        // High speed - render at ~15Hz
        nnVisualizer.renderInterval = 67;
    } else if (stepsPerSec > 500) {
        // Medium speed - render at ~30Hz
        nnVisualizer.renderInterval = 33;
    } else {
        // Slow training or visual mode - render at up to 60Hz for smoothness
        nnVisualizer.renderInterval = 16;
    }
}

/**
 * Collapse/expand NN visualization panel
 */
function toggleNNPanel() {
    const card = document.getElementById('nn-viz-card');
    const icon = document.getElementById('nn-viz-icon');
    const trigger = document.querySelector('[data-action="toggle-nn-panel"]');

    if (!card) return;

    card.classList.toggle('collapsed');
    const isExpanded = !card.classList.contains('collapsed');
    if (icon) {
        icon.textContent = isExpanded ? '▲' : '▼';
    }
    if (trigger) {
        trigger.setAttribute('aria-expanded', String(isExpanded));
    }
}

/**
 * Phase 2: Close neuron inspection panel
 */
function closeNeuronInspection() {
    const panel = document.getElementById('neuron-inspection-panel');
    if (panel) {
        panel.style.display = 'none';
        panel.innerHTML = '';
    }
}

function closeLayerAnalysis() {
    const panel = document.getElementById('layer-analysis-panel');
    if (panel) {
        panel.style.display = 'none';
        panel.innerHTML = '';
    }
}
