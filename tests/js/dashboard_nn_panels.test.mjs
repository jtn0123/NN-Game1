import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { test } from 'node:test';
import vm from 'node:vm';

function createElement(tagName = 'div') {
  const classes = new Set();
  const element = {
    tagName: tagName.toUpperCase(),
    children: [],
    classList: {
      add(...names) {
        names.forEach((name) => classes.add(name));
      },
      remove(...names) {
        names.forEach((name) => classes.delete(name));
      },
      contains(name) {
        return classes.has(name);
      },
    },
    className: '',
    dataset: {},
    id: '',
    parentNode: null,
    style: {},
    textContent: '',
    width: 0,
    height: 0,
    appendChild(child) {
      child.parentNode = this;
      this.children.push(child);
      return child;
    },
    replaceChildren(...children) {
      this.children.forEach((child) => {
        child.parentNode = null;
      });
      this.children = [];
      children.forEach((child) => this.appendChild(child));
    },
    setAttribute(name, value) {
      this[name] = String(value);
    },
    getContext() {
      return {};
    },
    set innerHTML(_value) {
      throw new Error('dashboard panels should not assign raw HTML');
    },
  };
  return element;
}

function loadPanels(documentRef) {
  function NeuralNetworkVisualizer() {}
  const context = vm.createContext({
    NeuralNetworkVisualizer,
    clampPercent(value) {
      return Math.max(0, Math.min(100, Number(value) || 0));
    },
    console: { warn() {} },
    document: documentRef,
    formatFixedValue(value, digits, fallback = 'N/A') {
      const numberValue = Number(value);
      return Number.isFinite(numberValue) ? numberValue.toFixed(digits) : fallback;
    },
  });
  vm.runInContext(
    readFileSync(resolve('src/web/static/dashboard_nn_panels.js'), 'utf8'),
    context,
    { filename: 'dashboard_nn_panels.js' },
  );
  return context.NeuralNetworkVisualizer;
}

test('neuron and layer panels render runtime labels as text nodes', () => {
  const neuronPanel = createElement('div');
  const layerPanel = createElement('div');
  const documentRef = {
    createElement,
    getElementById(id) {
      if (id === 'neuron-inspection-panel') return neuronPanel;
      if (id === 'layer-analysis-panel') return layerPanel;
      return null;
    },
  };
  const Visualizer = loadPanels(documentRef);
  const visualizer = new Visualizer();
  const hostileLabel = '<img src=x onerror=alert(1)>';

  visualizer.displayNeuronInspection({
    layer_name: hostileLabel,
    layer_idx: 1,
    neuron_idx: 2,
    current_activation: 0.25,
    incoming_weight_stats: { mean: 0.1, min: -0.2, max: 0.3 },
    outgoing_weight_stats: null,
    q_value_contributions: { '<script>': 0.5 },
    activation_history: [],
  });
  visualizer.displayLayerAnalysis({
    layer_name: hostileLabel,
    neuron_count: 3,
    dead_neuron_count: 0,
    saturated_neuron_count: 0,
    dead_neuron_percent: 0,
    saturated_percent: 0,
  });

  assert.equal(neuronPanel.children[0].children[0].textContent, `${hostileLabel} - Neuron #2`);
  assert.equal(layerPanel.children[0].children[0].textContent, hostileLabel);
});
