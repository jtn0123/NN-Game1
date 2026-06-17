import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { test } from 'node:test';
import vm from 'node:vm';

function createClassList() {
  const values = new Set();
  return {
    add(...classes) {
      classes.forEach((className) => values.add(className));
    },
    remove(...classes) {
      classes.forEach((className) => values.delete(className));
    },
    contains(className) {
      return values.has(className);
    },
    toggle(className, force) {
      const shouldAdd = force ?? !values.has(className);
      if (shouldAdd) {
        values.add(className);
      } else {
        values.delete(className);
      }
      return shouldAdd;
    },
  };
}

function createElement(id = '') {
  return {
    id,
    children: [],
    classList: createClassList(),
    style: {},
    dataset: {},
    textContent: '',
    value: '',
    scrollHeight: 0,
    scrollTop: 0,
    addEventListener() {},
    append(...children) {
      this.children.push(...children);
    },
    appendChild(child) {
      this.children.push(child);
      return child;
    },
    replaceChildren(...children) {
      this.children = children;
    },
    removeChild(child) {
      this.children = this.children.filter((candidate) => candidate !== child);
    },
    setAttribute(name, value) {
      this[name] = value;
    },
    getContext() {
      return {};
    },
    closest() {
      return null;
    },
  };
}

function createDashboardContext() {
  const errors = [];
  const elements = new Map([
    ['console-output', createElement('console-output')],
    ['console-container', createElement('console-container')],
    ['status-dot', createElement('status-dot')],
    ['status-text', createElement('status-text')],
  ]);
  const listeners = new Map();

  const documentRef = {
    addEventListener(event, handler) {
      listeners.set(event, handler);
    },
    createDocumentFragment() {
      return createElement('fragment');
    },
    createElement(tagName) {
      return createElement(tagName);
    },
    getElementById(id) {
      return elements.get(id) || null;
    },
    querySelector(selector) {
      if (selector === 'meta[name="dashboard-token"]') {
        return { content: 'test-token' };
      }
      if (selector === '.status-dot') {
        return elements.get('status-dot');
      }
      if (selector === '.status-text') {
        return elements.get('status-text');
      }
      return null;
    },
    querySelectorAll() {
      return [];
    },
  };

  const context = vm.createContext({
    AbortController: class {
      constructor() {
        this.signal = {};
      }
      abort() {}
    },
    DashboardCore: {
      readToken: () => 'test-token',
      withDashboardToken: (options) => options,
    },
    console: {
      error: (...args) => errors.push(args.join(' ')),
      log() {},
      warn() {},
    },
    document: documentRef,
    fetch: () => Promise.resolve({ json: () => Promise.resolve({}) }),
    setInterval: () => 1,
    clearInterval() {},
    setTimeout: () => 1,
    clearTimeout() {},
  });

  const chartsSource = readFileSync(resolve('src/web/static/dashboard_charts.js'), 'utf8');
  const source = readFileSync(resolve('src/web/static/app.js'), 'utf8');
  vm.runInContext(chartsSource, context, { filename: 'dashboard_charts.js' });
  vm.runInContext(source, context, { filename: 'app.js' });
  return { context, elements, errors, listeners };
}

test('initCharts reports a visible startup error when Chart.js is unavailable', () => {
  const { context, elements, errors } = createDashboardContext();

  assert.equal(context.initCharts(), false);

  assert.match(errors.join('\n'), /Chart\.js failed to load/);
  assert.equal(elements.get('console-output').children.length, 1);
  assert.equal(
    elements.get('console-output').children[0].children.at(-1).textContent,
    'Charts unavailable: Chart.js failed to load',
  );
});

test('connectSocket reports a visible startup error when Socket.IO is unavailable', () => {
  const { context, elements, errors } = createDashboardContext();

  assert.equal(context.connectSocket(), false);

  assert.match(errors.join('\n'), /Socket\.IO failed to load/);
  assert.equal(elements.get('status-text').textContent, 'Disconnected');
  assert.equal(
    elements.get('console-output').children[0].children.at(-1).textContent,
    'Live connection unavailable: Socket.IO failed to load',
  );
});

test('updateCharts is a no-op before chart instances exist', () => {
  const { context } = createDashboardContext();

  assert.doesNotThrow(() => context.updateCharts({
    scores: [10],
    losses: [0.5],
    q_values: [1.25],
  }, 1));
});
