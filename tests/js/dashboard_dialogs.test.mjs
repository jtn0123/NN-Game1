import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { test } from 'node:test';
import vm from 'node:vm';

function createDom() {
  const listeners = new Map();
  let documentRef;

  function createElement(tagName = 'div') {
    const element = {
      tagName: tagName.toUpperCase(),
      children: [],
      className: '',
      dataset: {},
      disabled: false,
      id: '',
      parentNode: null,
      style: {},
      textContent: '',
      appendChild(child) {
        child.parentNode = this;
        this.children.push(child);
        return child;
      },
      removeChild(child) {
        this.children = this.children.filter((candidate) => candidate !== child);
        child.parentNode = null;
      },
      replaceChildren(...children) {
        this.children.forEach((child) => {
          child.parentNode = null;
        });
        this.children = [];
        children.forEach((child) => this.appendChild(child));
      },
      addEventListener(type, handler) {
        this[`on${type}`] = handler;
      },
      setAttribute(name, value) {
        this[name] = String(value);
      },
      getAttribute(name) {
        return this[name];
      },
      focus() {
        documentRef.activeElement = this;
      },
      contains(candidate) {
        if (candidate === this) return true;
        return this.children.some((child) => child.contains?.(candidate));
      },
      closest(selector) {
        if (
          selector === 'button[data-choice]'
          && this.tagName === 'BUTTON'
          && Object.prototype.hasOwnProperty.call(this.dataset, 'choice')
        ) {
          return this;
        }
        return this.parentNode?.closest?.(selector) || null;
      },
      querySelector(selector) {
        return findFirst(this, (candidate) => {
          if (selector === '.primary, .danger, button') {
            return (
              candidate.tagName === 'BUTTON'
              || candidate.className.split(/\s+/).includes('primary')
              || candidate.className.split(/\s+/).includes('danger')
            );
          }
          return false;
        });
      },
      querySelectorAll() {
        return findAll(this, (candidate) => candidate.tagName === 'BUTTON' && !candidate.disabled);
      },
    };
    return element;
  }

  function findFirst(root, predicate) {
    for (const child of root.children) {
      if (predicate(child)) {
        return child;
      }
      const nested = findFirst(child, predicate);
      if (nested) {
        return nested;
      }
    }
    return null;
  }

  function findAll(root, predicate) {
    const results = [];
    root.children.forEach((child) => {
      if (predicate(child)) {
        results.push(child);
      }
      results.push(...findAll(child, predicate));
    });
    return results;
  }

  documentRef = {
    activeElement: null,
    body: createElement('body'),
    createElement,
    addEventListener(type, handler) {
      listeners.set(type, handler);
    },
    removeEventListener(type, handler) {
      if (listeners.get(type) === handler) {
        listeners.delete(type);
      }
    },
  };
  const opener = createElement('button');
  documentRef.activeElement = opener;
  return { documentRef, listeners, opener };
}

function loadDialogs(documentRef) {
  const context = vm.createContext({ document: documentRef });
  vm.runInContext(
    readFileSync(resolve('src/web/static/dashboard_dialogs.js'), 'utf8'),
    context,
    { filename: 'dashboard_dialogs.js' },
  );
  return context.DashboardDialogs;
}

test('DashboardDialogs traps tab focus and restores previous focus on close', async () => {
  const { documentRef, listeners, opener } = createDom();
  const dialogs = loadDialogs(documentRef);

  const resultPromise = dialogs.choose({
    title: 'Pick one',
    choices: [{ value: 'primary', label: 'Primary', variant: 'primary' }],
  });

  const backdrop = documentRef.body.children[0];
  const dialog = backdrop.children[0];
  const buttons = dialog.querySelectorAll('button');
  const primaryButton = buttons[0];
  const cancelButton = buttons[1];

  assert.equal(documentRef.activeElement, primaryButton);

  cancelButton.focus();
  let prevented = false;
  listeners.get('keydown')({
    key: 'Tab',
    shiftKey: false,
    preventDefault() {
      prevented = true;
    },
  });

  assert.equal(prevented, true);
  assert.equal(documentRef.activeElement, primaryButton);

  listeners.get('keydown')({ key: 'Escape', preventDefault() {} });

  assert.equal(await resultPromise, null);
  assert.equal(documentRef.activeElement, opener);
  assert.equal(documentRef.body.children.length, 0);
});

test('DashboardDialogs resolves cancel button clicks to null', async () => {
  const { documentRef } = createDom();
  const dialogs = loadDialogs(documentRef);

  const resultPromise = dialogs.choose({
    title: 'Pick one',
    choices: [{ value: 'primary', label: 'Primary', variant: 'primary' }],
  });

  const backdrop = documentRef.body.children[0];
  const dialog = backdrop.children[0];
  const footer = dialog.children.at(-1);
  const cancelButton = dialog.querySelectorAll('button')[1];

  footer.onclick({ target: cancelButton });

  assert.equal(await resultPromise, null);
});
