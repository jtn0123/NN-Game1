import assert from 'node:assert/strict';
import { createRequire } from 'node:module';
import { test } from 'node:test';

const require = createRequire(import.meta.url);
globalThis.DashboardCore = require('../../src/web/static/dashboard_core.js');
const DashboardModelList = require('../../src/web/static/dashboard_model_list.js');

class TextNode {
  constructor(text) {
    this.textContent = text;
    this.children = [];
  }
}

class Element {
  constructor(tagName, ownerDocument) {
    this.tagName = tagName;
    this.ownerDocument = ownerDocument;
    this.children = [];
    this.dataset = {};
    this.className = '';
    this.title = '';
    this.textContent = '';
  }

  append(...children) {
    this.children.push(...children);
  }

  replaceChildren(...children) {
    this.children = children;
  }

  text() {
    return [
      this.textContent,
      ...this.children.map((child) => (typeof child.text === 'function' ? child.text() : child.textContent)),
    ].join('');
  }
}

function createDocument() {
  return {
    createElement(tagName) {
      return new Element(tagName, this);
    },
    createTextNode(text) {
      return new TextNode(text);
    },
  };
}

test('renderModelList builds model rows with text nodes and opaque ids', () => {
  const documentRef = createDocument();
  const list = documentRef.createElement('div');

  DashboardModelList.renderModelList(list, [{
    id: 'snake:<best>.pth',
    name: '<script>alert("model")</script>',
    size: 1048576,
    modified_str: 'Today <now>',
    has_metadata: true,
    metadata: {
      episode: 1234,
      best_score: 99,
      avg_score_last_100: 12.3,
      save_reason: 'best<script>',
    },
    epsilon: 0.1234,
  }]);

  const item = list.children[0];
  const content = item.children[0];
  const deleteButton = item.children[1];

  assert.equal(content.dataset.modelId, 'snake:<best>.pth');
  assert.equal(deleteButton.dataset.modelId, 'snake:<best>.pth');
  assert.match(item.text(), /<script>alert\("model"\)<\/script>/);
  assert.match(item.text(), /Today <now>/);
  assert.equal(content.children[0].children[0].children[2].className, 'reason-badge bestscript');
});

test('renderModelList renders empty state without model rows', () => {
  const documentRef = createDocument();
  const list = documentRef.createElement('div');

  DashboardModelList.renderModelList(list, []);

  assert.equal(list.children.length, 1);
  assert.equal(list.children[0].className, 'no-models');
  assert.equal(list.children[0].textContent, 'No saved models found');
});

test('renderModelList displays legacy compatibility warnings', () => {
  const documentRef = createDocument();
  const list = documentRef.createElement('div');

  DashboardModelList.renderModelList(list, [{
    id: 'breakout:legacy.pth',
    name: 'legacy model',
    size: 1024,
    is_loadable: true,
    requires_unsafe_load: true,
    security_warning: 'Re-save after loading',
  }]);

  const name = list.children[0].children[0].children[0].children[0];
  const badge = name.children[name.children.length - 1];

  assert.equal(badge.className, 'reason-badge warning');
  assert.equal(badge.textContent, 'legacy');
  assert.equal(badge.title, 'Re-save after loading');
});
