import assert from 'node:assert/strict';
import { createRequire } from 'node:module';
import { test } from 'node:test';

const require = createRequire(import.meta.url);
const DashboardCore = require('../../src/web/static/dashboard_core.js');

test('withDashboardToken preserves headers and adds token', () => {
  const options = DashboardCore.withDashboardToken(
    { method: 'DELETE', headers: { 'Content-Type': 'application/json' } },
    'secret-token',
  );

  assert.equal(options.method, 'DELETE');
  assert.equal(options.headers['Content-Type'], 'application/json');
  assert.equal(options.headers['X-Dashboard-Token'], 'secret-token');
});

test('withDashboardToken does not mutate original options', () => {
  const original = { method: 'GET', headers: { Accept: 'application/json' } };
  const options = DashboardCore.withDashboardToken(original, 'new-token');

  assert.notEqual(options, original);
  assert.notEqual(options.headers, original.headers);
  assert.deepEqual(original.headers, { Accept: 'application/json' });
  assert.equal(options.headers['X-Dashboard-Token'], 'new-token');
});

test('readToken reads dashboard token from meta tag', () => {
  const documentRef = {
    querySelector(selector) {
      assert.equal(selector, 'meta[name="dashboard-token"]');
      return { content: 'meta-token' };
    },
  };

  assert.equal(DashboardCore.readToken(documentRef), 'meta-token');
});

test('authorizedControlPayload preserves existing payload fields', () => {
  assert.deepEqual(
    DashboardCore.authorizedControlPayload({ action: 'save_as', filename: 'x.pth' }, 'token'),
    { action: 'save_as', filename: 'x.pth', token: 'token' },
  );
});

test('authorizedControlPayload ignores non-object payloads', () => {
  assert.equal(DashboardCore.authorizedControlPayload(null, 'token'), null);
  assert.equal(DashboardCore.authorizedControlPayload('bad', 'token'), 'bad');
});

test('createAuthorizedSocket injects token into mutating events only', () => {
  const emitted = [];
  const socket = DashboardCore.createAuthorizedSocket((options) => ({
    options,
    emit(event, payload, ...args) {
      emitted.push({ event, payload, args });
    },
  }), 'socket-token');

  assert.equal(socket.options.auth.token, 'socket-token');

  socket.emit('control', { action: 'pause' }, 'ack');
  socket.emit('clear_logs', {});
  socket.emit('state_update', { episode: 1 });

  assert.deepEqual(emitted[0], {
    event: 'control',
    payload: { action: 'pause', token: 'socket-token' },
    args: ['ack'],
  });
  assert.deepEqual(emitted[1], {
    event: 'clear_logs',
    payload: { token: 'socket-token' },
    args: [],
  });
  assert.deepEqual(emitted[2], {
    event: 'state_update',
    payload: { episode: 1 },
    args: [],
  });
});

test('emitControl resolves explicit server acknowledgements', async () => {
  const emitted = [];
  const socket = {
    emit(event, payload, callback) {
      emitted.push({ event, payload });
      callback({ success: true, action: payload.action });
    },
  };

  const response = await DashboardCore.emitControl(socket, { action: 'save' }, { timeoutMs: 50 });

  assert.deepEqual(response, { success: true, action: 'save' });
  assert.deepEqual(emitted, [{ event: 'control', payload: { action: 'save' } }]);
});

test('emitControl reports disconnected sockets without emitting', async () => {
  let emitted = false;
  const socket = {
    connected: false,
    emit() {
      emitted = true;
    },
  };

  const response = await DashboardCore.emitControl(socket, { action: 'save' }, { timeoutMs: 50 });

  assert.deepEqual(response, { success: false, error: 'Not connected to server' });
  assert.equal(emitted, false);
});

test('emitControl times out when the server never acknowledges', async () => {
  const socket = {
    emit() {
      // Intentionally never calls the acknowledgement callback.
    },
  };

  const response = await DashboardCore.emitControl(socket, { action: 'save' }, {
    timeoutMs: 5,
    timeoutMessage: 'Timed out',
  });

  assert.deepEqual(response, { success: false, error: 'Timed out' });
});

test('controlErrorMessage normalizes failed acknowledgement text', () => {
  assert.equal(DashboardCore.controlErrorMessage({ success: true }), '');
  assert.equal(
    DashboardCore.controlErrorMessage({ success: false, error: 'Save failed' }),
    'Save failed',
  );
  assert.equal(DashboardCore.controlErrorMessage(null, 'Fallback'), 'Fallback');
});

test('model helpers prefer opaque ids and display filenames', () => {
  assert.equal(
    DashboardCore.modelId({ id: 'breakout:best.pth', path: '/tmp/best.pth' }),
    'breakout:best.pth',
  );
  assert.equal(DashboardCore.modelDisplayName('breakout:best.pth'), 'best.pth');
});

test('escape helpers encode html and attribute-sensitive characters', () => {
  const hostile = `<img src=x onerror="alert('x')">&`;

  assert.equal(
    DashboardCore.escapeHtml(hostile),
    '&lt;img src=x onerror=&quot;alert(&#39;x&#39;)&quot;&gt;&amp;',
  );
  assert.equal(DashboardCore.escapeHtmlAttribute(hostile), DashboardCore.escapeHtml(hostile));
});

test('modelListHtml renders empty and malformed model lists safely', () => {
  assert.equal(
    DashboardCore.modelListHtml([]),
    '<div class="no-models">No saved models found</div>',
  );
  assert.equal(
    DashboardCore.modelListHtml(null),
    '<div class="no-models">No saved models found</div>',
  );
});

test('modelListHtml escapes unsafe model fields and preserves opaque ids', () => {
  const html = DashboardCore.modelListHtml([{
    id: 'snake:<best>.pth',
    name: `<script>alert('model')</script>`,
    size: 1048576,
    modified_str: 'Today <now>',
    has_metadata: true,
    metadata: {
      episode: 1234,
      best_score: 99,
      avg_score_last_100: 12.345,
      save_reason: 'best',
    },
    epsilon: 0.12345,
  }]);

  assert.match(html, /data-model-id="snake:&lt;best&gt;.pth"/);
  assert.match(html, /&lt;script&gt;alert\(&#39;model&#39;\)&lt;\/script&gt;/);
  assert.match(html, /Today &lt;now&gt;/);
  assert.match(html, />1\.00 MB</);
  assert.match(html, />1,234</);
  assert.match(html, />12\.3</);
  assert.match(html, />0\.123</);
  assert.doesNotMatch(html, /<script>alert/);
});

test('modelListHtml marks unreadable checkpoints without load action', () => {
  const html = DashboardCore.modelListHtml([{
    id: 'bad:model.pth',
    name: 'bad model',
    size: -1,
    is_loadable: false,
    load_error: 'bad "checkpoint"',
  }]);

  assert.match(html, /model-item-invalid/);
  assert.doesNotMatch(html, /data-action="load-model"/);
  assert.match(html, /title="bad &quot;checkpoint&quot;"/);
  assert.match(html, />\? MB</);
});
