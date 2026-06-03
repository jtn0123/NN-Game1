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

test('model helpers prefer opaque ids and display filenames', () => {
  assert.equal(
    DashboardCore.modelId({ id: 'breakout:best.pth', path: '/tmp/best.pth' }),
    'breakout:best.pth',
  );
  assert.equal(DashboardCore.modelDisplayName('breakout:best.pth'), 'best.pth');
});
