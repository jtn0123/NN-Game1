import assert from 'node:assert/strict';
import { createRequire } from 'node:module';
import { test } from 'node:test';

const require = createRequire(import.meta.url);
const LauncherApp = require('../../src/web/static/launcher.js');

test('difficultyClass maps display difficulty to stable CSS class', () => {
  assert.equal(LauncherApp.difficultyClass('Easy'), 'easy');
  assert.equal(LauncherApp.difficultyClass('Medium-Hard'), 'hard');
  assert.equal(LauncherApp.difficultyClass('Unknown'), 'medium');
  assert.equal(LauncherApp.difficultyClass(null), 'medium');
});

test('startLabel formats selected mode and game id consistently', () => {
  assert.equal(LauncherApp.startLabel('ai', 'space_invaders'), 'TRAIN SPACE INVADERS');
  assert.equal(LauncherApp.startLabel('human', 'breakout'), 'PLAY BREAKOUT');
});
