import { chromium, expect } from '@playwright/test';
import { spawn } from 'node:child_process';
import { existsSync } from 'node:fs';
import net from 'node:net';

const DEFAULT_CHROME_PATH = '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome';

function freePort() {
  return new Promise((resolve, reject) => {
    const server = net.createServer();
    server.on('error', reject);
    server.listen(0, '127.0.0.1', () => {
      const address = server.address();
      server.close(() => resolve(address.port));
    });
  });
}

function waitForDashboardUrl(server, timeoutMs = 30000) {
  return new Promise((resolve, reject) => {
    let output = '';
    const timeout = setTimeout(() => {
      reject(new Error(`Dashboard server did not become ready.\n${output}`));
    }, timeoutMs);

    const consume = (chunk) => {
      output += chunk.toString();
      for (const line of output.split(/\r?\n/)) {
        if (line.startsWith('DASHBOARD_URL=')) {
          clearTimeout(timeout);
          resolve(line.slice('DASHBOARD_URL='.length).trim());
        }
      }
    };

    server.stdout.on('data', consume);
    server.stderr.on('data', consume);
    server.on('exit', (code, signal) => {
      clearTimeout(timeout);
      reject(new Error(`Dashboard server exited before ready: code=${code} signal=${signal}\n${output}`));
    });
  });
}

async function launchBrowser() {
  const configuredPath = process.env.PLAYWRIGHT_CHROME_PATH;
  const executablePath = configuredPath || (existsSync(DEFAULT_CHROME_PATH) ? DEFAULT_CHROME_PATH : undefined);

  try {
    return await chromium.launch({
      headless: true,
      ...(executablePath ? { executablePath } : {}),
    });
  } catch (error) {
    if (executablePath) {
      return chromium.launch({ headless: true });
    }
    throw error;
  }
}

async function main() {
  const port = await freePort();
  const token = `e2e-token-${Date.now()}`;
  const python = process.env.PYTHON || 'python';
  const server = spawn(python, ['tests/e2e/dashboard_smoke_server.py', '--port', String(port)], {
    cwd: process.cwd(),
    env: {
      ...process.env,
      NN_GAME_DASHBOARD_TOKEN: token,
      PYTHONUNBUFFERED: '1',
    },
    stdio: ['ignore', 'pipe', 'pipe'],
  });

  let browser;
  try {
    const dashboardUrl = await waitForDashboardUrl(server);
    browser = await launchBrowser();
    const page = await browser.newPage();
    const consoleErrors = [];
    const httpFailures = [];
    const pageErrors = [];

    page.on('console', (message) => {
      if (message.type() === 'error') {
        consoleErrors.push(message.text());
      }
    });
    page.on('response', (response) => {
      if (response.status() >= 400) {
        httpFailures.push(`${response.status()} ${response.url()}`);
      }
    });
    page.on('pageerror', (error) => pageErrors.push(error.message));

    await page.goto(dashboardUrl, { waitUntil: 'domcontentloaded' });
    await expect(page.locator('.status-text')).toHaveText('Connected', { timeout: 20000 });
    await expect(page.locator('#info-status')).toContainText(/Training|Starting|Idle/);

    await page.locator('[data-action="save-model"]').first().click();
    await expect(page.locator('#console-output')).toContainText('Save requested', { timeout: 5000 });

    await page.locator('[data-action="show-load-modal"]').first().click();
    await expect(page.locator('#load-modal')).toHaveClass(/visible/);
    await expect(page.locator('#model-list')).not.toContainText('Failed to load models');
    await page.locator('[data-action="hide-load-modal"]').click();
    await expect(page.locator('#load-modal')).not.toHaveClass(/visible/);

    await expect(page.locator('#game-select')).toHaveValue('breakout');

    const relevantHttpFailures = httpFailures.filter((entry) => !entry.endsWith('/favicon.ico'));
    const relevantConsoleErrors = consoleErrors.filter(
      (message) => message !== 'Failed to load resource: the server responded with a status of 404 (NOT FOUND)' || relevantHttpFailures.length > 0,
    );

    if (pageErrors.length > 0 || relevantConsoleErrors.length > 0 || relevantHttpFailures.length > 0) {
      throw new Error(
        [
          'Dashboard smoke produced browser errors.',
          ...relevantHttpFailures.map((message) => `http: ${message}`),
          ...pageErrors.map((message) => `pageerror: ${message}`),
          ...relevantConsoleErrors.map((message) => `console: ${message}`),
        ].join('\n'),
      );
    }
  } finally {
    if (browser) {
      await browser.close();
    }
    if (!server.killed) {
      server.kill('SIGTERM');
    }
  }
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
