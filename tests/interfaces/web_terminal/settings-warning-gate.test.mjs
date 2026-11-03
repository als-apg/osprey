// @ts-check
/**
 * Unit tests for the settings drawer's warning gate (settings-warning-gate.js):
 *   npx vitest run tests/interfaces/web_terminal/settings-warning-gate.test.mjs
 *
 * The gate decides whether the drawer may open: once per server session (the
 * `/health` session id stored under the ack key) the user is warned and must
 * proceed. `/health` is reached through api.js's fetchJSON, mocked here so each
 * test controls the session id or holds the request open. The module keeps its
 * re-entrancy flag at module scope, so every test imports a fresh copy.
 */

import { test, expect, describe, beforeEach, afterEach, vi } from 'vitest';

const health = vi.hoisted(() => ({
  /** @type {() => Promise<any>} */
  respond: () => Promise.resolve({ session_id: 'srv-1' }),
}));

vi.mock('../../../src/osprey/interfaces/web_terminal/static/js/api.js', () => ({
  fetchJSON: vi.fn(() => health.respond()),
}));

const GATE_MODULE =
  '../../../src/osprey/interfaces/web_terminal/static/js/settings-warning-gate.js';
const ACK_KEY = 'osprey-settings-warning-ack';

/** @returns {Promise<typeof import('../../../src/osprey/interfaces/web_terminal/static/js/settings-warning-gate.js')>} */
async function loadGate() {
  vi.resetModules();
  return import(GATE_MODULE);
}

/** The warning dialog's button, once the gate has built it. */
async function dialogButton(/** @type {string} */ selector) {
  await vi.waitFor(() => {
    if (!document.querySelector(selector)) throw new Error('dialog not built yet');
  });
  return /** @type {HTMLElement} */ (document.querySelector(selector));
}

beforeEach(() => {
  localStorage.clear();
  document.body.innerHTML = '';
  health.respond = () => Promise.resolve({ session_id: 'srv-1' });
});

afterEach(() => {
  vi.useRealTimers();
  localStorage.clear();
});

describe('runWarningGate', () => {
  test('resolves true without a dialog when the ack is preset for this server session', async () => {
    localStorage.setItem(ACK_KEY, 'srv-1');
    const { runWarningGate } = await loadGate();

    await expect(runWarningGate()).resolves.toBe(true);
    expect(document.querySelector('.settings-warning-overlay')).toBeNull();
  });

  test('Proceed resolves true and stores the server session under the ack key', async () => {
    const { runWarningGate } = await loadGate();

    const result = runWarningGate();
    (await dialogButton('.settings-warning-proceed')).click();

    await expect(result).resolves.toBe(true);
    expect(localStorage.getItem(ACK_KEY)).toBe('srv-1');
  });

  test('Cancel resolves false and stores nothing', async () => {
    const { runWarningGate } = await loadGate();

    const result = runWarningGate();
    (await dialogButton('.settings-warning-cancel')).click();

    await expect(result).resolves.toBe(false);
    expect(localStorage.getItem(ACK_KEY)).toBeNull();
  });

  test('Escape resolves false and stores nothing', async () => {
    const { runWarningGate } = await loadGate();

    const result = runWarningGate();
    await dialogButton('.settings-warning-cancel');
    document.dispatchEvent(new KeyboardEvent('keydown', { key: 'Escape' }));

    await expect(result).resolves.toBe(false);
    expect(localStorage.getItem(ACK_KEY)).toBeNull();
  });

  test('a second call while one is pending resolves false and builds no second dialog', async () => {
    const { runWarningGate } = await loadGate();

    const first = runWarningGate();
    await dialogButton('.settings-warning-proceed');

    await expect(runWarningGate()).resolves.toBe(false);
    expect(document.querySelectorAll('.settings-warning-overlay')).toHaveLength(1);

    /** @type {HTMLElement} */ (document.querySelector('.settings-warning-cancel')).click();
    await expect(first).resolves.toBe(false);
  });

  test('shows the dialog once the health check outlasts its timeout', async () => {
    vi.useFakeTimers();
    health.respond = () => new Promise(() => {});
    const { runWarningGate, HEALTH_CHECK_TIMEOUT_MS } = await loadGate();

    const result = runWarningGate();
    await vi.advanceTimersByTimeAsync(HEALTH_CHECK_TIMEOUT_MS - 1);
    expect(document.querySelector('.settings-warning-overlay')).toBeNull();

    await vi.advanceTimersByTimeAsync(1);
    expect(document.querySelector('.settings-warning-overlay')).not.toBeNull();

    /** @type {HTMLElement} */ (document.querySelector('.settings-warning-proceed')).click();
    await expect(result).resolves.toBe(true);
    // No server session was read, so nothing is remembered.
    expect(localStorage.getItem(ACK_KEY)).toBeNull();
  });
});

describe('initSettingsWarningGate', () => {
  /** @type {any} */ let drawer;
  /** @type {HTMLElement} */ let trigger;

  beforeEach(() => {
    document.body.innerHTML = `
      <button data-drawer-trigger="settings-drawer"></button>
      <div id="settings-drawer"></div>
    `;
    drawer = document.getElementById('settings-drawer');
    drawer.open = vi.fn(() => drawer.setAttribute('open', ''));
    drawer.close = vi.fn(() => drawer.removeAttribute('open'));
    trigger = /** @type {HTMLElement} */ (
      document.querySelector('[data-drawer-trigger="settings-drawer"]')
    );
  });

  test('the trigger click on an open drawer closes it without a dialog', async () => {
    const { initSettingsWarningGate } = await loadGate();
    initSettingsWarningGate(drawer);
    drawer.setAttribute('open', '');

    trigger.click();
    await Promise.resolve();

    expect(drawer.close).toHaveBeenCalledTimes(1);
    expect(drawer.open).not.toHaveBeenCalled();
    expect(document.querySelector('.settings-warning-overlay')).toBeNull();
  });

  test('the trigger click on a closed drawer opens it once the gate clears', async () => {
    localStorage.setItem(ACK_KEY, 'srv-1');
    const { initSettingsWarningGate } = await loadGate();
    initSettingsWarningGate(drawer);

    trigger.click();

    await vi.waitFor(() => expect(drawer.open).toHaveBeenCalledTimes(1));
    expect(document.querySelector('.settings-warning-overlay')).toBeNull();
  });
});
