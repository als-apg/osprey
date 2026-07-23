// @ts-check
/**
 * Unit tests for the Web Terminal Agent Settings panel's write path
 * (settings.js's `applySettings`, reached via `initSettings()`'s
 * confirm-button wiring):
 *   npx vitest run tests/interfaces/web_terminal/settings.test.mjs
 *
 * Covers the raw-mode PUT and form-mode PATCH `/api/config` calls being
 * prefix-aware via `window.__OSPREY_PREFIX__` (multi-user deployments) --
 * `fetchJSON` (api.js) already prefixes GET, but these are raw `fetch()`
 * calls with request options `fetchJSON` doesn't support, so settings.js
 * applies the shared `withPrefix` helper (imported from api.js) to these
 * paths directly.
 *
 * Only the confirm-button's direct click is driven here (not the
 * apply-button -> confirm-dialog gate) since `applySettings` is not
 * exported and the confirm button is wired straight to it -- clicking it
 * exercises the same function without needing to also drive the dialog's
 * visibility. The stubbed fetch responds not-ok, so `applySettings` throws
 * before reaching the post-save `restartTerminal()`/`startTerminal()` tail
 * (a real WebSocket/xterm.js round trip out of scope for this fetch-prefix
 * guard -- see terminal-resume.test.mjs for that surface).
 */

import { test, expect, describe, beforeEach, afterEach, vi } from 'vitest';

import {
  initSettings,
  openDrawerTab,
} from '../../../src/osprey/interfaces/web_terminal/static/js/settings.js';

/** Mount the minimal DOM `initSettings()` and `applySettings()` need. */
function mountFixture() {
  document.body.innerHTML = `
    <div id="settings-drawer"></div>
    <div id="tab-config">
      <button class="settings-mode-btn" data-mode="raw"></button>
      <button class="settings-mode-btn" data-mode="form"></button>
      <button class="settings-apply-btn"></button>
      <button class="settings-confirm-btn"></button>
      <button class="settings-cancel-btn"></button>
      <div class="settings-status"></div>
    </div>
    <textarea id="settings-raw-editor">model: anthropic/claude-sonnet</textarea>
  `;
}

/** A not-ok JSON response, so `applySettings` throws before its restart tail. */
function notOkResponse() {
  return {
    ok: false,
    status: 500,
    json: () => Promise.resolve({ detail: 'boom' }),
  };
}

beforeEach(() => {
  mountFixture();
  initSettings();
});

afterEach(() => {
  vi.unstubAllGlobals();
  delete window.__OSPREY_PREFIX__;
});

describe('applySettings: raw mode PUT', () => {
  test('prepends window.__OSPREY_PREFIX__ to the /api/config PUT (multi-user deployments)', async () => {
    window.__OSPREY_PREFIX__ = '/u/alice';
    const fetchMock = vi.fn(() => Promise.resolve(notOkResponse()));
    vi.stubGlobal('fetch', fetchMock);

    /** @type {HTMLElement} */ (
      document.querySelector('.settings-mode-btn[data-mode="raw"]')
    ).click();
    /** @type {HTMLElement} */ (document.querySelector('.settings-confirm-btn')).click();
    await new Promise((resolve) => setTimeout(resolve, 0));

    expect(fetchMock).toHaveBeenCalledWith('/u/alice/api/config', expect.objectContaining({
      method: 'PUT',
    }));
  });

  test('is a no-op (unprefixed) when window.__OSPREY_PREFIX__ is absent', async () => {
    const fetchMock = vi.fn(() => Promise.resolve(notOkResponse()));
    vi.stubGlobal('fetch', fetchMock);

    /** @type {HTMLElement} */ (
      document.querySelector('.settings-mode-btn[data-mode="raw"]')
    ).click();
    /** @type {HTMLElement} */ (document.querySelector('.settings-confirm-btn')).click();
    await new Promise((resolve) => setTimeout(resolve, 0));

    expect(fetchMock).toHaveBeenCalledWith('/api/config', expect.objectContaining({
      method: 'PUT',
    }));
  });
});

/**
 * The command-palette gate seam. `openDrawerTab` is the SINGLE gated entry the
 * palette (and `revealSetting`) build on: it must clear the warning gate first,
 * then activate a tab, and report whether the tab actually became active.
 * `runWarningGate` is internal, so its true-on-preset-ack decision is observed
 * through this seam (no warning overlay is built and the drawer opens).
 */
describe('openDrawerTab: gated tab activation', () => {
  const ACK_KEY = 'osprey-settings-warning-ack';

  /** @type {any} */ let drawer;
  /** @type {HTMLElement} */ let tab;

  /** Stub `/health` (reached via fetchJSON) to hand back a server session id.
   * @param {string} sessionId */
  function stubHealth(sessionId) {
    vi.stubGlobal('fetch', vi.fn((url) =>
      String(url).endsWith('/health')
        ? Promise.resolve({
            ok: true,
            status: 200,
            statusText: 'OK',
            json: () => Promise.resolve({ session_id: sessionId }),
          })
        : Promise.resolve(notOkResponse())));
  }

  beforeEach(() => {
    localStorage.clear();
    document.body.innerHTML = `
      <button data-drawer-trigger="settings-drawer"></button>
      <div id="settings-drawer">
        <div class="drawer-tab" data-tab="tab-config"></div>
      </div>
      <div id="tab-config"></div>
    `;
    drawer = document.getElementById('settings-drawer');
    // The real osprey-drawer supplies open()/close(); the fixture stubs them.
    drawer.open = vi.fn(() => drawer.setAttribute('open', ''));
    drawer.close = vi.fn(() => drawer.removeAttribute('open'));
    tab = /** @type {HTMLElement} */ (document.querySelector('.drawer-tab'));
    initSettings();
  });

  afterEach(() => {
    localStorage.clear();
  });

  test('clears the gate silently (no dialog, drawer opens) when the ack is preset for this server session', async () => {
    localStorage.setItem(ACK_KEY, 'srv-1');
    stubHealth('srv-1');

    await openDrawerTab('tab-config');

    // Preset ack => runWarningGate resolved true WITHOUT building a dialog...
    expect(document.querySelector('.settings-warning-overlay')).toBeNull();
    // ...and the caller (openDrawerTab) went on to open the drawer.
    expect(drawer.open).toHaveBeenCalled();
  });

  test('returns false without polling when the target tab never gains .active (unsaved-changes veto)', async () => {
    localStorage.setItem(ACK_KEY, 'srv-1');
    stubHealth('srv-1');

    const result = await openDrawerTab('tab-config');

    // Fixture tab has no component handler, so the click adds no `.active`.
    expect(tab.classList.contains('active')).toBe(false);
    expect(result).toBe(false);
  });
});
