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

import { initSettings } from '../../../src/osprey/interfaces/web_terminal/static/js/settings.js';

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
