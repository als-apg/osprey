// @ts-check
/**
 * Unit tests for session.js, the entry of the Activity page.
 *
 * session.js runs its page boot on import, so each case sets the page URL,
 * imports the module fresh, and reads what the boot did:
 *
 *   - theme role: standalone, the page owns its theme (a pick persists);
 *     framed (?embedded=true), it follows its host and never writes the
 *     host's preference
 *   - session scope: the page reads a canonical ?session_id= from its own URL,
 *     ignores a malformed one, and moves when a host frame of its own origin
 *     names another session
 *   - room: framed, the session list starts hidden
 *
 *   npx vitest run tests/interfaces/web_terminal/session.test.mjs
 */

import { test, expect, describe, afterEach, vi } from 'vitest';

import { shellMarkup, stubActivityFetch } from '../_support/activity-page.mjs';
import { byId } from '../_support/dom.mjs';

const ENTRY_PATH = '../../../src/osprey/interfaces/web_terminal/static/js/session.js';

/** @type {ReturnType<typeof stubActivityFetch>} */
let api;

/** Let the reads a boot or a message started land. */
const settle = () => new Promise((resolve) => setTimeout(resolve, 0));

/**
 * Boot session.js at `url` with a clean theme state, then return the
 * theme-manager instance that boot initialised (same module graph, no reset
 * between the two imports).
 *
 * @param {string} url
 */
async function bootAt(url) {
  localStorage.clear();
  document.documentElement.removeAttribute('data-theme');
  document.documentElement.removeAttribute('data-theme-mode');
  window.history.replaceState({}, '', url);
  document.body.innerHTML = shellMarkup();
  // The strip's own boot is session-strip.test.mjs's subject.
  document.getElementById('activity-strip')?.remove();
  api = stubActivityFetch();
  vi.resetModules();
  await import(ENTRY_PATH);
  await settle();
  return import('/design-system/js/theme-manager.js');
}

afterEach(() => {
  window.history.replaceState({}, '', '/');
  vi.unstubAllGlobals();
  document.body.classList.remove('embedded');
  document.body.innerHTML = '';
});

describe('theme role', () => {
  test('a theme picked on the page opened on its own persists', async () => {
    const ThemeManager = await bootAt('/static/session.html');

    ThemeManager.setTheme('light');

    expect(localStorage.getItem('osprey-theme')).toBe(
      JSON.stringify({ family: 'main', mode: 'light' })
    );
  });

  test('a framed page never writes the host preference', async () => {
    const ThemeManager = await bootAt('/static/session.html?embedded=true');

    ThemeManager.setTheme('light');

    expect(localStorage.getItem('osprey-theme')).toBeNull();
  });
});

describe('session scope', () => {
  const KEY = '11111111-2222-3333-4444-555555555555';
  const OTHER_KEY = 'aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee';

  test('a link naming a session scopes the first read to it', async () => {
    await bootAt(`/static/session.html?session_id=${KEY}`);

    expect(api.reads()[0]).toBe(`/api/session-activity?session_id=${KEY}`);
  });

  test('a malformed session_id is not read', async () => {
    await bootAt('/static/session.html?session_id=..%2F..%2Fx');

    expect(api.reads()).toEqual(['/api/session-activity']);
    expect(api.urls().some((url) => url.includes('session_id'))).toBe(false);
  });

  test('a host frame stays authoritative over the URL', async () => {
    await bootAt(`/static/session.html?session_id=${KEY}#${KEY}/tools`);
    const before = api.reads().length;

    window.postMessage(
      { type: 'osprey-session-change', session_id: OTHER_KEY },
      window.location.origin
    );
    await settle();

    expect(api.reads()[before]).toBe(`/api/session-activity?session_id=${OTHER_KEY}`);
    expect(window.location.hash).toBe('#/tools');
  });

  test('a frame of another origin moves nothing', async () => {
    await bootAt(`/static/session.html?session_id=${KEY}`);
    const before = api.reads().length;

    window.dispatchEvent(
      new MessageEvent('message', {
        data: { type: 'osprey-session-change', session_id: OTHER_KEY },
        origin: 'https://elsewhere.example',
      })
    );
    await settle();

    expect(api.reads()).toHaveLength(before);
  });

  test('a host frame naming no usable session falls back to the current one', async () => {
    await bootAt(`/static/session.html?session_id=${KEY}`);
    const before = api.reads().length;

    window.postMessage(
      { type: 'osprey-session-change', session_id: '../../x' },
      window.location.origin
    );
    await settle();

    expect(api.reads()[before]).toBe('/api/session-activity');
  });
});

describe('room', () => {
  test('opened on its own, the page shows its session list', async () => {
    await bootAt('/static/session.html');

    expect(byId('sb').dataset.open).toBe('true');
  });

  test('framed, the session list starts hidden', async () => {
    await bootAt('/static/session.html?embedded=true');

    expect(document.body.classList.contains('embedded')).toBe(true);
    expect(byId('sb').dataset.open).toBe('false');
  });
});
