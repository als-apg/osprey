// @ts-check
/**
 * Unit tests for session.js opened as a page of its own.
 *
 * session.js runs its page boot on import, so each case sets the page URL,
 * imports the module fresh, and reads what the boot did:
 *
 *   - theme role: standalone, the page owns its theme (a pick persists);
 *     framed (?embedded=true), it follows its host and never writes the
 *     host's preference
 *
 *   npx vitest run tests/interfaces/web_terminal/session-standalone.test.mjs
 */

import { test, expect, describe, afterEach, vi } from 'vitest';

const ENTRY_PATH = '../../../src/osprey/interfaces/web_terminal/static/js/session.js';

/** @type {import('vitest').Mock} */
let fetchStub;

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
  document.body.innerHTML = `
    <div class="refresh-dot" id="refresh-dot"></div>
    <nav id="nav"><button class="pill active" data-view="agents">Agents</button></nav>
    <section class="view active" id="view-agents"></section>
    <div class="toast" id="toast"></div>
  `;
  fetchStub = vi.fn(() => Promise.resolve({
    status: 200,
    ok: true,
    json: () => Promise.resolve({ total_events: 0, agents: [], tool_calls_by_agent: {} }),
  }));
  vi.stubGlobal('fetch', fetchStub);
  vi.resetModules();
  await import(ENTRY_PATH);
  return import('/design-system/js/theme-manager.js');
}

afterEach(() => {
  window.history.replaceState({}, '', '/');
  vi.unstubAllGlobals();
  document.body.innerHTML = '';
});

describe('theme role', () => {
  test('a theme picked on the page opened on its own persists', async () => {
    const ThemeManager = await bootAt('/static/session.html');

    ThemeManager.setTheme('light');

    expect(localStorage.getItem('osprey-theme')).toBe(
      JSON.stringify({ family: 'main', mode: 'light' }),
    );
  });

  test('a framed page never writes the host preference', async () => {
    const ThemeManager = await bootAt('/static/session.html?embedded=true');

    ThemeManager.setTheme('light');

    expect(localStorage.getItem('osprey-theme')).toBeNull();
  });
});
