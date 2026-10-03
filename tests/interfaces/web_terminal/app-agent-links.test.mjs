// @ts-check
/**
 * The hub wires links in agent answers (app.js's `initAgentLinks`):
 *   npx vitest run tests/interfaces/web_terminal/app-agent-links.test.mjs
 *
 * A click on a panel link in an agent answer reaches `navigateAndActivatePanel`
 * with the panel id and the linked URL, fragment included; whether that panel
 * is hosted is read from `getPanelStandaloneUrl` at click time. The routing
 * itself is pinned in agent-links.test.mjs; this file pins the wiring.
 *
 * Seams as in app-paste-bridge.test.mjs: terminal.js is mocked whole and app.js
 * is imported once, statically, with `initAgentLinks` exported so it can be
 * driven directly. panel-manager.js keeps its real exports except the two the
 * wiring reads.
 */

import { describe, expect, test, vi } from 'vitest';

// The host-chrome import graph below reaches bar-sync.js, which GETs the
// operator's bar layout at import time. Nothing serves this environment, so
// that request is answered here — from `vi.hoisted`, which runs before the
// static imports are evaluated and therefore before the GET is made. Any other
// URL is a dependency this file has not declared, and fails loudly.
vi.hoisted(() => {
  vi.stubGlobal('fetch', vi.fn(async (/** @type {string} */ url) => {
    if (url !== '/api/bar-items') throw new Error(`unstubbed fetch: ${url}`);
    return {
      ok: true,
      status: 200,
      json: async () => ({
        version: 1,
        rev: 0,
        header: [],
        status: [],
        header_visible: true,
        status_visible: true,
      }),
    };
  }));
});

/** Mutable stand-in for terminal.js, reachable from the hoisted vi.mock factory. */
const term = vi.hoisted(() => ({
  paste: vi.fn(),
  focus: vi.fn(),
  /** @type {(() => void)[]} */
  sessionListeners: [],
}));

vi.mock('../../../src/osprey/interfaces/web_terminal/static/js/terminal.js', () => ({
  initTerminal: vi.fn(),
  startTerminal: vi.fn(),
  stopTerminal: vi.fn(),
  restartTerminal: vi.fn(),
  switchSession: vi.fn(),
  setSessionLabel: vi.fn(),
  notifySessionChange: vi.fn(),
  clearStoredSessionId: vi.fn(),
  fitTerminal: vi.fn(),
  getTerminalInstance: () => null,
  getCurrentSessionId: () => null,
  focusTerminal: term.focus,
  pasteToTerminal: term.paste,
  /** @param {() => void} fn */
  onSessionChange: (fn) => term.sessionListeners.push(fn),
}));

const panels = vi.hoisted(() => ({ navigate: vi.fn() }));

vi.mock('../../../src/osprey/interfaces/web_terminal/static/js/panel-manager.js', async (importOriginal) => {
  const actual = /** @type {Record<string, unknown>} */ (await importOriginal());
  return {
    ...actual,
    navigateAndActivatePanel: panels.navigate,
    getPanelStandaloneUrl: (/** @type {string} */ id) => (id === 'okf' ? '/panel/okf' : null),
  };
});

import { initAgentLinks } from '../../../src/osprey/interfaces/web_terminal/static/js/app.js';

describe('initAgentLinks', () => {
  test('a panel link in an agent answer opens that panel in the workspace', () => {
    document.body.innerHTML =
      '<div id="operator-container"><div class="op-entry assistant">' +
      '<div class="osprey-md-rendered"><a href="panel/okf#devices/bpm">BPM</a></div>' +
      '</div></div>';
    initAgentLinks();

    const a = /** @type {HTMLAnchorElement} */ (document.querySelector('a'));
    a.click();

    expect(panels.navigate).toHaveBeenCalledTimes(1);
    expect(panels.navigate).toHaveBeenCalledWith('okf', '/panel/okf#devices/bpm');
  });
});
