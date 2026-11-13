// @ts-check
/**
 * The terminal's Agent bar `+` (app.js's `initNewSessionButton`):
 *   npx vitest run tests/interfaces/web_terminal/app-new-session.test.mjs
 *
 * It starts a new session the way every other way does, through the session
 * model's guarded `switchTo({new: true})` (which session-layout.js answers
 * with the Agent pane alone), and holds the button down until that settles.
 *
 * Seams as in app-agent-links.test.mjs: terminal.js is mocked whole and app.js
 * is imported once, statically; session-model.js keeps its real exports but
 * `switchTo`.
 */

import { expect, test, vi } from 'vitest';

vi.hoisted(() => {
  vi.stubGlobal(
    'fetch',
    vi.fn(async (/** @type {string} */ url) => {
      throw new Error(`unstubbed fetch: ${url}`);
    })
  );
});

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
  focusTerminal: vi.fn(),
  pasteToTerminal: vi.fn(),
  onSessionChange: vi.fn(),
}));

const model = vi.hoisted(() => ({ switchTo: vi.fn() }));

vi.mock(
  '../../../src/osprey/interfaces/web_terminal/static/js/session-model.js',
  async (importOriginal) => ({
    .../** @type {Record<string, unknown>} */ (await importOriginal()),
    switchTo: model.switchTo,
  })
);

vi.mock('../../../src/osprey/interfaces/web_terminal/static/js/sidebar-artifacts.js', () => ({
  initSidebarArtifacts: vi.fn(),
  teardownSidebarArtifacts: vi.fn(),
}));

import { initNewSessionButton } from '../../../src/osprey/interfaces/web_terminal/static/js/app.js';

test('+ starts a new session through the guarded switch, held down until it settles', async () => {
  /** @type {(moved: boolean) => void} */
  let finish = () => {};
  model.switchTo.mockImplementation(() => new Promise((resolve) => (finish = resolve)));
  document.body.innerHTML = '<button id="new-session-btn" type="button"></button>';
  initNewSessionButton();
  const btn = /** @type {HTMLButtonElement} */ (document.getElementById('new-session-btn'));

  btn.click();
  expect(model.switchTo).toHaveBeenCalledExactlyOnceWith({ new: true });
  expect(btn.disabled).toBe(true);

  finish(true);
  await vi.waitFor(() => expect(btn.disabled).toBe(false));
});
