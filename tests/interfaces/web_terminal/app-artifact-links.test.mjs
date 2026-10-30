// @ts-check
/**
 * The hub wires artifact ids in agent answers (app.js's `initArtifactLinks`):
 *   npx vitest run tests/interfaces/web_terminal/app-artifact-links.test.mjs
 *
 * A click asks the gallery, through the hub's panel proxy under the per-user
 * prefix, to focus the artifact, and opens the artifacts panel only once the
 * gallery has. An id the gallery refuses opens nothing and marks the control.
 * Which text becomes a control is pinned in chat-render.test.mjs.
 *
 * Seams as in app-agent-links.test.mjs: terminal.js is mocked whole and app.js
 * is imported once, statically; panel-manager.js keeps its real exports except
 * the one the wiring calls. sidebar-artifacts.js is mocked: app.js only calls
 * its init, which this file never reaches.
 */

import { beforeEach, describe, expect, test, vi } from 'vitest';

/** The gallery's answer to the next focus request. */
const gallery = vi.hoisted(() => ({
  status: 200,
  /** Reject the next focus request, as a dropped connection does. */
  fail: false,
  /** @type {{ url: string, init: RequestInit | undefined }[]} */
  focusCalls: [],
}));

vi.hoisted(() => {
  vi.stubGlobal(
    'fetch',
    vi.fn(async (/** @type {string} */ url, /** @type {RequestInit | undefined} */ init) => {
      if (url.endsWith('/panel/artifacts/api/focus')) {
        gallery.focusCalls.push({ url, init });
        if (gallery.fail) throw new TypeError('Failed to fetch');
        return { ok: gallery.status < 300, status: gallery.status, json: async () => ({}) };
      }
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

const panels = vi.hoisted(() => ({ navigate: vi.fn() }));

vi.mock(
  '../../../src/osprey/interfaces/web_terminal/static/js/panel-manager.js',
  async (importOriginal) => {
    const actual = /** @type {Record<string, unknown>} */ (await importOriginal());
    return { ...actual, navigateAndActivatePanel: panels.navigate };
  }
);

vi.mock('../../../src/osprey/interfaces/web_terminal/static/js/sidebar-artifacts.js', () => ({
  initSidebarArtifacts: vi.fn(),
  teardownSidebarArtifacts: vi.fn(),
}));

import { initArtifactLinks } from '../../../src/osprey/interfaces/web_terminal/static/js/app.js';

/** Mount one artifact control in an agent answer and wire it; returns the control. */
function mount() {
  document.body.innerHTML =
    '<div id="operator-container"><div class="op-entry assistant"><div class="osprey-md-rendered">' +
    '<button type="button" class="op-artifact-link" data-artifact-id="a57934cf00dd">artifact_id: a57934cf00dd</button>' +
    '</div></div></div>';
  initArtifactLinks();
  return /** @type {HTMLButtonElement} */ (document.querySelector('.op-artifact-link'));
}

describe('initArtifactLinks', () => {
  beforeEach(() => {
    gallery.focusCalls.length = 0;
    panels.navigate.mockClear();
    window.__OSPREY_PREFIX__ = '/u/alice';
  });

  test('a click focuses the artifact in the gallery, then opens the artifacts panel', async () => {
    gallery.status = 200;
    const control = mount();
    control.click();

    await vi.waitFor(() => expect(panels.navigate).toHaveBeenCalledWith('artifacts', ''));
    expect(gallery.focusCalls).toHaveLength(1);
    expect(gallery.focusCalls[0].url).toBe('/u/alice/panel/artifacts/api/focus');
    expect(gallery.focusCalls[0].init?.method).toBe('POST');
    expect(JSON.parse(String(gallery.focusCalls[0].init?.body))).toEqual({
      artifact_id: 'a57934cf00dd',
    });
    expect(control.dataset.missing).toBeUndefined();
  });

  test('an id the gallery does not hold opens nothing and marks the control', async () => {
    gallery.status = 404;
    const control = mount();
    control.click();

    await vi.waitFor(() => expect(control.dataset.missing).toBe('true'));
    expect(panels.navigate).not.toHaveBeenCalled();
  });

  test.each([
    [401, 'Your session has expired. Sign in again.'],
    [403, 'The server refused this request'],
    [500, 'The gallery could not be reached'],
  ])('a %i from the gallery opens nothing and says why in the title', async (status, title) => {
    gallery.status = status;
    const control = mount();
    control.click();

    await vi.waitFor(() => expect(control.title).toBe(title));
    expect(control.dataset.missing).toBeUndefined();
    expect(panels.navigate).not.toHaveBeenCalled();
  });

  test('a 404 says the artifact is not in the gallery', async () => {
    gallery.status = 404;
    const control = mount();
    control.click();

    await vi.waitFor(() => expect(control.title).toBe('This artifact is not in the gallery'));
  });

  test('a gallery that cannot be reached says so in the title', async () => {
    gallery.status = 0;
    gallery.fail = true;
    try {
      const control = mount();
      control.click();

      await vi.waitFor(() => expect(control.title).toBe('The gallery could not be reached'));
      expect(panels.navigate).not.toHaveBeenCalled();
    } finally {
      gallery.fail = false;
    }
  });
});
