// @ts-check
/**
 * Unit tests for the Web Terminal logout button (app.js's `initLogoutButton`):
 *   npx vitest run tests/interfaces/web_terminal/app-logout.test.mjs
 *
 * Real logout closes M2 (warm-PTY-inheritance across users/visits): a click
 * must (1) POST the server logout route (routes/websocket.py's
 * `logout_terminal`, prefix-aware via `window.__OSPREY_PREFIX__` so it
 * reaches this container's `/api/terminal/logout` under `/u/<user>/`),
 * (2) clear the client's stored PTY session id (terminal.js's
 * `clearStoredSessionId`), and only then (3) navigate to the landing URL —
 * in that order, so a fresh page load's `initTerminal()` finds nothing to
 * auto-resume (asserted directly against terminal.js in the last describe
 * block).
 *
 * app.js is imported once, statically: its own top-level imports (the
 * design-system custom element, panel-manager, settings, etc.) run at
 * import time regardless of what we test, and app.js's DOMContentLoaded
 * bootstrap never fires in this environment (the event has already passed
 * by the time a test module is evaluated) — `initLogoutButton` is exported
 * precisely so it can be driven directly here instead of relying on that
 * bootstrap.
 */

import { test, expect, describe, beforeEach, afterEach, vi } from 'vitest';
import { initLogoutButton } from '../../../src/osprey/interfaces/web_terminal/static/js/app.js';

const STORAGE_KEY = 'osprey-pty-session';

/** Render `#logout-btn` the way the server does when `landing_url` is set. */
function renderLogoutButton(/** @type {string} */ landingUrl) {
  document.body.innerHTML = `<button id="logout-btn" data-landing-url="${landingUrl}"></button>`;
  return /** @type {HTMLButtonElement} */ (document.getElementById('logout-btn'));
}

beforeEach(() => {
  localStorage.clear();
  delete window.__OSPREY_PREFIX__;
});

afterEach(() => {
  vi.unstubAllGlobals();
  vi.restoreAllMocks();
});

describe('initLogoutButton: no-op guards (unchanged from the nav-only version)', () => {
  test('does nothing when #logout-btn is absent from the DOM', () => {
    document.body.innerHTML = '';
    expect(() => initLogoutButton()).not.toThrow();
  });

  test('does nothing when the button has no data-landing-url (plain `osprey web`)', () => {
    document.body.innerHTML = '<button id="logout-btn"></button>';
    const btn = /** @type {HTMLButtonElement} */ (document.getElementById('logout-btn'));
    const fetchMock = vi.fn();
    vi.stubGlobal('fetch', fetchMock);

    initLogoutButton();
    btn.click();

    expect(fetchMock).not.toHaveBeenCalled();
  });

  test('refuses an unsafe landing_url: never calls fetch or navigates', () => {
    const btn = renderLogoutButton('javascript:alert(1)');
    const fetchMock = vi.fn();
    vi.stubGlobal('fetch', fetchMock);
    const assign = vi.fn();
    vi.stubGlobal('location', { origin: 'http://localhost:5000', assign });
    const errSpy = vi.spyOn(console, 'error').mockImplementation(() => {});

    initLogoutButton();
    btn.click();

    expect(fetchMock).not.toHaveBeenCalled();
    expect(assign).not.toHaveBeenCalled();
    expect(errSpy).toHaveBeenCalled();
    // The guard returns before the in-flight lock, so the button stays usable.
    expect(btn.disabled).toBe(false);
    expect(btn.hasAttribute('aria-busy')).toBe(false);
  });
});

describe('initLogoutButton: click flow', () => {
  test('POSTs the logout route, clears storage, then navigates -- in that order', async () => {
    localStorage.setItem(STORAGE_KEY, 'warm-session-id');
    const btn = renderLogoutButton('/landing');

    const fetchMock = vi.fn(async (/** @type {string} */ url, /** @type {any} */ opts) => {
      expect(url).toBe('/api/terminal/logout');
      expect(opts).toEqual({ method: 'POST' });
      // The stored pointer must still be intact when the request goes out —
      // the client only drops it after the server confirms.
      expect(localStorage.getItem(STORAGE_KEY)).toBe('warm-session-id');
      return { ok: true, status: 200, statusText: 'OK', json: async () => ({ status: 'ok' }) };
    });
    vi.stubGlobal('fetch', fetchMock);

    const assign = vi.fn((/** @type {string} */ url) => {
      // Cleared by the time navigation happens, so the landing -> return
      // round trip's initTerminal() has nothing to auto-resume.
      expect(localStorage.getItem(STORAGE_KEY)).toBeNull();
      expect(url).toBe('/landing');
    });
    vi.stubGlobal('location', { origin: 'http://localhost:5000', assign });

    initLogoutButton();
    btn.click();

    await vi.waitFor(() => expect(assign).toHaveBeenCalledTimes(1));
    expect(fetchMock).toHaveBeenCalledTimes(1);
    expect(localStorage.getItem(STORAGE_KEY)).toBeNull();
  });

  test('prepends window.__OSPREY_PREFIX__ to the logout route (multi-user deployments)', async () => {
    window.__OSPREY_PREFIX__ = '/u/alice';
    const btn = renderLogoutButton('/landing');

    const fetchMock = vi.fn(async () => ({
      ok: true,
      status: 200,
      statusText: 'OK',
      json: async () => ({ status: 'ok' }),
    }));
    vi.stubGlobal('fetch', fetchMock);
    const assign = vi.fn();
    vi.stubGlobal('location', { origin: 'http://localhost:5000', assign });

    initLogoutButton();
    btn.click();

    await vi.waitFor(() => expect(assign).toHaveBeenCalled());
    expect(fetchMock).toHaveBeenCalledWith('/u/alice/api/terminal/logout', { method: 'POST' });
  });

  test('locks the button (disabled + aria-busy) while the request is in flight', async () => {
    const btn = renderLogoutButton('/landing');

    // Hold the request open so the in-flight window is observable before nav.
    /** @type {() => void} */
    let releaseFetch = () => {};
    const fetchMock = vi.fn(
      () =>
        new Promise((resolve) => {
          releaseFetch = () =>
            resolve({ ok: true, status: 200, statusText: 'OK', json: async () => ({ status: 'ok' }) });
        })
    );
    vi.stubGlobal('fetch', fetchMock);
    const assign = vi.fn();
    vi.stubGlobal('location', { origin: 'http://localhost:5000', assign });

    initLogoutButton();
    btn.click();

    // Request dispatched, not yet resolved: button is locked and announced,
    // and a second click can't fire a second POST.
    await vi.waitFor(() => expect(fetchMock).toHaveBeenCalledTimes(1));
    expect(btn.disabled).toBe(true);
    expect(btn.getAttribute('aria-busy')).toBe('true');
    expect(assign).not.toHaveBeenCalled();

    releaseFetch();
    await vi.waitFor(() => expect(assign).toHaveBeenCalledTimes(1));
    expect(fetchMock).toHaveBeenCalledTimes(1);
  });

  test('a failed logout request still clears storage and navigates (best effort)', async () => {
    localStorage.setItem(STORAGE_KEY, 'warm-session-id');
    const btn = renderLogoutButton('/landing');

    vi.stubGlobal('fetch', vi.fn(async () => { throw new Error('network down'); }));
    const assign = vi.fn();
    vi.stubGlobal('location', { origin: 'http://localhost:5000', assign });
    vi.spyOn(console, 'error').mockImplementation(() => {});

    initLogoutButton();
    btn.click();

    await vi.waitFor(() => expect(assign).toHaveBeenCalledTimes(1));
    expect(localStorage.getItem(STORAGE_KEY)).toBeNull();
    expect(assign).toHaveBeenCalledWith('/landing');
  });
});

describe('post-logout: a fresh load does not auto-resume', () => {
  /** Minimal fake xterm.js Terminal -- just enough surface for initTerminal(). */
  class FakeTerminal {
    constructor() {
      this.cols = 80;
      this.rows = 24;
      this.options = {};
    }
    loadAddon() {}
    open() {}
    onData() {}
    onResize() {}
    write() {}
    reset() {}
    focus() {}
  }

  class FakeAddon {
    fit() {}
  }

  /** Captures the URL of the most recently constructed socket; never opens. */
  class FakeWebSocket {
    /** @param {string} url */
    constructor(url) {
      this.url = url;
      this.onopen = null;
      this.onmessage = null;
      this.onclose = null;
      FakeWebSocket.last = this;
    }
    send() {}
    close() {}
  }
  /** @type {FakeWebSocket|null} */
  FakeWebSocket.last = null;

  test('clearStoredSessionId (the logout step) leaves initTerminal() nothing to resume', async () => {
    localStorage.setItem(STORAGE_KEY, 'warm-session-id');

    document.body.innerHTML = '<div id="terminal-container"></div>';
    // happy-dom does not implement document.fonts; initTerminal() awaits its
    // `ready` promise to re-fit after web fonts load.
    // @ts-expect-error -- test stub, not a full FontFaceSet
    document.fonts = { ready: Promise.resolve() };
    vi.stubGlobal('Terminal', FakeTerminal);
    vi.stubGlobal('FitAddon', { FitAddon: FakeAddon });
    vi.stubGlobal('WebLinksAddon', { WebLinksAddon: class {} });
    vi.stubGlobal('WebSocket', FakeWebSocket);
    // xtermPalette() logs a console.error when the CSS custom properties it
    // reads are absent, which they are in this bare happy-dom document.
    vi.spyOn(console, 'error').mockImplementation(() => {});

    vi.resetModules();
    const terminal = await import(
      '../../../src/osprey/interfaces/web_terminal/static/js/terminal.js'
    );

    // This is exactly what app.js's initLogoutButton does before navigating.
    terminal.clearStoredSessionId();
    expect(localStorage.getItem(STORAGE_KEY)).toBeNull();

    terminal.initTerminal('terminal-container');

    // A fresh session -- no ?session_id=&mode=resume on the constructed
    // WebSocket URL -- because there is nothing left in storage to resume.
    const constructedUrl = /** @type {FakeWebSocket} */ (FakeWebSocket.last).url;
    expect(constructedUrl).not.toMatch(/mode=resume/);
    expect(constructedUrl).not.toMatch(/session_id=/);
  });
});
