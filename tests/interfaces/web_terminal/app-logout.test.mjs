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
 * auto-resume. That fresh load is proven end to end by
 * test_logout_resume_browser.py; a boot with an empty pointer is pinned by
 * session-pointer-boot.test.mjs.
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
  test.each([
    ['no logout button at all', ''],
    ['a button without data-landing-url (plain `osprey web`)', '<button id="logout-btn"></button>'],
  ])('does nothing with %s', (_case, markup) => {
    document.body.innerHTML = markup;
    const fetchMock = vi.fn();
    vi.stubGlobal('fetch', fetchMock);

    initLogoutButton();
    /** @type {HTMLButtonElement|null} */ (document.getElementById('logout-btn'))?.click();

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

  test('binds the display-menu copy too, and locks only the clicked button', async () => {
    // Log out renders twice — `#logout-btn` in the header identity menu and
    // `#display-menu-logout-btn` in the display menu's action row — and one
    // initLogoutButton() call wires both to the same flow.
    document.body.innerHTML =
      '<button id="logout-btn" data-landing-url="/landing"></button>' +
      '<button id="display-menu-logout-btn" data-landing-url="/landing"></button>';
    const chipBtn = /** @type {HTMLButtonElement} */ (document.getElementById('logout-btn'));
    const menuBtn = /** @type {HTMLButtonElement} */ (
      document.getElementById('display-menu-logout-btn')
    );

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
    menuBtn.click();

    await vi.waitFor(() => expect(assign).toHaveBeenCalledTimes(1));
    expect(fetchMock).toHaveBeenCalledWith('/api/terminal/logout', { method: 'POST' });
    // The in-flight lock lands on the control the operator clicked; the other
    // copy is left alone (every path navigates away regardless).
    expect(menuBtn.disabled).toBe(true);
    expect(menuBtn.getAttribute('aria-busy')).toBe('true');
    expect(chipBtn.disabled).toBe(false);
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

describe('initLogoutButton: auth-session chaining', () => {
  /** Every fetch the click made, in order. */
  function captureFetches() {
    /** @type {string[]} */
    const urls = [];
    const fetchMock = vi.fn(async (/** @type {string} */ url) => {
      urls.push(url);
      return { ok: true, status: 200, statusText: 'OK', json: async () => ({ status: 'ok' }) };
    });
    vi.stubGlobal('fetch', fetchMock);
    return { urls, fetchMock };
  }

  test('ends the auth session after the PTY teardown, before navigating', async () => {
    window.__OSPREY_PREFIX__ = '/u/alice';
    const btn = renderLogoutButton('/landing');
    const { urls } = captureFetches();
    const assign = vi.fn();
    vi.stubGlobal('location', { origin: 'http://localhost:5000', assign });

    initLogoutButton();
    btn.click();

    await vi.waitFor(() => expect(assign).toHaveBeenCalledTimes(1));
    // Order matters: the terminal is torn down first, so a session that
    // outlives the request cannot still be driving a live PTY.
    expect(urls).toEqual(['/u/alice/api/terminal/logout', '/auth/logout?user=alice']);
  });

  test('sends exactly one user parameter, percent-encoded', async () => {
    // A roster name carrying the query's own delimiters: unencoded, it would
    // smuggle a second `user` parameter into the sidecar request.
    window.__OSPREY_PREFIX__ = '/u/a&user=b';
    const btn = renderLogoutButton('/landing');
    const { urls } = captureFetches();
    vi.stubGlobal('location', { origin: 'http://localhost:5000', assign: vi.fn() });

    initLogoutButton();
    btn.click();

    await vi.waitFor(() => expect(urls).toHaveLength(2));
    const query = new URL(urls[1], 'http://localhost:5000').searchParams;
    // The route refuses a repeated `user` outright rather than picking one.
    expect(query.getAll('user')).toEqual(['a&user=b']);
    expect(urls[1]).toBe('/auth/logout?user=a%26user%3Db');
  });

  test("takes the user from the prefix's last segment, not from a spelled mount root", async () => {
    window.__OSPREY_PREFIX__ = '/m/carol';
    const btn = renderLogoutButton('/landing');
    const { urls } = captureFetches();
    vi.stubGlobal('location', { origin: 'http://localhost:5000', assign: vi.fn() });

    initLogoutButton();
    btn.click();

    await vi.waitFor(() => expect(urls).toHaveLength(2));
    expect(urls).toEqual(['/m/carol/api/terminal/logout', '/auth/logout?user=carol']);
  });

  test('a trailing slash on the prefix does not empty the user', async () => {
    window.__OSPREY_PREFIX__ = '/u/alice/';
    const btn = renderLogoutButton('/landing');
    const { urls } = captureFetches();
    vi.stubGlobal('location', { origin: 'http://localhost:5000', assign: vi.fn() });

    initLogoutButton();
    btn.click();

    await vi.waitFor(() => expect(urls).toHaveLength(2));
    expect(urls[1]).toBe('/auth/logout?user=alice');
  });

  test('carries same-origin credentials, or the session cookie never arrives', async () => {
    window.__OSPREY_PREFIX__ = '/u/alice';
    const btn = renderLogoutButton('/landing');
    const fetchMock = vi.fn(async () => ({
      ok: true,
      status: 200,
      statusText: 'OK',
      json: async () => ({ status: 'ok' }),
    }));
    vi.stubGlobal('fetch', fetchMock);
    vi.stubGlobal('location', { origin: 'http://localhost:5000', assign: vi.fn() });

    initLogoutButton();
    btn.click();

    await vi.waitFor(() => expect(fetchMock).toHaveBeenCalledTimes(2));
    expect(fetchMock).toHaveBeenLastCalledWith('/auth/logout?user=alice', {
      credentials: 'same-origin',
      cache: 'no-store',
    });
  });

  test('skips the sidecar entirely without a per-user prefix (plain `osprey web`)', async () => {
    const btn = renderLogoutButton('/landing');
    const { urls } = captureFetches();
    const assign = vi.fn();
    vi.stubGlobal('location', { origin: 'http://localhost:5000', assign });

    initLogoutButton();
    btn.click();

    await vi.waitFor(() => expect(assign).toHaveBeenCalledTimes(1));
    expect(urls).toEqual(['/api/terminal/logout']);
  });

  test('still navigates when the sidecar answers 404 (authentication is off)', async () => {
    // The regression this shape exists to prevent: `location /auth/` is only
    // rendered when `auth.method != "none"`, and the app cannot tell the two
    // postures apart, so a *navigation* would strand a no-auth deployment on a
    // 404 instead of the landing page.
    window.__OSPREY_PREFIX__ = '/u/alice';
    const btn = renderLogoutButton('/landing');
    vi.stubGlobal(
      'fetch',
      vi.fn(async (/** @type {string} */ url) =>
        url.startsWith('/auth/')
          ? { ok: false, status: 404, statusText: 'Not Found' }
          : { ok: true, status: 200, statusText: 'OK', json: async () => ({ status: 'ok' }) }
      )
    );
    const assign = vi.fn();
    vi.stubGlobal('location', { origin: 'http://localhost:5000', assign });
    const errSpy = vi.spyOn(console, 'error').mockImplementation(() => {});

    initLogoutButton();
    btn.click();

    await vi.waitFor(() => expect(assign).toHaveBeenCalledTimes(1));
    expect(assign).toHaveBeenCalledWith('/landing');
    // A 404 here is the expected answer, not a fault to shout about.
    expect(errSpy).not.toHaveBeenCalled();
  });

  test('still clears storage and navigates when the sidecar is unreachable', async () => {
    window.__OSPREY_PREFIX__ = '/u/alice';
    localStorage.setItem(STORAGE_KEY, 'warm-session-id');
    const btn = renderLogoutButton('/landing');
    vi.stubGlobal(
      'fetch',
      vi.fn(async (/** @type {string} */ url) => {
        if (url.startsWith('/auth/')) throw new Error('sidecar down');
        return { ok: true, status: 200, statusText: 'OK', json: async () => ({ status: 'ok' }) };
      })
    );
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
