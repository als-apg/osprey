// @ts-check
/**
 * Unit tests for the Web Terminal's sign-out wiring (app.js's `initSignOut`):
 *   npx vitest run tests/interfaces/web_terminal/app-logout.test.mjs
 *
 * Sign out is the sidebar footer's `#logout-btn`, and — on a deployment that
 * offers it (`html[data-landing-url]`) — the OSPREY mark too. Both ask first
 * (sign-out.js's dialog) and, on confirm, run logout.js's flow.
 *
 * Real logout closes M2 (warm-PTY-inheritance across users/visits): a
 * confirmed sign-out must (1) POST the server logout route
 * (routes/websocket.py's `logout_terminal`, prefix-aware via
 * `window.__OSPREY_PREFIX__` so it reaches this container's
 * `/api/terminal/logout` under `/u/<user>/`), (2) clear the client's stored
 * PTY session id (terminal.js's `clearStoredSessionId`), and only then (3)
 * navigate to the landing URL — in that order, so a fresh page load's
 * `initTerminal()` finds nothing to auto-resume. That fresh load is proven end
 * to end by test_logout_resume_browser.py; a boot with an empty pointer is
 * pinned by session-pointer-boot.test.mjs.
 *
 * app.js is imported once, statically: its own top-level imports (the
 * design-system custom element, panel-manager, settings, etc.) run at
 * import time regardless of what we test, and app.js's DOMContentLoaded
 * bootstrap never fires in this environment (the event has already passed
 * by the time a test module is evaluated) — `initSignOut` is exported
 * precisely so it can be driven directly here instead of relying on that
 * bootstrap.
 *
 * sidebar-artifacts.js is mocked: app.js only calls its init, which this file
 * never reaches.
 */

import { test, expect, describe, beforeEach, afterEach, vi } from 'vitest';

vi.mock('../../../src/osprey/interfaces/web_terminal/static/js/sidebar-artifacts.js', () => ({
  initSidebarArtifacts: vi.fn(),
  teardownSidebarArtifacts: vi.fn(),
}));

import { initSignOut } from '../../../src/osprey/interfaces/web_terminal/static/js/app.js';
import { dismissConfirm } from '../../../src/osprey/interfaces/web_terminal/static/js/posture-confirm.js';

const STORAGE_KEY = 'osprey-pty-session';

/** Render `#logout-btn` the way the server does when `landing_url` is set. */
function renderLogoutButton(/** @type {string} */ landingUrl) {
  document.body.innerHTML = `<button id="logout-btn" data-landing-url="${landingUrl}"></button>`;
  return /** @type {HTMLButtonElement} */ (document.getElementById('logout-btn'));
}

/** The sign-out dialog while it is up, or null. */
const dialog = () =>
  /** @type {HTMLElement|null} */ (
    document.querySelector('.posture-modal-overlay:not([data-closing])')
  );

/** Answer the sign-out dialog with Sign out. */
function confirmDialog() {
  const button = dialog()?.querySelector('.posture-modal-confirm');
  if (!(button instanceof HTMLButtonElement)) throw new Error('no sign-out dialog is up');
  button.click();
}

beforeEach(() => {
  localStorage.clear();
  delete window.__OSPREY_PREFIX__;
});

afterEach(() => {
  dismissConfirm();
  document.body.innerHTML = '';
  delete document.documentElement.dataset.landingUrl;
  vi.unstubAllGlobals();
  vi.restoreAllMocks();
});

describe('initSignOut: no-op guards', () => {
  test.each([
    ['no logout button at all', ''],
    ['a button without data-landing-url (plain `osprey web`)', '<button id="logout-btn"></button>'],
  ])('does nothing with %s', (_case, markup) => {
    document.body.innerHTML = markup;
    const fetchMock = vi.fn();
    vi.stubGlobal('fetch', fetchMock);

    initSignOut();
    /** @type {HTMLButtonElement|null} */ (document.getElementById('logout-btn'))?.click();

    expect(dialog()).toBeNull();
    expect(fetchMock).not.toHaveBeenCalled();
  });

  test('refuses an unsafe landing_url: never calls fetch or navigates', () => {
    const btn = renderLogoutButton('javascript:alert(1)');
    const fetchMock = vi.fn();
    vi.stubGlobal('fetch', fetchMock);
    const assign = vi.fn();
    vi.stubGlobal('location', { origin: 'http://localhost:5000', assign });
    const errSpy = vi.spyOn(console, 'error').mockImplementation(() => {});

    initSignOut();
    btn.click();
    confirmDialog();

    expect(fetchMock).not.toHaveBeenCalled();
    expect(assign).not.toHaveBeenCalled();
    expect(errSpy).toHaveBeenCalled();
    // The guard returns before the in-flight lock, so the button stays usable.
    expect(btn.disabled).toBe(false);
    expect(btn.hasAttribute('aria-busy')).toBe(false);
  });
});

describe('initSignOut: the footer row asks first', () => {
  test('a click opens the sign-out dialog and sends nothing', () => {
    const btn = renderLogoutButton('/landing');
    const fetchMock = vi.fn();
    vi.stubGlobal('fetch', fetchMock);

    initSignOut();
    btn.click();

    expect(dialog()?.querySelector('.posture-modal-title')?.textContent).toBe(
      'Sign out of OSPREY?'
    );
    expect(fetchMock).not.toHaveBeenCalled();
    expect(btn.disabled).toBe(false);
  });

  test('Cancel closes the dialog and signs nobody out', async () => {
    const btn = renderLogoutButton('/landing');
    const fetchMock = vi.fn();
    vi.stubGlobal('fetch', fetchMock);
    const assign = vi.fn();
    vi.stubGlobal('location', { origin: 'http://localhost:5000', assign });

    initSignOut();
    btn.click();
    const cancel = dialog()?.querySelector('.posture-modal-cancel');
    if (!(cancel instanceof HTMLButtonElement)) throw new Error('no sign-out dialog is up');
    cancel.click();

    await Promise.resolve();
    expect(dialog()).toBeNull();
    expect(fetchMock).not.toHaveBeenCalled();
    expect(assign).not.toHaveBeenCalled();
  });
});

describe('initSignOut: the OSPREY mark', () => {
  /**
   * The brand link as index.html renders it, with a fragment href so the
   * click this file lets through does not navigate the test page away.
   */
  function renderBrand() {
    document.body.insertAdjacentHTML(
      'beforeend',
      '<a class="sidebar-brand" href="#home" aria-label="OSPREY">OSPREY</a>'
    );
    return /** @type {HTMLAnchorElement} */ (document.querySelector('a.sidebar-brand'));
  }

  /** Click the brand; returns whether the default (navigation) was prevented. */
  function clickBrand(/** @type {HTMLAnchorElement} */ brand) {
    const event = new MouseEvent('click', { bubbles: true, cancelable: true });
    brand.dispatchEvent(event);
    return event.defaultPrevented;
  }

  test('with sign-out available it opens the dialog instead of navigating', async () => {
    document.documentElement.dataset.landingUrl = '/landing';
    const btn = renderLogoutButton('/landing');
    const brand = renderBrand();
    const fetchMock = vi.fn(async () => ({ ok: true, status: 200 }));
    vi.stubGlobal('fetch', fetchMock);
    const assign = vi.fn();
    vi.stubGlobal('location', { origin: 'http://localhost:5000', assign });

    initSignOut();
    expect(clickBrand(brand)).toBe(true);
    expect(dialog()?.querySelector('.posture-modal-title')?.textContent).toBe(
      'Sign out of OSPREY?'
    );
    expect(fetchMock).not.toHaveBeenCalled();

    confirmDialog();
    await vi.waitFor(() => expect(assign).toHaveBeenCalledWith('/landing'));
    expect(fetchMock).toHaveBeenCalledWith('/api/terminal/logout', { method: 'POST' });
    // The mark is not a button: no lock lands on the footer row it did not click.
    expect(btn.hasAttribute('aria-busy')).toBe(false);
  });

  test('without a landing URL it stays a plain home link', () => {
    const brand = renderBrand();

    initSignOut();

    expect(clickBrand(brand)).toBe(false);
    expect(dialog()).toBeNull();
  });
});

describe('initSignOut: confirmed click flow', () => {
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

    initSignOut();
    btn.click();
    confirmDialog();

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

    initSignOut();
    btn.click();
    confirmDialog();

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
            resolve({
              ok: true,
              status: 200,
              statusText: 'OK',
              json: async () => ({ status: 'ok' }),
            });
        })
    );
    vi.stubGlobal('fetch', fetchMock);
    const assign = vi.fn();
    vi.stubGlobal('location', { origin: 'http://localhost:5000', assign });

    initSignOut();
    btn.click();
    confirmDialog();

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

    vi.stubGlobal(
      'fetch',
      vi.fn(async () => {
        throw new Error('network down');
      })
    );
    const assign = vi.fn();
    vi.stubGlobal('location', { origin: 'http://localhost:5000', assign });
    vi.spyOn(console, 'error').mockImplementation(() => {});

    initSignOut();
    btn.click();
    confirmDialog();

    await vi.waitFor(() => expect(assign).toHaveBeenCalledTimes(1));
    expect(localStorage.getItem(STORAGE_KEY)).toBeNull();
    expect(assign).toHaveBeenCalledWith('/landing');
  });
});

describe('initSignOut: auth-session chaining', () => {
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

    initSignOut();
    btn.click();
    confirmDialog();

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

    initSignOut();
    btn.click();
    confirmDialog();

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

    initSignOut();
    btn.click();
    confirmDialog();

    await vi.waitFor(() => expect(urls).toHaveLength(2));
    expect(urls).toEqual(['/m/carol/api/terminal/logout', '/auth/logout?user=carol']);
  });

  test('a trailing slash on the prefix does not empty the user', async () => {
    window.__OSPREY_PREFIX__ = '/u/alice/';
    const btn = renderLogoutButton('/landing');
    const { urls } = captureFetches();
    vi.stubGlobal('location', { origin: 'http://localhost:5000', assign: vi.fn() });

    initSignOut();
    btn.click();
    confirmDialog();

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

    initSignOut();
    btn.click();
    confirmDialog();

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

    initSignOut();
    btn.click();
    confirmDialog();

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

    initSignOut();
    btn.click();
    confirmDialog();

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

    initSignOut();
    btn.click();
    confirmDialog();

    await vi.waitFor(() => expect(assign).toHaveBeenCalledTimes(1));
    expect(localStorage.getItem(STORAGE_KEY)).toBeNull();
    expect(assign).toHaveBeenCalledWith('/landing');
  });
});
