/**
 * Unit tests for theme-manager.js's per-user scoping of its localStorage
 * preference slot.
 *
 * theme-manager is the hub-role writer for the same key theme-boot.js reads
 * pre-paint, so the two must agree on the key or a persisted pick would be
 * invisible on the next reload. What is pinned here:
 *
 *   - a scoped page WRITES `osprey-theme--<scope>` and leaves the bare key alone
 *   - it READS that same key back, so a pick survives a reload
 *   - it never falls back to the polluted bare key when its own is missing
 *   - unscoped serving still uses the bare key, unchanged
 *   - the key is resolved per call, not captured at module load — the scope
 *     attribute is on the server-rendered document, which is in place before
 *     any of this runs, but a load-time capture would silently pin the wrong
 *     key for any page that imports the module before init
 *
 * theme-manager.js keeps role/preference state as module-level singletons, so
 * each test resets the module registry and re-imports fresh through the
 * `/design-system/js/*` alias (see vitest.config.js).
 *
 * Pure DOM/logic guard, happy-dom environment (configured globally):
 *   npx vitest run tests/interfaces/design_system/js/theme-manager-scope.test.mjs
 */

import { test, expect, describe, beforeEach, afterEach, vi } from 'vitest';
import { stubOSPreference } from './os-preference.mjs';

const SCOPE_ATTR = 'data-osprey-storage-scope';
const LEGACY_KEY = 'osprey-theme';

describe('theme-manager.js storage scoping', () => {
  /** @type {typeof import('/design-system/js/theme-manager.js')} */
  let ThemeManager;

  beforeEach(async () => {
    vi.resetModules();
    window.localStorage.clear();
    document.documentElement.removeAttribute('data-theme');
    document.documentElement.removeAttribute('data-theme-mode');
    document.documentElement.removeAttribute(SCOPE_ATTR);
    document.body.innerHTML = '';
    window.history.replaceState({}, '', '/');
    stubOSPreference(false);
    ThemeManager = await import('/design-system/js/theme-manager.js');
  });

  describe('writing', () => {
    test('a scoped hub persists under its own key and leaves the bare one alone', () => {
      document.documentElement.setAttribute(SCOPE_ATTR, 'bob');
      ThemeManager.initTheme({ role: 'hub' });

      ThemeManager.setTheme('retro-dark');

      expect(window.localStorage.getItem(`${LEGACY_KEY}--bob`)).toBe(
        JSON.stringify({ family: 'retro', mode: 'dark' })
      );
      expect(window.localStorage.getItem(LEGACY_KEY)).toBeNull();
    });

    test("one user's write does not land in another's slot", () => {
      document.documentElement.setAttribute(SCOPE_ATTR, 'carol');
      ThemeManager.initTheme({ role: 'hub' });

      ThemeManager.setTheme('retro-dark');

      expect(window.localStorage.getItem(`${LEGACY_KEY}--bob`)).toBeNull();
      expect(window.localStorage.getItem(`${LEGACY_KEY}--carol`)).not.toBeNull();
    });

    test('an unscoped hub still writes the bare legacy key', () => {
      ThemeManager.initTheme({ role: 'hub' });

      ThemeManager.setTheme('retro-dark');

      expect(window.localStorage.getItem(LEGACY_KEY)).toBe(
        JSON.stringify({ family: 'retro', mode: 'dark' })
      );
    });

    test('the key is resolved per call, not captured when the module loaded', () => {
      // The module was imported in beforeEach with no attribute present; a
      // load-time capture would write the bare key here.
      document.documentElement.setAttribute(SCOPE_ATTR, 'bob');
      ThemeManager.initTheme({ role: 'hub' });

      ThemeManager.setTheme('retro-dark');

      expect(window.localStorage.getItem(`${LEGACY_KEY}--bob`)).not.toBeNull();
      expect(window.localStorage.getItem(LEGACY_KEY)).toBeNull();
    });
  });

  describe('reading', () => {
    test("a scoped hub reads back what it wrote — the writer's key is the reader's", async () => {
      document.documentElement.setAttribute(SCOPE_ATTR, 'bob');
      ThemeManager.initTheme({ role: 'hub' });
      ThemeManager.setTheme('retro-dark');

      // A fresh module instance, as on the next page load.
      vi.resetModules();
      const reloaded = await import('/design-system/js/theme-manager.js');
      reloaded.initTheme({ role: 'hub' });

      expect(reloaded.getTheme()).toBe('retro-dark');
    });

    test('a scoped hub ignores a polluted bare key and takes the server default', () => {
      window.localStorage.setItem(LEGACY_KEY, JSON.stringify({ family: 'retro', mode: 'dark' }));
      document.documentElement.setAttribute(SCOPE_ATTR, 'bob');
      document.documentElement.setAttribute('data-theme', 'light');

      ThemeManager.initTheme({ role: 'hub' });

      expect(ThemeManager.getTheme()).toBe('light');
    });

    test("a scoped hub does not see another user's stored preference", () => {
      window.localStorage.setItem(
        `${LEGACY_KEY}--bob`,
        JSON.stringify({ family: 'retro', mode: 'dark' })
      );
      document.documentElement.setAttribute(SCOPE_ATTR, 'carol');
      document.documentElement.setAttribute('data-theme', 'light');

      ThemeManager.initTheme({ role: 'hub' });

      expect(ThemeManager.getTheme()).toBe('light');
    });

    test('an unscoped hub still reads the bare legacy key', () => {
      window.localStorage.setItem(LEGACY_KEY, JSON.stringify({ family: 'retro', mode: 'dark' }));
      document.documentElement.setAttribute('data-theme', 'light');

      ThemeManager.initTheme({ role: 'hub' });

      expect(ThemeManager.getTheme()).toBe('retro-dark');
    });
  });
});

/*
 * Opaque origin (C-OPAQUE / C-MSG): an agent-written artifact page or panel is
 * served sandboxed, so its origin is "null", localStorage throws on access, and
 * no hub message can match its origin. The follower must still take the theme
 * from the frame that embeds it, and the hub must address every frame with "*"
 * -- for the theme message only.
 */
describe('theme-manager.js under an opaque origin', () => {
  /** @type {typeof import('/design-system/js/theme-manager.js')} */
  let ThemeManager;

  beforeEach(async () => {
    vi.resetModules();
    window.localStorage.clear();
    document.documentElement.removeAttribute('data-theme');
    document.documentElement.removeAttribute('data-theme-mode');
    document.documentElement.removeAttribute(SCOPE_ATTR);
    document.body.innerHTML = '';
    window.history.replaceState({}, '', '/');
    stubOSPreference(false);
    ThemeManager = await import('/design-system/js/theme-manager.js');
  });

  afterEach(() => {
    vi.restoreAllMocks();
    vi.unstubAllGlobals();
  });

  /** Make this window an opaque document embedded by `hub`. */
  function becomeOpaqueFrame() {
    const hub = { postMessage: vi.fn() };
    vi.spyOn(window.location, 'origin', 'get').mockReturnValue('null');
    vi.stubGlobal('parent', hub);
    vi.spyOn(window, 'localStorage', 'get').mockImplementation(() => {
      throw new DOMException('opaque origin', 'SecurityError');
    });
    return hub;
  }

  /**
   * @param {any} source
   * @param {string} origin
   * @param {any} data
   */
  function deliver(source, origin, data) {
    window.dispatchEvent(new MessageEvent('message', { source, origin, data }));
  }

  test('a follower initialises without touching storage and keeps the boot theme', () => {
    becomeOpaqueFrame();
    document.documentElement.setAttribute('data-theme', 'retro-dark');

    expect(() => ThemeManager.initTheme({ role: 'follower' })).not.toThrow();
    expect(ThemeManager.getTheme()).toBe('retro-dark');
  });

  test('a hub whose storage throws still initialises and switches themes', () => {
    becomeOpaqueFrame();

    expect(() => ThemeManager.initTheme({ role: 'hub' })).not.toThrow();
    expect(() => ThemeManager.setTheme('retro-dark')).not.toThrow();
    expect(ThemeManager.getTheme()).toBe('retro-dark');
  });

  test('the follower applies the theme message from its parent, whatever its origin', () => {
    const hub = becomeOpaqueFrame();
    ThemeManager.initTheme({ role: 'follower' });

    deliver(hub, 'https://console.example', { type: 'osprey-theme-change', theme: 'retro-dark' });

    expect(ThemeManager.getTheme()).toBe('retro-dark');
    expect(document.documentElement.getAttribute('data-theme')).toBe('retro-dark');
  });

  test('the follower ignores the theme message from any window but its parent', () => {
    becomeOpaqueFrame();
    document.documentElement.setAttribute('data-theme', 'light');
    ThemeManager.initTheme({ role: 'follower' });

    deliver({ postMessage() {} }, 'https://evil.example', {
      type: 'osprey-theme-change',
      theme: 'retro-dark',
    });
    deliver(null, 'null', { type: 'osprey-theme-change', theme: 'retro-dark' });

    expect(ThemeManager.getTheme()).toBe('light');
  });

  test('an unknown theme id from the parent still never reaches data-theme', () => {
    const hub = becomeOpaqueFrame();
    ThemeManager.initTheme({ role: 'follower' });

    deliver(hub, 'https://console.example', { type: 'osprey-theme-change', theme: 'x" onload="' });

    expect(document.documentElement.getAttribute('data-theme')).not.toContain('onload');
  });

  test('a same-origin follower does not take a foreign-origin message from its parent', () => {
    const hub = { postMessage: vi.fn() };
    vi.stubGlobal('parent', hub);
    document.documentElement.setAttribute('data-theme', 'light');
    ThemeManager.initTheme({ role: 'follower' });

    deliver(hub, 'https://evil.example', { type: 'osprey-theme-change', theme: 'retro-dark' });

    expect(ThemeManager.getTheme()).toBe('light');
  });

  test('the hub posts the theme message to every frame with target "*"', () => {
    document.body.innerHTML = '<iframe></iframe><iframe sandbox="allow-scripts"></iframe>';
    const frames = [...document.querySelectorAll('iframe')];
    const spies = frames.map((f) =>
      vi.spyOn(/** @type {Window} */ (f.contentWindow), 'postMessage')
    );
    ThemeManager.initTheme({ role: 'hub' });

    ThemeManager.setTheme('retro-dark');

    for (const spy of spies) {
      expect(spy).toHaveBeenCalledWith({ type: 'osprey-theme-change', theme: 'retro-dark' }, '*');
    }
  });
});
