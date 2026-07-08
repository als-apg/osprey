// @ts-nocheck
// TODO(frontend-hardening Pn): remove & fix types when this interface is retrofitted (P2–P5)
/**
 * Unit tests for theme-manager.js's setTheme() -- the explicit user-toggle
 * path -- covering:
 *   - role-gated persistence (hub only) and broadcast (hub only)
 *   - the D15 behavior: BOTH roles strip a leftover `theme` param from the
 *     URL via history.replaceState, so it can't out-rank the next reload's
 *     OS/localStorage resolution
 *   - broadcast-driven applies (postMessage -> _handleMessage -> _applyTheme,
 *     the hub-bypasses-setTheme path) do NOT strip the param -- only the
 *     explicit-choice path (setTheme) does
 *
 * Pure DOM/logic guard, happy-dom environment (configured globally):
 *   npx vitest run tests/interfaces/design_system/js/theme-settheme.test.mjs
 *
 * theme-manager.js keeps role/preference/listener-attached state as
 * module-level singletons, so each test resets the module registry and
 * re-imports fresh through the `/design-system/js/*` alias (see
 * vitest.config.js) to get an untouched instance. This mirrors
 * alias-smoke.test.mjs's use of the absolute runtime specifier.
 */

import { test, expect, describe, beforeEach, vi } from 'vitest';

describe('setTheme()', () => {
  /** @type {typeof import('/design-system/js/theme-manager.js')} */
  let ThemeManager;

  beforeEach(async () => {
    vi.resetModules();
    ThemeManager = await import('/design-system/js/theme-manager.js');
    window.localStorage.clear();
    document.documentElement.removeAttribute('data-theme');
    document.body.innerHTML = '';
    window.history.replaceState({}, '', '/');
  });

  function spyOnBroadcast() {
    const iframe = document.createElement('iframe');
    document.body.appendChild(iframe);
    return vi.spyOn(iframe.contentWindow, 'postMessage').mockImplementation(() => {});
  }

  describe('follower role', () => {
    test('applies the requested theme', () => {
      ThemeManager.initTheme({ role: 'follower' });

      ThemeManager.setTheme('light');

      expect(ThemeManager.getTheme()).toBe('light');
      expect(document.documentElement.getAttribute('data-theme')).toBe('light');
    });

    test('never persists to localStorage', () => {
      ThemeManager.initTheme({ role: 'follower' });

      ThemeManager.setTheme('light');

      expect(window.localStorage.getItem('osprey-theme')).toBeNull();
    });

    test('never broadcasts to embedded iframes', () => {
      const postMessage = spyOnBroadcast();

      ThemeManager.initTheme({ role: 'follower' });
      ThemeManager.setTheme('light');

      expect(postMessage).not.toHaveBeenCalled();
    });

    test('strips a leftover ?theme= param from the URL, preserving other params', () => {
      window.history.replaceState({}, '', '/panel?theme=dark&foo=bar');
      ThemeManager.initTheme({ role: 'follower' });

      ThemeManager.setTheme('light');

      expect(window.location.search).not.toContain('theme');
      expect(window.location.search).toContain('foo=bar');
    });

    test('is a no-op on the URL when there was no ?theme= to begin with', () => {
      window.history.replaceState({}, '', '/panel?foo=bar');
      ThemeManager.initTheme({ role: 'follower' });

      ThemeManager.setTheme('light');

      expect(window.location.search).toBe('?foo=bar');
    });
  });

  describe('hub role', () => {
    test('persists the preference to localStorage', () => {
      ThemeManager.initTheme({ role: 'hub' });

      ThemeManager.setTheme('light');

      expect(window.localStorage.getItem('osprey-theme')).toBe('light');
    });

    test('broadcasts the resolved theme to embedded iframes', () => {
      const postMessage = spyOnBroadcast();

      ThemeManager.initTheme({ role: 'hub' });
      ThemeManager.setTheme('light');

      expect(postMessage).toHaveBeenCalledWith(
        { type: 'osprey-theme-change', theme: 'light' },
        window.location.origin
      );
    });

    test('strips a leftover ?theme= param from the URL', () => {
      window.history.replaceState({}, '', '/?theme=dark');
      ThemeManager.initTheme({ role: 'hub' });

      ThemeManager.setTheme('light');

      expect(window.location.search).not.toContain('theme');
    });
  });

  describe('broadcast-driven apply (postMessage, follower role)', () => {
    test('applies the broadcast theme but does NOT strip ?theme= from the URL', () => {
      window.history.replaceState({}, '', '/panel?theme=dark');
      ThemeManager.initTheme({ role: 'follower' });
      expect(ThemeManager.getTheme()).toBe('dark');

      window.dispatchEvent(
        new MessageEvent('message', {
          origin: window.location.origin,
          data: { type: 'osprey-theme-change', theme: 'light' },
        })
      );

      expect(ThemeManager.getTheme()).toBe('light');
      expect(window.location.search).toContain('theme=dark');
    });
  });
});
