/**
 * Unit tests for theme-manager.js's initTheme() first-visit resolution --
 * the hub must adopt the family theme-boot.js already applied (which honors
 * a server-configured web.theme), not displace it with DEFAULT_FAMILY.
 *
 * Pure DOM/logic guard, happy-dom environment (configured globally):
 *   npx vitest run tests/interfaces/design_system/js/theme-init-server-family.test.mjs
 *
 * theme-manager.js keeps role/preference state as module-level singletons,
 * so each test resets the module registry and re-imports fresh through the
 * `/design-system/js/*` alias (see vitest.config.js).
 */

import { test, expect, describe, beforeEach, vi } from 'vitest';

describe('initTheme() first visit (hub)', () => {
  /** @type {typeof import('/design-system/js/theme-manager.js')} */
  let ThemeManager;

  beforeEach(async () => {
    vi.resetModules();
    ThemeManager = await import('/design-system/js/theme-manager.js');
    window.localStorage.clear();
    document.documentElement.removeAttribute('data-theme');
    window.history.replaceState({}, '', '/');
  });

  test('adopts the boot-applied (server-configured) family when nothing is stored', () => {
    // theme-boot.js has already resolved the server-stamped web.theme family
    // and applied a concrete id pre-paint.
    document.documentElement.setAttribute('data-theme', 'high-contrast-dark');

    ThemeManager.initTheme({ role: 'hub' });

    // Mode stays 'auto' (OS decides light/dark) but the family must survive.
    expect(ThemeManager.getTheme()).toMatch(/^high-contrast-/);
  });

  test('falls back to the default family when no theme was applied pre-init', () => {
    ThemeManager.initTheme({ role: 'hub' });

    expect(ThemeManager.getTheme()).toMatch(/^(dark|light)$/);
  });

  test('a stored preference still outranks the boot-applied family', () => {
    document.documentElement.setAttribute('data-theme', 'high-contrast-dark');
    window.localStorage.setItem(
      'osprey-theme',
      JSON.stringify({ family: 'main', mode: 'light' })
    );

    ThemeManager.initTheme({ role: 'hub' });

    expect(ThemeManager.getTheme()).toBe('light');
  });
});

describe('initTheme() first visit (hub): a server-pinned mode', () => {
  /** @type {typeof import('/design-system/js/theme-manager.js')} */
  let ThemeManager;

  /**
   * Force the OS preference so a pin can be shown to beat it. Without this the
   * assertions would pass for the wrong reason whenever the host's own OS
   * preference happened to agree with the pin.
   * @param {boolean} prefersDark
   */
  function stubOSPreference(prefersDark) {
    vi.stubGlobal('matchMedia', (/** @type {string} */ query) => ({
      matches: query.includes('dark') ? prefersDark : !prefersDark,
      media: query,
      addEventListener() {},
      removeEventListener() {},
      addListener() {},
      removeListener() {},
    }));
  }

  beforeEach(async () => {
    vi.resetModules();
    vi.unstubAllGlobals();
    ThemeManager = await import('/design-system/js/theme-manager.js');
    window.localStorage.clear();
    document.documentElement.removeAttribute('data-theme');
    document.documentElement.removeAttribute('data-theme-mode');
    window.history.replaceState({}, '', '/');
  });

  test('honors data-theme-mode over the OS preference', () => {
    // The deployment configured `web.theme: high-contrast-light`; the
    // operator's OS says dark. Config wins on a first visit.
    stubOSPreference(true);
    document.documentElement.setAttribute('data-theme', 'high-contrast-light');
    document.documentElement.setAttribute('data-theme-mode', 'light');

    ThemeManager.initTheme({ role: 'hub' });

    expect(ThemeManager.getTheme()).toBe('high-contrast-light');
  });

  test('a pinned dark survives a light OS preference', () => {
    stubOSPreference(false);
    document.documentElement.setAttribute('data-theme', 'high-contrast-dark');
    document.documentElement.setAttribute('data-theme-mode', 'dark');

    ThemeManager.initTheme({ role: 'hub' });

    expect(ThemeManager.getTheme()).toBe('high-contrast-dark');
  });

  test('without the attribute the OS still decides the mode', () => {
    // `web.theme: high-contrast` names a palette, not a mode.
    stubOSPreference(false);
    document.documentElement.setAttribute('data-theme', 'high-contrast-dark');

    ThemeManager.initTheme({ role: 'hub' });

    expect(ThemeManager.getTheme()).toBe('high-contrast-light');
  });

  test('an unrecognized data-theme-mode is ignored, not obeyed', () => {
    stubOSPreference(false);
    document.documentElement.setAttribute('data-theme', 'high-contrast-dark');
    document.documentElement.setAttribute('data-theme-mode', 'sepia');

    ThemeManager.initTheme({ role: 'hub' });

    expect(ThemeManager.getTheme()).toBe('high-contrast-light');
  });

  test('a stored preference still outranks a server pin', () => {
    // Config sets the DEFAULT, not a lock: once the operator has chosen,
    // their choice must survive every reload.
    stubOSPreference(true);
    document.documentElement.setAttribute('data-theme', 'high-contrast-light');
    document.documentElement.setAttribute('data-theme-mode', 'light');
    window.localStorage.setItem('osprey-theme', JSON.stringify({ family: 'main', mode: 'dark' }));

    ThemeManager.initTheme({ role: 'hub' });

    expect(ThemeManager.getTheme()).toBe('dark');
  });

  test('followers ignore data-theme-mode entirely', () => {
    // A follower has no preference of its own — it trusts the applied
    // data-theme and the hub's broadcasts. Reading the pin here would let a
    // panel resolve independently of its hub.
    stubOSPreference(true);
    document.documentElement.setAttribute('data-theme', 'high-contrast-light');
    document.documentElement.setAttribute('data-theme-mode', 'light');

    ThemeManager.initTheme({ role: 'follower' });

    expect(ThemeManager.getTheme()).toBe('high-contrast-light');
  });
});
