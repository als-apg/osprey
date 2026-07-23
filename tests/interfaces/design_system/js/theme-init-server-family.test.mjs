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
      JSON.stringify({ family: 'osprey', mode: 'light' })
    );

    ThemeManager.initTheme({ role: 'hub' });

    expect(ThemeManager.getTheme()).toBe('light');
  });
});
