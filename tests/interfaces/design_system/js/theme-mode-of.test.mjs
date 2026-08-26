/**
 * Unit tests for theme-manager.js's `modeOf()` — the fleet's one derivation of
 * a concrete theme id's light/dark side.
 *
 * Pure logic guard, happy-dom environment (configured globally):
 *   npx vitest run tests/interfaces/design_system/js/theme-mode-of.test.mjs
 *
 * `<osprey-theme-switcher>` and `<osprey-display-menu>` each used to carry a
 * private copy of this helper (reading `THEMES` themselves); both now import
 * it, and the identity tests at the bottom keep it that way.
 */

import { test, expect, describe } from 'vitest';

import * as ThemeManager from '/design-system/js/theme-manager.js';
import { THEMES } from '/design-system/js/tokens.js';

describe('modeOf', () => {
  test('returns the declared mode of every theme in the registry', () => {
    for (const theme of THEMES) {
      expect(ThemeManager.modeOf(theme.id)).toBe(theme.mode);
    }
  });

  test('covers both modes (the registry is not single-sided)', () => {
    const modes = new Set(THEMES.map((t) => ThemeManager.modeOf(t.id)));
    expect(modes).toEqual(new Set(['light', 'dark']));
  });

  test('is null for an unknown id and for null itself', () => {
    expect(ThemeManager.modeOf('no-such-theme')).toBeNull();
    expect(ThemeManager.modeOf(null)).toBeNull();
  });

  test('reflects getTheme() once a theme is applied', () => {
    ThemeManager.initTheme({ role: 'hub' });
    ThemeManager.setTheme('dark');
    expect(ThemeManager.modeOf(ThemeManager.getTheme())).toBe('dark');
    ThemeManager.toggleTheme();
    expect(ThemeManager.modeOf(ThemeManager.getTheme())).toBe('light');
  });
});
