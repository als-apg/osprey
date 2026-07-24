/**
 * Unit tests for the web terminal's header display menu
 * (web_terminal/static/js/display-menu.js).
 *
 * Pure DOM/logic guard, happy-dom environment (configured globally):
 *   npx vitest run tests/interfaces/web_terminal/display-menu.test.mjs
 *
 * Covers:
 *   - graceful no-op when the dot/card markup is absent
 *   - open/close grammar: dot click toggles, outside click and Escape close
 *     (Escape returns focus to the dot), aria-expanded mirrors state
 *   - the Appearance row: data-active mirrors the resolved light/dark mode,
 *     picking the other side drives theme-manager's toggleTheme(), and
 *     re-picking the active side no-ops
 *   - the Theme row: one pill per THEMES family (deduped, declaration
 *     order), the active family's pill carries .active/aria-pressed, and a
 *     pill click drives setFamily()
 *   - a View-row segment click closes the card (the mode flip itself is
 *     app.js's initModeToggle(), out of scope here)
 *
 * Module-identity note (same as theme-switcher.test.mjs): this file imports
 * theme-manager.js and display-menu.js exactly once, at the top, and resets
 * *state* between tests (localStorage.clear() + a fresh initTheme()) so the
 * test's ThemeManager handle is the same singleton the menu's handlers are
 * bound to.
 */

import { test, expect, describe, beforeEach, afterEach } from 'vitest';

import * as ThemeManager from '/design-system/js/theme-manager.js';
import { THEMES } from '/design-system/js/tokens.js';

import { initDisplayMenu, themeFamilies, familyLabel } from '../../../src/osprey/interfaces/web_terminal/static/js/display-menu.js';

/** The index.html display-menu fixture (family pills are built by the module). */
const FIXTURE = `
  <div class="display-menu" id="display-menu">
    <button class="display-menu-dot" id="display-menu-btn" type="button"
            aria-haspopup="true" aria-expanded="false"></button>
    <div class="display-menu-card" id="display-menu-card" role="group" aria-label="Display preferences">
      <div class="display-menu-seg display-menu-appearance" id="appearance-toggle" role="group">
        <button class="display-seg-option" data-appearance="light" type="button" aria-pressed="false">Light</button>
        <button class="display-seg-option" data-appearance="dark" type="button" aria-pressed="false">Dark</button>
      </div>
      <div class="display-menu-seg" id="mode-toggle" role="group">
        <button class="display-seg-option mode-segment" data-mode="expert" type="button">Expert</button>
        <button class="display-seg-option mode-segment" data-mode="simple" type="button">Simple</button>
      </div>
      <div class="display-menu-families" id="family-picker" role="group"></div>
    </div>
  </div>
  <div id="outside"></div>
`;

/** @param {string} selector */
function qs(selector) {
  const el = document.querySelector(selector);
  if (!el) throw new Error(`fixture element not found: ${selector}`);
  return /** @type {HTMLElement} */ (el);
}

function mountAndInit() {
  document.body.innerHTML = FIXTURE;
  initDisplayMenu();
}

describe('display-menu', () => {
  beforeEach(() => {
    window.localStorage.clear();
    document.documentElement.removeAttribute('data-theme');
    document.body.innerHTML = '';
    window.history.replaceState({}, '', '/');
    // Deterministic start: dark mode of the default family.
    ThemeManager.initTheme();
    ThemeManager.setTheme('dark');
  });

  afterEach(() => {
    document.body.innerHTML = '';
  });

  describe('init', () => {
    test('no-ops without the dot/card markup', () => {
      document.body.innerHTML = '<div id="something-else"></div>';
      expect(() => initDisplayMenu()).not.toThrow();
    });
  });

  describe('open/close grammar', () => {
    test('dot click opens the card and mirrors aria-expanded; second click closes', () => {
      mountAndInit();
      const dot = qs('#display-menu-btn');
      const card = qs('#display-menu-card');

      dot.click();
      expect(card.classList.contains('open')).toBe(true);
      expect(dot.getAttribute('aria-expanded')).toBe('true');

      dot.click();
      expect(card.classList.contains('open')).toBe(false);
      expect(dot.getAttribute('aria-expanded')).toBe('false');
    });

    test('an outside click closes the card; a click inside does not', () => {
      mountAndInit();
      qs('#display-menu-btn').click();

      qs('#appearance-toggle').click();
      expect(qs('#display-menu-card').classList.contains('open')).toBe(true);

      qs('#outside').click();
      expect(qs('#display-menu-card').classList.contains('open')).toBe(false);
      expect(qs('#display-menu-btn').getAttribute('aria-expanded')).toBe('false');
    });

    test('Escape closes the card and returns focus to the dot', () => {
      mountAndInit();
      qs('#display-menu-btn').click();

      document.dispatchEvent(new KeyboardEvent('keydown', { key: 'Escape' }));
      expect(qs('#display-menu-card').classList.contains('open')).toBe(false);
      expect(document.activeElement).toBe(qs('#display-menu-btn'));
    });

    test('a View-row segment click closes the card', () => {
      mountAndInit();
      qs('#display-menu-btn').click();

      qs('#mode-toggle .mode-segment[data-mode="simple"]').click();
      expect(qs('#display-menu-card').classList.contains('open')).toBe(false);
    });
  });

  describe('Appearance row', () => {
    test('data-active and aria-pressed mirror the resolved mode', () => {
      mountAndInit();
      const seg = qs('#appearance-toggle');
      expect(seg.getAttribute('data-active')).toBe('dark');
      expect(qs('[data-appearance="dark"]').getAttribute('aria-pressed')).toBe('true');
      expect(qs('[data-appearance="light"]').getAttribute('aria-pressed')).toBe('false');
    });

    test('picking the other side toggles the theme within the family', () => {
      mountAndInit();
      const before = ThemeManager.getTheme();

      qs('[data-appearance="light"]').click();
      const after = ThemeManager.getTheme();
      expect(after).not.toBe(before);
      expect(THEMES.find((t) => t.id === after)?.mode).toBe('light');
      expect(qs('#appearance-toggle').getAttribute('data-active')).toBe('light');
    });

    test('re-picking the already-active side no-ops', () => {
      mountAndInit();
      const before = ThemeManager.getTheme();

      qs('[data-appearance="dark"]').click();
      expect(ThemeManager.getTheme()).toBe(before);
    });
  });

  describe('Theme row (family pills)', () => {
    test('renders one pill per family, deduped, in THEMES declaration order', () => {
      mountAndInit();
      const pills = Array.from(document.querySelectorAll('.display-family-pill'));
      const expected = themeFamilies();
      expect(pills.map((p) => /** @type {HTMLElement} */ (p).dataset.family)).toEqual(
        expected.map((f) => f.id)
      );
      expect(pills.map((p) => p.textContent)).toEqual(expected.map((f) => f.label));
    });

    test('the active family pill carries .active and aria-pressed', () => {
      mountAndInit();
      const family = ThemeManager.getFamily();
      const active = qs(`.display-family-pill[data-family="${family}"]`);
      expect(active.classList.contains('active')).toBe(true);
      expect(active.getAttribute('aria-pressed')).toBe('true');
    });

    test('a pill click drives setFamily() and moves the active marker', () => {
      mountAndInit();
      const current = ThemeManager.getFamily();
      const other = themeFamilies().find((f) => f.id !== current);
      if (!other) throw new Error('THEMES declares only one family; test needs two');

      qs(`.display-family-pill[data-family="${other.id}"]`).click();
      expect(ThemeManager.getFamily()).toBe(other.id);
      const pill = qs(`.display-family-pill[data-family="${other.id}"]`);
      expect(pill.classList.contains('active')).toBe(true);
      expect(qs(`.display-family-pill[data-family="${current}"]`).classList.contains('active')).toBe(false);
    });
  });

  describe('familyLabel', () => {
    test('title-cases hyphen-separated family ids', () => {
      expect(familyLabel('osprey')).toBe('Osprey');
      expect(familyLabel('high-contrast')).toBe('High Contrast');
    });
  });
});
