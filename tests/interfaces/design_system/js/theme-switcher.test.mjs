/**
 * Unit tests for <osprey-theme-switcher>
 * (design_system/static/js/components/osprey-theme-switcher.js).
 *
 * Pure DOM/logic guard, happy-dom environment (configured globally):
 *   npx vitest run tests/interfaces/design_system/js/theme-switcher.test.mjs
 *
 * Covers (Task 1.9 family-picker rewrite):
 *   - the `customElements.get()` registration guard (a repeat module
 *     evaluation must not attempt to re-define the tag)
 *   - the rendered markup: a family `<select>` listing every family from
 *     `THEMES` (deduped) plus an in-family mode toggle `<button>` -- no
 *     dependency on any external `#theme-toggle`-shaped page markup
 *   - the element is fully self-contained: it wires its own handlers
 *     straight to theme-manager.js's real `setFamily()`/`toggleTheme()`
 *     and stays in sync via `subscribe()`, with no page-level init beyond
 *     mounting the tag
 *   - light DOM (no shadow root)
 *   - the D15 embedded-hidden behavior: the shared
 *     `body.embedded osprey-theme-switcher { display: none }` rule is
 *     injected once by the component itself, and actually takes effect
 *
 * Imported via the absolute `/design-system/js/*` alias (see
 * vitest.config.js), matching alias-smoke.test.mjs's pattern -- this is
 * the same specifier interfaces mount the component with at runtime.
 *
 * Module-identity note (unlike theme-settheme.test.mjs): `customElements`
 * is a real, process-global registry that Vitest's module registry reset
 * (`vi.resetModules()`) cannot touch. `OspreyThemeSwitcher` only ever
 * really gets *defined* once per process -- the registration guard test
 * below exists precisely because a second module-evaluation cycle must
 * not attempt (and fail) to redefine the tag. That means the one
 * `OspreyThemeSwitcher` class that will ever back every element created
 * in this file has its `setFamily`/`getFamily`/`getTheme`/`toggleTheme`/
 * `subscribe` imports bound to whichever theme-manager.js module instance
 * was live the first time this file's top-level imports ran. Re-importing
 * theme-manager.js per test via `vi.resetModules()` (as
 * theme-settheme.test.mjs does) would silently decouple the test's own
 * `ThemeManager` handle from the instance every switcher element actually
 * talks to. So this file imports both modules exactly once, at the top,
 * and resets *state* between tests instead: `localStorage.clear()` + a
 * fresh `initTheme()` call (which fully recomputes preference/current-id
 * from the now-empty storage and a reset URL) gives each test a clean,
 * deterministic starting point against the one shared singleton.
 */

import { test, expect, describe, beforeEach, afterEach, vi } from 'vitest';

import { qs } from '../../_support/dom.mjs';

import * as ThemeManager from '/design-system/js/theme-manager.js';
import '/design-system/js/components/osprey-theme-switcher.js';

describe('<osprey-theme-switcher>', () => {
  beforeEach(() => {
    window.localStorage.clear();
    document.documentElement.removeAttribute('data-theme');
    document.body.innerHTML = '';
    document.body.className = '';
    window.history.replaceState({}, '', '/');
  });

  afterEach(() => {
    document.body.innerHTML = '';
    document.body.className = '';
  });

  describe('registration', () => {
    test('registers the osprey-theme-switcher custom element', () => {
      expect(customElements.get('osprey-theme-switcher')).toBeDefined();
    });

    test('a repeat module evaluation does not attempt to re-define the tag', async () => {
      // customElements.define() throws on a second registration of the
      // same tag name; the module's `!customElements.get(...)` guard must
      // prevent that from ever being reached on a second evaluation. This
      // is the one place in this file a module-registry reset is safe: it
      // only asserts define() isn't called again, and never touches a
      // switcher element or theme-manager.js state.
      const defineSpy = vi.spyOn(customElements, 'define');
      vi.resetModules();
      await import('/design-system/js/components/osprey-theme-switcher.js');
      expect(defineSpy).not.toHaveBeenCalled();
      defineSpy.mockRestore();
    });
  });

  describe('rendered markup', () => {
    test('renders a family <select> listing every family from THEMES, deduped', () => {
      const el = document.createElement('osprey-theme-switcher');
      document.body.appendChild(el);

      const select = qs(el, '.theme-switcher-family', HTMLSelectElement);
      const values = Array.from(select.options).map((o) => o.value);

      expect(values).toEqual(['osprey', 'high-contrast']);
    });

    test('derives a human family label by title-casing the family id', () => {
      const el = document.createElement('osprey-theme-switcher');
      document.body.appendChild(el);

      const select = qs(el, '.theme-switcher-family', HTMLSelectElement);
      const labels = Array.from(select.options).map((o) => o.textContent);

      expect(labels).toEqual(['Osprey', 'High Contrast']);
    });

    test('renders an in-family mode toggle button', () => {
      const el = document.createElement('osprey-theme-switcher');
      document.body.appendChild(el);

      const button = qs(el, '.theme-switcher-mode', HTMLButtonElement);
      expect(button.tagName).toBe('BUTTON');
      expect(qs(el, '.theme-switcher-icon-sun').tagName.toLowerCase()).toBe('svg');
      expect(qs(el, '.theme-switcher-icon-moon').tagName.toLowerCase()).toBe('svg');
    });

    test('no external #theme-toggle-shaped markup is required or produced', () => {
      const el = document.createElement('osprey-theme-switcher');
      document.body.appendChild(el);

      expect(document.getElementById('theme-toggle')).toBeNull();
      expect(document.getElementById('theme-icon-sun')).toBeNull();
      expect(document.getElementById('theme-icon-moon')).toBeNull();
    });

    test('re-connecting an already-rendered instance does not duplicate the markup', () => {
      const el = document.createElement('osprey-theme-switcher');
      document.body.appendChild(el);
      document.body.removeChild(el);
      document.body.appendChild(el);

      expect(el.querySelectorAll('.theme-switcher-family').length).toBe(1);
      expect(el.querySelectorAll('.theme-switcher-mode').length).toBe(1);
    });

    test('two instances on the same page each own independent, non-colliding controls', () => {
      const first = document.createElement('osprey-theme-switcher');
      const second = document.createElement('osprey-theme-switcher');
      document.body.appendChild(first);
      document.body.appendChild(second);

      const firstSelect = qs(first, '.theme-switcher-family', HTMLSelectElement);
      const secondSelect = qs(second, '.theme-switcher-family', HTMLSelectElement);
      expect(firstSelect).not.toBe(secondSelect);
    });

    test('is light DOM (no shadow root), so tokens.css and theme-manager.js still reach it', () => {
      const el = document.createElement('osprey-theme-switcher');
      document.body.appendChild(el);

      expect(el.shadowRoot).toBeNull();
    });
  });

  describe('self-contained wiring against the real theme-manager', () => {
    test('selecting a family in the picker calls setFamily() and re-themes the page live', () => {
      ThemeManager.initTheme({ role: 'hub' });
      const el = document.createElement('osprey-theme-switcher');
      document.body.appendChild(el);

      const select = qs(el, '.theme-switcher-family', HTMLSelectElement);
      select.value = 'high-contrast';
      select.dispatchEvent(new Event('change', { bubbles: true }));

      expect(ThemeManager.getFamily()).toBe('high-contrast');
      const appliedId = /** @type {string} */ (ThemeManager.getTheme());
      expect(document.documentElement.getAttribute('data-theme')).toBe(appliedId);
      expect(appliedId.startsWith('high-contrast')).toBe(true);
    });

    test('picking a family preserves the current mode preference (setFamily, not setTheme)', () => {
      ThemeManager.initTheme({ role: 'hub' });
      ThemeManager.setTheme('light'); // explicit light mode, family osprey
      const el = document.createElement('osprey-theme-switcher');
      document.body.appendChild(el);

      const select = qs(el, '.theme-switcher-family', HTMLSelectElement);
      select.value = 'high-contrast';
      select.dispatchEvent(new Event('change', { bubbles: true }));

      expect(ThemeManager.getTheme()).toBe('high-contrast-light');
    });

    test('clicking the mode button calls toggleTheme(), flipping mode WITHIN the active family', () => {
      ThemeManager.initTheme({ role: 'hub' });
      ThemeManager.setTheme('high-contrast-dark');
      const el = document.createElement('osprey-theme-switcher');
      document.body.appendChild(el);

      const button = qs(el, '.theme-switcher-mode', HTMLButtonElement);
      button.click();

      expect(ThemeManager.getFamily()).toBe('high-contrast');
      expect(ThemeManager.getTheme()).toBe('high-contrast-light');
    });

    test('reflects an externally-applied theme change via subscribe() (family select + mode button sync)', () => {
      ThemeManager.initTheme({ role: 'hub' });
      const el = document.createElement('osprey-theme-switcher');
      document.body.appendChild(el);

      // Changed through the real API directly, not through this element's
      // own controls -- proves the sync comes from subscribe(), not from
      // the click/change handlers.
      ThemeManager.setTheme('high-contrast-light');

      const select = qs(el, '.theme-switcher-family', HTMLSelectElement);
      const button = qs(el, '.theme-switcher-mode', HTMLButtonElement);
      expect(select.value).toBe('high-contrast');
      expect(button.getAttribute('aria-pressed')).toBe('true');
      expect(button.getAttribute('aria-label')).toBe('Switch to dark theme');
    });

    test('reflects the already-applied theme at connect time (element mounted after initTheme())', () => {
      ThemeManager.initTheme({ role: 'hub' });
      ThemeManager.setTheme('high-contrast-dark');

      const el = document.createElement('osprey-theme-switcher');
      document.body.appendChild(el);

      const select = qs(el, '.theme-switcher-family', HTMLSelectElement);
      expect(select.value).toBe('high-contrast');
    });

    test('reflects the theme once initTheme() resolves, when mounted before initTheme() runs', () => {
      const el = document.createElement('osprey-theme-switcher');
      document.body.appendChild(el);

      // Mirrors real page order: the tag upgrades (and connects) as soon
      // as the parser reaches it, before a later init script calls
      // initTheme() -- the subscribe() registered at connect must still
      // catch initTheme()'s first apply.
      ThemeManager.initTheme({ role: 'hub' });
      ThemeManager.setTheme('high-contrast-dark');

      const select = qs(el, '.theme-switcher-family', HTMLSelectElement);
      expect(select.value).toBe('high-contrast');
    });

    test('does not keep reacting to theme changes after being disconnected', () => {
      ThemeManager.initTheme({ role: 'hub' });
      const el = document.createElement('osprey-theme-switcher');
      document.body.appendChild(el);
      document.body.removeChild(el);

      // Should not throw even though the element is no longer connected.
      expect(() => ThemeManager.setTheme('high-contrast-dark')).not.toThrow();
    });
  });

  describe('embedded-hidden CSS rule (D15)', () => {
    test('injects the shared body.embedded osprey-theme-switcher rule', () => {
      document.body.appendChild(document.createElement('osprey-theme-switcher'));

      const styles = Array.from(document.querySelectorAll('style'))
        .map((s) => s.textContent)
        .join('\n');
      expect(styles).toContain('body.embedded osprey-theme-switcher');
      expect(styles).toContain('display: none');
    });

    test('a second instance does not duplicate the injected style tag', () => {
      document.body.appendChild(document.createElement('osprey-theme-switcher'));
      document.body.appendChild(document.createElement('osprey-theme-switcher'));

      const matching = Array.from(document.querySelectorAll('style')).filter((s) =>
        (s.textContent || '').includes('osprey-theme-switcher')
      );
      expect(matching.length).toBe(1);
    });

    test('is hidden (computed display:none) when body carries the embedded class', () => {
      document.body.classList.add('embedded');
      const el = document.createElement('osprey-theme-switcher');
      document.body.appendChild(el);

      expect(getComputedStyle(el).display).toBe('none');
    });

    test('is not hidden when body has no embedded class', () => {
      const el = document.createElement('osprey-theme-switcher');
      document.body.appendChild(el);

      expect(getComputedStyle(el).display).not.toBe('none');
    });
  });
});
