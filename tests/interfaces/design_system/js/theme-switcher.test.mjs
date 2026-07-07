/**
 * Unit tests for <osprey-theme-switcher>
 * (design_system/static/js/components/osprey-theme-switcher.js).
 *
 * Pure DOM/logic guard, happy-dom environment (configured globally):
 *   npx vitest run tests/interfaces/design_system/js/theme-switcher.test.mjs
 *
 * Covers:
 *   - the `customElements.get()` registration guard (a repeat module
 *     evaluation must not attempt to re-define the tag)
 *   - the rendered `#theme-toggle` / `#theme-icon-sun` / `#theme-icon-moon`
 *     markup theme-manager.js binds by id (contract unchanged)
 *   - the D15 embedded-hidden behavior: the shared
 *     `body.embedded osprey-theme-switcher { display: none }` rule is
 *     injected once by the component itself, and actually takes effect
 *
 * Imported via the absolute `/design-system/js/*` alias (see
 * vitest.config.js), matching alias-smoke.test.mjs's pattern -- this is
 * the same specifier interfaces mount the component with at runtime.
 */

import { test, expect, describe, afterEach, vi } from 'vitest';

import '/design-system/js/components/osprey-theme-switcher.js';

describe('<osprey-theme-switcher>', () => {
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
      // prevent that from ever being reached on a second evaluation.
      const defineSpy = vi.spyOn(customElements, 'define');
      vi.resetModules();
      await import('/design-system/js/components/osprey-theme-switcher.js');
      expect(defineSpy).not.toHaveBeenCalled();
      defineSpy.mockRestore();
    });
  });

  describe('rendered markup', () => {
    test('renders #theme-toggle / #theme-icon-sun / #theme-icon-moon on connect', () => {
      const el = document.createElement('osprey-theme-switcher');
      document.body.appendChild(el);

      const button = el.querySelector('#theme-toggle');
      const sunIcon = el.querySelector('#theme-icon-sun');
      const moonIcon = el.querySelector('#theme-icon-moon');

      expect(button).toBeTruthy();
      expect(button.tagName).toBe('BUTTON');
      expect(button.classList.contains('header-icon-btn')).toBe(true);
      expect(sunIcon).toBeTruthy();
      expect(sunIcon.tagName.toLowerCase()).toBe('svg');
      expect(moonIcon).toBeTruthy();
      expect(moonIcon.tagName.toLowerCase()).toBe('svg');
    });

    test('the rendered ids are resolvable from document (theme-manager.js looks them up via document.getElementById, not the host)', () => {
      const el = document.createElement('osprey-theme-switcher');
      document.body.appendChild(el);

      expect(document.getElementById('theme-toggle')).toBe(el.querySelector('#theme-toggle'));
      expect(document.getElementById('theme-icon-sun')).toBe(el.querySelector('#theme-icon-sun'));
      expect(document.getElementById('theme-icon-moon')).toBe(el.querySelector('#theme-icon-moon'));
    });

    test('re-connecting an already-rendered instance does not duplicate the button', () => {
      const el = document.createElement('osprey-theme-switcher');
      document.body.appendChild(el);
      document.body.removeChild(el);
      document.body.appendChild(el);

      expect(el.querySelectorAll('#theme-toggle').length).toBe(1);
    });

    test('two instances on the same page each render their own button', () => {
      const first = document.createElement('osprey-theme-switcher');
      const second = document.createElement('osprey-theme-switcher');
      document.body.appendChild(first);
      document.body.appendChild(second);

      expect(first.querySelector('#theme-toggle')).toBeTruthy();
      expect(second.querySelector('#theme-toggle')).toBeTruthy();
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
