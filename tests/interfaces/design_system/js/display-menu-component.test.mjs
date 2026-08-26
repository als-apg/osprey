/**
 * Unit tests for <osprey-display-menu>
 * (design_system/static/js/components/osprey-display-menu.js).
 *
 * Pure DOM/logic guard, happy-dom environment (configured globally):
 *   npx vitest run tests/interfaces/design_system/js/display-menu-component.test.mjs
 *
 * Named for the element rather than the file it lives beside, because the
 * hub's own (still-separate) popover already owns the `display-menu.test.mjs`
 * basename under tests/interfaces/web_terminal/.
 *
 * Covers:
 *   - the `customElements.get()` registration guard
 *   - the three FIXED rows (Appearance / View / Theme) render, with the
 *     family pills coming from theme-manager.js's `themeFamilies()`
 *   - the Appearance row drives theme-manager.js's `toggleTheme()`
 *   - the View row's three-step mechanism: persist under
 *     `localStorage['osprey-ui-mode']`, strip a leftover `?mode=`, and post
 *     `osprey-mode-change` to its OWN window
 *   - the Theme pills call `setFamily()` and reflect `subscribe()` notifications
 *   - the popover grammar: trigger toggles, outside click and Escape dismiss
 *   - the optional Settings entry — present only with `settings-drawer`, and
 *     carrying `<osprey-drawer>`'s delegated `data-drawer` trigger contract
 *   - the D15 embedded-hidden rule, injected exactly once across instances
 *
 * Module-identity note (same as theme-switcher.test.mjs): `customElements` is
 * a process-global registry Vitest's module-registry reset cannot touch, so
 * the one `OspreyDisplayMenu` class that backs every element created here has
 * its theme-manager.js / frame-params.js imports bound to whichever module
 * instances were live the first time this file's top-level imports ran.
 * Re-importing them per test would silently decouple the test's handles from
 * the instances the element actually talks to — so this file imports each
 * module exactly once, at the top, and resets *state* between tests instead.
 */

import { test, expect, describe, beforeEach, afterEach, vi } from 'vitest';

import { qs } from '../../_support/dom.mjs';

import * as ThemeManager from '/design-system/js/theme-manager.js';
import '/design-system/js/components/osprey-display-menu.js';

const MODE_KEY = 'osprey-ui-mode';

/**
 * Mount a fresh menu, optionally naming a settings drawer. The attribute is
 * set before connect because it is read once at render (matching
 * `<osprey-drawer>`'s once-at-connect `resizable` precedent).
 * @param {string} [settingsDrawer]
 * @returns {HTMLElement}
 */
function mount(settingsDrawer) {
  const el = document.createElement('osprey-display-menu');
  if (settingsDrawer !== undefined) el.setAttribute('settings-drawer', settingsDrawer);
  document.body.appendChild(el);
  return /** @type {HTMLElement} */ (el);
}

describe('<osprey-display-menu>', () => {
  beforeEach(() => {
    window.localStorage.clear();
    document.documentElement.removeAttribute('data-theme');
    document.documentElement.removeAttribute('data-ui-mode');
    document.body.innerHTML = '';
    document.body.className = '';
    window.history.replaceState({}, '', '/');
    ThemeManager.initTheme({ role: 'hub' });
  });

  afterEach(() => {
    vi.restoreAllMocks();
    document.body.innerHTML = '';
    document.body.className = '';
  });

  describe('registration', () => {
    test('registers the osprey-display-menu custom element', () => {
      expect(customElements.get('osprey-display-menu')).toBeDefined();
    });

    test('a repeat module evaluation does not attempt to re-define the tag', async () => {
      // customElements.define() throws on a second registration of the same
      // tag; the module's `!customElements.get(...)` guard must prevent that
      // from ever being reached. Safe to reset the module registry here: this
      // test asserts only that define() isn't called again.
      const defineSpy = vi.spyOn(customElements, 'define');
      vi.resetModules();
      await import('/design-system/js/components/osprey-display-menu.js');
      expect(defineSpy).not.toHaveBeenCalled();
      defineSpy.mockRestore();
    });

    test('is light DOM (no shadow root), so tokens.css still reaches it', () => {
      expect(mount().shadowRoot).toBeNull();
    });
  });

  describe('fixed rows', () => {
    test('renders the three headings in order: Appearance, View, Theme', () => {
      const el = mount();
      const headings = Array.from(el.querySelectorAll('.display-menu-heading')).map(
        (h) => h.textContent
      );
      expect(headings).toEqual(['Appearance', 'View', 'Theme']);
    });

    test('the Appearance row offers exactly light and dark', () => {
      const el = mount();
      const options = Array.from(
        el.querySelectorAll('.display-menu-appearance .display-seg-option')
      ).map((o) => (o instanceof HTMLElement ? o.dataset.appearance : null));
      expect(options).toEqual(['light', 'dark']);
    });

    test('the View row offers exactly expert and simple', () => {
      const el = mount();
      const options = Array.from(
        el.querySelectorAll('.display-menu-view .display-seg-option')
      ).map((o) => (o instanceof HTMLElement ? o.dataset.mode : null));
      expect(options).toEqual(['expert', 'simple']);
    });

    test('the Theme row renders one pill per family, from themeFamilies()', () => {
      const el = mount();
      const pills = Array.from(el.querySelectorAll('.display-family-pill'));
      expect(pills.map((p) => (p instanceof HTMLElement ? p.dataset.family : null))).toEqual(
        ThemeManager.themeFamilies().map((f) => f.id)
      );
      expect(pills.map((p) => p.textContent)).toEqual(
        ThemeManager.themeFamilies().map((f) => f.label)
      );
    });

    test('two instances on one page own independent, non-colliding controls', () => {
      const first = mount();
      const second = mount();
      expect(qs(first, '.display-menu-card')).not.toBe(qs(second, '.display-menu-card'));
      // No ids anywhere in the rendered markup — that is what makes the above
      // true (the hub's id-bound copy could only ever have one instance).
      expect(first.querySelectorAll('[id]').length).toBe(0);
    });

    test('re-connecting an already-rendered instance does not duplicate the markup', () => {
      const el = mount();
      document.body.removeChild(el);
      document.body.appendChild(el);
      expect(el.querySelectorAll('.display-menu-card').length).toBe(1);
      expect(el.querySelectorAll('.display-family-pill').length).toBe(
        ThemeManager.themeFamilies().length
      );
    });
  });

  describe('Appearance row', () => {
    test('picking the opposite appearance calls theme-manager toggleTheme()', () => {
      const toggleSpy = vi.spyOn(ThemeManager, 'toggleTheme');
      ThemeManager.setTheme('high-contrast-dark');
      const el = mount();

      qs(el, '.display-menu-appearance .display-seg-option[data-appearance="light"]').click();

      expect(toggleSpy).toHaveBeenCalledTimes(1);
      // ...and the flip stays WITHIN the active family (setFamily semantics).
      expect(ThemeManager.getTheme()).toBe('high-contrast-light');
    });

    test('re-picking the already-active appearance is a no-op', () => {
      const toggleSpy = vi.spyOn(ThemeManager, 'toggleTheme');
      ThemeManager.setTheme('high-contrast-dark');
      const el = mount();

      qs(el, '.display-menu-appearance .display-seg-option[data-appearance="dark"]').click();

      expect(toggleSpy).not.toHaveBeenCalled();
      expect(ThemeManager.getTheme()).toBe('high-contrast-dark');
    });

    test('reflects the active appearance, including an externally-applied change', () => {
      const el = mount();
      ThemeManager.setTheme('main-light');

      const seg = qs(el, '.display-menu-appearance');
      expect(seg.getAttribute('data-active')).toBe('light');
      expect(
        qs(seg, '.display-seg-option[data-appearance="light"]').getAttribute('aria-pressed')
      ).toBe('true');
      expect(
        qs(seg, '.display-seg-option[data-appearance="dark"]').getAttribute('aria-pressed')
      ).toBe('false');
    });

    test('a pick leaves the card open', () => {
      const el = mount();
      qs(el, '.display-menu-trigger').click();
      qs(el, '.display-menu-appearance .display-seg-option[data-appearance="light"]').click();
      expect(qs(el, '.display-menu-card').classList.contains('open')).toBe(true);
    });
  });

  describe('View row', () => {
    test('persists the pick under localStorage[osprey-ui-mode]', () => {
      const el = mount();
      qs(el, '.display-menu-view .display-seg-option[data-mode="simple"]').click();
      expect(window.localStorage.getItem(MODE_KEY)).toBe('simple');

      qs(el, '.display-menu-view .display-seg-option[data-mode="expert"]').click();
      expect(window.localStorage.getItem(MODE_KEY)).toBe('expert');
    });

    test('posts osprey-mode-change to its OWN window, at the page origin', () => {
      const postSpy = vi.spyOn(window, 'postMessage');
      const el = mount();

      qs(el, '.display-menu-view .display-seg-option[data-mode="simple"]').click();

      expect(postSpy).toHaveBeenCalledWith(
        { type: 'osprey-mode-change', mode: 'simple' },
        window.location.origin
      );
    });

    test('strips a leftover one-shot ?mode= so it cannot out-rank the pick on reload', () => {
      window.history.replaceState({}, '', '/?mode=expert&embedded=true');
      const el = mount();

      qs(el, '.display-menu-view .display-seg-option[data-mode="simple"]').click();

      const params = new URLSearchParams(window.location.search);
      expect(params.has('mode')).toBe(false);
      expect(params.get('embedded')).toBe('true');
    });

    test('does not stamp data-ui-mode itself — the page listener owns that', () => {
      // The stamp path lives in each page's app.js 'message' listener, which
      // this suite never imports — so nothing here would stamp even without
      // the mock. Suppressing delivery anyway pins the claim to the element
      // itself: the null assertion stays meaningful even if another suite
      // sharing this window ever registers such a listener.
      vi.spyOn(window, 'postMessage').mockImplementation(() => {});
      const el = mount();

      qs(el, '.display-menu-view .display-seg-option[data-mode="simple"]').click();

      expect(document.documentElement.getAttribute('data-ui-mode')).toBeNull();
    });

    test('reflects the pick immediately, without waiting for the broadcast', () => {
      vi.spyOn(window, 'postMessage').mockImplementation(() => {});
      const el = mount();

      qs(el, '.display-menu-view .display-seg-option[data-mode="simple"]').click();

      const seg = qs(el, '.display-menu-view');
      expect(seg.getAttribute('data-active')).toBe('simple');
      expect(qs(seg, '.display-seg-option[data-mode="simple"]').getAttribute('aria-pressed')).toBe(
        'true'
      );
      expect(qs(seg, '.display-seg-option[data-mode="expert"]').getAttribute('aria-pressed')).toBe(
        'false'
      );
    });

    test('reflects the mode mode-boot.js already stamped, at connect time', () => {
      document.documentElement.setAttribute('data-ui-mode', 'simple');
      const el = mount();
      expect(qs(el, '.display-menu-view').getAttribute('data-active')).toBe('simple');
    });
  });

  describe('Theme row', () => {
    test('clicking a family pill calls setFamily(), preserving the light/dark mode', () => {
      ThemeManager.setTheme('main-light');
      const el = mount();

      qs(el, '.display-family-pill[data-family="high-contrast"]').click();

      expect(ThemeManager.getFamily()).toBe('high-contrast');
      expect(ThemeManager.getTheme()).toBe('high-contrast-light');
    });

    test('marks the active family pill, including on an external change', () => {
      const el = mount();
      ThemeManager.setFamily('high-contrast');

      const active = qs(el, '.display-family-pill[data-family="high-contrast"]');
      expect(active.classList.contains('active')).toBe(true);
      expect(active.getAttribute('aria-pressed')).toBe('true');
      expect(
        qs(el, '.display-family-pill[data-family="main"]').getAttribute('aria-pressed')
      ).toBe('false');
    });
  });

  describe('popover behavior', () => {
    test('the trigger opens and closes the card, mirroring aria-expanded', () => {
      const el = mount();
      const trigger = qs(el, '.display-menu-trigger');
      const card = qs(el, '.display-menu-card');

      expect(card.classList.contains('open')).toBe(false);
      expect(trigger.getAttribute('aria-expanded')).toBe('false');

      trigger.click();
      expect(card.classList.contains('open')).toBe(true);
      expect(trigger.getAttribute('aria-expanded')).toBe('true');

      trigger.click();
      expect(card.classList.contains('open')).toBe(false);
      expect(trigger.getAttribute('aria-expanded')).toBe('false');
    });

    test('an outside click dismisses the card', () => {
      const el = mount();
      const outside = document.createElement('button');
      document.body.appendChild(outside);

      qs(el, '.display-menu-trigger').click();
      outside.click();

      expect(qs(el, '.display-menu-card').classList.contains('open')).toBe(false);
    });

    test('a click inside the card does not dismiss it', () => {
      const el = mount();
      qs(el, '.display-menu-trigger').click();
      qs(el, '.display-menu-heading').click();
      expect(qs(el, '.display-menu-card').classList.contains('open')).toBe(true);
    });

    test('Escape dismisses the card and returns focus to the trigger', () => {
      const el = mount();
      const trigger = qs(el, '.display-menu-trigger');
      trigger.click();

      document.dispatchEvent(new KeyboardEvent('keydown', { key: 'Escape', bubbles: true }));

      expect(qs(el, '.display-menu-card').classList.contains('open')).toBe(false);
      expect(document.activeElement).toBe(trigger);
    });

    test('opening one instance dismisses another that was already open', () => {
      const first = mount();
      const second = mount();

      qs(first, '.display-menu-trigger').click();
      qs(second, '.display-menu-trigger').click();

      expect(qs(first, '.display-menu-card').classList.contains('open')).toBe(false);
      expect(qs(second, '.display-menu-card').classList.contains('open')).toBe(true);
    });

    test('disconnecting an open menu stops it listening for outside clicks', () => {
      const el = mount();
      qs(el, '.display-menu-trigger').click();
      document.body.removeChild(el);

      const outside = document.createElement('button');
      document.body.appendChild(outside);
      expect(() => outside.click()).not.toThrow();
      expect(qs(el, '.display-menu-card').classList.contains('open')).toBe(false);
    });
  });

  describe('optional Settings entry', () => {
    test('is absent without the settings-drawer attribute', () => {
      const el = mount();
      expect(el.querySelector('.display-menu-settings')).toBeNull();
      expect(el.querySelector('.display-menu-actions')).toBeNull();
    });

    test('appears with settings-drawer, carrying osprey-drawer\'s data-drawer contract', () => {
      const el = mount('settings-drawer');
      const button = qs(el, '.display-menu-settings', HTMLButtonElement);

      expect(button.dataset.drawer).toBe('settings-drawer');
      expect(button.textContent).toContain('Settings');
    });

    test('names whatever drawer id the attribute carries', () => {
      const el = mount('ariel-config-drawer');
      expect(qs(el, '.display-menu-settings').dataset.drawer).toBe('ariel-config-drawer');
    });

    test('is the one row that closes the card, since it moves to another surface', () => {
      const el = mount('settings-drawer');
      qs(el, '.display-menu-trigger').click();
      qs(el, '.display-menu-settings').click();
      expect(qs(el, '.display-menu-card').classList.contains('open')).toBe(false);
    });

    test('hands focus to the persistent trigger so the drawer can restore it there', () => {
      // osprey-drawer records document.activeElement (bubble-phase, after this
      // component's own click listener) as its return-focus target; the entry
      // itself is inside the now-hidden card, so the trigger must hold focus.
      const el = mount('settings-drawer');
      qs(el, '.display-menu-trigger').click();
      qs(el, '.display-menu-settings').click();
      expect(document.activeElement).toBe(qs(el, '.display-menu-trigger'));
    });
  });

  describe('projected action slot', () => {
    /**
     * Mount a menu whose markup carries page-owned action children — the
     * hub's Settings (data-drawer-trigger, NOT data-drawer) and Log out.
     * @param {string} [settingsDrawer]
     */
    function mountWithActions(settingsDrawer) {
      const el = document.createElement('osprey-display-menu');
      if (settingsDrawer !== undefined) el.setAttribute('settings-drawer', settingsDrawer);
      el.innerHTML = `
        <button class="display-menu-settings" id="page-settings" type="button"
                data-drawer-trigger="settings-drawer">Settings</button>
        <button class="display-menu-logout" id="page-logout" type="button">Log out</button>
      `;
      document.body.appendChild(el);
      return /** @type {HTMLElement} */ (el);
    }

    test('moves the tag\'s children into the action row, last in the card, in order', () => {
      const el = mountWithActions();
      const card = qs(el, '.display-menu-card');
      const actions = qs(el, '.display-menu-actions');

      expect(card.lastElementChild).toBe(actions);
      expect(Array.from(actions.children).map((c) => c.id)).toEqual(['page-settings', 'page-logout']);
      // Nothing leaks outside the card: the trigger and the card are the only
      // direct children left.
      expect(Array.from(el.children).map((c) => c.className)).toEqual([
        'display-menu-trigger',
        'display-menu-card',
      ]);
    });

    test('the built-in Settings entry leads the projected children when both are present', () => {
      const el = mountWithActions('settings-drawer');
      const actions = qs(el, '.display-menu-actions');
      const first = /** @type {HTMLElement} */ (actions.firstElementChild);

      expect(first.dataset.drawer).toBe('settings-drawer');
      expect(first.id).toBe('');
      expect(actions.lastElementChild?.id).toBe('page-logout');
    });

    test('projected children keep their own attributes and get no component contract', () => {
      const el = mountWithActions();
      const settings = qs(el, '#page-settings');

      expect(settings.dataset.drawerTrigger).toBe('settings-drawer');
      expect(settings.hasAttribute('data-drawer')).toBe(false);
    });

    test('a projected child\'s click leaves the card open (the page decides)', () => {
      // The hub's Log out must NOT be closed away: its aria-busy state has to
      // stay visible while the logout POST is in flight. The hub closes on its
      // own Settings click by calling closeMenu() itself.
      const el = mountWithActions();
      qs(el, '.display-menu-trigger').click();

      qs(el, '#page-logout').click();
      expect(qs(el, '.display-menu-card').classList.contains('open')).toBe(true);

      qs(el, '#page-settings').click();
      expect(qs(el, '.display-menu-card').classList.contains('open')).toBe(true);

      /** @type {any} */ (el).closeMenu();
      expect(qs(el, '.display-menu-card').classList.contains('open')).toBe(false);
    });

    test('survives disconnect and reconnect without re-projecting', () => {
      const el = mountWithActions();
      el.remove();
      document.body.appendChild(el);

      expect(el.querySelectorAll('#page-logout')).toHaveLength(1);
      expect(qs(el, '.display-menu-actions').lastElementChild?.id).toBe('page-logout');
    });
  });

  describe('embedded-hidden CSS rule (D15)', () => {
    test('injects the shared body.embedded osprey-display-menu rule itself', () => {
      mount();
      const styles = Array.from(document.querySelectorAll('style'))
        .map((s) => s.textContent)
        .join('\n');
      expect(styles).toContain('body.embedded osprey-display-menu');
      expect(styles).toContain('display: none');
    });

    test('a second instance does not duplicate the injected style tag', () => {
      mount();
      mount();
      const matching = Array.from(document.querySelectorAll('style')).filter((s) =>
        (s.textContent || '').includes('body.embedded osprey-display-menu')
      );
      expect(matching.length).toBe(1);
    });

    test('is hidden (computed display:none) when body carries the embedded class', () => {
      document.body.classList.add('embedded');
      expect(getComputedStyle(mount()).display).toBe('none');
    });

    test('is not hidden when body has no embedded class', () => {
      expect(getComputedStyle(mount()).display).not.toBe('none');
    });
  });
});
