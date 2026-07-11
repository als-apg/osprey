// @ts-check
/**
 * <osprey-theme-switcher> — the shared theme control every standalone
 * fleet page mounts once (D15, see
 * docs/superpowers/frontend-foundation/PROGRAM.md).
 *
 * Task 1.9 rewrites this element from a binary sun/moon `#theme-toggle`
 * button into a **family picker**: `THEMES` (tokens.js) groups into
 * families (a family is a `{light, dark}` pair — the built-in `osprey`
 * family and the WCAG-AAA `high-contrast` family today), and this element
 * renders one control that lists every family plus a second control that
 * flips light/dark WITHIN whichever family is active. Picking a family
 * calls theme-manager.js's `setFamily()`; the mode control calls
 * `toggleTheme()`. Neither concrete id nor mode is ever hand-resolved here
 * — theme-manager.js is the single source of truth for what "active"
 * means; this element only reads it back via `getFamily()`/`getTheme()`
 * and `subscribe()`.
 *
 * Fully self-contained (unlike the pre-1.9 element, which rendered fixed
 * `#theme-toggle`/`#theme-icon-sun`/`#theme-icon-moon` markup that
 * theme-manager.js bound by *global* id, so exactly one instance's ids
 * could ever resolve): `connectedCallback` renders this instance's own
 * markup scoped under `this` (classes, not ids — see the multi-instance
 * note below), wires its own `change`/`click` handlers straight to
 * theme-manager.js's `setFamily()`/`toggleTheme()`, and keeps itself in
 * sync by calling theme-manager.js's exported `subscribe()`. A host page
 * needs nothing beyond `<osprey-theme-switcher></osprey-theme-switcher>`
 * (plus this module and theme-manager.js's own `initTheme()` call) — no
 * external `#theme-toggle`-shaped markup, no page-level wiring.
 *
 * Multi-instance: every element instance queries its *own* subtree
 * (`this.querySelector`), so N instances on one page each own independent
 * `<select>`/`<button>` nodes with no id collisions, and each instance
 * registers its own `subscribe()` callback — all of them stay in sync
 * because theme-manager.js notifies every subscriber on every apply.
 *
 * Component conventions followed (D13, locked by `<osprey-drawer>`):
 *   - `osprey-` tag prefix; one custom element per file; filename == tag
 *     name (this file defines exactly `osprey-theme-switcher`).
 *   - Home: design_system/static/js/components/; interfaces import it via
 *     the absolute `/design-system/js/components/osprey-theme-switcher.js`
 *     mount path.
 *   - Light DOM only -- no attachShadow(). Nothing here needs to reach
 *     outside its own subtree, but tokens.css custom properties must
 *     still cascade in, which shadow DOM would not prevent anyway (custom
 *     properties pierce shadow boundaries) -- light DOM is kept simply to
 *     match every other component in this tree.
 *   - Registration guard: `customElements.define` is only ever reached
 *     behind `!customElements.get(...)`, so a double side-effect import is
 *     safe.
 *   - Token-only styling: every color this component's injected stylesheet
 *     sets is a `var(--…)` custom property from tokens.css; no raw hex.
 *     Non-color layout values (sizes, radii) are plain px, matching every
 *     other interface's own `.header-icon-btn`-style rules -- tokens.css
 *     does not yet emit `--space-*`/`--radius-*` custom properties (see
 *     core.json's `space`/`radius` primitives, generator-internal only
 *     today), so there is no token to reference for those.
 *
 * D15 embedded-hidden default: per D14's dual-mode contract, a standalone
 * page shows its own theme switcher; an embedded panel defers all chrome
 * to the hub and hides it (`body.embedded`, set by frame-params.js's
 * `applyEmbedded()`). This component injects the shared
 * `body.embedded osprey-theme-switcher { display: none }` rule itself --
 * once, however many instances end up on a page -- since there is no
 * shared stylesheet every page already includes to add it to instead (no
 * build step, no bundler).
 *
 * @module components/osprey-theme-switcher
 */

import {
  getFamily,
  getTheme,
  setFamily,
  subscribe,
  toggleTheme,
} from '/design-system/js/theme-manager.js';
import { THEMES } from '/design-system/js/tokens.js';

/** @typedef {{id: string, label: string, mode: string, family: string}} ThemeEntry */

const STYLE_ID = 'osprey-theme-switcher-style';

// tokens.js is plain (unchecked) generated JS -- cast to the documented
// shape rather than relying on tsc's inference of the literal it emits
// (same rationale as theme-manager.js's own `_themes` cast).
const _themes = /** @type {ThemeEntry[]} */ (THEMES);

/**
 * Human label for a family id: title-case each hyphen-separated word
 * ('osprey' -> 'Osprey', 'high-contrast' -> 'High Contrast'). THEMES
 * carries a label per concrete theme id (e.g. 'High Contrast Dark') but no
 * separate family-level label, so this derives one from the family id
 * itself -- stripping a common mode-word suffix off two member labels
 * would break the moment a family's dark/light labels don't share one
 * (e.g. a themed pair named for its own light/dark variants rather than a
 * shared family word).
 * @param {string} family
 * @returns {string}
 */
function _familyLabel(family) {
  return family
    .split('-')
    .map((word) => (word.length ? word[0].toUpperCase() + word.slice(1) : word))
    .join(' ');
}

/**
 * The available families, deduped, in `THEMES` declaration order -- the
 * same order theme-manager.js's `DEFAULT_FAMILY` fallback uses, so the
 * first `<option>` here is always that same fallback family.
 * @returns {{id: string, label: string}[]}
 */
function _families() {
  /** @type {Map<string, string>} */
  const seen = new Map();
  for (const theme of _themes) {
    if (!seen.has(theme.family)) seen.set(theme.family, _familyLabel(theme.family));
  }
  return Array.from(seen, ([id, label]) => ({ id, label }));
}

/**
 * The `mode` ('dark'|'light') of a concrete theme id, or null if `id` is
 * not a recognized theme (including `null` itself, e.g. before
 * theme-manager.js's `initTheme()` has resolved an initial theme).
 * @param {string|null} id
 * @returns {string|null}
 */
function _modeOfId(id) {
  if (id === null) return null;
  const theme = _themes.find((entry) => entry.id === id);
  return theme ? theme.mode : null;
}

/**
 * This instance's markup: a family `<select>` (one `<option>` per family)
 * and a mode toggle `<button>` reusing the same sun/moon glyphs the
 * pre-1.9 binary toggle used. Classes only -- see the multi-instance note
 * in the module docstring for why no ids appear here.
 * @returns {string}
 */
function _renderTemplate() {
  const options = _families()
    .map((family) => `<option value="${family.id}">${family.label}</option>`)
    .join('');
  return `
    <select class="theme-switcher-family" aria-label="Theme family">${options}</select>
    <button type="button" class="theme-switcher-mode" title="Toggle light/dark theme" aria-label="Switch to light theme" aria-pressed="false">
      <svg class="theme-switcher-icon-sun" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="5"/><line x1="12" y1="1" x2="12" y2="3"/><line x1="12" y1="21" x2="12" y2="23"/><line x1="4.22" y1="4.22" x2="5.64" y2="5.64"/><line x1="18.36" y1="18.36" x2="19.78" y2="19.78"/><line x1="1" y1="12" x2="3" y2="12"/><line x1="21" y1="12" x2="23" y2="12"/><line x1="4.22" y1="19.78" x2="5.64" y2="18.36"/><line x1="18.36" y1="5.64" x2="19.78" y2="4.22"/></svg>
      <svg class="theme-switcher-icon-moon" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="display:none"><path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z"/></svg>
    </button>
  `;
}

/**
 * Inject the shared embedded-hidden rule plus this component's own
 * (token-only) skin exactly once, however many `<osprey-theme-switcher>`
 * instances end up on the page. Idempotent -- a no-op once the style tag
 * already exists.
 */
function _ensureStyleInjected() {
  if (document.getElementById(STYLE_ID)) return;
  const style = document.createElement('style');
  style.id = STYLE_ID;
  style.textContent = `
    body.embedded osprey-theme-switcher { display: none; }
    osprey-theme-switcher {
      display: inline-flex;
      align-items: center;
      gap: 6px;
    }
    osprey-theme-switcher .theme-switcher-family {
      background: var(--bg-elevated);
      color: var(--text-primary);
      border: 1px solid var(--border-default);
      border-radius: 4px;
      font-family: var(--font-display);
      font-size: 12px;
      line-height: 1.4;
      padding: 3px 6px;
      cursor: pointer;
    }
    osprey-theme-switcher .theme-switcher-family:hover,
    osprey-theme-switcher .theme-switcher-family:focus-visible {
      border-color: var(--border-accent);
    }
    osprey-theme-switcher .theme-switcher-mode {
      background: transparent;
      border: 1px solid transparent;
      border-radius: 4px;
      color: var(--text-muted);
      width: 28px;
      height: 28px;
      display: flex;
      align-items: center;
      justify-content: center;
      cursor: pointer;
      padding: 0;
      flex: none;
    }
    osprey-theme-switcher .theme-switcher-mode:hover,
    osprey-theme-switcher .theme-switcher-mode:focus-visible {
      color: var(--color-accent-light);
      border-color: var(--border-accent);
      background: var(--accent-tint-06);
    }
  `;
  document.head.appendChild(style);
}

/**
 * Light-DOM family-picker + in-family mode toggle. Renders its own markup,
 * wires its own handlers straight to theme-manager.js, and keeps itself in
 * sync via `subscribe()`. See the module docstring for the full contract.
 */
export class OspreyThemeSwitcher extends HTMLElement {
  constructor() {
    super();
    /** True once this instance has rendered its markup. */
    this._rendered = false;
    /** @type {HTMLSelectElement|null} */
    this._familySelect = null;
    /** @type {HTMLButtonElement|null} */
    this._modeButton = null;
    /** @type {HTMLElement|null} */
    this._sunIcon = null;
    /** @type {HTMLElement|null} */
    this._moonIcon = null;
    /** Unsubscribe callback from theme-manager.js's `subscribe()`, while connected. */
    /** @type {(() => void)|null} */
    this._unsubscribe = null;
    this._onFamilyChange = this._onFamilyChange.bind(this);
    this._onModeToggle = this._onModeToggle.bind(this);
    this._syncUI = this._syncUI.bind(this);
  }

  connectedCallback() {
    _ensureStyleInjected();
    if (this._rendered) {
      // Reconnect after disconnect (idempotent, matching <osprey-drawer>'s
      // reconnect contract): re-render nothing (markup and handlers are
      // still attached to the same nodes), just resume tracking theme
      // changes -- disconnectedCallback tore the subscription down.
      this._subscribeToThemeManager();
      // Catch up on any theme change missed while disconnected: subscribe()
      // only delivers future applies, so reconcile the UI to the current
      // (family, mode) immediately, mirroring the first-connect ordering.
      this._syncUI();
      return;
    }
    this._rendered = true;
    this.innerHTML = _renderTemplate();

    this._familySelect = /** @type {HTMLSelectElement} */ (
      this.querySelector('.theme-switcher-family')
    );
    this._modeButton = /** @type {HTMLButtonElement} */ (
      this.querySelector('.theme-switcher-mode')
    );
    this._sunIcon = this.querySelector('.theme-switcher-icon-sun');
    this._moonIcon = this.querySelector('.theme-switcher-icon-moon');

    this._familySelect.addEventListener('change', this._onFamilyChange);
    this._modeButton.addEventListener('click', this._onModeToggle);

    this._syncUI();
    this._subscribeToThemeManager();
  }

  disconnectedCallback() {
    if (this._unsubscribe) {
      this._unsubscribe();
      this._unsubscribe = null;
    }
  }

  _subscribeToThemeManager() {
    if (this._unsubscribe) return; // already subscribed
    this._unsubscribe = subscribe(this._syncUI);
  }

  _onFamilyChange() {
    if (!this._familySelect) return;
    setFamily(this._familySelect.value);
  }

  _onModeToggle() {
    toggleTheme();
  }

  /**
   * Reflect theme-manager.js's current (family, mode) into this instance's
   * markup. Called once at connect (covers the case where `initTheme()`
   * already ran before this element upgraded) and on every subsequent
   * `subscribe()` notification, including notifications where the applied
   * id is unchanged (theme-manager.js's hidden-iframe protocol re-notifies
   * unconditionally) -- re-applying the same family/mode here is cheap and
   * this element's correctness never depends on deduping it.
   */
  _syncUI() {
    if (!this._familySelect || !this._modeButton) return;

    const family = getFamily();
    if (family !== null && this._familySelect.value !== family) {
      this._familySelect.value = family;
    }

    const mode = _modeOfId(getTheme());
    const isLight = mode === 'light';
    if (this._sunIcon) this._sunIcon.style.display = isLight ? 'none' : 'block';
    if (this._moonIcon) this._moonIcon.style.display = isLight ? 'block' : 'none';

    const targetMode = isLight ? 'dark' : 'light';
    this._modeButton.setAttribute('aria-label', `Switch to ${targetMode} theme`);
    this._modeButton.setAttribute('aria-pressed', isLight ? 'true' : 'false');
  }
}

if (!customElements.get('osprey-theme-switcher')) {
  customElements.define('osprey-theme-switcher', OspreyThemeSwitcher);
}
