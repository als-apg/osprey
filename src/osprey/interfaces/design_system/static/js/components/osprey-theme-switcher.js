// @ts-check
/**
 * <osprey-theme-switcher> — the shared theme-toggle control every
 * standalone fleet page mounts once (D15, see
 * docs/superpowers/frontend-foundation/PROGRAM.md). Promoted to a component
 * because the same three-id button markup was on track to be hand-copied
 * into 7 pages -- past the D13 duplication bar `<osprey-drawer>` was
 * promoted at.
 *
 * Renders exactly the `#theme-toggle` / `#theme-icon-sun` / `#theme-icon-moon`
 * markup theme-manager.js already binds (its `initTheme()` ->
 * `_wireToggleButton()` wires the click handler, `_updateToggleUI()` flips
 * the icon visibility and aria-label) -- this component only stops that
 * markup from being duplicated; the binding contract is unchanged. It
 * carries no theme-switching logic of its own and must be paired with a
 * page that calls `initTheme()` after this element has connected and
 * rendered (true as long as this element -- or its own `<script
 * type="module">` -- appears before, or is imported by, the page's
 * init script: module scripts execute in document order after the DOM
 * that contains them has fully parsed, so `customElements.define()`
 * upgrades every already-parsed instance, synchronously calling
 * `connectedCallback` and rendering the button, before a later module's
 * top-level code runs).
 *
 * Adopter note: remove any hand-written `#theme-toggle` markup when
 * mounting this element -- this component renders its own, and a
 * duplicate id would make `document.getElementById('theme-toggle')`
 * resolve to whichever copy is first in document order.
 *
 * Component conventions followed (D13, locked by `<osprey-drawer>`):
 *   - `osprey-` tag prefix; one custom element per file; filename == tag
 *     name (this file defines exactly `osprey-theme-switcher`).
 *   - Home: design_system/static/js/components/; interfaces import it via
 *     the absolute `/design-system/js/components/osprey-theme-switcher.js`
 *     mount path.
 *   - Light DOM only -- no attachShadow(). theme-manager.js's
 *     `document.getElementById` lookups (against `document`, not this
 *     host) keep working unchanged.
 *   - Registration guard: `customElements.define` is only ever reached
 *     behind `!customElements.get(...)`, so a double side-effect import is
 *     safe.
 *   - Tokens consumed, never redefined: this component ships no skin of
 *     its own. `.header-icon-btn` and the icon SVGs stay exactly as
 *     styled by each interface's own CSS today (same skin/behavior split
 *     `<osprey-drawer>` uses) -- nothing here reaches into
 *     `--color-*`/`--bg-*` tokens directly. The one rule this component
 *     does inject (below) is a structural default, not a design token.
 *
 * D15 embedded-hidden default: per D14's dual-mode contract, a standalone
 * page shows its own theme switcher; an embedded panel defers all chrome
 * to the hub and hides it (`body.embedded`, set by frame-params.js's
 * `applyEmbedded()`). Rather than have all 7 fleet pages hand-copy a
 * `body.embedded osprey-theme-switcher { display: none }` rule into their
 * own CSS the way today's per-interface `body.embedded .app-header`-style
 * rules are duplicated, this component injects that one shared rule
 * itself -- once, however many instances end up on a page -- since there
 * is no shared stylesheet every page already includes to add it to
 * instead (no build step, no bundler).
 *
 * @module components/osprey-theme-switcher
 */

const STYLE_ID = 'osprey-theme-switcher-style';

// The exact button/icon markup web_terminal's index.html carried before
// this component existed (theme-manager.js's ids/classes, unchanged).
const TEMPLATE = `
  <button class="header-icon-btn" id="theme-toggle" title="Toggle theme" aria-label="Switch to light theme">
    <svg id="theme-icon-sun" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="5"/><line x1="12" y1="1" x2="12" y2="3"/><line x1="12" y1="21" x2="12" y2="23"/><line x1="4.22" y1="4.22" x2="5.64" y2="5.64"/><line x1="18.36" y1="18.36" x2="19.78" y2="19.78"/><line x1="1" y1="12" x2="3" y2="12"/><line x1="21" y1="12" x2="23" y2="12"/><line x1="4.22" y1="19.78" x2="5.64" y2="18.36"/><line x1="18.36" y1="5.64" x2="19.78" y2="4.22"/></svg>
    <svg id="theme-icon-moon" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="display:none"><path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z"/></svg>
  </button>
`;

/**
 * Inject the shared embedded-hidden rule exactly once, however many
 * `<osprey-theme-switcher>` instances end up on the page. Idempotent --
 * a no-op once the style tag already exists.
 */
function _ensureStyleInjected() {
  if (document.getElementById(STYLE_ID)) return;
  const style = document.createElement('style');
  style.id = STYLE_ID;
  style.textContent = 'body.embedded osprey-theme-switcher { display: none; }';
  document.head.appendChild(style);
}

/**
 * Light-DOM theme-switcher element. Renders the toggle button markup
 * theme-manager.js binds by id; carries no theme-switching logic itself.
 * See the module docstring above for the full authoring contract.
 */
export class OspreyThemeSwitcher extends HTMLElement {
  constructor() {
    super();
    /** True once this instance has rendered its button markup. */
    this._rendered = false;
  }

  connectedCallback() {
    _ensureStyleInjected();
    // Idempotent: a reconnect (disconnect + reconnect without removal)
    // must not re-render and duplicate the button.
    if (this._rendered) return;
    this._rendered = true;
    this.innerHTML = TEMPLATE;
  }
}

if (!customElements.get('osprey-theme-switcher')) {
  customElements.define('osprey-theme-switcher', OspreyThemeSwitcher);
}
