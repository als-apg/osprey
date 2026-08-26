// @ts-check
/**
 * <osprey-display-menu> — the shared standalone settings popover: one quiet
 * faders button that collapses every display preference behind a popover
 * card (Appearance, View, Theme), with an optional Settings entry.
 *
 * This is THE display popover of the fleet: the web-terminal hub, standalone
 * ARIEL and every standalone panel mount this one element. It is deliberately
 * page-agnostic: it carries no page-specific markup, ids, or logic, and its
 * only coupling points are the two shared preference keys, theme-manager.js,
 * the `osprey-mode-change` message it posts to its own window, the
 * `settings-drawer` attribute, and the projected action slot below.
 *
 * Rows (FIXED — there is no row-configuration API, and none is added until
 * a consumer actually needs one):
 *   - **Appearance** (Light/Dark) — flips light/dark WITHIN the active theme
 *     family via theme-manager.js's `toggleTheme()`. Like the hub's popover
 *     it never hand-resolves a concrete theme id: theme-manager.js is the
 *     single source of truth and this element reads state back through
 *     `getTheme()`/`getFamily()` + `subscribe()`.
 *   - **View** (Expert/Simple) — the mechanism app.js's `initModeToggle()`
 *     uses, minus the hub-only dock/panel follow-ups: persist the explicit
 *     choice under `localStorage['osprey-ui-mode']`, call frame-params.js's
 *     `stripQueryMode()` so a leftover one-shot `?mode=` deep-link cannot
 *     out-rank that choice on the next reload, then post
 *     `{type: 'osprey-mode-change', mode}` to this element's OWN window.
 *     Stamping `data-ui-mode` is deliberately NOT done here: the page's
 *     same-origin `osprey-mode-change` listener owns that, and a
 *     same-document post carries `event.origin === location.origin`, so the
 *     hub-broadcast path and the standalone path are the same path. This
 *     element registers frame-params.js's shared `onModeChange()` receive
 *     side for itself as well, so the View row reflects what was actually
 *     applied — and so a page that mounts the element without wiring a
 *     listener of its own still gets a working mode flip.
 *   - **Theme** (family pills) — one pill per family from theme-manager.js's
 *     `themeFamilies()`, applied with `setFamily()` so the current light/dark
 *     preference survives the family change.
 *
 * Only the Settings entry is optional, and only by presence of the
 * `settings-drawer="<drawer-id>"` attribute: it renders a Settings button
 * carrying `data-drawer="<drawer-id>"`, which is exactly the delegated
 * trigger contract `<osprey-drawer>` already installs document-wide (see
 * osprey-drawer.js's `_installDelegatedHandlersOnce`). This element
 * therefore does not import, reference, or need osprey-drawer.js — a page
 * that mounts a drawer gets the open behavior for free, and a page that does
 * not simply omits the attribute. The attribute is read once at render
 * (matching `<osprey-drawer>`'s own once-at-connect `resizable` precedent),
 * not observed.
 *
 * The action row is also where a page projects its own chrome. Any element
 * children the tag carries in the page's markup are moved, in order, into the
 * card's `.display-menu-actions` row after the three rows — after the
 * built-in Settings entry when both are present. This is the light-DOM
 * equivalent of a slot, and it is the whole of the page-specific surface: the
 * web-terminal hub projects its own Settings button (which must keep the
 * `data-drawer-trigger` contract settings.js's warning gate binds to, not
 * the component's `data-drawer`) and its Log out button this way, and keeps
 * their ids, handlers and skins where they always were. Projected children
 * get no behavior from this element — it neither closes the card on their
 * click nor styles anything beyond the shared `.display-menu-settings` look;
 * a page that wants the card closed calls `closeMenu()` on the element.
 *
 * Only the built-in Settings row closes the card. Appearance, View, and Theme
 * picks all leave it open: every row renders identically either way, so the
 * card an operator is looking at survives the pick and they can compare (or
 * go straight back) without re-opening the menu. Opening a drawer, by
 * contrast, moves them to a different surface entirely.
 *
 * Component conventions followed (D13, locked by `<osprey-drawer>`):
 *   - `osprey-` tag prefix; one custom element per file; filename == tag name
 *     (this file defines exactly `osprey-display-menu`).
 *   - Home: design_system/static/js/components/; interfaces import it via the
 *     absolute `/design-system/js/components/osprey-display-menu.js` mount
 *     path, and its own imports use the same absolute `/design-system/js/...`
 *     specifiers `<osprey-theme-switcher>` and `<osprey-drawer>` use.
 *   - Light DOM only — no attachShadow(), matching every other component in
 *     this tree.
 *   - Registration guard: `customElements.define` is only ever reached behind
 *     `!customElements.get(...)`, so a double side-effect import is safe.
 *   - Multi-instance safe: the template is CLASS-only (no ids), every lookup
 *     is scoped to `this`, and each instance registers its own `subscribe()`
 *     callback, so N instances on one page stay in sync with each other
 *     through theme-manager.js rather than through shared markup.
 *   - Token-only styling: every value the injected stylesheet sets for a
 *     color, type step, weight, radius, or duration is a custom-property
 *     reference to tokens.css. The handful of raw px left are pure layout
 *     (control box sizes, the thumb inset, the card's min-width) — the axes
 *     tokens.css deliberately does not cover.
 *
 * D15 embedded-hidden default: a standalone page shows its own chrome; an
 * embedded panel defers all chrome to the hub and hides it (`body.embedded`,
 * set by frame-params.js's `applyEmbedded()`). Like
 * `<osprey-theme-switcher>`, this component injects that rule itself — once,
 * however many instances end up on a page — since there is no shared
 * stylesheet every page already includes to add it to instead.
 *
 * @module components/osprey-display-menu
 */

import { onModeChange, stripQueryMode } from '/design-system/js/frame-params.js';
import {
  getFamily,
  getTheme,
  modeOf,
  setFamily,
  subscribe,
  themeFamilies,
  toggleTheme,
} from '/design-system/js/theme-manager.js';

const STYLE_ID = 'osprey-display-menu-style';

/** Shared, origin-scoped Expert/Simple key — the one mode-boot.js resolves from. */
const MODE_STORAGE_KEY = 'osprey-ui-mode';

/**
 * The UI mode currently stamped on `<html>` by mode-boot.js (pre-paint) or by
 * a since-applied `osprey-mode-change`. Anything other than `'simple'` reads
 * as `'expert'`, matching both mode-boot.js's default and frame-params.js's
 * `onModeChange()` normalization.
 *
 * @returns {'expert'|'simple'}
 */
function _currentUiMode() {
  return document.documentElement.getAttribute('data-ui-mode') === 'simple'
    ? 'simple'
    : 'expert';
}

/**
 * This instance's markup: the faders trigger plus the popover card's three
 * fixed rows. Classes only — see the multi-instance note in the module
 * docstring. The Settings entry is not here; it is built programmatically
 * (and only when `settings-drawer` is present) so the attribute's value is
 * never interpolated into an HTML string.
 *
 * @returns {string}
 */
function _renderTemplate() {
  return `
    <button type="button" class="display-menu-trigger" title="Display preferences"
            aria-label="Display preferences" aria-haspopup="true" aria-expanded="false">
      <svg class="display-menu-trigger-icon" aria-hidden="true" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="4" y1="21" x2="4" y2="14"/><line x1="4" y1="10" x2="4" y2="3"/><line x1="12" y1="21" x2="12" y2="12"/><line x1="12" y1="8" x2="12" y2="3"/><line x1="20" y1="21" x2="20" y2="16"/><line x1="20" y1="12" x2="20" y2="3"/><line x1="1" y1="14" x2="7" y2="14"/><line x1="9" y1="8" x2="15" y2="8"/><line x1="17" y1="16" x2="23" y2="16"/></svg>
    </button>
    <div class="display-menu-card" role="group" aria-label="Display preferences">
      <div class="display-menu-row">
        <span class="display-menu-heading">Appearance</span>
        <div class="display-menu-seg display-menu-appearance" role="group" aria-label="Light or dark appearance">
          <button class="display-seg-option" data-appearance="light" type="button" aria-pressed="false">
            <svg class="display-seg-icon" aria-hidden="true" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="5"/><line x1="12" y1="1" x2="12" y2="3"/><line x1="12" y1="21" x2="12" y2="23"/><line x1="4.22" y1="4.22" x2="5.64" y2="5.64"/><line x1="18.36" y1="18.36" x2="19.78" y2="19.78"/><line x1="1" y1="12" x2="3" y2="12"/><line x1="21" y1="12" x2="23" y2="12"/><line x1="4.22" y1="19.78" x2="5.64" y2="18.36"/><line x1="18.36" y1="5.64" x2="19.78" y2="4.22"/></svg>Light
          </button>
          <button class="display-seg-option" data-appearance="dark" type="button" aria-pressed="false">
            <svg class="display-seg-icon" aria-hidden="true" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z"/></svg>Dark
          </button>
        </div>
      </div>
      <div class="display-menu-row">
        <span class="display-menu-heading">View</span>
        <div class="display-menu-seg display-menu-view" role="group" aria-label="UI mode">
          <button class="display-seg-option" data-mode="expert" type="button" aria-pressed="false">
            <span class="display-seg-icon" aria-hidden="true">&#x269b;</span>Expert
          </button>
          <button class="display-seg-option" data-mode="simple" type="button" aria-pressed="false">
            <span class="display-seg-icon" aria-hidden="true">&#x25c9;</span>Simple
          </button>
        </div>
      </div>
      <div class="display-menu-row">
        <span class="display-menu-heading">Theme</span>
        <div class="display-menu-families" role="group" aria-label="Theme family"></div>
      </div>
    </div>
  `;
}

/**
 * Inject the shared embedded-hidden rule plus this component's own
 * (token-only) skin exactly once, however many `<osprey-display-menu>`
 * instances end up on the page. Idempotent — a no-op once the style tag
 * already exists.
 *
 * Every rule past the embedded-hidden one is scoped under the tag name, so a
 * page's own stylesheet can style what it projects into the action row (the
 * hub's Log out button) under the same scope without the two colliding.
 */
function _ensureStyleInjected() {
  if (document.getElementById(STYLE_ID)) return;
  const style = document.createElement('style');
  style.id = STYLE_ID;
  style.textContent = `
    body.embedded osprey-display-menu { display: none; }
    osprey-display-menu {
      position: relative;
      display: flex;
      flex-shrink: 0;
    }
    osprey-display-menu .display-menu-trigger {
      width: 28px;
      height: 28px;
      display: flex;
      align-items: center;
      justify-content: center;
      padding: 0;
      color: var(--color-accent);
      background: var(--bg-panel);
      border: 1px solid var(--border-default);
      border-radius: var(--radius-full);
      cursor: pointer;
      transition: color var(--duration-fast) ease,
        border-color var(--duration-fast) ease,
        background var(--duration-fast) ease;
    }
    osprey-display-menu .display-menu-trigger-icon { display: block; }
    osprey-display-menu .display-menu-trigger:hover,
    osprey-display-menu .display-menu-trigger[aria-expanded="true"] {
      color: var(--color-accent-light);
      border-color: var(--border-accent);
      background: var(--bg-elevated);
    }
    osprey-display-menu .display-menu-card {
      display: none;
      position: absolute;
      top: calc(100% + var(--space-2));
      right: 0;
      min-width: 230px;
      padding: var(--space-3);
      flex-direction: column;
      gap: var(--space-3);
      background: var(--bg-secondary);
      border: 1px solid var(--border-default);
      border-radius: var(--radius-xl);
      box-shadow: var(--shadow-dropdown);
      z-index: var(--z-dropdown);
    }
    osprey-display-menu .display-menu-card.open {
      display: flex;
      animation: osprey-display-menu-in var(--duration-fast) ease;
    }
    @keyframes osprey-display-menu-in {
      from { opacity: 0; transform: translateY(-4px); }
      to { opacity: 1; transform: none; }
    }
    osprey-display-menu .display-menu-row {
      display: flex;
      flex-direction: column;
      gap: var(--space-1);
    }
    osprey-display-menu .display-menu-heading {
      font-family: var(--font-display);
      font-size: var(--text-sm);
      font-weight: var(--weight-semibold);
      text-transform: uppercase;
      letter-spacing: 0.08em;
      color: var(--text-muted);
    }
    osprey-display-menu .display-menu-seg {
      position: relative;
      display: grid;
      grid-template-columns: 1fr 1fr;
      padding: 3px;
      background: var(--bg-panel);
      border: 1px solid var(--border-default);
      border-radius: var(--radius-lg);
    }
    osprey-display-menu .display-menu-seg::before {
      content: '';
      position: absolute;
      top: 3px;
      bottom: 3px;
      left: 3px;
      width: calc(50% - 3px);
      background: var(--bg-elevated);
      border-radius: var(--radius-md);
      transition: transform var(--duration-base) ease;
    }
    osprey-display-menu .display-menu-seg[data-active="dark"]::before,
    osprey-display-menu .display-menu-seg[data-active="simple"]::before {
      transform: translateX(100%);
    }
    osprey-display-menu .display-seg-option {
      position: relative;
      display: inline-flex;
      align-items: center;
      justify-content: center;
      gap: var(--space-1);
      padding: 5px 10px;
      font-family: var(--font-display);
      font-size: var(--text-base);
      font-weight: var(--weight-regular);
      color: var(--text-secondary);
      background: transparent;
      border: none;
      cursor: pointer;
      white-space: nowrap;
      transition: color var(--duration-fast) ease;
    }
    osprey-display-menu .display-seg-option:hover { color: var(--text-primary); }
    osprey-display-menu .display-seg-option[aria-pressed="true"] {
      color: var(--text-primary);
      font-weight: var(--weight-medium);
    }
    osprey-display-menu .display-seg-icon {
      font-size: var(--text-md);
      line-height: var(--leading-none);
    }
    osprey-display-menu .display-menu-families {
      display: flex;
      flex-wrap: wrap;
      gap: var(--space-1);
    }
    osprey-display-menu .display-family-pill {
      padding: 4px 10px;
      font-family: var(--font-display);
      font-size: var(--text-base);
      color: var(--text-secondary);
      background: transparent;
      border: 1px solid var(--border-default);
      border-radius: var(--radius-full);
      cursor: pointer;
      transition: color var(--duration-fast) ease,
        border-color var(--duration-fast) ease,
        background var(--duration-fast) ease;
    }
    osprey-display-menu .display-family-pill:hover {
      color: var(--text-primary);
      border-color: var(--border-accent);
    }
    osprey-display-menu .display-family-pill.active {
      color: var(--text-primary);
      border-color: var(--color-accent);
      background: var(--accent-tint-06);
    }
    osprey-display-menu .display-menu-actions {
      display: flex;
      gap: var(--space-2);
      padding-top: var(--space-3);
      border-top: 1px solid var(--border-default);
    }
    osprey-display-menu .display-menu-settings {
      flex: 1;
      display: inline-flex;
      align-items: center;
      justify-content: center;
      gap: var(--space-1);
      padding: 6px 10px;
      font-family: var(--font-display);
      font-size: var(--text-base);
      color: var(--text-secondary);
      background: var(--bg-panel);
      border: 1px solid var(--border-default);
      border-radius: var(--radius-md);
      cursor: pointer;
      transition: color var(--duration-fast) ease,
        border-color var(--duration-fast) ease,
        background var(--duration-fast) ease;
    }
    osprey-display-menu .display-menu-settings:hover,
    osprey-display-menu .display-menu-settings.active {
      color: var(--text-primary);
      border-color: var(--border-accent);
      background: var(--bg-elevated);
    }
    @media (prefers-reduced-motion: reduce) {
      osprey-display-menu .display-menu-card.open { animation: none; }
      osprey-display-menu .display-menu-seg::before { transition: none; }
    }
  `;
  document.head.appendChild(style);
}

/**
 * Light-DOM display-preferences popover. Renders its own markup, wires its own
 * handlers straight to theme-manager.js and the shared mode key, and keeps
 * itself in sync via `subscribe()` and `onModeChange()`. See the module
 * docstring for the full contract.
 */
export class OspreyDisplayMenu extends HTMLElement {
  constructor() {
    super();
    /** True once this instance has rendered its markup. */
    this._rendered = false;
    /** True once this instance registered the mode-broadcast receive side. */
    this._modeListenerBound = false;
    /** @type {HTMLButtonElement|null} */
    this._trigger = null;
    /** @type {HTMLElement|null} */
    this._card = null;
    /** @type {HTMLElement|null} */
    this._appearanceSeg = null;
    /** @type {HTMLElement|null} */
    this._viewSeg = null;
    /** @type {HTMLElement|null} */
    this._familyPicker = null;
    /** Unsubscribe callback from theme-manager.js's `subscribe()`, while connected. */
    /** @type {(() => void)|null} */
    this._unsubscribe = null;
    this._onDocumentClick = this._onDocumentClick.bind(this);
    this._onDocumentKeydown = this._onDocumentKeydown.bind(this);
    this._syncUI = this._syncUI.bind(this);
  }

  connectedCallback() {
    _ensureStyleInjected();
    if (this._rendered) {
      // Reconnect after disconnect (idempotent, matching <osprey-drawer>'s
      // reconnect contract): markup and handlers are still attached to the
      // same nodes, so only resume tracking — disconnectedCallback tore the
      // theme subscription down — and reconcile any change missed meanwhile.
      this._subscribeToThemeManager();
      this._syncUI();
      return;
    }
    this._rendered = true;
    // The page's own action chrome, declared as this tag's children. Captured
    // before the template replaces them; re-homed into the action row below.
    const projected = Array.from(this.children);
    this.innerHTML = _renderTemplate();

    this._trigger = /** @type {HTMLButtonElement} */ (
      this.querySelector('.display-menu-trigger')
    );
    this._card = /** @type {HTMLElement} */ (this.querySelector('.display-menu-card'));
    this._appearanceSeg = /** @type {HTMLElement} */ (
      this.querySelector('.display-menu-appearance')
    );
    this._viewSeg = /** @type {HTMLElement} */ (this.querySelector('.display-menu-view'));
    this._familyPicker = /** @type {HTMLElement} */ (
      this.querySelector('.display-menu-families')
    );

    this._buildFamilyPills();
    this._buildActions(projected);

    this._trigger.addEventListener('click', (event) => {
      event.stopPropagation();
      if (this.isOpen()) this.closeMenu();
      else this.openMenu();
    });
    this._appearanceSeg.addEventListener('click', (event) => this._onAppearancePick(event));
    this._viewSeg.addEventListener('click', (event) => this._onViewPick(event));

    this._syncUI();
    this._subscribeToThemeManager();
    this._listenForModeBroadcasts();
  }

  disconnectedCallback() {
    // Closing first is what removes the document-level outside-click/Escape
    // listeners — a menu removed while open would otherwise keep them, and
    // keep this element alive, for the life of the document.
    this.closeMenu();
    if (this._unsubscribe) {
      this._unsubscribe();
      this._unsubscribe = null;
    }
  }

  // ---- Popover ----

  /** @returns {boolean} whether the popover card is currently showing. */
  isOpen() {
    return this._card !== null && this._card.classList.contains('open');
  }

  /** Show the popover card and start listening for the ways it can be dismissed. */
  openMenu() {
    if (!this._card || !this._trigger || this.isOpen()) return;
    this._card.classList.add('open');
    this._trigger.setAttribute('aria-expanded', 'true');
    // Capture-phase, matching the hub popover and panel-add-menu.js: an
    // outside click dismisses the card before it does anything else.
    document.addEventListener('click', this._onDocumentClick, true);
    document.addEventListener('keydown', this._onDocumentKeydown, true);
  }

  /** Hide the popover card. Idempotent — safe to call unconditionally. */
  closeMenu() {
    document.removeEventListener('click', this._onDocumentClick, true);
    document.removeEventListener('keydown', this._onDocumentKeydown, true);
    if (!this._card || !this._trigger) return;
    this._card.classList.remove('open');
    this._trigger.setAttribute('aria-expanded', 'false');
  }

  /** @param {MouseEvent} event */
  _onDocumentClick(event) {
    if (event.target instanceof Node && !this.contains(event.target)) this.closeMenu();
  }

  /** @param {KeyboardEvent} event */
  _onDocumentKeydown(event) {
    if (event.key !== 'Escape') return;
    this.closeMenu();
    this._trigger?.focus();
  }

  // ---- Rows ----

  /**
   * One pill per theme family, built once from theme-manager.js's
   * `themeFamilies()`. `textContent`, never innerHTML — the labels derive from
   * generated token data.
   */
  _buildFamilyPills() {
    if (!this._familyPicker) return;
    for (const family of themeFamilies()) {
      const pill = document.createElement('button');
      pill.type = 'button';
      pill.className = 'display-family-pill';
      pill.dataset.family = family.id;
      pill.textContent = family.label;
      pill.setAttribute('aria-pressed', 'false');
      pill.addEventListener('click', () => setFamily(family.id));
      this._familyPicker.appendChild(pill);
    }
  }

  /**
   * Build the card's action row: the optional built-in Settings entry (only
   * when `settings-drawer` names a drawer) followed by the page's projected
   * children, in their markup order. No row at all when there is nothing to
   * put in it, so a page with neither gets no stray hairline.
   *
   * The Settings button carries `data-drawer`, `<osprey-drawer>`'s own
   * delegated trigger contract, so the drawer component owns the open — this
   * element dismisses the card and parks focus on the persistent trigger,
   * since opening a drawer is a move to another surface. Built with DOM calls
   * so the attribute value is never interpolated into markup.
   *
   * @param {Element[]} projected The tag's original element children.
   */
  _buildActions(projected) {
    const drawerId = this.getAttribute('settings-drawer');
    if ((!drawerId && projected.length === 0) || !this._card) return;

    const actions = document.createElement('div');
    actions.className = 'display-menu-actions';
    if (drawerId) actions.appendChild(this._buildSettingsEntry(drawerId));
    for (const child of projected) actions.appendChild(child);
    this._card.appendChild(actions);
  }

  /**
   * The built-in Settings entry for `drawerId` — see `_buildActions`.
   * @param {string} drawerId
   * @returns {HTMLButtonElement}
   */
  _buildSettingsEntry(drawerId) {
    const button = document.createElement('button');
    button.type = 'button';
    button.className = 'display-menu-settings';
    button.dataset.drawer = drawerId;
    button.title = 'Open settings';

    const icon = document.createElement('span');
    icon.className = 'display-menu-settings-icon';
    icon.setAttribute('aria-hidden', 'true');
    icon.textContent = '⚙';
    button.appendChild(icon);
    button.appendChild(document.createTextNode('Settings'));
    button.addEventListener('click', () => {
      // Close, then hand focus to the trigger: <osprey-drawer>'s delegated
      // [data-drawer] handler is bubble-phase on document, so it runs after
      // this listener and records document.activeElement as its return-focus
      // target. By then the card is display:none — without the hand-off the
      // drawer would try to restore focus into a hidden subtree and keyboard
      // users would land on <body> when the drawer closes.
      this.closeMenu();
      this._trigger?.focus();
    });
    return button;
  }

  /**
   * Appearance pick: flip light/dark within the active family, only when the
   * pick differs from the current mode (re-clicking the active side no-ops
   * rather than toggling away from it).
   * @param {Event} event
   */
  _onAppearancePick(event) {
    const target =
      event.target instanceof Element ? event.target.closest('.display-seg-option') : null;
    if (!(target instanceof HTMLElement)) return;
    const want = target.dataset.appearance;
    if (want !== 'light' && want !== 'dark') return;
    if (want !== modeOf(getTheme())) toggleTheme();
  }

  /**
   * View pick: persist the explicit choice, drop a leftover one-shot `?mode=`,
   * then broadcast it to this element's own window for the page's same-origin
   * listener (this element's included) to apply. Mirrors app.js's
   * `initModeToggle()` minus its hub-only dock/panel follow-ups.
   * @param {Event} event
   */
  _onViewPick(event) {
    const target =
      event.target instanceof Element ? event.target.closest('.display-seg-option') : null;
    if (!(target instanceof HTMLElement)) return;
    const mode = target.dataset.mode;
    if (mode !== 'expert' && mode !== 'simple') return;

    try {
      window.localStorage.setItem(MODE_STORAGE_KEY, mode);
    } catch { /* storage blocked — the mode still applies for this session */ }
    stripQueryMode();
    window.postMessage({ type: 'osprey-mode-change', mode }, window.location.origin);
    // Reflect the pick now rather than waiting for the message to round-trip:
    // the post is asynchronous, and the row must not look unchanged in
    // between. The listener's own sync then confirms what was applied.
    this._syncView(mode);
  }

  // ---- State reflection ----

  _subscribeToThemeManager() {
    if (this._unsubscribe) return; // already subscribed
    this._unsubscribe = subscribe(this._syncUI);
  }

  /**
   * Register frame-params.js's shared `osprey-mode-change` receive side once
   * per instance. `onModeChange()` has no unsubscribe (it is designed for a
   * page's one-time boot wiring), so this is deliberately bound once and left
   * bound across disconnect/reconnect rather than re-registered — the
   * callback is a pure UI sync and is harmless on a disconnected element.
   */
  _listenForModeBroadcasts() {
    if (this._modeListenerBound) return;
    this._modeListenerBound = true;
    onModeChange((mode) => this._syncView(mode));
  }

  /**
   * Reflect theme-manager.js's current (family, mode) plus the active UI mode
   * into this instance's markup: the segmented controls' thumb position rides
   * `data-active` (CSS moves it), and the active family pill gets `.active`.
   * Called once at connect (covering an `initTheme()` that already ran before
   * this element upgraded) and on every subsequent theme apply.
   */
  _syncUI() {
    const mode = modeOf(getTheme());
    if (this._appearanceSeg && mode !== null) {
      this._appearanceSeg.setAttribute('data-active', mode);
      for (const option of this._appearanceSeg.querySelectorAll('.display-seg-option')) {
        const active = option instanceof HTMLElement && option.dataset.appearance === mode;
        option.setAttribute('aria-pressed', active ? 'true' : 'false');
      }
    }

    const family = getFamily();
    if (this._familyPicker && family !== null) {
      for (const pill of this._familyPicker.querySelectorAll('.display-family-pill')) {
        const active = pill instanceof HTMLElement && pill.dataset.family === family;
        pill.classList.toggle('active', active);
        pill.setAttribute('aria-pressed', active ? 'true' : 'false');
      }
    }

    this._syncView();
  }

  /**
   * Reflect the UI mode into the View row. Defaults to whatever is stamped on
   * `<html>`; an explicit argument is the mode just picked, applied ahead of
   * the broadcast round-trip.
   * @param {'expert'|'simple'} [mode]
   */
  _syncView(mode) {
    if (!this._viewSeg) return;
    const resolved = mode ?? _currentUiMode();
    this._viewSeg.setAttribute('data-active', resolved);
    for (const option of this._viewSeg.querySelectorAll('.display-seg-option')) {
      const active = option instanceof HTMLElement && option.dataset.mode === resolved;
      option.setAttribute('aria-pressed', active ? 'true' : 'false');
    }
  }
}

if (!customElements.get('osprey-display-menu')) {
  customElements.define('osprey-display-menu', OspreyDisplayMenu);
}
