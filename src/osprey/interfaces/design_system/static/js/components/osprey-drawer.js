// @ts-check
/**
 * <osprey-drawer> — OSPREY's first Custom Element, and the reference
 * implementation of the component conventions every component added after
 * it must follow (D12/D13, see docs/superpowers/frontend-foundation/PROGRAM.md).
 *
 * This file carries the base drawer behavior, dialog accessibility, and the
 * web_terminal opt-in superset: open()/close()/toggle(); single-active
 * exclusivity (opening one closes any other open drawer); backdrop
 * click-to-close; Escape-to-close; a focus trap and `role="dialog"`
 * semantics; background isolation via `inert` while open; a resizable
 * drag handle (opt-in via the `resizable` attribute); tabbed panels
 * (opt-in purely by the presence of `.drawer-tab`/`.drawer-tab-panel`
 * markup — no attribute needed, since ariel simply has none); and an
 * unsaved-changes guard that can block closing and switching tabs. Every
 * opt-in feature is inert unless its attribute or markup is present, so
 * ariel (base only) sees zero behavior change.
 *
 * Component conventions (decide-once, binding for every future component):
 *   - `osprey-` tag prefix; one custom element per file; filename == tag
 *     name (this file defines exactly `osprey-drawer`).
 *   - Home: design_system/static/js/components/; interfaces import it via
 *     the absolute `/design-system/js/components/osprey-drawer.js` mount
 *     path (the same absolute-mount convention theme-manager.js and
 *     frame-params.js already use for `/design-system/js/...`).
 *   - Light DOM only — no attachShadow(). Existing global stylesheets and
 *     `document.getElementById` lookups on the drawer's projected content
 *     (panel ids, etc.) keep working unchanged; the element carries no skin
 *     of its own — dimensions, glow, and all other appearance stay in each
 *     interface's own global drawer.css.
 *   - Registration guard: `customElements.define` is only ever reached
 *     behind `!customElements.get(...)`, so a double side-effect import
 *     (`import '/design-system/js/components/osprey-drawer.js'`) is safe.
 *   - Canonical state lives on the `open` boolean attribute — deliberately
 *     not a same-named JS property, which would collide with the
 *     open()/close()/toggle() methods below. `open()`/`close()` validate
 *     BEFORE mutating (see the guard-veto note below), then mutate; a
 *     `_committingOpen`/`_committingClose` flag tells the synchronous
 *     `attributeChangedCallback` reaction that validation already
 *     happened, so it applies side effects directly rather than
 *     re-validating. `attributeChangedCallback` remains the single place
 *     the actual open/close side effects (backdrop sync, background
 *     inert, focus trap, events) are applied — it is also a backstop for
 *     a bypass that mutates the attribute directly
 *     (`el.setAttribute`/`removeAttribute`/`toggleAttribute('open', ...)`
 *     without going through a method): in that case it validates
 *     after-the-fact and, if invalid, undoes the mutation via a
 *     `_vetoRestoring`-guarded re-entrant call that applies no side
 *     effects and fires no events, because the drawer's state never
 *     really changed from an outside observer's perspective. Either way
 *     there is exactly one place that decides what "open" means.
 *     `connectedCallback` and `disconnectedCallback` apply/undo the same
 *     open-state side effects when an already-open drawer is (re)connected
 *     or removed without a matching attribute change.
 *   - Events: dispatched `{ bubbles: true, composed: true }` on the
 *     contract-specified target: `drawer:open`/`drawer:close` on the host,
 *     `drawer:tab-activate` on the newly active panel. Fired for every
 *     drawer, including ariel's, which never listens for them — the
 *     proposal freezes these as the element's public event API, and firing
 *     them unconditionally is simpler than an opt-in gate and not a
 *     behavior change for anyone not listening.
 *   - Tokens are consumed via CSS custom-property references (design-system
 *     tokens) if a component needs any inline style at all, never redefined.
 *     This component needs none: it is pure
 *     behavior, and visual state is entirely CSS-selector driven by each
 *     interface's own skin.
 *   - Dialog accessibility (decide-once, binding for every future modal-like
 *     component): `role="dialog"`, `aria-modal="true"` and `tabindex="-1"`
 *     are set once, unconditionally, on connect — they describe what the
 *     element *is*, not its momentary open state. `aria-labelledby` is
 *     resolved once (see `_resolveTitleId` below) and left untouched if a
 *     consumer already supplied one. Opening traps Tab/Shift+Tab inside the
 *     drawer, moves focus to the first focusable descendant (or the host
 *     itself as a fallback), and marks every other top-level sibling of
 *     `<body>` — except the shared backdrop — `inert` (with an `aria-hidden`
 *     fallback for engines that don't yet implement `inert`). Closing
 *     restores focus to whatever triggered the open and lifts the
 *     background isolation. Only elements this component itself inerted are
 *     ever un-inerted, so an app's own unrelated `inert` usage is untouched.
 *   - Unsaved-changes guard (`registerUnsavedGuard`, decide-once): a guard
 *     returning false (or throwing — `_checkUnsavedChangesSafely` treats a
 *     throw as "unsaved changes present," fail closed, and logs it) must
 *     block EVERY path that removes the `open` attribute — not only
 *     `.close()` itself. `close()` checks the guard BEFORE mutating: a
 *     blocked close never touches the attribute at all, so it is a true
 *     no-op — zero events, zero re-applied side effects, the captured
 *     return-focus target untouched. This covers `.close()`, `.toggle()`,
 *     backdrop-click, and Escape (all of which call `.close()`).
 *     `attributeChangedCallback`'s own guard check is only a backstop for
 *     a bare `el.removeAttribute('open')` that bypasses `.close()`
 *     entirely; that backstop path can't prevent the mutation (it already
 *     happened by the time the reaction runs), so it undoes it instead,
 *     `_vetoRestoring`-guarded so the undo applies no side effects and
 *     fires no events either. Either way there is no bypass, matching the
 *     "fail closed" framing for this drawer's safety-adjacent content.
 *     Single-active-under-veto policy: `open()` refuses to open (a true
 *     no-op, `false` returned) if any other currently-open drawer's guard
 *     would refuse to close — opening one drawer never force-closes
 *     another that has unsaved changes. The same guard also blocks
 *     switching tabs, mirroring old web_terminal/drawer.js exactly.
 *   - Resizable (opt-in via the boolean `resizable` attribute, checked once
 *     at connect — not reactive, matching how old web_terminal/drawer.js's
 *     `initDrawerResize()` wired whatever markup existed at page-init):
 *     inert unless both the attribute is present AND a `.drawer-resize-handle`
 *     descendant exists to wire. The drag itself is the design system's
 *     shared splitter — the same module behind every other resizable pane in
 *     the product — configured `anchor: 'end'` (the drawer is right-anchored,
 *     so dragging left widens it) and `sizing: 'box'` (it is `position: fixed`,
 *     not a flex item). It persists to the exact `localStorage` key old
 *     web_terminal/drawer.js used (`osprey-drawer-width`), and the 320px /
 *     90vw clamp is applied uniformly on both restore and drag, with the
 *     ceiling resolved per clamp so a window resize cannot leave it stale. A
 *     drag left in progress when the drawer closes or disconnects is discarded
 *     WITHOUT persisting: an in-progress, never-released drag was never a
 *     deliberate choice of width, and persisting it would silently corrupt the
 *     next restore.
 *   - Tabbed panels: switching updates `.drawer-tab`/`.drawer-tab-panel`
 *     `.active` classes exactly as old web_terminal/drawer.js did (own CSS
 *     is unaffected — no attribute-based selectors introduced for this).
 *     Gated purely by the presence of `.drawer-tab` markup, not an
 *     attribute — ariel has none, so this is naturally inert there.
 *   - Not carried over from old web_terminal/drawer.js: marking the
 *     `[data-drawer]` trigger button `.active` while its drawer is open.
 *     That was a page-level visual nicety with no external listener
 *     depending on it (unlike the events above); the interface that owns
 *     the trigger button can now do it itself by listening for
 *     `drawer:open`/`drawer:close` on the host, which decouples that
 *     concern from this shared component.
 *
 * @module components/osprey-drawer
 */

// ---- Module state ----

import { initSplitter } from '/design-system/js/splitter.js';

const BACKDROP_ID = 'drawer-backdrop';

// Guards the document-level delegated handlers below so they are installed
// exactly once no matter how many <osprey-drawer> instances connect.
let _delegatedHandlersInstalled = false;

// Fallback suffix counter for _resolveTitleId, only ever used when a host
// has no id of its own to derive a deterministic one from.
let _titleIdCounter = 0;

// Elements this component itself marked inert while a drawer was open, so
// closing only ever restores what this component touched.
/** @type {Set<HTMLElement>} */
const _inertedByUs = new Set();

const FOCUSABLE_SELECTOR = [
  'a[href]',
  'button:not([disabled])',
  'textarea:not([disabled])',
  'input:not([disabled])',
  'select:not([disabled])',
  '[tabindex]:not([tabindex="-1"])',
  '[contenteditable="true"]',
].join(',');

// `resizable` width persistence: key and bounds reused verbatim from old
// web_terminal/static/js/drawer.js's initDrawerResize()/_doOpenDrawer().
const DRAWER_WIDTH_STORAGE_KEY = 'osprey-drawer-width';
const DRAWER_WIDTH_MIN_PX = 320;
const DRAWER_WIDTH_MAX_VIEWPORT_FRACTION = 0.9;

// Clamping now lives in the shared splitter, which resolves both bounds on
// every clamp rather than capturing them — the 90vw ceiling moves with the
// window, and a value captured at construction would go stale.

// ---- Public API ----

/**
 * Light-DOM drawer element. See the module docstring above for the full
 * authoring contract.
 */
export class OspreyDrawer extends HTMLElement {
  static get observedAttributes() {
    return ['open'];
  }

  constructor() {
    super();
    /** Element to refocus on close, captured when this drawer opened. */
    /** @type {Element|null} */
    this._returnFocusTarget = null;
    /** The Tab-trap keydown listener while installed, else null. */
    /** @type {((event: KeyboardEvent) => void)|null} */
    this._focusTrapHandler = null;
    /** Registered unsaved-changes guards; all must return true to proceed. */
    /** @type {Array<() => boolean>} */
    this._unsavedGuards = [];
    /** The shared splitter driving the resize handle, once `_wireResizable` finds one. */
    /** @type {import('../splitter.js').SplitterApi|null} */
    this._splitter = null;
    /** True only inside the synchronous toggleAttribute() call that close() makes after its own guard check already passed -- tells attributeChangedCallback the closing branch's guard check was already done, so it isn't repeated. */
    /** @type {boolean} */
    this._committingClose = false;
    /** Same idea as `_committingClose`, for open()'s own pre-validated exclusivity check. */
    /** @type {boolean} */
    this._committingOpen = false;
    /** True only inside a backstop veto's own attribute mutation, so that mutation's re-entrant attributeChangedCallback call is recognized as an echo of itself and does nothing (no cascade). */
    /** @type {boolean} */
    this._vetoRestoring = false;
  }

  connectedCallback() {
    this._ensureStaticAria();
    this._wireResizable();
    // A custom element's connectedCallback can fire before its own
    // light-DOM children (e.g. .drawer-title, .drawer-resize-handle) have
    // been parsed and appended, depending on how the parser streams the
    // subtree in. Retry once microtasks flush -- by then the whole subtree
    // is guaranteed present -- so aria-labelledby resolves and resizable
    // wires even for a drawer that's connected in one shot. Both retried
    // methods are idempotent, so this is a no-op wherever the first
    // attempt already succeeded.
    queueMicrotask(() => {
      this._ensureStaticAria();
      this._wireResizable();
    });
    _installDelegatedHandlersOnce();
    if (this.hasAttribute('open')) this._applyOpenSideEffects();
  }

  disconnectedCallback() {
    this._unwireResizable();
    // Removing a still-open drawer from the DOM (without a close() call)
    // doesn't change its `open` attribute, so attributeChangedCallback never
    // fires. Apply the closed-state side effects explicitly (this includes
    // removing the focus trap) or the backdrop/background-inert/focus-trap
    // would keep reflecting a drawer that no longer exists. The focus trap
    // can only ever be installed while `open` is set, so this is the only
    // path that needs to run.
    if (this.hasAttribute('open')) this._applyClosedSideEffects();
  }

  /**
   * @param {string} name
   * @param {string|null} oldValue
   * @param {string|null} newValue
   */
  attributeChangedCallback(name, oldValue, newValue) {
    if (name !== 'open') return;
    // A backstop veto (below) mutates the attribute right back to undo an
    // invalid transition. That mutation re-enters this same callback,
    // synchronously, before the veto's own call returns. Recognize that
    // echo and do nothing: the state never really changed, so there is
    // nothing further to apply.
    if (this._vetoRestoring) return;

    if (newValue === null) {
      if (this._committingClose) {
        // close() already checked the guard before making this mutation --
        // see close() below. Trust it; do not invoke the guard a second
        // time for the same close attempt.
        this._committingClose = false;
        this._applyClosedSideEffects();
        return;
      }
      // Reached only via a bypass that skipped close()'s own guard check
      // (e.g. a bare `el.removeAttribute('open')` called directly, rather
      // than through close()/toggle()/backdrop/Escape, all of which call
      // close()). Back-stop check, fail closed: if blocked, put the
      // attribute back without cascading -- no _applyClosedSideEffects,
      // no drawer:close, because nothing about this drawer's open state
      // ever really changed.
      if (!this._checkUnsavedChangesSafely()) {
        this._vetoRestoring = true;
        try {
          this.setAttribute('open', '');
        } finally {
          this._vetoRestoring = false;
        }
        return;
      }
      this._applyClosedSideEffects();
      return;
    }

    if (this._committingOpen) {
      // open() already pre-validated exclusivity before making this
      // mutation -- see open() below.
      this._committingOpen = false;
      this._commitOpen();
      return;
    }
    // Bypass path, mirroring the closing branch above (e.g. a bare
    // `el.setAttribute('open', '')`/`toggleAttribute('open', true)`).
    // Fail closed (rubric B): refuse to open while any other open
    // drawer's guard would refuse to yield -- see the module docstring's
    // single-active-under-veto policy. Undo without cascading: this
    // drawer never really opens.
    const blocked = _openDrawers().some(
      (other) => other !== this && !other._checkUnsavedChangesSafely()
    );
    if (blocked) {
      this._vetoRestoring = true;
      try {
        this.toggleAttribute('open', false);
      } finally {
        this._vetoRestoring = false;
      }
      return;
    }
    this._commitOpen();
  }

  /**
   * Close every other open drawer (already guard-validated by the caller,
   * either open()'s pre-check or the bypass branch above) and apply this
   * drawer's own open side effects.
   */
  _commitOpen() {
    for (const other of _openDrawers()) {
      if (other !== this) other.close();
    }
    this._applyOpenSideEffects();
  }

  /**
   * Open this drawer. Idempotent — a no-op (returning true) if already
   * open. Refused — a true no-op for this drawer, `false` returned,
   * nothing about it ever appears open — if any other currently-open
   * drawer's unsaved-changes guard would refuse to close (fail closed;
   * see the module docstring's single-active-under-veto policy).
   * @returns {boolean} true if this drawer ended up open.
   */
  open() {
    if (this.hasAttribute('open')) return true;
    const blockingOther = _openDrawers().find((other) => !other._checkUnsavedChangesSafely());
    if (blockingOther) return false;
    this._committingOpen = true;
    this.toggleAttribute('open', true);
    return this.hasAttribute('open');
  }

  /**
   * Close this drawer. Idempotent — a no-op (returning true) if already
   * closed. Blocked by a registered unsaved-changes guard returning false
   * (or throwing, treated as blocking — fail closed), in which case this
   * is a true no-op: the attribute is never touched, so no cascade of side
   * effects or events ever fires, and this returns `false` (mirrors old
   * web_terminal/drawer.js's `closeDrawer()` return contract).
   * @returns {boolean} true if the drawer ended up closed.
   */
  close() {
    if (!this.hasAttribute('open')) return true;
    if (!this._checkUnsavedChangesSafely()) return false;
    this._committingClose = true;
    this.toggleAttribute('open', false);
    return !this.hasAttribute('open');
  }

  /** Toggle this drawer's open state. Subject to the same guard as open()/close(). */
  toggle() {
    return this.hasAttribute('open') ? this.close() : this.open();
  }

  // ---- Dialog semantics (set once; describe what the element is) ----

  /**
   * Set the static dialog role/semantics exactly once each (idempotent —
   * every one of these is guarded, so none ever overwrites a value a
   * consumer already supplied).
   */
  _ensureStaticAria() {
    if (!this.hasAttribute('role')) this.setAttribute('role', 'dialog');
    if (!this.hasAttribute('aria-modal')) this.setAttribute('aria-modal', 'true');
    if (!this.hasAttribute('tabindex')) this.setAttribute('tabindex', '-1');
    if (!this.hasAttribute('aria-labelledby')) {
      const titleId = this._resolveTitleId();
      if (titleId) this.setAttribute('aria-labelledby', titleId);
    }
  }

  /**
   * Resolve (and assign, if missing) the id of this drawer's title element,
   * so `aria-labelledby` has a resolvable accessible name. Resolution order:
   *   1. The first `.drawer-title` descendant — the convention both of the
   *      pinned drawers (ariel, web_terminal) already use for their header.
   *   2. None found — `aria-labelledby` is simply omitted; the dialog still
   *      functions, just without a resolvable accessible name. Acceptable
   *      degradation for a generic base component used outside the two
   *      pinned interfaces.
   * When a title element is found without its own id, one is assigned
   * deterministically: `${this.id}-title` when this drawer has an id (both
   * pinned drawers do), else a module-unique fallback.
   * @returns {string|null}
   */
  _resolveTitleId() {
    const title = this.querySelector('.drawer-title');
    if (!title) return null;
    if (!title.id) {
      title.id = this.id ? `${this.id}-title` : `osprey-drawer-title-${++_titleIdCounter}`;
    }
    return title.id;
  }

  // ---- Focus trap ----

  /** Every focusable, actually-rendered descendant, in document order. */
  _focusableElements() {
    return Array.from(this.querySelectorAll(FOCUSABLE_SELECTOR)).filter(
      /** @returns {el is HTMLElement} */
      (el) => el instanceof HTMLElement && el.offsetParent !== null
    );
  }

  /** Move focus to the first focusable descendant, or the host as a fallback. */
  _moveFocusIn() {
    const [first] = this._focusableElements();
    if (first) {
      first.focus();
    } else {
      this.focus();
    }
  }

  /** Install the Tab/Shift+Tab wraparound. Idempotent — guarded. */
  _installFocusTrap() {
    if (this._focusTrapHandler) return;
    this._focusTrapHandler = (event) => {
      if (event.key !== 'Tab') return;
      const focusable = this._focusableElements();
      if (focusable.length === 0) {
        event.preventDefault();
        return;
      }
      const first = focusable[0];
      const last = focusable[focusable.length - 1];
      const current = document.activeElement;
      const wrapsBackward = event.shiftKey && (current === first || !this.contains(current));
      const wrapsForward = !event.shiftKey && (current === last || !this.contains(current));
      if (wrapsBackward) {
        event.preventDefault();
        last.focus();
      } else if (wrapsForward) {
        event.preventDefault();
        first.focus();
      }
    };
    this.addEventListener('keydown', this._focusTrapHandler);
  }

  /** Remove the Tab/Shift+Tab wraparound. Idempotent — a no-op if absent. */
  _removeFocusTrap() {
    if (!this._focusTrapHandler) return;
    this.removeEventListener('keydown', this._focusTrapHandler);
    this._focusTrapHandler = null;
  }

  // ---- Unsaved-changes guard (opt-in via registerUnsavedGuard) ----

  /**
   * Register an unsaved-changes guard. A guard returning false blocks
   * every path that removes the `open` attribute (see the module
   * docstring's guard-veto note) and blocks switching away from the
   * active tab. Multiple guards may be registered; all must return true
   * to proceed.
   * @param {() => boolean} guardFn
   */
  registerUnsavedGuard(guardFn) {
    this._unsavedGuards.push(guardFn);
  }

  /**
   * Whether it is safe to close this drawer or switch its active tab: true
   * if every registered guard returns true, or if none are registered.
   * @returns {boolean}
   */
  _checkUnsavedChanges() {
    return this._unsavedGuards.every((guard) => guard());
  }

  /**
   * Like `_checkUnsavedChanges()`, but a guard that throws is treated as
   * "unsaved changes present" (fail closed) instead of letting the
   * exception propagate mid-transition and leave this drawer (or, via
   * exclusivity, another drawer) in a half-changed state. Logged so a
   * buggy guard is still debuggable; the drawer itself stays fully usable.
   * @returns {boolean}
   */
  _checkUnsavedChangesSafely() {
    try {
      return this._checkUnsavedChanges();
    } catch (error) {
      console.error(
        'osprey-drawer: an unsaved-changes guard threw; treating as unsaved changes present (fail closed)',
        error
      );
      return false;
    }
  }

  // ---- Tabbed panels (opt-in purely by `.drawer-tab` markup presence) ----

  /**
   * Switch to a different tab panel, mirroring old web_terminal/drawer.js's
   * initDrawerTabs click handler exactly: a no-op if `tab` is already
   * active or its target panel can't be resolved; blocked by an unsaved-
   * changes guard; deactivates every other tab/panel in this drawer and
   * activates the target; fires `drawer:tab-activate` on the newly active
   * panel.
   * @param {HTMLElement} tab
   */
  _activateTab(tab) {
    if (tab.classList.contains('active')) return;
    const targetId = tab.dataset.tab;
    if (!targetId) return;
    const targetPanel = document.getElementById(targetId);
    if (!targetPanel) return;
    if (!this._checkUnsavedChangesSafely()) return;

    this.querySelectorAll('.drawer-tab').forEach((t) => t.classList.remove('active'));
    this.querySelectorAll('.drawer-tab-panel').forEach((p) => p.classList.remove('active'));

    tab.classList.add('active');
    targetPanel.classList.add('active');

    targetPanel.dispatchEvent(new CustomEvent('drawer:tab-activate', { bubbles: true, composed: true }));
  }

  // ---- Resizable (opt-in via the `resizable` attribute) ----

  /**
   * Wire the resize handle if, and only if, the `resizable` attribute is
   * present AND a `.drawer-resize-handle` descendant exists to wire. Checked
   * once here (connect time, retried once on the connectedCallback microtask
   * and again on open — see those callers) rather than reactively. Idempotent:
   * a no-op once already wired, and a complete no-op when `resizable` is
   * absent — no handle lookup side effect, no listeners.
   *
   * The drag is the design system's shared splitter, the same module behind
   * every other resizable pane in the product. Three of its options carry what
   * is specific to a drawer:
   *
   * - `anchor: 'end'` — the drawer is right-anchored, so dragging *left*
   *   widens it;
   * - `sizing: 'box'` — the drawer is `position: fixed`, not a flex item, so a
   *   flex-basis on it would write nothing observable;
   * - a function-valued `max` — the ceiling is 90vw, which moves when the
   *   window resizes, so a number captured at construction would go stale and
   *   let the drawer be dragged past its limit.
   */
  _wireResizable() {
    if (this._splitter) return;
    if (!this.hasAttribute('resizable')) return;
    const handle = this.querySelector('.drawer-resize-handle');
    if (!(handle instanceof HTMLElement)) return;
    this._splitter = initSplitter({
      handle,
      pane: this,
      storageKey: DRAWER_WIDTH_STORAGE_KEY,
      axis: 'x',
      anchor: 'end',
      sizing: 'box',
      min: () => DRAWER_WIDTH_MIN_PX,
      max: () => window.innerWidth * DRAWER_WIDTH_MAX_VIEWPORT_FRACTION,
      // A drawer already has a close button, so "collapse" here means snap
      // back to the narrowest useful width rather than vanish.
      collapsedSize: DRAWER_WIDTH_MIN_PX,
    });
  }

  /**
   * Undo whatever _wireResizable installed. Idempotent — safe unconditionally.
   *
   * `destroy()` deliberately does not persist: a drawer closed or disconnected
   * mid-drag was never a deliberate choice of width, and persisting it would
   * corrupt the next restore. It also clears `data-splitter-dragging`, which
   * would otherwise suppress the drawer's own close transition.
   */
  _unwireResizable() {
    if (this._splitter) this._splitter.destroy();
    this._splitter = null;
  }

  /**
   * Apply the persisted width, clamped to the current bounds. A Storage read
   * that throws (e.g. disabled storage) is swallowed inside the splitter and
   * falls back to the drawer's default width, because the rest of open's side
   * effects (inert, focus trap, drawer:open) must still run.
   */
  _restorePersistedWidth() {
    if (this._splitter) this._splitter.restoreSize();
  }

  // ---- Open/close side effects (the single source of truth) ----

  /** Everything that must happen while this drawer is open. */
  _applyOpenSideEffects() {
    this._returnFocusTarget = document.activeElement;
    // Retry static-aria resolution and resizable wiring: connectedCallback
    // can, depending on parser/insertion timing, run before this drawer's
    // projected light-DOM content (.drawer-title, .drawer-resize-handle)
    // exists yet, and open() can in principle be called before the
    // connectedCallback microtask retry fires. Both retried methods are
    // idempotent, so this only ever does work when an earlier attempt
    // found nothing yet.
    this._ensureStaticAria();
    this._wireResizable();
    this._restorePersistedWidth();
    _syncGlobalOpenState();
    this._installFocusTrap();
    this._moveFocusIn();
    this.dispatchEvent(new CustomEvent('drawer:open', { bubbles: true, composed: true }));
    const activePanel = this.querySelector('.drawer-tab-panel.active');
    if (activePanel) {
      activePanel.dispatchEvent(new CustomEvent('drawer:tab-activate', { bubbles: true, composed: true }));
    }
  }

  /** Everything that must happen once this drawer is closed. */
  _applyClosedSideEffects() {
    this._removeFocusTrap();
    // Tear the splitter down rather than merely stopping a drag: destroy()
    // discards an unreleased drag without persisting it and clears
    // `data-splitter-dragging`, which would otherwise suppress this drawer's
    // own close transition. _applyOpenSideEffects re-wires on the way back in.
    this._unwireResizable();
    _syncGlobalOpenState();
    const target = this._returnFocusTarget;
    this._returnFocusTarget = null;
    if (target instanceof HTMLElement && document.contains(target)) {
      target.focus();
    }
    this.dispatchEvent(new CustomEvent('drawer:close', { bubbles: true, composed: true }));
  }
}

// ---- Backdrop + exclusivity helpers ----

/** @returns {HTMLElement|null} */
function _backdrop() {
  return document.getElementById(BACKDROP_ID);
}

/**
 * @param {Element} el
 * @returns {el is OspreyDrawer}
 */
function _isOspreyDrawer(el) {
  return el instanceof OspreyDrawer;
}

/** Every currently-open, connected `<osprey-drawer>`. */
function _openDrawers() {
  return Array.from(document.querySelectorAll('osprey-drawer[open]')).filter(_isOspreyDrawer);
}

/** Reflect whether any drawer is open onto the shared backdrop element. */
function _syncBackdrop() {
  const backdrop = _backdrop();
  if (!backdrop) return;
  backdrop.toggleAttribute('open', _openDrawers().length > 0);
}

/**
 * Reflect whether any drawer is open onto the rest of the page: every
 * top-level sibling of `<body>` other than a `<osprey-drawer>` itself or the
 * shared backdrop becomes `inert` (+ `aria-hidden` fallback) while a drawer
 * is open, and exactly the elements this function inerted are restored once
 * none are.
 */
function _syncBackgroundInert() {
  const backdrop = _backdrop();

  if (_openDrawers().length > 0) {
    for (const child of Array.from(document.body.children)) {
      if (!(child instanceof HTMLElement)) continue;
      if (child instanceof OspreyDrawer || child === backdrop) continue;
      if (_inertedByUs.has(child)) continue;
      child.toggleAttribute('inert', true);
      child.setAttribute('aria-hidden', 'true');
      _inertedByUs.add(child);
    }
  } else {
    for (const child of _inertedByUs) {
      child.toggleAttribute('inert', false);
      child.removeAttribute('aria-hidden');
    }
    _inertedByUs.clear();
  }
}

/** Recompute both of the "is any drawer open" side effects together. */
function _syncGlobalOpenState() {
  _syncBackdrop();
  _syncBackgroundInert();
}

/**
 * Close every currently open drawer. Exclusivity keeps this to at most one
 * in practice, but this tolerates markup that starts with more than one
 * `open` attribute present.
 */
function _closeAllOpenDrawers() {
  for (const drawer of _openDrawers()) drawer.close();
}

// ---- Delegated document-level handlers (installed once) ----

/**
 * Install the document-level delegated handlers exactly once: `[data-drawer]`
 * triggers, `.drawer-close-btn`, `.drawer-tab`, backdrop click, and Escape —
 * regardless of how many `<osprey-drawer>` instances exist or
 * connect/disconnect. `.drawer-tab` clicks are handled here too (rather
 * than a per-instance listener) for the same reason as the others: one
 * delegated handler regardless of instance count. Ariel has no
 * `.drawer-tab` elements, so this branch simply never matches there.
 */
function _installDelegatedHandlersOnce() {
  if (_delegatedHandlersInstalled) return;
  _delegatedHandlersInstalled = true;

  document.addEventListener('click', (event) => {
    if (!(event.target instanceof Element)) return;

    const trigger = event.target.closest('[data-drawer]');
    if (trigger instanceof HTMLElement && trigger.dataset.drawer) {
      const target = document.getElementById(trigger.dataset.drawer);
      if (target instanceof OspreyDrawer) target.toggle();
      return;
    }

    const closeBtn = event.target.closest('.drawer-close-btn');
    if (closeBtn) {
      const host = closeBtn.closest('osprey-drawer');
      if (host instanceof OspreyDrawer) host.close();
      return;
    }

    const tab = event.target.closest('.drawer-tab');
    if (tab instanceof HTMLElement) {
      const host = tab.closest('osprey-drawer');
      if (host instanceof OspreyDrawer) host._activateTab(tab);
      return;
    }

    if (event.target.closest('.drawer-backdrop')) {
      _closeAllOpenDrawers();
    }
  });

  document.addEventListener('keydown', (event) => {
    if (event.key !== 'Escape') return;
    if (_openDrawers().length === 0) return;
    event.preventDefault();
    _closeAllOpenDrawers();
  });
}

// ---- Registration ----

if (!customElements.get('osprey-drawer')) {
  customElements.define('osprey-drawer', OspreyDrawer);
}
