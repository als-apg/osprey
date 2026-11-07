// @ts-check
/* OSPREY Design System — shared context menu.
 *
 * A small cursor-anchored popover listing verbs in words: the right-click menu
 * every OSPREY surface opens. It began as the web terminal's rail menu
 * (web_terminal/static/js/panel-context-menu.js, now a thin wrapper over this
 * module) and moved here for the same reason the pane splitter did: the panels
 * are served by different processes, they cannot import each other's code, and
 * `/design-system/` is the one path all of them resolve. One definition keeps
 * the dismissal and focus behaviour identical everywhere.
 *
 * Purely presentational: it owns the popover DOM, open/close, dismissal, focus
 * bookkeeping and viewport clamping. It holds no state of its callers and
 * issues no fetches — the caller passes fully-bound `run` closures, so this
 * module never knows what a verb does. One menu at a time: opening replaces
 * any open menu.
 *
 * The keyboard path is the caller's: macOS fires no `contextmenu` event from
 * the menu key or Shift+F10, so a surface that wants keyboard access handles
 * those keydowns itself and calls the same opener with the element's rect.
 *
 * Dismissal contract — the menu closes on:
 *   - pointerdown outside it (capture phase, so the press dismisses first);
 *     a press on the `triggerEl` that opened it is exempt, so that button's
 *     click can toggle the menu closed (see {@link openContextMenu}),
 *   - Escape or Tab,
 *   - running an item (the menu closes BEFORE the action),
 *   - window `blur` — a click into an iframe never reaches this document,
 *   - window `resize`,
 *   - a scroll that actually moves the anchor: capture-phase `scroll` whose
 *     target is the document or an ancestor of `anchorEl`. Scoping matters —
 *     an unscoped listener would dismiss on any inner scroller's own scrolls,
 *     the terminal's streaming output above all.
 *
 * Documents — the menu opens in the anchor's document (`document` with no
 * anchor), which may be a window of its own (the web terminal's detached panes):
 * it mounts on that body, clamps to that viewport, and hears dismissal on that
 * document and its window. The press and scroll targets may belong to that
 * window's realm, so nothing here tests them with `instanceof Node`.
 *
 * Focus contract — opening records the document's `activeElement` and focuses the
 * first enabled row. That focus is handed back only on the KEYBOARD dismissals
 * (Escape/Tab) and on close-before-run, where it is restored before the action
 * runs so an action that moves focus itself still wins. Pointer, blur, scroll
 * and resize dismissals never move focus: blur in particular fires AFTER focus
 * entered a clicked iframe, and restoring would yank it straight back out.
 *
 * DOM contract, for the default class prefix `osprey-context` (base.css skins
 * it; a caller passing `classPrefix` gets the same structure under its own
 * names, which is how the web terminal keeps its `rail-context-*` skin):
 *
 *   <div class="osprey-context-menu" role="menu" aria-label="…">
 *     <div class="osprey-context-heading">Columns</div>
 *     <button class="osprey-context-item" role="menuitem" type="button">
 *       <span class="osprey-context-label">Only section <code>SR01C</code></span>
 *       <span class="osprey-context-glyph" aria-hidden="true">⎘</span>
 *     </button>
 *     <button class="osprey-context-item checked" role="menuitemcheckbox"
 *             aria-checked="true">…<span class="osprey-context-glyph">✓</span></button>
 *     <div class="osprey-context-divider" role="separator"></div>
 *     <button class="osprey-context-item danger" …>
 *   </div>
 */

/**
 * One menu row. A `divider: true` entry renders a separator and a `heading`
 * entry a caption; their other fields are ignored.
 * @typedef {object} MenuItem
 * @property {boolean} [divider]  - render a separator, not a button
 * @property {string} [heading]   - render an inert caption over the rows below it
 * @property {string} [label]     - the verb, in words
 * @property {string} [code]      - a value the verb acts on, drawn monospace after the label
 * @property {string} [glyph]     - trailing glyph tying the row to the surface's iconography
 * @property {boolean} [checked]  - a toggle row: true draws ✓, false draws nothing; undefined is a plain verb
 * @property {boolean} [disabled] - render the row inert (aria-disabled, no run)
 * @property {string} [tip]       - a hover tip, e.g. why a disabled row is disabled
 * @property {boolean} [danger]   - destructive verb — hover styles error-red
 * @property {() => void} [run]   - the caller-bound action; the menu closes first
 */

/**
 * @typedef {object} OpenOptions
 * @property {number} x - viewport x of the cursor (or of the element, from the keyboard)
 * @property {number} y - viewport y
 * @property {HTMLElement | null} [anchorEl] - the surface the menu belongs to; scopes
 *   the scroll dismissal (omitting it leaves only document scrolls)
 * @property {HTMLElement | null} [triggerEl] - the button that opened the menu (a ⋯).
 *   The menu then sits under it, right edges aligned, instead of at x/y; the
 *   button carries `aria-expanded` while the menu is open; a press on it
 *   does not count as an outside press, and opening again from the same
 *   button while its menu is open closes the menu instead (a toggle).
 * @property {string} ariaLabel - the menu's accessible name
 * @property {MenuItem[]} items - the rows, in order
 * @property {string} [classPrefix] - class-name stem, `osprey-context` by default
 */

const DEFAULT_PREFIX = 'osprey-context';

/** Space between a trigger button and the menu hanging under it, in px. */
const TRIGGER_GAP = 4;

/** @type {HTMLElement | null} the single open menu, or null */
let menuEl = null;

/** @type {string} the class stem of the open menu */
let prefix = DEFAULT_PREFIX;

/** @type {HTMLElement | null} the surface the menu belongs to (scroll scoping) */
let anchorEl = null;

/** @type {HTMLElement | null} the button the open menu hangs under, if any */
let triggerEl = null;

/** @type {HTMLElement | null} what had focus when the menu opened */
let previousFocus = null;

/** @type {Document | null} the document the open menu lives in */
let ownerDoc = null;

/** @type {(Window & typeof globalThis) | null} that document's window */
let ownerView = null;

/**
 * @param {unknown} target
 * @returns {target is Node} whichever realm the node belongs to
 */
function isNode(target) {
  return typeof (/** @type {any} */ (target)?.nodeType) === 'number';
}

/** @returns {string} selector for the open menu's enabled rows */
function enabledRows() {
  return `.${prefix}-item:not([aria-disabled="true"])`;
}

/** @param {MouseEvent} e */
function onDocPointerDown(e) {
  if (!menuEl || !isNode(e.target) || menuEl.contains(e.target)) return;
  // The trigger's own press is the toggle's job (see openContextMenu), not an
  // outside press: dismissing here would reopen the menu on the click.
  if (triggerEl && triggerEl.contains(e.target)) return;
  dismiss(false);
}

/** @param {KeyboardEvent} e */
function onDocKeydown(e) {
  if (!menuEl) return;
  // Escape and Tab both leave the menu by keyboard, so both hand focus back.
  // Tab's default is cancelled: the operator resumes tabbing from where they
  // were, not from wherever a removed menu row left the sequence.
  if (e.key === 'Escape' || e.key === 'Tab') {
    e.preventDefault();
    e.stopPropagation();
    dismiss(true);
    return;
  }
  // Arrow navigation over the enabled rows — the menu's only focusables.
  if (e.key === 'ArrowDown' || e.key === 'ArrowUp') {
    e.preventDefault();
    e.stopPropagation();
    const rows = /** @type {HTMLElement[]} */ ([...menuEl.querySelectorAll(enabledRows())]);
    if (!rows.length) return;
    const idx = rows.indexOf(/** @type {HTMLElement} */ (ownerDoc?.activeElement));
    const next =
      e.key === 'ArrowDown'
        ? rows[(idx + 1 + rows.length) % rows.length]
        : rows[(idx - 1 + rows.length) % rows.length];
    next.focus();
  }
}

/** The window lost focus — usually a click landing inside an iframe. */
function onWindowBlur() {
  dismiss(false);
}

/** A resize moves every anchor; a cursor-anchored popover cannot follow. */
function onWindowResize() {
  dismiss(false);
}

/**
 * Close only when the scroll actually moved the menu's anchor: the document
 * scrolled, or a container holding the anchor did.
 * @param {Event} e
 */
function onDocScroll(e) {
  if (!menuEl) return;
  if (e.target === ownerDoc) {
    dismiss(false);
    return;
  }
  if (anchorEl && isNode(e.target) && e.target.contains(anchorEl)) dismiss(false);
}

/**
 * Tear down the open menu, optionally handing focus back to whatever held it
 * when the menu opened.
 * @param {boolean} restoreFocus
 */
function dismiss(restoreFocus) {
  if (!menuEl) return;
  menuEl.remove();
  menuEl = null;
  anchorEl = null;
  triggerEl?.setAttribute('aria-expanded', 'false');
  triggerEl = null;
  const restoreTo = previousFocus;
  previousFocus = null;
  ownerDoc?.removeEventListener('pointerdown', onDocPointerDown, true);
  ownerDoc?.removeEventListener('keydown', onDocKeydown, true);
  ownerDoc?.removeEventListener('scroll', onDocScroll, true);
  ownerView?.removeEventListener('blur', onWindowBlur);
  ownerView?.removeEventListener('resize', onWindowResize);
  ownerDoc = null;
  ownerView = null;
  if (restoreFocus && restoreTo && restoreTo.isConnected) restoreTo.focus();
}

/**
 * Close the open menu, if any, without moving focus: the surface it belongs to is
 * going away.
 * @returns {void}
 */
export function closeContextMenu() {
  dismiss(false);
}

/**
 * Build one row element.
 * @param {MenuItem} item
 * @returns {HTMLElement}
 */
function buildRow(item) {
  if (item.divider) {
    const hr = document.createElement('div');
    hr.className = `${prefix}-divider`;
    hr.setAttribute('role', 'separator');
    return hr;
  }
  if (item.heading !== undefined) {
    const caption = document.createElement('div');
    caption.className = `${prefix}-heading`;
    caption.setAttribute('role', 'presentation');
    caption.textContent = item.heading;
    return caption;
  }

  const toggle = item.checked !== undefined;
  const btn = document.createElement('button');
  btn.type = 'button';
  btn.className =
    `${prefix}-item` + (item.danger ? ' danger' : '') + (item.checked ? ' checked' : '');
  btn.setAttribute('role', toggle ? 'menuitemcheckbox' : 'menuitem');
  if (toggle) btn.setAttribute('aria-checked', String(Boolean(item.checked)));

  // Text nodes only: labels and values routinely carry server-supplied names.
  const label = document.createElement('span');
  label.className = `${prefix}-label`;
  label.textContent = item.label ?? '';
  if (item.code) {
    label.append(' ');
    const code = document.createElement('code');
    code.textContent = item.code;
    label.appendChild(code);
  }
  btn.appendChild(label);

  const glyphText = toggle ? (item.checked ? '✓' : '') : (item.glyph ?? '');
  if (glyphText) {
    const glyph = document.createElement('span');
    glyph.className = `${prefix}-glyph`;
    glyph.setAttribute('aria-hidden', 'true');
    glyph.textContent = glyphText;
    btn.appendChild(glyph);
  }

  if (item.tip) btn.title = item.tip;
  if (item.disabled) {
    btn.setAttribute('aria-disabled', 'true');
  } else {
    const run = item.run;
    // Close BEFORE running: an action that moves focus must not race the
    // dismissal. Focus goes back first for the same reason — whatever the
    // action does with focus lands last and therefore wins.
    btn.addEventListener('click', () => {
      dismiss(true);
      run?.();
    });
  }
  return btn;
}

/**
 * Open the context menu at a viewport position, replacing any open menu.
 *
 * Every child is assembled through DOM APIs (never innerHTML). After insertion
 * the menu is clamped so it never overflows the viewport (opened near an edge
 * it flips inward), and the first enabled row takes focus. With `triggerEl`
 * the menu hangs under that button instead, right edges aligned, and a
 * second open from the same button closes it.
 *
 * @param {OpenOptions} opts
 * @returns {void}
 */
export function openContextMenu(opts) {
  const { ariaLabel, items } = opts;
  if (menuEl && opts.triggerEl && opts.triggerEl === triggerEl) {
    dismiss(false);
    return;
  }
  // Capture the focus to hand back BEFORE tearing down any open menu — when
  // one menu replaces another the active element is the old menu's row, so
  // the earlier capture is the one worth keeping.
  const doc = (opts.anchorEl ?? opts.triggerEl)?.ownerDocument ?? document;
  const view = /** @type {Window & typeof globalThis} */ (doc.defaultView ?? window);
  const active = (ownerDoc ?? doc).activeElement;
  const held = menuEl && isNode(active) && menuEl.contains(active) ? previousFocus : active;
  dismiss(false);
  // Not `instanceof HTMLElement`: the active element may be of a window's realm.
  previousFocus = /** @type {HTMLElement | null} */ (held);
  anchorEl = opts.anchorEl ?? null;
  triggerEl = opts.triggerEl ?? null;
  ownerDoc = doc;
  ownerView = view;
  prefix = opts.classPrefix || DEFAULT_PREFIX;

  const menu = document.createElement('div');
  menu.className = `${prefix}-menu`;
  menu.setAttribute('role', 'menu');
  menu.setAttribute('aria-label', ariaLabel);
  // A right-click on the popover itself (or the menu key on a focused row)
  // must not stack the browser's native menu on top of this one.
  menu.addEventListener('contextmenu', (e) => e.preventDefault());

  for (const item of items) menu.appendChild(buildRow(item));

  doc.body.appendChild(menu);
  menuEl = menu;
  place(menu, opts, view);
  triggerEl?.setAttribute('aria-expanded', 'true');

  // Capture-phase pointerdown (not click) so a drag started outside also
  // dismisses, and before the outside target acts on the press. Scroll is
  // capture-phase because element scrolls do not bubble.
  doc.addEventListener('pointerdown', onDocPointerDown, true);
  doc.addEventListener('keydown', onDocKeydown, true);
  doc.addEventListener('scroll', onDocScroll, true);
  view.addEventListener('blur', onWindowBlur);
  view.addEventListener('resize', onWindowResize);

  const first = /** @type {HTMLElement | null} */ (menu.querySelector(enabledRows()));
  first?.focus();
}

/**
 * Position the inserted menu. At the cursor: clamped into the viewport, so a
 * menu opened near the right or bottom edge flips to the cursor's other side
 * rather than overflowing. Under a trigger button: right edges aligned, a
 * small gap below it, flipped above it when the bottom edge is short of
 * room. Measured after insertion, at x/y, so the menu's own size is known.
 * @param {HTMLElement} menu
 * @param {OpenOptions} opts
 * @param {Window} view  the window the menu lives in
 */
function place(menu, opts, view) {
  const { x, y } = opts;
  menu.style.left = `${x}px`;
  menu.style.top = `${y}px`;
  const rect = menu.getBoundingClientRect();
  if (opts.triggerEl) {
    const r = opts.triggerEl.getBoundingClientRect();
    const below = r.bottom + TRIGGER_GAP;
    const fits = below + rect.height <= view.innerHeight;
    menu.style.left = `${Math.max(0, r.right - rect.width)}px`;
    menu.style.top = `${Math.max(0, fits ? below : r.top - rect.height - TRIGGER_GAP)}px`;
    return;
  }
  if (rect.right > view.innerWidth) menu.style.left = `${Math.max(0, x - rect.width)}px`;
  if (rect.bottom > view.innerHeight) menu.style.top = `${Math.max(0, y - rect.height)}px`;
}

/**
 * Open the menu for a surface from the keyboard: at the element's lower-left
 * corner, since there is no cursor. Shift+F10 and the ContextMenu key reach a
 * caller as keydowns (macOS fires no `contextmenu` for them); this is the one
 * place their position is decided.
 * @param {HTMLElement} el - the focused surface
 * @param {Omit<OpenOptions, 'x' | 'y'>} opts
 * @returns {void}
 */
export function openContextMenuAt(el, opts) {
  const rect = el.getBoundingClientRect();
  openContextMenu({ ...opts, x: rect.left + 8, y: rect.bottom, anchorEl: opts.anchorEl ?? el });
}

/**
 * Whether a keydown is a request for the context menu (Shift+F10 or the
 * ContextMenu key).
 * @param {KeyboardEvent} e
 * @returns {boolean}
 */
export function isContextMenuKey(e) {
  return e.key === 'ContextMenu' || (e.key === 'F10' && e.shiftKey);
}
