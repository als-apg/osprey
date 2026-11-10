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
 * Passthrough: a `contextmenu` listener asks {@link declinesToBrowser} first
 * and, when it answers true, returns without `preventDefault` so the browser's
 * own menu opens — Shift+right-click anywhere, and any right-click in a text
 * field (the browser's menu has spelling, paste and input methods). Shift only
 * counts for pointer-originated events: on Windows/Linux the menu key and
 * Shift+F10 fire `contextmenu` themselves, Shift+F10 with Shift held.
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
 * A surface may redraw while its menu is open, replacing the element focus came
 * from: the hand-back then finds the element drawn in its place by the same
 * identity (its `id` and the {@link IDENTITY} data attributes, or, lacking
 * any, its classes under the nearest ancestor that has one), searching from
 * the nearest of its former ancestors still on the page outward. With no
 * replacement, focus goes to the nearest of those ancestors that takes focus,
 * and stays put when none does. No surface opts in.
 *
 * DOM contract, for the default class prefix `osprey-context` (base.css skins
 * it; a caller passing `classPrefix` gets the same structure under its own
 * names, which is how the web terminal keeps its `rail-context-*` skin):
 *
 *   <div class="osprey-context-menu" role="menu" aria-label="…">
 *     <div class="osprey-context-heading" role="presentation">Columns</div>
 *     <button class="osprey-context-item" role="menuitem" type="button">
 *       <span class="osprey-context-glyph" aria-hidden="true">⎘</span>
 *       <span class="osprey-context-label">Only section <code>SR01C</code></span>
 *       <span class="osprey-context-hint">⌘C</span>
 *     </button>
 *     <button class="osprey-context-item checked" role="menuitemcheckbox"
 *             aria-checked="true"><span class="osprey-context-glyph">✓</span>…</button>
 *     <button class="osprey-context-item" role="menuitem" type="button">
 *       <span class="osprey-context-glyph-blank" aria-hidden="true"></span>…
 *     </button>
 *     <div class="osprey-context-divider" role="separator"></div>
 *     <button class="osprey-context-item danger" …>
 *   </div>
 *
 * Every row is three columns: a fixed-width leading glyph, the label, and an
 * optional right-aligned hint. A row with no glyph (or an unchecked toggle)
 * keeps the leading column as an empty `-glyph-blank` slot so every label in
 * the menu starts at the same x; the slot has its own class so `-glyph`
 * still matches only rows that draw one.
 *
 * Submenus — a row with `submenu` items (any depth) ends in a fourth column,
 * `<span class="osprey-context-submenu" aria-hidden="true">▸</span>`, carries
 * `aria-haspopup="menu"` and `aria-expanded`, and runs nothing itself: it
 * opens a child `<div class="osprey-context-menu" role="menu" data-depth="1">`
 * named after the row. The child opens on hover (after {@link HOVER_DELAY}),
 * on click, and on Enter, Space or ArrowRight (which also focus its first
 * row); it mounts on the root menu's document beside its row and flips to
 * the row's other side at either viewport edge. ArrowLeft and Escape close
 * one level and focus its parent row again; Escape on the root closes the
 * menu as before. Every open level counts as inside the menu — a press on a
 * child row runs it, arrows move among the focused level's rows only — and
 * every whole-menu dismissal above, and {@link closeContextMenu}, closes the
 * whole chain.
 *
 * Home and End focus the focused level's first and last enabled row. An
 * unmodified printable key is typeahead: keys within {@link TYPEAHEAD_RESET}
 * ms build a prefix, and focus moves to the level's next enabled row whose
 * label starts with it (case-insensitive, wrapping). Both are consumed —
 * typeahead keys even without a match — so page shortcuts stay silent.
 */

/**
 * One menu row. A `divider: true` entry renders a separator and a `heading`
 * entry a caption; their other fields are ignored.
 * @typedef {object} MenuItem
 * @property {boolean} [divider]  - render a separator, not a button
 * @property {string} [heading]   - render an inert caption over the rows below it
 * @property {string} [label]     - the verb, in words
 * @property {string} [code]      - a value the verb acts on, drawn monospace after the label
 * @property {string} [glyph]     - leading glyph tying the row to the surface's iconography
 * @property {string} [hint]      - right-aligned secondary text (a shortcut, a count), muted
 * @property {boolean} [checked]  - a toggle row: true draws ✓, false draws nothing; undefined is a plain verb
 * @property {boolean} [disabled] - render the row inert (aria-disabled, no run)
 * @property {string} [tip]       - a hover tip, e.g. why a disabled row is disabled
 * @property {boolean} [danger]   - destructive verb — hover styles error-red
 * @property {() => void} [run]   - the caller-bound action; the menu closes first
 * @property {MenuItem[]} [submenu] - rows of a nested menu this row opens instead of running
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

/** How long the pointer rests on a submenu row before its submenu opens, in ms. */
const HOVER_DELAY = 140;

/** How far a submenu overlaps its row horizontally / rises above it, in px. */
const SUBMENU_OVERLAP = 2;
const SUBMENU_RISE = 4;

/** ms after the last typed key before the typeahead prefix starts over */
const TYPEAHEAD_RESET = 500;

/**
 * How long after a menu-key keydown (ContextMenu, Shift+F10) a `contextmenu`
 * event counts as keyboard-originated, in ms.
 */
const MENU_KEY_WINDOW = 500;

/** `Date.now()` of the last menu-key keydown, or -Infinity. */
let lastMenuKeyAt = -Infinity;

/**
 * Documents whose keydowns are recorded for {@link declinesToBrowser}.
 * @type {WeakSet<Document>}
 */
const keyWatched = new WeakSet();

/** @type {HTMLElement | null} the single open (root) menu, or null */
let menuEl = null;

/**
 * The open submenus, outermost first: `subs[i]` is the level at depth i + 1,
 * opened from `row` in the level above; `dir` is the side it opened on
 * (1 right, -1 left), which its own children prefer.
 * @type {{ menu: HTMLElement, row: HTMLElement, dir: number }[]}
 */
const subs = [];

/** @type {ReturnType<typeof setTimeout> | null} pending hover-open of a submenu */
let hoverTimer = null;

/**
 * Keyboard opener of each enabled submenu row: opens its child and focuses
 * the child's first row.
 * @type {WeakMap<HTMLElement, () => void>}
 */
const submenuOpeners = new WeakMap();

/** The typeahead prefix typed so far, and the timer that clears it. */
let typed = '';
/** @type {ReturnType<typeof setTimeout> | null} */
let typedTimer = null;

/** @type {string} the class stem of the open menu */
let prefix = DEFAULT_PREFIX;

/** @type {HTMLElement | null} the surface the menu belongs to (scroll scoping) */
let anchorEl = null;

/** @type {HTMLElement | null} the button the open menu hangs under, if any */
let triggerEl = null;

/** @type {HTMLElement | null} what had focus when the menu opened */
let previousFocus = null;

/**
 * How to find `previousFocus` again once a redraw has replaced it: its
 * ancestors when the menu opened, nearest first, and a selector for the
 * element drawn in its place.
 * @typedef {{ancestors: Element[], selector: string | null}} FocusTrail
 * @type {FocusTrail | null}
 */
let focusTrail = null;

/** The attributes that name what an element stands for, not how it looks. */
const IDENTITY = [
  'id',
  'data-id',
  'data-session-id',
  'data-panel-id',
  'data-tree-pin',
  'data-group-head',
  'data-group',
];

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

/** @returns {HTMLElement[]} every open level, the root first */
function levels() {
  return menuEl ? [menuEl, ...subs.map((s) => s.menu)] : [];
}

/**
 * @param {HTMLElement} level - one open menu level
 * @returns {HTMLElement[]} that level's enabled rows (a child level is a
 *   separate element on the body, so a level never lists its children's rows)
 */
function rowsOf(level) {
  return /** @type {HTMLElement[]} */ ([...level.querySelectorAll(enabledRows())]);
}

/**
 * @param {unknown} node
 * @returns {number} depth of the open level holding `node`, or -1
 */
function depthOf(node) {
  if (!isNode(node)) return -1;
  return levels().findIndex((level) => level.contains(node));
}

function cancelHover() {
  if (hoverTimer !== null) clearTimeout(hoverTimer);
  hoverTimer = null;
}

/**
 * Close the levels at `depth` and deeper (depth ≥ 1; the root stays open).
 * @param {number} depth
 */
function closeLevelsFrom(depth) {
  for (const { menu, row } of subs.splice(Math.max(0, depth - 1))) {
    menu.remove();
    row.setAttribute('aria-expanded', 'false');
  }
}

/**
 * Open the submenu of `row` (a row of the level at `depth`), replacing any
 * submenu a sibling row had open. Opening an already-open submenu again only
 * (optionally) moves focus into it.
 * @param {HTMLElement} row
 * @param {MenuItem[]} items
 * @param {number} depth - the depth of the level holding `row`
 * @param {boolean} focusFirst - focus the child's first enabled row (keyboard)
 */
function openSubmenu(row, items, depth, focusFirst) {
  cancelHover();
  if (!menuEl || !ownerDoc || !ownerView) return;
  let child = subs[depth]?.row === row ? subs[depth].menu : null;
  if (!child) {
    closeLevelsFrom(depth + 1);
    const parent = levels()[depth];
    if (!parent) return;
    const menu = ownerDoc.createElement('div');
    menu.className = `${prefix}-menu`;
    menu.setAttribute('role', 'menu');
    menu.setAttribute('aria-label', row.querySelector(`.${prefix}-label`)?.textContent ?? '');
    menu.dataset.depth = String(depth + 1);
    menu.addEventListener('contextmenu', (e) => e.preventDefault());
    for (const item of items) menu.appendChild(buildRow(item, depth + 1));
    ownerDoc.body.appendChild(menu);
    const dir = placeSubmenu(menu, row, depth === 0 ? 1 : subs[depth - 1].dir, ownerView);
    subs.push({ menu, row, dir });
    row.setAttribute('aria-expanded', 'true');
    child = menu;
  }
  if (focusFirst) rowsOf(child)[0]?.focus();
}

/**
 * Close the level at `depth` (≥ 1) and everything under it, and focus the row
 * that opened it.
 * @param {number} depth
 */
function closeLevel(depth) {
  const row = subs[depth - 1]?.row;
  closeLevelsFrom(depth);
  row?.focus();
}

function resetTypeahead() {
  if (typedTimer !== null) clearTimeout(typedTimer);
  typedTimer = null;
  typed = '';
}

/**
 * Typeahead over one level: extend the prefix with `char` and focus the next
 * enabled row whose label starts with it (case-insensitive, wrapping). A
 * fresh prefix searches past the focused row; a longer one may stay on it.
 * Repeating one character cycles through the rows starting with it.
 * @param {HTMLElement} level
 * @param {string} char
 */
function typeahead(level, char) {
  if (typedTimer !== null) clearTimeout(typedTimer);
  typed += char.toLowerCase();
  typedTimer = setTimeout(resetTypeahead, TYPEAHEAD_RESET);
  const cycling = [...typed].every((c) => c === typed[0]);
  const needle = cycling ? typed[0] : typed;
  const rows = rowsOf(level);
  if (!rows.length) return;
  const idx = rows.indexOf(/** @type {HTMLElement} */ (ownerDoc?.activeElement));
  const start = needle.length > 1 ? Math.max(0, idx) : idx + 1;
  for (let i = 0; i < rows.length; i++) {
    const candidate = rows[(start + i) % rows.length];
    const label = candidate.querySelector(`.${prefix}-label`)?.textContent ?? '';
    if (label.toLowerCase().startsWith(needle)) {
      candidate.focus();
      return;
    }
  }
}

/** @param {MouseEvent} e */
function onDocPointerDown(e) {
  if (!menuEl || !isNode(e.target) || depthOf(e.target) >= 0) return;
  // The trigger's own press is the toggle's job (see openContextMenu), not an
  // outside press: dismissing here would reopen the menu on the click.
  if (triggerEl && triggerEl.contains(e.target)) return;
  dismiss(false);
}

/** @param {KeyboardEvent} e */
function onDocKeydown(e) {
  if (!menuEl) return;
  const active = /** @type {HTMLElement | null} */ (ownerDoc?.activeElement ?? null);
  // The level the keys act on: the one holding focus, else the root.
  const depth = Math.max(0, depthOf(active));
  const level = levels()[depth];
  // Escape closes one level (the deepest open one), handing focus to the row
  // that opened it; on the root it leaves the menu like Tab. Both keyboard
  // exits hand focus back. Tab's default is cancelled: the operator resumes
  // tabbing from where they were, not from wherever a removed menu row left
  // the sequence.
  if (e.key === 'Escape' || e.key === 'Tab') {
    e.preventDefault();
    e.stopPropagation();
    if (e.key === 'Escape' && subs.length) closeLevel(subs.length);
    else dismiss(true);
    return;
  }
  if (e.key === 'ArrowLeft') {
    if (depth === 0) return;
    e.preventDefault();
    e.stopPropagation();
    closeLevel(depth);
    return;
  }
  if (e.key === 'ArrowRight' || e.key === 'Enter' || e.key === ' ') {
    const opener = active && level.contains(active) ? submenuOpeners.get(active) : undefined;
    if (!opener) return; // Enter/Space on a plain row: the button's own click runs it
    e.preventDefault();
    e.stopPropagation();
    opener();
    return;
  }
  // Arrow navigation over the focused level's enabled rows — the menu's only
  // focusables.
  if (e.key === 'ArrowDown' || e.key === 'ArrowUp') {
    e.preventDefault();
    e.stopPropagation();
    const rows = rowsOf(level);
    if (!rows.length) return;
    const idx = rows.indexOf(/** @type {HTMLElement} */ (ownerDoc?.activeElement));
    const next =
      e.key === 'ArrowDown'
        ? rows[(idx + 1 + rows.length) % rows.length]
        : rows[(idx - 1 + rows.length) % rows.length];
    next.focus();
    return;
  }
  // Home/End: the level's first/last enabled row, wherever focus is.
  if (e.key === 'Home' || e.key === 'End') {
    e.preventDefault();
    e.stopPropagation();
    const rows = rowsOf(level);
    rows[e.key === 'Home' ? 0 : rows.length - 1]?.focus();
    return;
  }
  // Typeahead: an unmodified printable key is the menu's, matched or not, so
  // a page's single-key shortcuts never fire under an open menu.
  if (e.key.length === 1 && e.key !== ' ' && !e.ctrlKey && !e.metaKey && !e.altKey) {
    e.preventDefault();
    e.stopPropagation();
    typeahead(level, e.key);
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
  // A long submenu scrolling its own rows moves no anchor.
  if (depthOf(e.target) >= 0) return;
  if (e.target === ownerDoc) {
    dismiss(false);
    return;
  }
  if (anchorEl && isNode(e.target) && e.target.contains(anchorEl)) dismiss(false);
}

/**
 * @param {Element} el
 * @returns {string} `el`'s identity attributes as a selector, '' when it has none
 */
function identityOf(el) {
  return IDENTITY.filter((a) => el.hasAttribute(a))
    .map((a) => `[${a}="${CSS.escape(el.getAttribute(a) ?? '')}"]`)
    .join('');
}

/**
 * @param {Element | null} el - what had focus when the menu opened
 * @returns {FocusTrail | null}
 */
function trailOf(el) {
  if (!el || !el.parentElement) return null;
  /** @type {Element[]} */
  const ancestors = [];
  for (let a = /** @type {Element | null} */ (el.parentElement); a;) {
    ancestors.push(a);
    a = a.parentElement;
  }
  const tag = el.tagName.toLowerCase();
  const own = identityOf(el);
  if (own) return { ancestors, selector: tag + own };
  const classes = [...el.classList].map((c) => `.${CSS.escape(c)}`).join('');
  if (!classes) return { ancestors, selector: null };
  const named = ancestors.find((a) => identityOf(a));
  const scope = named ? `${named.tagName.toLowerCase()}${identityOf(named)} ` : '';
  return { ancestors, selector: scope + tag + classes };
}

/**
 * Where focus goes back to when `el` left the page during the menu: the
 * element drawn in its place, else the nearest former ancestor that takes focus.
 * @param {FocusTrail} trail
 * @returns {HTMLElement | null}
 */
function standIn(trail) {
  const alive = trail.ancestors.filter((a) => a.isConnected);
  if (trail.selector) {
    for (const a of alive) {
      const found = a.querySelector(trail.selector);
      if (found) return /** @type {HTMLElement} */ (found);
    }
  }
  const takesFocus = (/** @type {any} */ a) =>
    (a.tabIndex >= 0 || a.hasAttribute('tabindex')) && !a.hasAttribute('disabled');
  return /** @type {HTMLElement | null} */ (alive.find(takesFocus) ?? null);
}

/**
 * Tear down the open menu, optionally handing focus back to whatever held it
 * when the menu opened.
 * @param {boolean} restoreFocus
 */
function dismiss(restoreFocus) {
  if (!menuEl) return;
  cancelHover();
  resetTypeahead();
  closeLevelsFrom(1);
  menuEl.remove();
  menuEl = null;
  anchorEl = null;
  triggerEl?.setAttribute('aria-expanded', 'false');
  triggerEl = null;
  const restoreTo = previousFocus;
  const trail = focusTrail;
  previousFocus = null;
  focusTrail = null;
  ownerDoc?.removeEventListener('pointerdown', onDocPointerDown, true);
  ownerDoc?.removeEventListener('keydown', onDocKeydown, true);
  ownerDoc?.removeEventListener('scroll', onDocScroll, true);
  ownerView?.removeEventListener('blur', onWindowBlur);
  ownerView?.removeEventListener('resize', onWindowResize);
  ownerDoc = null;
  ownerView = null;
  if (!restoreFocus || !restoreTo) return;
  if (restoreTo.isConnected) restoreTo.focus();
  else if (trail) standIn(trail)?.focus();
}

/**
 * Close the open menu and every submenu, if any, without moving focus: the
 * surface it belongs to is going away.
 * @returns {void}
 */
export function closeContextMenu() {
  dismiss(false);
}

/**
 * Build one row element.
 * @param {MenuItem} item
 * @param {number} depth - depth of the level the row belongs to (0 = root)
 * @returns {HTMLElement}
 */
function buildRow(item, depth) {
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

  // Leading column: the glyph, or an empty slot of the same width so the
  // labels of glyph-less rows line up with the rest.
  const glyphText = toggle ? (item.checked ? '✓' : '') : (item.glyph ?? '');
  const glyph = document.createElement('span');
  glyph.className = glyphText ? `${prefix}-glyph` : `${prefix}-glyph-blank`;
  glyph.setAttribute('aria-hidden', 'true');
  glyph.textContent = glyphText;
  btn.appendChild(glyph);

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

  if (item.hint) {
    const hint = document.createElement('span');
    hint.className = `${prefix}-hint`;
    hint.textContent = item.hint;
    btn.appendChild(hint);
  }

  const submenu = item.submenu;
  if (submenu) {
    btn.setAttribute('aria-haspopup', 'menu');
    btn.setAttribute('aria-expanded', 'false');
    const arrow = document.createElement('span');
    arrow.className = `${prefix}-submenu`;
    arrow.setAttribute('aria-hidden', 'true');
    arrow.textContent = '▸';
    btn.appendChild(arrow);
  }

  if (item.tip) btn.title = item.tip;
  if (item.disabled) {
    btn.setAttribute('aria-disabled', 'true');
  } else if (submenu) {
    // A submenu row runs nothing: it opens its child — at once on a click,
    // after a short rest of the pointer, and from the keyboard (onDocKeydown).
    submenuOpeners.set(btn, () => openSubmenu(btn, submenu, depth, true));
    btn.addEventListener('click', () => openSubmenu(btn, submenu, depth, false));
    btn.addEventListener('pointerenter', () => {
      cancelHover();
      hoverTimer = setTimeout(() => openSubmenu(btn, submenu, depth, false), HOVER_DELAY);
    });
    btn.addEventListener('pointerleave', cancelHover);
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
  if (!submenu) {
    // Resting on any other row of this level closes a sibling's submenu.
    btn.addEventListener('pointerenter', () => {
      cancelHover();
      closeLevelsFrom(depth + 1);
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
  const replacing = depthOf(active) >= 0;
  const held = replacing ? previousFocus : active;
  const trail = replacing ? focusTrail : trailOf(active);
  dismiss(false);
  // Not `instanceof HTMLElement`: the active element may be of a window's realm.
  previousFocus = /** @type {HTMLElement | null} */ (held);
  focusTrail = trail;
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

  for (const item of items) menu.appendChild(buildRow(item, 0));

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
 * Position an inserted submenu beside its row: on side `prefer` (1 right,
 * -1 left — the side its parent opened on, so a cascade keeps going one
 * way), flipped to the row's other side when that edge of the viewport is
 * short of room, and clamped inside it when neither side fits. Vertically
 * level with the row, lifted to stay above the bottom edge.
 * @param {HTMLElement} menu
 * @param {HTMLElement} row
 * @param {number} prefer
 * @param {Window} view  the root menu's window
 * @returns {number} the side it opened on
 */
function placeSubmenu(menu, row, prefer, view) {
  const r = row.getBoundingClientRect();
  menu.style.left = `${r.right}px`;
  menu.style.top = `${r.top}px`;
  const { width, height } = menu.getBoundingClientRect();
  const right = r.right - SUBMENU_OVERLAP;
  const left = r.left - width + SUBMENU_OVERLAP;
  const fitsRight = right + width <= view.innerWidth;
  const fitsLeft = left >= 0;
  let dir = prefer;
  if (dir > 0 && !fitsRight && fitsLeft) dir = -1;
  else if (dir < 0 && !fitsLeft && fitsRight) dir = 1;
  const x = Math.min(Math.max(0, dir > 0 ? right : left), Math.max(0, view.innerWidth - width));
  const y = Math.max(0, Math.min(r.top - SUBMENU_RISE, view.innerHeight - height));
  menu.style.left = `${x}px`;
  menu.style.top = `${y}px`;
  return dir;
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

/**
 * Record menu-key keydowns in `doc`, once per document, so a `contextmenu` the
 * key fires right after is known to be keyboard-originated. Capture phase:
 * a surface that stops the keydown still lets it be seen.
 * @param {Document | null | undefined} doc
 * @returns {void}
 */
function watchMenuKeys(doc) {
  if (!doc || keyWatched.has(doc) || typeof doc.addEventListener !== 'function') return;
  keyWatched.add(doc);
  doc.addEventListener(
    'keydown',
    (e) => {
      if (isContextMenuKey(e)) lastMenuKeyAt = Date.now();
    },
    true
  );
}

if (typeof document !== 'undefined') watchMenuKeys(document);

/** `input` types that take typed text, where the browser's own menu belongs. */
const TEXT_INPUT_TYPES = new Set([
  '',
  'text',
  'search',
  'email',
  'url',
  'tel',
  'password',
  'number',
]);

/**
 * Whether `el` (or the element holding it) is a text-entry field: a text-type
 * `input`, a `textarea`, or inside contenteditable content. xterm's helper
 * textarea is not one — xterm moves it under the cursor on right-click, so it
 * stands in for the terminal, not for a field.
 * @param {unknown} target
 * @returns {boolean}
 */
function isTextField(target) {
  if (!isNode(target)) return false;
  /** @type {any} */
  const el = target.nodeType === 1 ? target : target.parentElement;
  if (!el || typeof el.tagName !== 'string') return false;
  const tag = el.tagName.toLowerCase();
  if (tag === 'textarea') return !el.classList?.contains('xterm-helper-textarea');
  if (tag === 'input') {
    return TEXT_INPUT_TYPES.has(String(el.getAttribute('type') ?? '').toLowerCase());
  }
  return el.isContentEditable === true;
}

/**
 * Whether a `contextmenu` event originated from a pointer, not from the menu
 * key or Shift+F10: no menu-key keydown in the last {@link MENU_KEY_WINDOW} ms,
 * and either a secondary-button press or a pointer type. Keyboard-fired and
 * synthetic events carry `button` 0 and no `pointerType`.
 * @param {MouseEvent} event
 * @returns {boolean}
 */
function isPointerOriginated(event) {
  const sinceMenuKey = Date.now() - lastMenuKeyAt;
  // A clock stepped backwards (negative gap) is not a recent key press.
  if (sinceMenuKey >= 0 && sinceMenuKey < MENU_KEY_WINDOW) return false;
  const pointerType = /** @type {any} */ (event).pointerType;
  return event.button === 2 || (typeof pointerType === 'string' && pointerType !== '');
}

/**
 * Whether a context-menu request belongs to the browser rather than to an
 * OSPREY menu: a pointer-originated `contextmenu` with Shift held, or any
 * request whose target is a text-entry field (xterm's helper textarea
 * excepted). Callers return without `preventDefault` when this is true.
 * Keyboard-originated requests (menu key, Shift+F10) never decline for Shift.
 * @param {Event} event - a `contextmenu` event, or a keydown/synthetic event
 *   standing for one
 * @returns {boolean}
 */
export function declinesToBrowser(event) {
  const view = /** @type {any} */ (event)?.view;
  watchMenuKeys(view?.document);
  const mouse = /** @type {MouseEvent} */ (event);
  if (event.type === 'contextmenu' && mouse.shiftKey && isPointerOriginated(mouse)) return true;
  return isTextField(event.target);
}
