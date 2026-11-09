// @ts-check
/* OSPREY Web Terminal — Context menus for the shell's places
 *
 * One dispatcher for every place of the shell that has a context menu but no
 * element of its own to listen on (the chat, the terminal, ...). A place is
 * registered once, with {@link registerPlace}; {@link installPlaceMenus} wires a
 * document — the hub's, and every pop-out window's, since the Agent pane moves
 * into one.
 *
 *   pointer   — a `contextmenu` the browser does not keep (declinesToBrowser),
 *               not already handled by a surface of its own (defaultPrevented),
 *               opens the menu of the innermost registered place holding the
 *               target, at the cursor.
 *   keyboard  — the menu key or Shift+F10 on a focused element opens that
 *               element's place menu at the element. Never declines for Shift.
 *
 * A place whose build returns no rows opens nothing and leaves the event alone,
 * so the browser's own menu shows. The listeners sit on the document's window,
 * in the bubble phase, so every surface's own listener (on an element or on the
 * document itself) has had its turn — and its `preventDefault` — first.
 *
 * Imports only the design system's context-menu.js: pane-window.js imports this
 * module and must stay a leaf of the panel stack.
 */

import {
  declinesToBrowser,
  isContextMenuKey,
  openContextMenu,
  openContextMenuAt,
} from '/design-system/js/context-menu.js';

/** @typedef {import('/design-system/js/context-menu.js').MenuItem} MenuItem */

/**
 * @typedef {object} Place
 * @property {string | ((el: Element) => boolean)} match - a CSS selector the
 *   place's element matches, or a predicate on an element
 * @property {(el: Element, event: Event) => MenuItem[]} build - the rows for the
 *   place's element; none opens nothing
 * @property {string | ((el: Element) => string)} ariaLabel - the menu's name
 */

/** @type {Place[]} the registered places, in registration order */
let places = [];

/**
 * Documents already wired, mapped to the controller that removes their
 * listeners. A pop-out reload brings a fresh Document, which is wired anew.
 * @type {WeakMap<Document, AbortController>}
 */
let installed = new WeakMap();

/**
 * Register a place. The innermost registered place holding an event's target
 * wins; between two places matching the same element, the first registered.
 * @param {Place} place
 * @returns {() => void} removes the place again
 */
export function registerPlace(place) {
  places.push(place);
  return () => {
    places = places.filter((p) => p !== place);
  };
}

/**
 * Whether `el` is an element. Not `instanceof`: a pop-out's elements belong to
 * that window's realm.
 * @param {unknown} node
 * @returns {node is Element}
 */
function isElement(node) {
  return !!node && /** @type {any} */ (node).nodeType === 1;
}

/**
 * @param {Place} place
 * @param {Element} el
 * @returns {boolean}
 */
function matches(place, el) {
  if (typeof place.match === 'function') return place.match(el);
  return typeof el.matches === 'function' && el.matches(place.match);
}

/**
 * The innermost registered place holding `target`, and its element.
 * @param {unknown} target
 * @returns {{ place: Place, el: Element } | null}
 */
function findPlace(target) {
  if (!target) return null;
  /** @type {any} */
  let el = isElement(target) ? target : /** @type {any} */ (target).parentElement;
  for (; isElement(el); el = el.parentElement) {
    for (const place of places) if (matches(place, el)) return { place, el };
  }
  return null;
}

/**
 * The rows and name for a place's element.
 * @param {{ place: Place, el: Element }} found
 * @param {Event} event
 */
function menuFor({ place, el }, event) {
  const items = place.build(el, event) ?? [];
  const ariaLabel = typeof place.ariaLabel === 'function' ? place.ariaLabel(el) : place.ariaLabel;
  return { items, ariaLabel };
}

/** @param {MouseEvent} e */
function onContextMenu(e) {
  if (e.defaultPrevented || declinesToBrowser(e)) return;
  const found = findPlace(e.target);
  if (!found) return;
  const { items, ariaLabel } = menuFor(found, e);
  if (items.length === 0) return;
  e.preventDefault();
  openContextMenu({
    x: e.clientX,
    y: e.clientY,
    anchorEl: /** @type {HTMLElement} */ (found.el),
    ariaLabel,
    items,
  });
}

/** @param {KeyboardEvent} e */
function onKeyDown(e) {
  if (e.defaultPrevented || !isContextMenuKey(e) || declinesToBrowser(e)) return;
  const found = findPlace(e.target);
  if (!found) return;
  const { items, ariaLabel } = menuFor(found, e);
  if (items.length === 0) return;
  // Stops the native menu a Windows/Firefox key would fire next.
  e.preventDefault();
  openContextMenuAt(/** @type {HTMLElement} */ (found.el), { ariaLabel, items });
}

/**
 * Wire `doc` for the registered places. Idempotent per document: the hub calls
 * it at boot, pane-window on every dress of a pop-out (again after a reload).
 * @param {Document | null | undefined} doc
 * @returns {void}
 */
export function installPlaceMenus(doc) {
  if (!doc || installed.has(doc)) return;
  const target = doc.defaultView ?? doc;
  if (typeof target.addEventListener !== 'function') return;
  const controller = new AbortController();
  installed.set(doc, controller);
  const { signal } = controller;
  target.addEventListener('contextmenu', /** @type {EventListener} */ (onContextMenu), { signal });
  target.addEventListener('keydown', /** @type {EventListener} */ (onKeyDown), { signal });
}

/**
 * Tests only: forget every place and unwire the given documents.
 * @param {Document[]} [docs]
 * @returns {void}
 */
export function resetPlaceMenus(docs = []) {
  for (const doc of docs) installed.get(doc)?.abort();
  places = [];
  installed = new WeakMap();
}
