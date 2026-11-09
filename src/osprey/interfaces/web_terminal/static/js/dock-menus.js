// @ts-check
/* OSPREY Web Terminal — Context menu of the empty dock slot
 *
 * One place, registered with the shell's place dispatcher (place-menu.js): the
 * zero-tile dock.
 *
 *   Open panel ▸ · Reset layout
 *
 * The empty dock slot is dockview's empty area: #dock-root while the layout
 * holder holds no layout (the watermark dockview draws there belongs to it).
 * In fallback mode (no dockview, no holder) it is the "no active panel" card
 * in #right-panel; dock-workspace hides that card's panel whenever dockview is
 * up, so the card matches only while there is no holder. A tile body, a sash
 * or a placeholder never matches: while any tile exists, #dock-root is not the
 * empty slot, so their walk up the tree finds no place.
 *
 * Open panel lists the rail's service panels that are neither in a tile nor in
 * a window of their own ({@link untiledPanels}), in sidebar order. A panel
 * that never started, failed to start or stopped answering stays listed,
 * disabled, with the reason. Opening one from the empty dock goes through the
 * rail's Open verb, whose open rule mints the first tile. Reset layout is the
 * destructive row, last, behind a divider.
 *
 * The slot is made focusable (tabindex="-1") so the menu key reaches it; a
 * click on it focuses it.
 *
 * Imports only leaf modules: panel-menu-policy (which panel-manager imports)
 * reads untiledPanels() from here, so panel-manager's verbs are injected by
 * app.js instead of imported. No import-time side effects.
 */

import { registerPlace } from './place-menu.js';
import { getLayoutHolder, isDetached } from './dock-apply.js';
import { BUTTON_SELECTOR, parked } from './rail-row-state.js';
import { TERMINAL_RAIL_ID } from './panel-catalog.js';
import { tileOf } from '/static/vendor/dock-engine.mjs';

/** @typedef {import('/design-system/js/context-menu.js').MenuItem} MenuItem */

/** The menu's accessible name. */
const ARIA_LABEL = 'Dock';

/** The dock's root element (dockview's managed grid). */
const DOCK_ROOT_ID = 'dock-root';

/** The fallback's "no active panel" card, inside #right-panel. */
const FALLBACK_CARD_SELECTOR = '#right-panel .artifacts-empty-state';

/** The disabled row Open panel shows when nothing is left to open. */
export const NO_PANELS_LABEL = 'No panels to open';

/** Why a panel that has not answered yet cannot be opened. */
export const NOT_STARTED_TIP = 'This panel has not started yet.';

/** Why a panel whose start failed cannot be opened (without a server message). */
export const FAILED_TIP = 'This panel failed to start.';

/** Why a panel that stopped answering cannot be opened. */
export const UNREACHABLE_TIP = 'This panel stopped answering.';

/** Why a panel reported unhealthy by the panel manager cannot be opened. */
export const UNHEALTHY_TIP = 'This panel is not available right now.';

/**
 * @typedef {object} DockMenuDeps
 * @property {(id: string) => void} [activate] - opens a panel (the rail's Open verb)
 * @property {(id: string) => boolean} [isHealthy] - whether a panel may be opened
 * @property {(id: string) => string} [labelOf] - a panel's label
 * @property {() => void} [reset] - resets the dock layout
 */

/**
 * @typedef {object} UntiledPanel
 * @property {string} id
 * @property {string} label
 * @property {boolean} [disabled]
 * @property {string} [tip] - why it is disabled
 */

/** @type {DockMenuDeps} the verbs app.js injected */
let deps = {};

/**
 * Whether `node` is an element. Not `instanceof`: a pop-out's elements belong
 * to that window's realm.
 * @param {unknown} node
 * @returns {node is Element}
 */
function isElement(node) {
  return !!node && /** @type {any} */ (node).nodeType === 1;
}

/**
 * The panel the fallback surfaces, read off #right-panel (panel-manager writes
 * `data-active-panel` there), or null.
 * @param {Document} doc
 * @returns {string | null}
 */
function fallbackActiveId(doc) {
  return doc.getElementById('right-panel')?.getAttribute('data-active-panel') || null;
}

/**
 * Whether a panel holds a tile: a tile in the dock, or the fallback's surfaced
 * panel when there is no dock.
 * @param {string} id
 * @param {Document} doc
 * @returns {boolean}
 */
function isTiled(id, doc) {
  const holder = getLayoutHolder();
  if (holder) return tileOf(holder.layout ?? null, id) !== null;
  return fallbackActiveId(doc) === id;
}

/**
 * Why a rail row cannot be opened, or null when it can.
 * @param {Element} row
 * @param {string} id
 * @returns {string | null}
 */
function disabledReason(row, id) {
  if (row.classList.contains('failed')) {
    return row.getAttribute('data-title-status') || FAILED_TIP;
  }
  if (row.classList.contains('unreachable')) return UNREACHABLE_TIP;
  if (row.classList.contains('disabled')) return NOT_STARTED_TIP;
  if (deps.isHealthy && !deps.isHealthy(id)) return UNHEALTHY_TIP;
  return null;
}

/**
 * A rail row's label: what the operator reads in the sidebar, else the
 * injected label, else the id.
 * @param {Element} row
 * @param {string} id
 * @returns {string}
 */
function labelFor(row, id) {
  const shown = row.querySelector('.osprey-nav-label')?.textContent?.trim();
  return shown || deps.labelOf?.(id) || id;
}

/**
 * The rail's service panels that are neither in a tile nor detached, in
 * sidebar order. Panels that cannot be opened are listed disabled, with the
 * reason. Read live on every call; no side effects.
 * @param {Document} [doc]
 * @returns {UntiledPanel[]}
 */
export function untiledPanels(doc = document) {
  const rail = doc.getElementById('panel-rail');
  if (!rail) return [];
  /** @type {UntiledPanel[]} */
  const out = [];
  for (const row of rail.querySelectorAll(`${BUTTON_SELECTOR}[data-panel-id]`)) {
    if (parked(row)) continue;
    const id = row.getAttribute('data-panel-id') ?? '';
    if (!id || id === TERMINAL_RAIL_ID) continue;
    if (isDetached(id) || isTiled(id, doc)) continue;
    const tip = disabledReason(row, id);
    out.push(
      tip ? { id, label: labelFor(row, id), disabled: true, tip } : { id, label: labelFor(row, id) }
    );
  }
  return out;
}

/**
 * Reset the dock layout: the injected verb, else dock-workspace's, loaded
 * lazily (it pulls the page in), as the sidebar header does.
 */
function resetLayout() {
  if (deps.reset) {
    deps.reset();
    return;
  }
  void import('./dock-workspace.js').then((dock) => dock.resetDockLayout());
}

/**
 * The empty dock slot's rows: Open panel ▸ (one disabled row when nothing is
 * left to open), then Reset layout, destructive and last.
 * @param {Document} [doc]
 * @returns {MenuItem[]}
 */
export function buildEmptyDockItems(doc = document) {
  const panels = untiledPanels(doc);
  /** @type {MenuItem[]} */
  const submenu = panels.length
    ? panels.map((p) => ({
        label: p.label,
        ...(p.disabled ? { disabled: true, tip: p.tip } : {}),
        run: () => deps.activate?.(p.id),
      }))
    : [{ label: NO_PANELS_LABEL, disabled: true }];
  return [
    { label: 'Open panel', submenu },
    { divider: true },
    { label: 'Reset layout', danger: true, run: resetLayout },
  ];
}

/**
 * Whether `el` is the empty dock slot: #dock-root while the holder holds no
 * layout, or the fallback card while there is no holder.
 * @param {Element} el
 * @returns {boolean}
 */
export function isEmptyDockSlot(el) {
  const holder = getLayoutHolder();
  if (el.id === DOCK_ROOT_ID) return holder !== null && (holder.layout ?? null) === null;
  if (typeof el.matches === 'function' && el.matches(FALLBACK_CARD_SELECTOR))
    return holder === null;
  return false;
}

/**
 * The empty dock slot holding `target`, or null.
 * @param {unknown} target
 * @returns {Element | null}
 */
function slotOf(target) {
  /** @type {any} */
  let el = isElement(target) ? target : /** @type {any} */ (target)?.parentElement;
  for (; isElement(el); el = el.parentElement) if (isEmptyDockSlot(el)) return el;
  return null;
}

/**
 * Register the empty dock slot's place and make the slot focusable.
 * @param {DockMenuDeps & { doc?: Document }} [options]
 * @returns {() => void} unregisters the place and removes the listener
 */
export function installDockMenus({ doc = document, ...injected } = {}) {
  deps = injected;
  doc.getElementById(DOCK_ROOT_ID)?.setAttribute('tabindex', '-1');

  const unregister = registerPlace({
    match: isEmptyDockSlot,
    ariaLabel: ARIA_LABEL,
    build: (el) => buildEmptyDockItems(el.ownerDocument ?? doc),
  });

  /** A click on the empty slot focuses it, so the menu key opens its menu. */
  function onClick(/** @type {MouseEvent} */ e) {
    const slot = /** @type {HTMLElement | null} */ (slotOf(e.target));
    if (!slot) return;
    if (!slot.hasAttribute('tabindex')) slot.setAttribute('tabindex', '-1');
    slot.focus({ preventScroll: true });
  }
  doc.addEventListener('click', onClick);

  return () => {
    unregister();
    doc.removeEventListener('click', onClick);
    deps = {};
  };
}
