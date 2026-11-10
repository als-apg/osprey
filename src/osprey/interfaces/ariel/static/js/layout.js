// @ts-check
/**
 * ARIEL panel layout: the one screen's panes, splitters, stacked mode and
 * drawers.
 *
 * The screen is three pane roots inside `.app-container` (index.html):
 * `#ariel-rail` | `#ariel-list-col` (the list plus the compose host over it)
 * | `#ariel-reader`. This module is the only writer of `stacked`, and maps
 * these store keys onto the DOM in both directions:
 *
 * - `railCollapsed`: the rail splitter. A drag, a double-click or Enter on
 *   the handle writes it; a write from elsewhere (`[`, the ⋯ menu) is applied
 *   to the splitter. The shared splitter has no snap, so the rail's `min` is
 *   the icon-strip width and the snap lives in `onCommit`: an expanded
 *   release below RAIL_SNAP restores the last usable width and then collapses
 *   to the strip, so expanding returns a usable rail. Default collapsed.
 * - `stacked`: a ResizeObserver on the panel root (below STACK_BELOW px the
 *   list and reader share one column and `narrowView` picks which shows).
 *   Back in the reader's pane head writes `narrowView:'list'`.
 * - `drawer`: `#status-drawer` and `#settings-drawer`. A store write opens or
 *   closes them; every open/close the drawers do themselves (Esc, backdrop,
 *   an unsaved-changes veto) is written
 *   back from their `drawer:open` / `drawer:close` events, never assumed.
 *   `initSettings` removes `#settings-drawer` when the config panel is
 *   disabled, so it must run first; `drawer:'settings'` is then a no-op.
 *
 * Root attributes CSS keys off (layout.css): `data-rail-collapsed`,
 * `data-stacked`, `data-narrow-view` on `.app-container`.
 *
 * Widths persist under `scopedStorageKey`; the keys keep their `-embedded`
 * suffix so an operator's stored sizes survive. The list column's width is the
 * only list/reader split stored: the reader takes the remaining space.
 */

import { initSplitter, parseStoredState } from '/design-system/js/splitter.js';
import { scopedStorageKey } from '/design-system/js/storage-scope.js';
import { icon } from '/design-system/js/icons.js';
import { getState, setState, subscribe } from './store.js';

/** @typedef {import('./store.js').State} State */
/** @typedef {'status'|'settings'} DrawerName */

/** Icon-strip width: the rail's collapsed size and its drag minimum. */
export const RAIL_STRIP = 48;
/** An expanded rail released narrower than this snaps to the icon strip. */
export const RAIL_SNAP = 110;
export const RAIL_DEFAULT = 208;
export const RAIL_MAX = 360;
export const LIST_MIN = 260;
export const LIST_MAX = 760;
/** Panel widths below this stack list and reader into one column. */
export const STACK_BELOW = 640;

/** @returns {string} storage key of the rail's size + collapsed record */
export function railStorageKey() {
  return scopedStorageKey('ariel-rail-embedded');
}

/** @returns {string} storage key of the list column's width */
export function listStorageKey() {
  return scopedStorageKey('ariel-list-width-embedded');
}

/** @param {string} key @returns {string|null} */
function readStored(key) {
  try {
    return localStorage.getItem(key);
  } catch {
    return null;
  }
}

/** @type {ReadonlyArray<[DrawerName, string]>} name, element id */
const DRAWERS = [
  ['status', 'status-drawer'],
  ['settings', 'settings-drawer'],
];

/**
 * The rail's splitter with its snap: an expanded release below RAIL_SNAP
 * restores the last usable width and collapses to the strip.
 * @param {HTMLElement|null} handle
 * @param {HTMLElement|null} pane
 * @param {HTMLElement|null} app
 * @returns {{apply: (want: boolean) => void, isCollapsed: () => boolean, destroy: () => void}}
 */
function wireRail(handle, pane, app) {
  const railKey = railStorageKey();
  const storedRail = parseStoredState(readStored(railKey), true).size;
  /** Last expanded width at or above the snap threshold; what expand returns to. */
  let lastUsable =
    storedRail !== null && storedRail >= RAIL_SNAP ? Math.min(storedRail, RAIL_MAX) : RAIL_DEFAULT;
  /** True while this module drives the rail itself, so its commits are not re-read. */
  let drivingRail = false;

  const rail = initSplitter({
    handle,
    pane,
    storageKey: railKey,
    min: RAIL_STRIP,
    max: RAIL_MAX,
    collapsedSize: RAIL_STRIP,
    defaultCollapsed: true,
    initialSize: lastUsable,
    onCommit: (size, collapsed) => {
      if (drivingRail) return;
      if (collapsed) {
        setState({ railCollapsed: true });
        return;
      }
      if (size < RAIL_SNAP) {
        drivingRail = true;
        try {
          rail.applySize(lastUsable);
          rail.collapse();
        } finally {
          drivingRail = false;
        }
        setState({ railCollapsed: true });
        return;
      }
      lastUsable = size;
      setState({ railCollapsed: false });
    },
  });
  rail.restoreSize();
  if (!rail.isCollapsed() && (storedRail === null || storedRail < RAIL_SNAP)) {
    rail.applySize(lastUsable);
  }

  return {
    /** @param {boolean} want */
    apply: (want) => {
      if (want !== rail.isCollapsed()) {
        drivingRail = true;
        try {
          if (want) {
            rail.collapse();
          } else {
            rail.applySize(lastUsable);
            rail.expand();
          }
        } finally {
          drivingRail = false;
        }
      }
      app?.toggleAttribute('data-rail-collapsed', want);
    },
    isCollapsed: () => rail.isCollapsed(),
    destroy: () => rail.destroy(),
  };
}

/**
 * The list/reader splitter. The list never collapses: the shared handle toggles
 * on double-click, which here would leave an empty column, so it springs
 * straight back.
 * @param {HTMLElement|null} handle
 * @param {HTMLElement|null} pane
 * @returns {{destroy: () => void}}
 */
function wireListSplitter(handle, pane) {
  const list = initSplitter({
    handle,
    pane,
    storageKey: listStorageKey(),
    min: LIST_MIN,
    max: LIST_MAX,
    isEnabled: () => !getState().stacked,
    onCommit: (_size, collapsed) => {
      if (collapsed) list.expand();
    },
  });
  list.restoreSize();
  if (list.isCollapsed()) list.expand();
  return list;
}

/**
 * `stacked` from the panel's width.
 * @param {HTMLElement|null} app
 * @returns {() => void} stop observing
 */
function observeStacked(app) {
  if (!app || typeof ResizeObserver !== 'function') return () => {};
  const observer = new ResizeObserver((entries) => {
    const entry = entries[entries.length - 1];
    if (!entry) return;
    const stacked = entry.contentRect.width < STACK_BELOW;
    if (stacked !== getState().stacked) setState({ stacked });
  });
  observer.observe(app);
  return () => observer.disconnect();
}

/**
 * Map `drawer` onto the drawer elements both ways: their own opens and closes
 * are written back from events, and `apply` drives them from a store write.
 * @param {(id: string) => HTMLElement|null} byId
 * @param {AbortSignal} signal
 * @returns {{current: () => DrawerName|null, apply: (want: DrawerName|null) => void}}
 */
function wireDrawers(byId, signal) {
  /** @type {Map<DrawerName, HTMLElement & {open?: () => boolean, close?: () => boolean}>} */
  const drawers = new Map();
  for (const [name, id] of DRAWERS) {
    const el = byId(id);
    if (!el) continue;
    drawers.set(name, /** @type {any} */ (el));
    el.addEventListener('drawer:open', () => setState({ drawer: name }), { signal });
    el.addEventListener(
      'drawer:close',
      () => {
        if (getState().drawer === name) setState({ drawer: null });
      },
      { signal }
    );
  }

  /** @param {HTMLElement} el @returns {boolean} */
  const isOpen = (el) => el.hasAttribute('open');

  /** @returns {DrawerName|null} the drawer actually open */
  const current = () => {
    for (const [name, el] of drawers) if (isOpen(el)) return name;
    return null;
  };

  return {
    current,
    apply: (want) => {
      if (want !== null && !drawers.has(want)) return; // e.g. Settings removed: no-op
      let refused = false;
      for (const [name, el] of drawers) {
        if (name !== want && isOpen(el) && typeof el.close === 'function' && !el.close()) {
          refused = true;
        }
      }
      if (want !== null && !refused) {
        const el = /** @type {HTMLElement & {open?: () => boolean}} */ (drawers.get(want));
        if (!isOpen(el) && typeof el.open === 'function') el.open();
      }
      const actual = current();
      if (actual !== getState().drawer) setState({ drawer: actual });
    },
  };
}

/**
 * @typedef {object} LayoutApi
 * @property {() => void} destroy  Detach every listener, observer and subscription.
 */

/**
 * Wire the screen. Call once at boot, after `initSettings` (which may remove
 * `#settings-drawer`) and before the router applies the first hash.
 * Missing elements degrade to no-ops.
 *
 * @param {ParentNode} [root] where to find the screen's elements
 * @returns {LayoutApi}
 */
export function initLayout(root = document) {
  /** @param {string} id */
  const byId = (id) => /** @type {HTMLElement|null} */ (root.querySelector(`#${id}`));
  const app = /** @type {HTMLElement|null} */ (root.querySelector('.app-container'));
  const controller = new AbortController();
  const { signal } = controller;

  const rail = wireRail(byId('ariel-rail-splitter'), byId('ariel-rail'), app);
  const list = wireListSplitter(byId('ariel-list-splitter'), byId('ariel-list-col'));
  const stopObserving = observeStacked(app);

  const back = byId('ariel-reader-back');
  if (back) {
    if (!back.querySelector('svg')) back.prepend(icon('chevron-left'));
    back.addEventListener('click', () => setState({ narrowView: 'list' }), { signal });
  }

  const drawers = wireDrawers(byId, signal);

  /** @param {State} state */
  const stampRoot = (state) => {
    if (!app) return;
    app.toggleAttribute('data-stacked', state.stacked);
    app.dataset.narrowView = state.narrowView;
  };

  const unsubscribe = subscribe((_next, _prev, changed) => {
    const state = getState();
    if (changed.has('railCollapsed')) rail.apply(state.railCollapsed);
    if (changed.has('stacked') || changed.has('narrowView')) stampRoot(state);
    if (changed.has('drawer')) drawers.apply(state.drawer);
  });

  // Publish what is on screen now: the restored rail, and any drawer the
  // markup opened before boot.
  const collapsedNow = rail.isCollapsed();
  app?.toggleAttribute('data-rail-collapsed', collapsedNow);
  if (getState().railCollapsed !== collapsedNow) setState({ railCollapsed: collapsedNow });
  stampRoot(getState());
  const openNow = drawers.current();
  if (openNow !== null && getState().drawer !== openNow) setState({ drawer: openNow });

  return {
    destroy: () => {
      controller.abort();
      rail.destroy();
      list.destroy();
      stopObserving();
      unsubscribe();
    },
  };
}
