// @ts-check
/**
 * ARIEL panel layout: the one screen's panes, splitters, stacked mode,
 * drawers and gear menu.
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
 *   to the strip, so expanding returns a usable rail. Default collapsed
 *   embedded, expanded standalone.
 * - `stacked`: a ResizeObserver on the panel root (below STACK_BELOW px the
 *   list and reader share one column and `narrowView` picks which shows).
 *   Back in the reader's pane head writes `narrowView:'list'`.
 * - `drawer`: `#status-drawer` and `#settings-drawer`. A store write opens or
 *   closes them; every open/close the drawers do themselves (Esc, backdrop,
 *   the display menu's Settings entry, an unsaved-changes veto) is written
 *   back from their `drawer:open` / `drawer:close` events, never assumed.
 *   `initSettings` removes `#settings-drawer` when the config panel is
 *   disabled, so it must run first; `drawer:'settings'` is then a no-op.
 *
 * Root attributes CSS keys off (layout.css): `data-rail-collapsed`,
 * `data-stacked`, `data-narrow-view` on `.app-container`.
 *
 * Widths persist per frame kind (embedded vs standalone) under
 * `scopedStorageKey`, so the hub tile and the standalone page never
 * overwrite each other. The list column's width is the only list/reader
 * split stored: the reader takes the remaining space.
 */

import { initSplitter, parseStoredState } from '/design-system/js/splitter.js';
import { scopedStorageKey } from '/design-system/js/storage-scope.js';
import { isEmbedded } from '/design-system/js/frame-params.js';
import { closePopover, togglePopover } from '/design-system/js/popover.js';
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

/** @returns {'embedded'|'standalone'} */
function frameKind() {
  return isEmbedded() ? 'embedded' : 'standalone';
}

/** @returns {string} storage key of the rail's size + collapsed record */
export function railStorageKey() {
  return scopedStorageKey(`ariel-rail-${frameKind()}`);
}

/** @returns {string} storage key of the list column's width */
export function listStorageKey() {
  return scopedStorageKey(`ariel-list-width-${frameKind()}`);
}

/** @param {string} key @returns {string|null} */
function readStored(key) {
  try {
    return localStorage.getItem(key);
  } catch {
    return null;
  }
}

/** @type {ReadonlyArray<[DrawerName, string, string]>} name, element id, menu label */
const DRAWERS = [
  ['status', 'status-drawer', 'Status'],
  ['settings', 'settings-drawer', 'Settings'],
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
  const embedded = isEmbedded();
  const railKey = railStorageKey();
  const storedRail = parseStoredState(readStored(railKey), embedded).size;
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
    defaultCollapsed: embedded,
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
 * @returns {{has: (name: DrawerName) => boolean, current: () => DrawerName|null,
 *   apply: (want: DrawerName|null) => void}}
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
    has: (name) => drawers.has(name),
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
 * The gear button and its popover menu: one entry per drawer that exists.
 * @param {HTMLElement|null} gear
 * @param {HTMLElement|null} menu
 * @param {(name: DrawerName) => boolean} hasDrawer
 * @param {AbortSignal} signal
 */
function wireGearMenu(gear, menu, hasDrawer, signal) {
  if (!gear || !menu) return;
  if (!gear.querySelector('svg')) gear.prepend(icon('settings'));
  menu.replaceChildren();
  for (const [name, , label] of DRAWERS) {
    if (!hasDrawer(name)) continue;
    const item = document.createElement('button');
    item.type = 'button';
    item.className = 'osprey-popover-item';
    item.setAttribute('role', 'menuitem');
    item.dataset.arielDrawer = name;
    item.textContent = label;
    item.addEventListener(
      'click',
      () => {
        closePopover(menu);
        setState({ drawer: name });
      },
      { signal }
    );
    menu.append(item);
  }
  gear.addEventListener('click', () => togglePopover(gear, menu, { place: 'below-end' }), {
    signal,
  });
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
  wireGearMenu(byId('ariel-gear-btn'), byId('ariel-gear-menu'), drawers.has, signal);

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
