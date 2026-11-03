// @ts-check
/* OSPREY Web Terminal — Rail Drag (drag a sidebar row into the workspace)
 *
 * A sidebar row is one more drag source of the dock, decided by the same
 * engine as a header tab or a tile. At drag start the row becomes an engine
 * source — `{new: panel}`, or `{tab: panel}` when the panel already has a tab
 * in the held layout (so it moves, never duplicates) — and goes into module
 * state through dock-tab's dockDragStart, the state the dock root's listener
 * (dock-drop.js) reads. From there a row drag is every other drag: the same
 * shield, the same live-drag hold on the layout holder, the same hitTest and
 * preview and the same planDrop result. A row makes a NEW tile: the bridge
 * admits its edge and dock-edge targets and refuses a tile's header and centre
 * (rail-drag.js never places anything itself).
 *
 * The row is a plain HTML5 drag source: panel-rail.js sets `draggable` and
 * forwards dragstart/dragend through panel-menu-policy to railDragStart /
 * railDragEnd.
 *
 * Landing: once a drop has put the panel in the held layout and the held apply
 * has rebuilt the dock, `onRailDrop(id, {wasOnScreen})` runs, so the caller can
 * reveal the panel through the activation tail every other open uses (health
 * check, iframe, focus).
 *
 * Lost gestures: HTML5 delivers dragend only to the drag's SOURCE element, and
 * a row removed mid-drag (an SSE rail change, another client) never gets it. A
 * capture-phase document failsafe, armed per drag, ends the drag on dragend,
 * mouseup or pointerup (native drags suppress the last two until the pointer
 * is released). A drop ends it one turn later, after the dock root's own drop
 * handler has read the source.
 */

import { getDockApi } from './dock-workspace.js';
import { getLayoutHolder } from './dock-apply.js';
import { clearDockDrag, dockDragSource, dockDragStart } from './dock-tab.js';
import { keysOf } from '/static/vendor/dock-engine.mjs';

/** @typedef {import('/static/vendor/dock-engine.mjs').Source} Source */
/** @typedef {import('/static/vendor/dock-engine.mjs').Layout} Layout */
/** @typedef {import('./dock-apply.js').LayoutHolder} LayoutHolder */

/** DataTransfer type carrying the dragged row's panel id. */
export const RAIL_DRAG_MIME = 'application/x-osprey-panel';

/** How many turns the landing waits for a live drag to end before running anyway. */
const SETTLE_TURNS = 20;

/**
 * @typedef {object} RailDragOptions
 * @property {(id: string, info: { wasOnScreen: boolean }) => void} [onRailDrop]
 *   runs once a rail drop has placed the panel and the dock shows it
 * @property {(id: string) => string} [labelOf]  the name a dropped row's tab is
 *   built with (the row's label); the id when absent
 * @property {() => LayoutHolder | null} [holder]  the page's holder; getLayoutHolder by default
 */

/** @type {RailDragOptions['onRailDrop'] | null} */
let landingHandler = null;
/** @type {(id: string) => string} */
let labelOf = (id) => id;
/** @type {() => LayoutHolder | null} */
let holderOf = getLayoutHolder;

/**
 * The rail drag in progress: its panel id and the engine source it became.
 * @type {{ id: string, source: Source } | null}
 */
let current = null;

/**
 * Register the landing handler (and, for tests, the holder getter). Safe to
 * call again; the last call wins.
 * @param {RailDragOptions} [options]
 */
export function initRailDrag(options = {}) {
  landingHandler = options.onRailDrop ?? null;
  labelOf = options.labelOf ?? ((id) => id);
  holderOf = options.holder ?? getLayoutHolder;
}

/** @returns {string | null} the panel id of the rail drag in progress */
export function railDragSource() {
  return current?.id ?? null;
}

/**
 * The engine source a row becomes: its tab when the held layout already has
 * the panel, else a new tab.
 * @param {string} id
 * @param {LayoutHolder | null} holder
 * @returns {Source}
 */
export function railSourceOf(id, holder) {
  const layout = holder?.layout ?? null;
  return layout !== null && keysOf(layout).includes(id) ? { tab: id } : { new: id };
}

// --- the failsafe and the landing watch -----------------------------------------------

const IMMEDIATE_TERMINATORS = ['dragend', 'mouseup', 'pointerup'];

/** @type {(() => void) | null} */
let disarm = null;

/**
 * Arm the document listeners for one drag: the terminators, and the landing
 * watch around the dock root's drop (capture before it, bubble after it).
 * @param {{ id: string, source: Source }} drag
 */
function arm(drag) {
  /** @type {Layout | null} */
  let before = null;
  const end = () => railDragEnd();
  const beforeDrop = () => {
    before = holderOf()?.layout ?? null;
  };
  const afterDrop = () => {
    const holder = holderOf();
    const after = holder?.layout ?? null;
    if (holder && after !== null && after !== before && keysOf(after).includes(drag.id)) {
      land(holder, drag.id, 'tab' in drag.source);
    }
    setTimeout(() => {
      if (current === drag) railDragEnd();
    }, 0);
  };
  for (const type of IMMEDIATE_TERMINATORS) document.addEventListener(type, end, true);
  document.addEventListener('drop', beforeDrop, true);
  document.addEventListener('drop', afterDrop, false);
  disarm = () => {
    for (const type of IMMEDIATE_TERMINATORS) document.removeEventListener(type, end, true);
    document.removeEventListener('drop', beforeDrop, true);
    document.removeEventListener('drop', afterDrop, false);
    disarm = null;
  };
}

/**
 * Run the landing handler once the drag is over and the held apply has run:
 * a turn after the holder stops being live, since its apply is queued then.
 * @param {LayoutHolder} holder
 * @param {string} id
 * @param {boolean} wasOnScreen
 */
function land(holder, id, wasOnScreen) {
  const handler = landingHandler;
  if (!handler) return;
  let turns = 0;
  const step = () => {
    if (holder.dragLive && turns++ < SETTLE_TURNS) setTimeout(step, 0);
    else setTimeout(() => handler(id, { wasOnScreen }), 0);
  };
  setTimeout(step, 0);
}

// --- start and end ----------------------------------------------------------------------

/**
 * Begin a row drag: the row becomes the dock's drag source (the bridge raises
 * the shield and holds the holder while the drag is live). Returns false —
 * cancelling the drag (panel-rail preventDefaults) — in fallback mode, where
 * there is no dock to drop into.
 * @param {string} id  the panel's rail id (its layout key)
 * @param {DataTransfer | null} dataTransfer
 * @returns {boolean} whether the drag may proceed
 */
export function railDragStart(id, dataTransfer) {
  if (!getDockApi()) return false;
  if (current) railDragEnd();
  const holder = holderOf();
  const source = railSourceOf(id, holder);
  // A new tab is built with the row's name, as every other open names it.
  if ('new' in source) holder?.setPanelRecord(id, { title: labelOf(id) });
  const drag = { id, source };
  current = drag;
  dockDragStart(source, dataTransfer);
  dataTransfer?.setData(RAIL_DRAG_MIME, id);
  arm(drag);
  return true;
}

/** End a row drag (drop, cancel or failsafe): forget it, clear the dock's
 *  source unless another drag has taken over, disarm the failsafe. Idempotent. */
export function railDragEnd() {
  disarm?.();
  const drag = current;
  current = null;
  if (drag && dockDragSource() === drag.source) clearDockDrag();
}
