// @ts-check
/* OSPREY Web Terminal — Dock placement rules
 *
 * Where a panel lands when it enters the dock, as pure operations on a dock
 * layout: each takes the held layout and the holder's context and returns the
 * layout after, whether it changed and the tile to focus (dock-apply.js's
 * `run`). Nothing here reads or mutates the dock; the holder applies the
 * result.
 *
 * THE OPEN RULE. A panel that is opened — by a sidebar row click, by the
 * agent, by a link, by the "+" menu — lands as a NEW TAB, IN FRONT, in the
 * service tile the operator used last ({@link openTab}): nothing resizes, and
 * the tab it covers is one click away. A panel already in the dock is shown
 * where it is. With no service tile at all, the first one opens LEFT of the
 * agent pane at the classic 60/40 split, or against the dock's left edge when
 * the agent pane is closed too ({@link firstTile}) — the agent pane is always
 * its own tile and never the target.
 *
 *   - OPEN AT (a drop on a tile edge or a dock edge, the row's "Open in a new
 *     tile") splits where the gesture said, and moves a panel that already
 *     holds a tab ({@link openAt}) — the one way to make a new tile on purpose;
 *   - the terminal reopens at the dock's right edge, its home side
 *     ({@link reopenTerminal}), and closes as its tile ({@link closeKey}).
 */

import {
  findTile,
  isTile,
  itemFractions,
  mintTileId,
  planDrop,
  removeTab,
  tileIdsOf,
  tileOf,
} from '/static/vendor/dock-engine.mjs';
import { TERMINAL_KEY } from './dock-apply.js';

/** @typedef {import('/static/vendor/dock-engine.mjs').Layout} Layout */
/** @typedef {import('/static/vendor/dock-engine.mjs').Tile} Tile */
/** @typedef {import('/static/vendor/dock-engine.mjs').DockPx} DockPx */
/** @typedef {import('/static/vendor/dock-engine.mjs').Side} Side */
/** @typedef {import('./dock-apply.js').OpResult} OpResult */
/** @typedef {import('./dock-apply.js').OpContext} OpContext */

/**
 * Where a panel is placed relative to a tile, or to the dock when no tile is
 * named: the four sides.
 * @typedef {object} Position
 * @property {string} [referenceTile]  the tile to split; the dock's edge when absent
 * @property {Side} direction
 */

/** Share of the row the first service tile takes beside the terminal — the
 *  service side of the classic split (terminal 40% / services 60%). */
export const SERVICE_SHARE = 0.6;
/** Share of the row the terminal takes when it reopens at the dock's edge. */
export const TERMINAL_SHARE = 0.4;

const SIDES = new Set(['left', 'right', 'top', 'bottom']);

/**
 * @param {Layout|null} layout
 * @returns {OpResult}
 */
function unchanged(layout) {
  return { layout, changed: false };
}

/**
 * The service tile the operator used last — where an open lands: the focused
 * tile when it shows a service, else the last tile that did, else the first
 * tile in tree order showing a service. Null when no tile shows a service.
 * The agent pane's tile is never chosen because the agent pane is in it.
 * @param {Layout|null} layout
 * @param {OpContext} ctx
 * @returns {Tile|null}
 */
export function serviceTileTarget(layout, ctx) {
  for (const id of [ctx.focusedTile, ctx.lastServiceTile]) {
    const tile = id === null ? null : findTile(layout, id);
    if (tile && tile.showing !== TERMINAL_KEY) return tile;
  }
  for (const id of tileIdsOf(layout)) {
    const tile = findTile(layout, id);
    if (tile && tile.showing !== TERMINAL_KEY) return tile;
  }
  return null;
}

/**
 * `layout` with tile `id` taking `share` of its parent split; its siblings
 * keep their proportions over the rest. A root tile has no split to share.
 * @param {Layout} layout
 * @param {string} id
 * @param {number} share
 * @returns {Layout}
 */
function withShare(layout, id, share) {
  if (isTile(layout)) return layout;
  const index = layout.items.findIndex((item) => isTile(item) && item.tile === id);
  if (index === -1) {
    return { ...layout, items: layout.items.map((item) => withShare(item, id, share)) };
  }
  const fractions = itemFractions(layout.items);
  const rest = fractions.reduce((sum, f, i) => (i === index ? sum : sum + f), 0);
  const items = layout.items.map((item, i) => ({
    ...item,
    size: i === index ? share : rest > 0 ? (fractions[i] / rest) * (1 - share) : 0,
  }));
  return { ...layout, items };
}

/**
 * The first service tile: left of the terminal at the classic split, or the
 * dock's left edge when the terminal tile is closed too, or alone in an empty
 * dock.
 * @param {Layout|null} layout
 * @param {string} key
 * @param {DockPx} dock
 * @returns {OpResult}
 */
export function firstTile(layout, key, dock) {
  if (layout === null) {
    const tile = mintTileId(null);
    return { layout: { tile, tabs: [key], showing: key }, changed: true, focus: tile };
  }
  const terminal = tileOf(layout, TERMINAL_KEY);
  const target = terminal
    ? { tile: terminal.tile, edge: /** @type {Side} */ ('left') }
    : { outer: /** @type {Side} */ ('left') };
  const out = planDrop(layout, { new: key }, target, dock);
  const home = out.changed ? tileOf(out.layout, key) : null;
  if (!out.changed || !home || out.layout === null) return unchanged(layout);
  const next = terminal ? withShare(out.layout, home.tile, SERVICE_SHARE) : out.layout;
  return { layout: next, changed: true, focus: home.tile };
}

/**
 * Open `key` by the open rule: a panel already in the dock stays where it is
 * (nothing changes here — the caller brings its tab forward, and the focused
 * tile follows that); otherwise it lands as a new tab, in front, in the
 * service tile the operator used last — nothing resizes — or, with no service
 * tile, as the first service tile beside the agent pane.
 * @param {Layout|null} layout
 * @param {string} key
 * @param {OpContext} ctx
 * @returns {OpResult}
 */
export function openTab(layout, key, ctx) {
  if (tileOf(layout, key)) return unchanged(layout);
  const target = serviceTileTarget(layout, ctx);
  if (!target || layout === null) return firstTile(layout, key, ctx.dock);
  const out = planDrop(layout, { new: key }, { tile: target.tile, zone: 'centre' }, ctx.dock);
  if (!out.changed || !tileOf(out.layout, key)) return unchanged(layout);
  return { layout: out.layout, changed: true, focus: target.tile };
}

/**
 * Open `key` at an explicit position — a new tile split off the reference
 * tile on the given side, or against the dock's edge when none is named — and
 * MOVE it there when it already holds a tile. Two guarded no-ops: a position
 * naming the panel's own tile (splitting a tile off itself), and a panel
 * already docked with no position at all. Without a position a panel not yet
 * docked opens beside the focused tile, else as the first service tile.
 * @param {Layout|null} layout
 * @param {string} key
 * @param {Position|null} position
 * @param {OpContext} ctx
 * @returns {OpResult}
 */
export function openAt(layout, key, position, ctx) {
  const home = tileOf(layout, key);
  if (!position) {
    if (home) return unchanged(layout);
    const focused = ctx.focusedTile === null ? null : findTile(layout, ctx.focusedTile);
    if (!focused || layout === null) return firstTile(layout, key, ctx.dock);
    position = { referenceTile: focused.tile, direction: 'right' };
  }
  if (!SIDES.has(position.direction)) return unchanged(layout);
  if (home && position.referenceTile === home.tile) return unchanged(layout);
  if (position.referenceTile !== undefined && !findTile(layout, position.referenceTile)) {
    return unchanged(layout);
  }
  const target =
    position.referenceTile === undefined
      ? { outer: position.direction }
      : { tile: position.referenceTile, edge: position.direction };
  const out = planDrop(layout, { new: key }, target, ctx.dock);
  const landed = out.changed ? tileOf(out.layout, key) : null;
  if (!out.changed || !landed) return unchanged(layout);
  return { layout: out.layout, changed: true, focus: landed.tile };
}

/**
 * Bring the terminal back at the dock's right edge, its home side, at its
 * share of the width. An open terminal changes nothing.
 * @param {Layout|null} layout
 * @param {DockPx} dock
 * @returns {OpResult}
 */
export function reopenTerminal(layout, dock) {
  const home = tileOf(layout, TERMINAL_KEY);
  if (home) return { layout, changed: false, focus: home.tile };
  const out = planDrop(layout, { new: TERMINAL_KEY }, { outer: 'right' }, dock);
  const landed = out.changed ? tileOf(out.layout, TERMINAL_KEY) : null;
  if (!out.changed || !landed || out.layout === null) return unchanged(layout);
  return {
    layout: withShare(out.layout, landed.tile, TERMINAL_SHARE),
    changed: true,
    focus: landed.tile,
  };
}

/**
 * Take `key` out of the dock: its tab leaves its tile, and a tile left empty
 * closes, handing its space to its neighbour (the engine's leave rule). No
 * close record is kept. A key not in the dock changes nothing.
 * @param {Layout|null} layout
 * @param {string} key
 * @returns {OpResult}
 */
export function closeKey(layout, key) {
  if (!tileOf(layout, key)) return unchanged(layout);
  return { layout: removeTab(layout, key).layout, changed: true };
}

/**
 * {@link closeKey} for several keys at once, as one change.
 * @param {Layout|null} layout
 * @param {Iterable<string>} keys
 * @returns {OpResult}
 */
export function closeKeys(layout, keys) {
  let next = layout;
  let changed = false;
  for (const key of keys) {
    const out = closeKey(next, key);
    if (out.changed) {
      next = out.layout;
      changed = true;
    }
  }
  return { layout: next, changed };
}

/**
 * The default arrangement: the terminal card alone in one tile. Services join
 * it left of the terminal at the classic split when they open.
 * @returns {Layout}
 */
export function defaultLayout() {
  const tile = mintTileId(null);
  return { tile, tabs: [TERMINAL_KEY], showing: TERMINAL_KEY };
}
