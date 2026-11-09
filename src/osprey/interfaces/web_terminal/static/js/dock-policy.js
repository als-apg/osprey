// @ts-check
/* OSPREY Web Terminal — Dock Policy: one panel per tile
 *
 * The engine's grammar (the vendored dock-engine, CANOPY's) lets a tile hold
 * several panels; OSPREY's dock holds ONE. The rule is OSPREY's, so it lives
 * here, above the engine and apart from it, as two pure functions on a layout:
 *
 *   - {@link crowdedTile} names the first tile that breaks the rule, which is
 *     what the holder (dock-apply.js) refuses an operation's result for and
 *     what an arrangement is refused for, by the one name
 *     {@link ONE_PANEL_PER_TILE} the server uses too;
 *   - {@link onePanelPerTile} MIGRATES a layout that breaks it — a dock stored
 *     by the release that had tabs — keeping in each tile the panel it was
 *     showing: the others leave the dock (they keep their sidebar rows) and
 *     are listed, so a caller can say what left.
 *
 * Nothing here reads or mutates the dock, and nothing is imported but the
 * engine, so every dock module may use it without pulling the page in.
 */

import { isTile } from '/static/vendor/dock-engine.mjs';

/** @typedef {import('/static/vendor/dock-engine.mjs').Layout} Layout */
/** @typedef {import('/static/vendor/dock-engine.mjs').Tile} Tile */

/** The refusal (named, and it changes nothing): a tile would hold several panels. */
export const ONE_PANEL_PER_TILE = 'one_panel_per_tile';

/**
 * Every tile of a layout, in tree order.
 * @param {Layout|null} layout
 * @returns {Tile[]}
 */
function tilesOf(layout) {
  if (layout === null) return [];
  if (isTile(layout)) return [layout];
  return layout.items.flatMap(tilesOf);
}

/**
 * The first tile, in tree order, holding more than one panel; null when every
 * tile holds one (an empty dock holds none).
 * @param {Layout|null} layout
 * @returns {Tile|null}
 */
export function crowdedTile(layout) {
  return tilesOf(layout).find((tile) => tile.tabs.length > 1) ?? null;
}

/**
 * `layout` with every tile reduced to the panel it shows. A layout that
 * already keeps the rule is returned as is (the same object), with nothing
 * evicted; the input is never mutated.
 * @param {Layout|null} layout
 * @returns {{ layout: Layout|null, evicted: string[] }}  the panels that left,
 *   in tree order
 */
export function onePanelPerTile(layout) {
  if (crowdedTile(layout) === null) return { layout, evicted: [] };
  /** @type {string[]} */
  const evicted = [];
  /** @param {Layout} node @returns {Layout} */
  const keep = (node) => {
    if (!isTile(node)) return { ...node, items: node.items.map(keep) };
    if (node.tabs.length === 1) return node;
    evicted.push(...node.tabs.filter((key) => key !== node.showing));
    return { ...node, tabs: [node.showing] };
  };
  return { layout: keep(/** @type {Layout} */ (layout)), evicted };
}
