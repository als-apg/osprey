// @ts-check
/* OSPREY Web Terminal — the tile ⋯ menu's shape
 *
 * A tile has ONE ⋯ menu (the button in its bar and a right-click on the bar
 * open the same one), with two sections:
 *
 *   This panel — the panel's own rows: its contributed `menu` entries, the
 *     rows the module owning its content registered (setServiceMenuExtras),
 *     and whatever the bar's fold ladder took out of a narrow tile. The
 *     section and its heading are absent when there is nothing to show;
 *     destructive rows (`danger`) go last, after a divider, in their own
 *     order; an entry's `separator` draws a divider before it.
 *   Tile — the hub's verbs on the tile (open beside, move, pop out, reload,
 *     link, help, rail membership, layout). Its heading shows only when a
 *     "This panel" section sits above it; alone it needs none.
 *
 * This module is the one place that shape lives. It is pure: rows in, rows
 * out, for the design-system context menu (context-menu.js) to render.
 * panel-menu-policy.js decides WHICH rows each section holds.
 */

/**
 * @typedef {import('/design-system/js/context-menu.js').MenuItem} MenuItem
 * @typedef {MenuItem & { separator?: boolean }} PanelRow  a "This panel" row;
 *   `separator` asks for a divider before it (a contributed menu entry's field)
 */

/** The section captions, as the menu shows them. */
export const THIS_PANEL_HEADING = 'This panel';
export const TILE_HEADING = 'Tile';

/**
 * Compose the two sections into one row list.
 * @param {{ thisPanel: PanelRow[], tile: MenuItem[] }} sections
 * @returns {MenuItem[]}
 */
export function composeTileMenu({ thisPanel, tile }) {
  const panelRows = tidy(dangerLast(thisPanel));
  const tileRows = tidy(tile);
  if (!panelRows.length) return tileRows;
  return [{ heading: THIS_PANEL_HEADING }, ...panelRows, { heading: TILE_HEADING }, ...tileRows];
}

/**
 * Expand `separator` flags into divider rows and move destructive rows to
 * the end, after a divider of their own. A danger row's own `separator`
 * is spent on that divider: the danger block is already set apart.
 * @param {PanelRow[]} rows
 * @returns {MenuItem[]}
 */
function dangerLast(rows) {
  /** @type {MenuItem[]} */
  const safe = [];
  /** @type {MenuItem[]} */
  const danger = [];
  for (const row of rows) {
    const { separator, ...item } = row;
    if (item.danger) {
      danger.push(item);
      continue;
    }
    if (separator) safe.push({ divider: true });
    safe.push(item);
  }
  return danger.length ? [...safe, { divider: true }, ...danger] : safe;
}

/**
 * Drop dividers that separate nothing: at either end, or doubled.
 * @param {MenuItem[]} rows
 * @returns {MenuItem[]}
 */
function tidy(rows) {
  /** @type {MenuItem[]} */
  const out = [];
  for (const row of rows) {
    if (row.divider && (!out.length || out[out.length - 1].divider)) continue;
    out.push(row);
  }
  while (out.length && out[out.length - 1].divider) out.pop();
  return out;
}
