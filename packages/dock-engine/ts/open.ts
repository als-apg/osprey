// C-DROP: opening items and sets into the dock, built on planDrop.
//
//   openItem(layout, key, ctx)                    -> OpenResult
//   openSet(layout, members, as, collection, ctx) -> OpenResult
//
// openItem (PROPOSAL rule 7, the CANOPY form):
//   - an empty dock gets one tile holding the item;
//   - a Target from `ctx.pickTarget` (OSPREY's rule 7 supplies one) is planned
//     through planDrop, an open key moving there as a tab would;
//   - else an open key is shown where it is and its tile gets focus;
//   - else the item becomes the showing tab of the focused tile, no size change.
// openSet (rules 15/16):
//   - `tabs`: one new tile holding every member, split off the focused tile's
//     right edge (`from` = the focused tile);
//   - `side`: the focused tile's span shared equally among it and the members,
//     in a row, else a near-square grid (cols = ceil(sqrt n), rows =
//     ceil(n / cols), filled row by row, the focused tile first); when no cell
//     would get `dock.minPx` an agent is refused (`too_many_for_side`) and a
//     person's click falls back to `tabs` with that notice;
//   - `layout`: the collection layout replaces the dock, skipped members pruned,
//     tile ids minted from t1 in tree order; the record holds the dock before.
//
// Pure: never mutates its input. Bundled into dock-engine.mjs, so it imports
// only sibling model modules.
import { findTile, planDrop, splitTile, tileOf } from './drop.js';
import type { DockPx, Target } from './drop.js';
import { geometry } from './geometry.js';
import { isTile, keysOf, mintTileId, normalize, normalizeCollection, tileIdsOf } from './layout.js';
import type { CollectionLayout, Layout, NormalizedCollection, Tile } from './layout.js';

/** What the dock passes an open: structurally the cases' EngineContext. */
export interface OpenContext {
  focusedTile: string | null;
  dock: DockPx;
  pickTarget?: (layout: Layout | null, key: string) => Target | null;
  caller?: 'agent' | 'person';
  /** Members the dock cannot open (missing, unavailable, over the tab cap). */
  skip?: string[];
}

/** The record an `as: 'layout'` open leaves: the whole dock it replaced. */
export interface LayoutRecord {
  kind: 'layout';
  layout: Layout | null;
}

export interface OpenResult {
  layout: Layout | null;
  changed: boolean;
  focus?: string | null;
  record?: LayoutRecord;
  refusal?: string;
  notice?: string;
  skipped?: string[];
}

export const TOO_MANY_FOR_SIDE = 'too_many_for_side';

// --- tree helpers ---------------------------------------------------------------

function sized<T extends Layout>(node: T, size: number | undefined): T {
  const out = { ...node };
  delete out.size;
  return size === undefined ? out : { ...out, size };
}

/** A fresh tree with tile `id` replaced by `fn(tile)`. */
function replaceTile(layout: Layout, id: string, fn: (tile: Tile) => Layout): Layout {
  if (isTile(layout)) return layout.tile === id ? fn(layout) : layout;
  return { ...layout, items: layout.items.map((item) => replaceTile(item, id, fn)) };
}

/** The focused tile, else the first tile in tree order. */
function focusedOf(layout: Layout, focusedTile: string | null): Tile {
  const focused = focusedTile === null ? null : findTile(layout, focusedTile);
  return focused ?? findTile(layout, tileIdsOf(layout)[0]!)!;
}

function idSeries(layout: Layout | null, count: number): string[] {
  const first = Number(mintTileId(layout).slice(1));
  return Array.from({ length: count }, (_, i) => `t${first + i}`);
}

function newTile(id: string, tabs: string[]): Tile {
  return { tile: id, tabs, showing: tabs[0]! };
}

// --- openItem -------------------------------------------------------------------

/** Open one item by rule 7; `focus` names the tile that shows it. */
export function openItem(layout: Layout | null, key: string, ctx: OpenContext): OpenResult {
  if (layout === null) {
    return { layout: newTile('t1', [key]), changed: true, focus: 't1' };
  }

  const picked = ctx.pickTarget?.(layout, key) ?? null;
  if (picked !== null) {
    const out = planDrop(layout, { new: key }, picked, ctx.dock);
    const holder = out.changed ? tileOf(out.layout, key) : null;
    if (holder !== null) return { layout: out.layout, changed: true, focus: holder.tile };
  }

  const home = tileOf(layout, key);
  if (home !== null) {
    if (home.showing === key) return { layout, changed: false, focus: home.tile };
    const shown = replaceTile(layout, home.tile, (tile) => ({ ...tile, showing: key }));
    return { layout: shown, changed: true, focus: home.tile };
  }

  const focused = focusedOf(layout, ctx.focusedTile);
  const out = planDrop(layout, { new: key }, { tile: focused.tile, zone: 'centre' }, ctx.dock);
  return { layout: out.layout, changed: out.changed, focus: focused.tile };
}

// --- openSet --------------------------------------------------------------------

function openAsTabs(layout: Layout | null, members: string[], ctx: OpenContext): OpenResult {
  if (layout === null) return { layout: newTile('t1', members), changed: true, focus: 't1' };
  const focused = focusedOf(layout, ctx.focusedTile);
  const id = mintTileId(layout);
  const split = splitTile(
    layout,
    focused.tile,
    'right',
    { ...newTile(id, members), from: focused.tile },
    ctx.dock
  );
  if (split !== null) return { layout: split, changed: true, focus: id };
  // No room for a new tile: the members join the focused tile as tabs instead.
  const joined = replaceTile(layout, focused.tile, (tile) => ({
    ...tile,
    tabs: [...tile.tabs, ...members],
    showing: members[0]!,
  }));
  return { layout: joined, changed: true, focus: focused.tile };
}

function rowOf(cells: Tile[], size: number | undefined): Layout {
  return sized({ split: 'row', items: cells.map((cell) => sized(cell, 1 / cells.length)) }, size);
}

function openAsSide(layout: Layout | null, members: string[], ctx: OpenContext): OpenResult {
  const { width, height } = ctx.dock.px;
  const minPx = ctx.dock.minPx;
  if (layout === null) {
    const ids = idSeries(null, members.length);
    const cells = members.map((m, i) => newTile(ids[i]!, [m]));
    return { layout: normalize(rowOf(cells, undefined)), changed: true, focus: ids[0]! };
  }

  const focused = focusedOf(layout, ctx.focusedTile);
  const rect = geometry(layout, { x: 0, y: 0, width, height }, minPx).tiles[focused.tile]!;
  const n = members.length + 1;
  const ids = idSeries(layout, members.length);
  const cells = [sized(focused, undefined), ...members.map((m, i) => newTile(ids[i]!, [m]))];

  let block: Layout | null = null;
  if (rect.width / n >= minPx) {
    block = rowOf(cells, focused.size);
  } else {
    const cols = Math.ceil(Math.sqrt(n));
    const rows = Math.ceil(n / cols);
    if (rect.width / cols >= minPx && rect.height / rows >= minPx) {
      const items: Layout[] = [];
      for (let r = 0; r < rows; r++)
        items.push(rowOf(cells.slice(r * cols, (r + 1) * cols), 1 / rows));
      block = sized({ split: 'column', items }, focused.size);
    }
  }

  if (block === null) {
    if (ctx.caller === 'person') {
      return { ...openAsTabs(layout, members, ctx), notice: TOO_MANY_FOR_SIDE };
    }
    return { layout, changed: false, refusal: TOO_MANY_FOR_SIDE };
  }
  const placed = block;
  return {
    layout: normalize(replaceTile(layout, focused.tile, () => placed)),
    changed: true,
    focus: ids[0]!,
  };
}

/** The collection with skipped keys pruned; emptied tiles and splits dropped. */
function prune(node: CollectionLayout<string>, skip: Set<string>): CollectionLayout<string> | null {
  if ('split' in node) {
    const items = node.items
      .map((item) => prune(item, skip))
      .filter((item): item is CollectionLayout<string> => item !== null);
    return items.length === 0 ? null : { ...node, items };
  }
  const tabs = node.tabs.filter((key) => !skip.has(key));
  if (tabs.length === 0) return null;
  const { showing, ...rest } = node;
  return showing !== undefined && tabs.includes(showing)
    ? { ...rest, tabs, showing }
    : { ...rest, tabs };
}

/** A normalised collection as a dock tree, tile ids minted from t1 in tree order. */
function toDock(node: NormalizedCollection<string>, next: { n: number }): Layout {
  if ('split' in node) {
    return sized(
      { split: node.split, items: node.items.map((item) => toDock(item, next)) },
      node.size
    );
  }
  const tile: Tile = { tile: `t${next.n++}`, tabs: [...node.tabs], showing: node.showing };
  return sized(tile, node.size);
}

function openAsLayout(
  layout: Layout | null,
  collection: CollectionLayout<string>,
  skipped: string[]
): OpenResult {
  const pruned = prune(collection, new Set(skipped));
  if (pruned === null) return { layout, changed: false, skipped };
  const dock = normalize(toDock(normalizeCollection(pruned), { n: 1 }));
  return {
    layout: dock,
    changed: true,
    focus: tileIdsOf(dock)[0]!,
    record: { kind: 'layout', layout },
    skipped,
  };
}

/**
 * Open a set of members `as` tabs, side by side, or by its collection layout.
 * Members in `ctx.skip` are left out and reported in `skipped`.
 */
export function openSet(
  layout: Layout | null,
  members: string[],
  as: 'tabs' | 'side' | 'layout',
  collection: CollectionLayout<string> | undefined,
  ctx: OpenContext
): OpenResult {
  const skip = new Set(ctx.skip ?? []);
  const skipped = members.filter((key) => skip.has(key));
  if (as === 'layout' && collection !== undefined) {
    return openAsLayout(layout, collection, skipped);
  }

  // Tabs and side add to the dock: a member already open is not opened twice.
  const open = new Set(keysOf(layout));
  const fresh = members.filter((key) => !skip.has(key) && !open.has(key));
  const extra = skipped.length > 0 ? { skipped } : {};
  if (fresh.length === 0) return { layout, changed: false, ...extra };
  const out = as === 'side' ? openAsSide(layout, fresh, ctx) : openAsTabs(layout, fresh, ctx);
  return { ...out, ...extra };
}
