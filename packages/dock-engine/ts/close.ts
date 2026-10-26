// C-DROP: closing and reopening, built on the removeTab/removeTile/splitTile
// primitives and openItem.
//
//   closeTab(layout, key)          -> CloseResult   (one tab or tile record)
//   closeTile(layout, id)          -> CloseResult   (one tile record)
//   closeOthers(layout, key)       -> CloseResult   (one `tabs` group record)
//   closeRight(layout, key)        -> CloseResult   (one `tabs` group record)
//   closeAll(layout)               -> CloseResult   (one `layout` group record)
//   reopen(layout, record, ctx)    -> CloseResult
//
// Closing (PROPOSAL rules 5 and 8):
//   - a hidden tab goes and nothing else changes; the showing tab hands
//     `showing` to the next tab, else the previous one;
//   - a tile's last tab removes the tile under the C-LAYOUT leave rule and
//     records the tile with the neighbour that took its space, its side, and
//     the tile's `share` of its parent split along that split's axis (none for
//     the only tile);
//   - closeOthers/closeRight act within the key's tile; the kept key is shown
//     when the showing tab was closed. Group records list tab records in
//     ascending pre-close index;
//   - a close that closes nothing is `changed: false` with a null record.
// Reopening:
//   - a tab goes back into its tile at its index (clamped) and is shown;
//   - a tile splits its recorded neighbour's recorded side, keeping its id when
//     free, and takes its recorded `share` of the split it lands in from the
//     neighbour only: clamped so both keep `dock.minPx`, no other tile moves.
//     A record without a share (stored before shares) halves the neighbour.
//     An empty dock gets the tile alone;
//   - a `tabs` group restores each tab at its index, ascending, and shows the one
//     recorded as showing, else keeps the tile's showing tab;
//   - a `layout` group replaces the dock;
//   - when the tile or neighbour is gone (or the split is infeasible), rule 7
//     through openItem.
//
// Pure: never mutates its input. Bundled into dock-engine.mjs, so it imports
// only sibling model modules.
import { findTile, removeTab, removeTile, splitTile, tileOf } from './drop.js';
import type { Side } from './drop.js';
import { axisOf, geometry, itemFractions } from './geometry.js';
import { isTile, keysOf, mintTileId, tileIdsOf } from './layout.js';
import type { Layout, Split, Tile } from './layout.js';
import { openItem } from './open.js';
import type { OpenContext } from './open.js';

// --- records ----------------------------------------------------------------------

/** One closed tab: where it sat and whether it was showing. */
export interface TabRecord {
  kind: 'tab';
  key: string;
  tile: string;
  index: number;
  showing: boolean;
}

/**
 * One closed tile: its tabs, the neighbour (and side of it) that took its
 * space, and its share (0..1) of its parent split along that split's axis.
 */
export interface TileRecord {
  kind: 'tile';
  tile: string;
  tabs: string[];
  showing: string;
  neighbour: string | null;
  side: Side | null;
  /** Absent for the only tile, and in records stored before shares. */
  share?: number;
}

/** A group close within one tile, in ascending pre-close index. */
export interface TabsGroupRecord {
  kind: 'tabs';
  records: TabRecord[];
}

/** closeAll (and an `as: 'layout'` open): the whole dock it replaced. */
export interface LayoutGroupRecord {
  kind: 'layout';
  layout: Layout | null;
}

export type CloseRecord = TabRecord | TileRecord | TabsGroupRecord | LayoutGroupRecord;

/** What the dock passes a reopen: the same context as an open. */
export type CloseContext = OpenContext;

export interface CloseResult {
  layout: Layout | null;
  changed: boolean;
  record?: CloseRecord | null;
  focus?: string | null;
}

// --- helpers ----------------------------------------------------------------------

/** Rounded to the 4 decimals sizes are stored at. */
function round4(value: number): number {
  return Math.round(value * 10000) / 10000;
}

/** The split directly holding tile `id` and its index there, or null (root or unknown). */
function parentOf(layout: Layout, id: string): { split: Split; index: number } | null {
  if (isTile(layout)) return null;
  for (let i = 0; i < layout.items.length; i++) {
    const item = layout.items[i]!;
    if (isTile(item)) {
      if (item.tile === id) return { split: layout, index: i };
    } else {
      const found = parentOf(item, id);
      if (found !== null) return found;
    }
  }
  return null;
}

/** A fresh tree with `parent` (found by identity) replaced by `next`. */
function replaceSplit(layout: Layout, parent: Split, next: Split): Layout {
  if (layout === parent) return next;
  if (isTile(layout)) return layout;
  return { ...layout, items: layout.items.map((item) => replaceSplit(item, parent, next)) };
}

function unchanged(layout: Layout | null): CloseResult {
  return { layout, changed: false, record: null };
}

/** A fresh tree with tile `id` replaced by `fn(tile)`. */
function replaceTile(layout: Layout, id: string, fn: (tile: Tile) => Tile): Layout {
  if (isTile(layout)) return layout.tile === id ? fn(layout) : layout;
  return { ...layout, items: layout.items.map((item) => replaceTile(item, id, fn)) };
}

/** Drop `closing` from `tile`, showing `keep` when the showing tab goes. */
function closeWithin(layout: Layout, tile: Tile, keep: string, closing: Set<string>): CloseResult {
  if (closing.size === 0) return unchanged(layout);
  const records: TabRecord[] = tile.tabs.flatMap((key, index) =>
    closing.has(key)
      ? [{ kind: 'tab' as const, key, tile: tile.tile, index, showing: tile.showing === key }]
      : []
  );
  const next = replaceTile(layout, tile.tile, (t) => ({
    ...t,
    tabs: t.tabs.filter((key) => !closing.has(key)),
    showing: closing.has(t.showing) ? keep : t.showing,
  }));
  return { layout: next, changed: true, record: { kind: 'tabs', records } };
}

// --- closes -----------------------------------------------------------------------

/** Close one tab (rule 8); its tile's last tab closes the tile. */
export function closeTab(layout: Layout | null, key: string): CloseResult {
  const tile = tileOf(layout, key);
  if (layout === null || tile === null) return unchanged(layout);
  if (tile.tabs.length === 1) return closeTile(layout, tile.tile);
  const removal = removeTab(layout, key);
  const record: TabRecord = {
    kind: 'tab',
    key,
    tile: tile.tile,
    index: tile.tabs.indexOf(key),
    showing: tile.showing === key,
  };
  return { layout: removal.layout, changed: true, record };
}

/** Close a tile and every tab in it, by the leave rule. */
export function closeTile(layout: Layout | null, id: string): CloseResult {
  const tile = findTile(layout, id);
  if (layout === null || tile === null) return unchanged(layout);
  const removal = removeTile(layout, id);
  const parent = parentOf(layout, id);
  const record: TileRecord = {
    kind: 'tile',
    tile: tile.tile,
    tabs: [...tile.tabs],
    showing: tile.showing,
    neighbour: removal.neighbour,
    side: removal.side,
  };
  if (parent !== null) record.share = round4(itemFractions(parent.split.items)[parent.index]!);
  return { layout: removal.layout, changed: true, record };
}

/** Close every other tab of the key's tile (rule 15). */
export function closeOthers(layout: Layout | null, key: string): CloseResult {
  const tile = tileOf(layout, key);
  if (layout === null || tile === null) return unchanged(layout);
  return closeWithin(layout, tile, key, new Set(tile.tabs.filter((k) => k !== key)));
}

/** Close the tabs right of the key in its tile (rule 15). */
export function closeRight(layout: Layout | null, key: string): CloseResult {
  const tile = tileOf(layout, key);
  if (layout === null || tile === null) return unchanged(layout);
  return closeWithin(layout, tile, key, new Set(tile.tabs.slice(tile.tabs.indexOf(key) + 1)));
}

/** Close the whole dock; the record holds it. */
export function closeAll(layout: Layout | null): CloseResult {
  if (layout === null) return unchanged(layout);
  return { layout: null, changed: true, record: { kind: 'layout', layout } };
}

// --- reopen -----------------------------------------------------------------------

/** Insert `key` into tile `id` at `index` (clamped); shown when `show`. */
function insertTab(layout: Layout, id: string, key: string, index: number, show: boolean): Layout {
  return replaceTile(layout, id, (t) => {
    const tabs = [...t.tabs];
    tabs.splice(Math.min(index, tabs.length), 0, key);
    return { ...t, tabs, showing: show ? key : t.showing };
  });
}

/** Show `key` in the tile holding it. */
function show(layout: Layout, key: string): Layout {
  const home = tileOf(layout, key)!;
  return replaceTile(layout, home.tile, (t) => ({ ...t, showing: key }));
}

function reopenTab(layout: Layout | null, record: TabRecord, ctx: CloseContext): CloseResult {
  const home = layout === null ? null : findTile(layout, record.tile);
  if (layout === null || home === null || keysOf(layout).includes(record.key)) {
    const out = openItem(layout, record.key, ctx);
    return { layout: out.layout, changed: out.changed, focus: out.focus ?? null };
  }
  const next = insertTab(layout, home.tile, record.key, record.index, true);
  return { layout: next, changed: true, focus: home.tile };
}

function reopenTabs(layout: Layout | null, group: TabsGroupRecord, ctx: CloseContext): CloseResult {
  let next = layout;
  let changed = false;
  let focus: string | null = null;
  const ordered = [...group.records].sort((a, b) => a.index - b.index);
  for (const record of ordered) {
    const home = next === null ? null : findTile(next, record.tile);
    if (next !== null && home !== null && !keysOf(next).includes(record.key)) {
      next = insertTab(next, home.tile, record.key, record.index, false);
      focus = home.tile;
      changed = true;
    } else {
      const out = openItem(next, record.key, ctx);
      next = out.layout;
      focus = out.focus ?? focus;
      changed = changed || out.changed;
    }
  }
  // openItem shows each fallback tab; restore the recorded showing tab last.
  const shown = ordered.find((r) => r.showing);
  if (next !== null && shown !== undefined && tileOf(next, shown.key) !== null) {
    next = show(next, shown.key);
    changed = true;
  }
  return { layout: next, changed, focus };
}

/**
 * After an edge split put tile `id` beside `neighbour`, give `id` `share` of the
 * split holding both, taken from the neighbour only. Clamped so each keeps
 * `dock.minPx`; when the two together have less than twice that (the split
 * already borrowed from elsewhere) the split is left as it is.
 */
function restoreShare(
  layout: Layout,
  id: string,
  neighbour: string,
  share: number,
  dock: CloseContext['dock']
): Layout {
  const held = parentOf(layout, id);
  const other = parentOf(layout, neighbour);
  if (held === null || other === null || held.split !== other.split) return layout;
  const { split, index } = held;
  const axis = axisOf(split.split);
  const rects = geometry(
    layout,
    { x: 0, y: 0, width: dock.px.width, height: dock.px.height },
    dock.minPx
  ).tiles;
  const pairPx = rects[id]![axis] + rects[neighbour]![axis];
  if (pairPx < 2 * dock.minPx) return layout;
  const fractions = itemFractions(split.items);
  const pair = fractions[index]! + fractions[other.index]!;
  const wantPx = (share / pair) * pairPx;
  const px = Math.min(Math.max(wantPx, dock.minPx), pairPx - dock.minPx);
  const mine = round4((px / pairPx) * pair);
  const items = split.items.map((item, i) => {
    if (i === index) return { ...item, size: mine };
    if (i === other.index) return { ...item, size: round4(pair - mine) };
    return item;
  });
  return replaceSplit(layout, split, { ...split, items });
}

function reopenTile(layout: Layout | null, record: TileRecord, ctx: CloseContext): CloseResult {
  const tabs = record.tabs.filter((key) => !keysOf(layout).includes(key));
  if (tabs.length === 0) return { layout, changed: false };
  const showing = tabs.includes(record.showing) ? record.showing : tabs[0]!;
  if (layout === null) {
    return { layout: { tile: record.tile, tabs, showing }, changed: true, focus: record.tile };
  }

  const id = tileIdsOf(layout).includes(record.tile) ? mintTileId(layout) : record.tile;
  if (record.neighbour !== null && record.side !== null && findTile(layout, record.neighbour)) {
    const tile: Tile = { tile: id, tabs, showing, from: record.neighbour };
    const split = splitTile(layout, record.neighbour, record.side, tile, ctx.dock);
    if (split !== null) {
      const next =
        typeof record.share === 'number' && record.share > 0 && record.share < 1
          ? restoreShare(split, id, record.neighbour, record.share, ctx.dock)
          : split;
      return { layout: next, changed: true, focus: id };
    }
  }

  // Rule 7: the first tab opens as any item would; the rest join its tile.
  const first = openItem(layout, tabs[0]!, ctx);
  let next = first.layout!;
  const holder = tileOf(next, tabs[0]!)!;
  for (const key of tabs.slice(1)) {
    next = insertTab(next, holder.tile, key, Number.POSITIVE_INFINITY, false);
  }
  return { layout: show(next, showing), changed: true, focus: holder.tile };
}

/** Reopen a close record (rule 8), falling back to rule 7 through openItem. */
export function reopen(layout: Layout | null, record: CloseRecord, ctx: CloseContext): CloseResult {
  switch (record.kind) {
    case 'tab':
      return reopenTab(layout, record, ctx);
    case 'tabs':
      return reopenTabs(layout, record, ctx);
    case 'tile':
      return reopenTile(layout, record, ctx);
    case 'layout':
      return {
        layout: record.layout,
        changed: JSON.stringify(record.layout) !== JSON.stringify(layout),
      };
  }
}
