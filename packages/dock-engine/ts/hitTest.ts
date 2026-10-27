// C-DROP hit precedence: which Target a pointer at `point` means while
// dragging `source` over the dock, or null for no target.
//
// Every point of a tile is a target:
//   1. a tile's header row (its measured bar, the one-tile strip, and anything
//      between the tile's top and its content) means "add as tab" at the
//      pointer's index; right of the last tab that is an append. The bar is
//      checked before the off-dock test, since the strip lies outside `dockRect`;
//   2. off the dock: null;
//   3. the dock-edge strips, STRIP_PX deep: the dock's outermost left, right and
//      bottom STRIP_PX, then, in a top-row tile, the first STRIP_PX of content
//      under its header row. Each means an outer drop on that side: a new
//      full-span tile of a third of the dock;
//   4. over tile content (tile rect minus its bar): the left or right edge
//      within EDGE_RATIO of the content's width (at least EDGE_PX, at most
//      half), then the top or bottom edge the same way along its height, so
//      left and right win at the corners;
//   5. the tile's centre.
//
// The zones are the same whichever tile the drag started in: a neighbour's band
// facing the source tile splits it like any other band. An edge highlights, and
// the drop lands in, exactly half of the hovered tile; an axis too short to
// halve into two MIN_PX tiles has no edges, so its points fall through to the
// other axis or the centre. A self-drop (the dragged tile, or a tab alone in
// its tile, landing on its own tile, or the only tile on a dock-edge strip)
// gets null.
//
// Pure and dependency-free apart from ./layout.js and ./geometry.js: bundled
// into dock-engine.mjs. Never mutates its inputs.

import { geometry } from './geometry.js';
import type { Rect } from './geometry.js';
import { isTile, tileIdsOf } from './layout.js';
import type { Layout, Tile } from './layout.js';

// --- the drop vocabulary (C-DROP; structurally identical to cases.ts) --------

export type Side = 'left' | 'right' | 'top' | 'bottom';

export type Source = { new: string } | { tab: string } | { tile: string };

export type Target =
  | { tile: string; zone: 'centre' }
  | { tile: string; zone: 'tab'; index: number }
  | { tile: string; edge: Side }
  | { border: { tile: string; side: Side } }
  | { outer: Side };

/** Per tile, the bar and tab rects the dock measured (with one tile, the strip's). */
export type TabRects = Record<string, { barRect: Rect; tabRects: Rect[] }>;

export interface Point {
  x: number;
  y: number;
}

// --- constants --------------------------------------------------------------

/** The tile floor every dock enforces. */
export const MIN_PX = 40;
/** Smallest edge depth; larger tiles use EDGE_RATIO of their content. */
export const EDGE_PX = 80;
/** Edge depth as a share of the tile content along that axis. */
export const EDGE_RATIO = 0.31;
/** Depth of the dock-edge strips that mean an outer drop. */
export const STRIP_PX = 48;

// --- helpers ----------------------------------------------------------------

function inside(r: Rect, p: Point): boolean {
  return p.x >= r.x && p.x < r.x + r.width && p.y >= r.y && p.y < r.y + r.height;
}

function insideClosed(r: Rect, p: Point): boolean {
  return p.x >= r.x && p.x <= r.x + r.width && p.y >= r.y && p.y <= r.y + r.height;
}

function findTile(layout: Layout, id: string): Tile | null {
  if (isTile(layout)) return layout.tile === id ? layout : null;
  for (const item of layout.items) {
    const found = findTile(item, id);
    if (found) return found;
  }
  return null;
}

function tileOfKey(layout: Layout, key: string): Tile | null {
  if (isTile(layout)) return layout.tabs.includes(key) ? layout : null;
  for (const item of layout.items) {
    const found = tileOfKey(item, key);
    if (found) return found;
  }
  return null;
}

/**
 * The tile a source moves as a whole: a `{tile}` source, or a tab alone in its
 * tile (`{new}` of an already open key acts as `{tab}`). Null when the source
 * moves a single tab out of a multi-tab tile, or is genuinely new.
 */
function movingTile(layout: Layout, source: Source): string | null {
  if ('tile' in source) return findTile(layout, source.tile) ? source.tile : null;
  const key = 'tab' in source ? source.tab : source.new;
  const tile = tileOfKey(layout, key);
  return tile && tile.tabs.length === 1 ? tile.tile : null;
}

/** The tile's content: its rect minus its bar when the bar lies inside it. */
function contentOf(rect: Rect, bars: TabRects[string] | undefined): Rect {
  if (!bars) return rect;
  const bar = bars.barRect;
  const barBottom = bar.y + bar.height;
  if (bar.y < rect.y - 1 || bar.y >= rect.y + rect.height || barBottom <= rect.y) return rect;
  const top = Math.min(barBottom, rect.y + rect.height);
  return { x: rect.x, y: top, width: rect.width, height: rect.y + rect.height - top };
}

/** Halving `span` px leaves each half at least MIN_PX. */
export function halvable(span: number): boolean {
  return Math.round(span / 2) >= MIN_PX;
}

// --- zones ------------------------------------------------------------------

/** The tab gap at `x`: the count of tabs whose centre lies left of it. */
function tabIndex(tabRects: Rect[], x: number): number {
  return tabRects.filter((r) => r.x + r.width / 2 < x).length;
}

function barHit(layout: Layout, tabRects: TabRects, point: Point): Target | null {
  for (const id of tileIdsOf(layout)) {
    const bars = tabRects[id];
    if (!bars || !inside(bars.barRect, point)) continue;
    return { tile: id, zone: 'tab', index: tabIndex(bars.tabRects, point.x) };
  }
  return null;
}

/** The edge band depth along a content span: max(EDGE_PX, EDGE_RATIO), at most half. */
function depthOf(span: number): number {
  return Math.min(Math.max(EDGE_PX, span * EDGE_RATIO), span / 2);
}

function edgeHit(rect: Rect, content: Rect, point: Point): Side | null {
  if (halvable(rect.width)) {
    const depth = depthOf(content.width);
    if (point.x - content.x < depth) return 'left';
    if (point.x - content.x > content.width - depth) return 'right';
  }
  if (halvable(rect.height)) {
    const depth = depthOf(content.height);
    if (point.y - content.y < depth) return 'top';
    if (point.y - content.y > content.height - depth) return 'bottom';
  }
  return null;
}

/** The dock-edge strip under `point` in tile `rect`, or null; see rule 3. */
function stripHit(dockRect: Rect, rect: Rect, content: Rect, point: Point): Side | null {
  if (point.x < dockRect.x + STRIP_PX) return 'left';
  if (point.x > dockRect.x + dockRect.width - STRIP_PX) return 'right';
  if (point.y > dockRect.y + dockRect.height - STRIP_PX) return 'bottom';
  const topRow = rect.y <= dockRect.y + 1;
  if (topRow && point.y < content.y + STRIP_PX) return 'top';
  return null;
}

// --- hitTest ----------------------------------------------------------------

/**
 * The Target a drop of `source` at `point` would mean, or null: off the dock
 * or a self-drop. Tile rects come from `geometry` on `dockRect` with the 40 px
 * floor; bars and tabs from the dock's `tabRects`.
 */
export function hitTest(
  layout: Layout,
  dockRect: Rect,
  tabRects: TabRects,
  point: Point,
  source: Source
): Target | null {
  const self = movingTile(layout, source);
  const unlessSelf = (target: Target): Target | null =>
    'tile' in target && target.tile === self ? null : target;

  // 1. Bars (and the one-tile strip) win, before the off-dock test.
  const bar = barHit(layout, tabRects, point);
  if (bar) return unlessSelf(bar);

  // 2. Off the dock.
  if (!insideClosed(dockRect, point)) return null;

  const g = geometry(layout, dockRect, MIN_PX);
  const id =
    Object.keys(g.tiles).find((t) => inside(g.tiles[t]!, point)) ??
    Object.keys(g.tiles).find((t) => insideClosed(g.tiles[t]!, point));
  if (id === undefined) return null;
  const rect = g.tiles[id]!;
  const content = contentOf(rect, tabRects[id]);

  // 1, continued: the rest of the header row.
  if (point.y < content.y) {
    const index = tabIndex(tabRects[id]?.tabRects ?? [], point.x);
    return unlessSelf({ tile: id, zone: 'tab', index });
  }

  // 3. Dock-edge strips; the only tile moving out to a strip stays put.
  const strip = stripHit(dockRect, rect, content, point);
  if (strip) return self !== null && isTile(layout) ? null : { outer: strip };

  // 4. Edge, then 5. centre.
  const edge = edgeHit(rect, content, point);
  return unlessSelf(edge ? { tile: id, edge } : { tile: id, zone: 'centre' });
}
