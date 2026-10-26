// C-DROP result rect: the rect the drop highlight paints for a drop of
// `source` on `target` — the part of the hovered tile the dragged item lands
// in (the cmux placeholder), or the new full-span strip for an outer drop.
//
//   resultRect(layout, dockRect, tabRects, source, target): ResultRect | null
//
// Rules:
//   - centre and tab targets: the target tile's own rect, plus the tab
//     insertion marker — the left edge of the tab at the insertion index, or
//     the right edge of the last tab when inserting at the end (a centre drop
//     appends); the marker spans the tile's bar (the strip with one tile);
//   - edge targets: that half of the target tile's rect, bar included, as the
//     edge split lays it out; no marker. When the dragged item's own tile
//     stays (a tab out of a multi-tab tile, a new item), it lands exactly
//     there; a tile that empties gives its space away first (rule 5), so the
//     landed tile can be larger. An axis too short to halve has no rect;
//   - outer targets (the dock-edge strips): the new full-span strip the dragged
//     item lands in, planDrop's result laid out on `dockRect`; no marker;
//   - border targets have no pointer zone, so no highlight: null;
//   - an unknown target tile, or a drop planDrop refuses, has no rect: null.
//
// Pure and dependency-free apart from sibling model modules: bundled into
// dock-engine.mjs. Never mutates its inputs.

import { planDrop, tileOf } from './drop.js';
import type { DockPx, Side, Source, Target } from './drop.js';
import { geometry } from './geometry.js';
import type { Rect } from './geometry.js';
import { MIN_PX, halvable } from './hitTest.js';
import type { TabRects } from './hitTest.js';
import type { Layout } from './layout.js';

/** The highlight for one drop (structurally identical to cases.ts). */
export interface ResultRect {
  rect: Rect;
  /** The tab insertion gap, for centre and tab targets; null otherwise. */
  marker: { x: number; y: number; height: number } | null;
}

type Marker = ResultRect['marker'];

/** The insertion gap at `index` in a tile's measured bar, or null with no bar. */
function markerAt(tabRects: TabRects, tile: string, index: number | null): Marker {
  const bar = tabRects[tile];
  if (bar === undefined) return null;
  const tabs = bar.tabRects;
  const at = index === null ? tabs.length : Math.max(0, Math.min(index, tabs.length));
  let x: number;
  if (at < tabs.length) x = tabs[at]!.x;
  else if (tabs.length > 0) x = tabs[tabs.length - 1]!.x + tabs[tabs.length - 1]!.width;
  else x = bar.barRect.x;
  return { x, y: bar.barRect.y, height: bar.barRect.height };
}

/** A key no layout holds: the stand-in whose landing the edge highlight shows. */
const PROBE = '\u0000highlight';

/**
 * The `edge` half of tile `id` as an edge split lays it out: the landed rect
 * of a new item dropped there, so a split that flattens into its parent rounds
 * the same way. Null when the tile's span along that axis cannot be halved.
 */
function halfOf(layout: Layout, dockRect: Rect, dock: DockPx, id: string, edge: Side): Rect | null {
  const rect = geometry(layout, dockRect, MIN_PX).tiles[id]!;
  if (!halvable(edge === 'left' || edge === 'right' ? rect.width : rect.height)) return null;
  const after = planDrop(layout, { new: PROBE }, { tile: id, edge }, dock).layout;
  const landed = tileOf(after, PROBE);
  return landed === null ? null : { ...geometry(after!, dockRect, MIN_PX).tiles[landed.tile]! };
}

/** The key or tile id whose holder the highlight follows. */
function holderOf(after: Layout | null, source: Source): string | null {
  if ('tile' in source) return source.tile;
  const key = 'tab' in source ? source.tab : source.new;
  return tileOf(after, key)?.tile ?? null;
}

/** The new strip an outer drop of `source` lands in, or null when it would not land. */
function stripOf(
  layout: Layout,
  dockRect: Rect,
  dock: DockPx,
  source: Source,
  side: Side
): Rect | null {
  const result = planDrop(layout, source, { outer: side }, dock);
  if (!result.changed || result.layout === null) return null;
  const holder = holderOf(result.layout, source);
  const rect =
    holder === null ? undefined : geometry(result.layout, dockRect, MIN_PX).tiles[holder];
  return rect === undefined ? null : { ...rect };
}

/**
 * The part of the target tile the dragged item lands in after dropping
 * `source` on `target`, with the tab insertion marker for centre and tab
 * targets, or the new strip of an outer drop; null when the drop would not
 * land or has no pointer zone.
 */
export function resultRect(
  layout: Layout,
  dockRect: Rect,
  tabRects: TabRects,
  source: Source,
  target: Target
): ResultRect | null {
  const dock: DockPx = { px: { width: dockRect.width, height: dockRect.height }, minPx: MIN_PX };
  if ('outer' in target) {
    const strip = stripOf(layout, dockRect, dock, source, target.outer);
    return strip === null ? null : { rect: strip, marker: null };
  }
  if (!('tile' in target)) return null;
  const rect = geometry(layout, dockRect, MIN_PX).tiles[target.tile];
  if (rect === undefined) return null;
  if ('zone' in target) {
    const index = target.zone === 'tab' ? target.index : null;
    return { rect: { ...rect }, marker: markerAt(tabRects, target.tile, index) };
  }
  if (!planDrop(layout, source, target, dock).changed) return null;
  const half = halfOf(layout, dockRect, dock, target.tile, target.edge);
  return half === null ? null : { rect: half, marker: null };
}
