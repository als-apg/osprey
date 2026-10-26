// geometry: where every tile of a C-LAYOUT tree sits on the dock, and where
// the interior borders between sibling subtrees run.
//
// The pixel arithmetic reproduces dockview-core 7.0.2's splitview, so a hit
// test or highlight computed here agrees with the rendered dock within 2 px:
//
//   - each child of a split gets clamp(round(fraction * span), its minimum)
//     (`splitview.js` layout with saved proportions);
//   - the remaining difference (span - sum) is then handed out from the LAST
//     child to the first, each child clamped to its minimum
//     (`distributeEmptySpace`);
//   - a tile's minimum is `minPx` on both axes; a split's minimum along its
//     own axis is the sum of its children's, across it the largest
//     (`branchNode.js` minimumSize / minimumOrthogonalSize).
//
// dockview's maximum sizes are unbounded for our groups, and the docks set no
// gaps, so the tile rects tile the dock rect exactly when it is feasible.
//
// The px -> fraction conversion (`fractionsOf`) lives here too: the dockview
// serializer and the holder's `resized` sync both read dockview's px sizes
// back into C-LAYOUT fractions through it.
//
// Pure and dependency-free apart from ./layout.js: bundled into dock-engine.mjs.

import { isTile } from './layout.js';
import type { Direction, Layout } from './layout.js';

// --- types ------------------------------------------------------------------

/** A rectangle in the dock's coordinate space (whatever space `dockRect` is in). */
export interface Rect {
  x: number;
  y: number;
  width: number;
  height: number;
}

/** A straight line segment; a border's line is vertical in a row, horizontal in a column. */
export interface Line {
  x1: number;
  y1: number;
  x2: number;
  y2: number;
}

export type Axis = 'width' | 'height';

/** One interior border: the boundary between two adjacent siblings of a split. */
export interface Border {
  /** Direction of the split the border lies in ('row': a vertical line). */
  split: Direction;
  /** Item indexes from the root down to that split ([] when it is the root). */
  path: number[];
  /** The border lies between `items[index - 1]` and `items[index]`. */
  index: number;
  /** Tile ids of the subtree before the border (left or above), in tree order. */
  before: string[];
  /** Tile ids of the subtree after the border (right or below), in tree order. */
  after: string[];
  line: Line;
}

export interface Geometry {
  /** Each tile's rect, keyed by tile id. */
  tiles: Record<string, Rect>;
  /** Every interior border, in tree order (a split's own borders before its children's). */
  borders: Border[];
  /** False when the dock cannot give every tile `minPx` on both axes. */
  feasible: boolean;
}

// --- sizes ------------------------------------------------------------------

/** The axis a split of this direction lays its items along. */
export function axisOf(direction: Direction): Axis {
  return direction === 'row' ? 'width' : 'height';
}

/**
 * The smallest px a subtree can take along `axis` under dockview's minimum
 * constraints, with `minPx` per tile.
 */
export function minimumSize(node: Layout<unknown>, axis: Axis, minPx: number): number {
  if (isTile(node)) return minPx;
  const mins = node.items.map((item) => minimumSize(item, axis, minPx));
  return axisOf(node.split) === axis ? mins.reduce((a, b) => a + b, 0) : Math.max(...mins);
}

/**
 * The fractions of a split's items: given sizes kept, missing sizes share the
 * remainder equally, then scaled to sum to 1 (the C-LAYOUT reading of sizes).
 */
export function itemFractions(items: readonly { size?: number }[]): number[] {
  const given = items.filter((item) => typeof item.size === 'number' && item.size > 0);
  const givenSum = given.reduce((sum, item) => sum + (item.size as number), 0);
  const missing = items.length - given.length;
  const share = missing > 0 ? Math.max(0, 1 - givenSum) / missing : 0;
  const raw = items.map((item) =>
    typeof item.size === 'number' && item.size > 0 ? item.size : share
  );
  const total = raw.reduce((a, b) => a + b, 0);
  return total > 0 ? raw.map((r) => r / total) : items.map(() => 1 / items.length);
}

/**
 * Lay `span` px out over items with these fractions and minimums, exactly as
 * dockview's splitview does: round and clamp each, then hand the difference
 * out from the last item to the first, each clamped to its minimum. When the
 * minimums exceed the span the items stay at their minimums (overflow).
 */
export function layoutSizes(
  fractions: readonly number[],
  span: number,
  minimums: readonly number[]
): number[] {
  const sizes = fractions.map((f, i) => Math.max(Math.round(f * span), minimums[i] ?? 0));
  let delta = span - sizes.reduce((a, b) => a + b, 0);
  for (let i = sizes.length - 1; delta !== 0 && i >= 0; i--) {
    const next = Math.max(sizes[i]! + delta, minimums[i] ?? 0);
    delta -= next - sizes[i]!;
    sizes[i] = next;
  }
  return sizes;
}

/**
 * px sizes (as dockview reports them for one split's items) -> C-LAYOUT
 * fractions. Each fraction is rounded to 4 decimals and the last absorbs the
 * rounding, so the result sums to exactly 1. A zero or empty total gives
 * equal shares.
 */
export function fractionsOf(pxSizes: readonly number[]): number[] {
  const n = pxSizes.length;
  if (n === 0) return [];
  const total = pxSizes.reduce((a, b) => a + Math.max(0, b), 0);
  const raw = total > 0 ? pxSizes.map((px) => Math.max(0, px) / total) : pxSizes.map(() => 1 / n);
  const rounded = raw.map((f) => Math.round(f * 10000) / 10000);
  const head = rounded.slice(0, -1).reduce((a, b) => a + b, 0);
  rounded[n - 1] = Math.round((1 - head) * 10000) / 10000;
  return rounded;
}

// --- geometry ---------------------------------------------------------------

function tileIds(node: Layout<unknown>, out: string[] = []): string[] {
  if (isTile(node)) out.push(node.tile);
  else for (const item of node.items) tileIds(item, out);
  return out;
}

/**
 * Each tile's rect and each interior border of `layout` laid out on
 * `dockRect`, with dockview's `minPx` minimum on every tile.
 */
export function geometry(layout: Layout<unknown>, dockRect: Rect, minPx: number): Geometry {
  const tiles: Record<string, Rect> = {};
  const borders: Border[] = [];

  const place = (node: Layout<unknown>, rect: Rect, path: number[]): void => {
    if (isTile(node)) {
      tiles[node.tile] = rect;
      return;
    }
    const axis = axisOf(node.split);
    const minimums = node.items.map((item) => minimumSize(item, axis, minPx));
    const span = axis === 'width' ? rect.width : rect.height;
    const sizes = layoutSizes(itemFractions(node.items), span, minimums);
    let offset = axis === 'width' ? rect.x : rect.y;
    node.items.forEach((item, i) => {
      const size = sizes[i]!;
      if (i > 0) {
        const line: Line =
          axis === 'width'
            ? { x1: offset, y1: rect.y, x2: offset, y2: rect.y + rect.height }
            : { x1: rect.x, y1: offset, x2: rect.x + rect.width, y2: offset };
        borders.push({
          split: node.split,
          path,
          index: i,
          before: tileIds(node.items[i - 1]!),
          after: tileIds(item),
          line,
        });
      }
      offset += size;
    });
    offset = axis === 'width' ? rect.x : rect.y;
    node.items.forEach((item, i) => {
      const size = sizes[i]!;
      const child: Rect =
        axis === 'width'
          ? { x: offset, y: rect.y, width: size, height: rect.height }
          : { x: rect.x, y: offset, width: rect.width, height: size };
      place(item, child, [...path, i]);
      offset += size;
    });
  };

  place(layout, { ...dockRect }, []);
  const feasible =
    dockRect.width >= minimumSize(layout, 'width', minPx) &&
    dockRect.height >= minimumSize(layout, 'height', minPx);
  return { tiles, borders, feasible };
}
