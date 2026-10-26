// C-DROP: planning a drop on a C-LAYOUT tree, and the tree primitives every
// other operation (close, reopen, open) builds on.
//
//   planDrop(layout, source, target, dock) -> { layout, changed }
//   replaceKey(layout, oldKey, newKey)     -> { layout, changed }
//
// The rules (PROPOSAL rules 1-6), as the C-CASES pin them:
//   1. centre / tab bar: the item becomes a tab of that tile and is shown;
//   2. edge: the target tile splits into two equal halves (`from` = target);
//   3. border: a new tile between the two sides, 1/n of that split; the others
//      keep their ratios, and an item leaving that same split keeps the ratios
//      it had before the drag; self-drops change nothing;
//   4. outer: a full-span tile of one third, the rest share two thirds;
//   5. leaving: a tile that empties gives its space to one neighbour (the
//      C-LAYOUT leave rule: its `from` subtree, else previous, else next);
//   6. a whole tile moves with its tabs and keeps which one is showing.
// A tile that would get less than `dock.minPx` is clamped to it, the target's
// leave-rule neighbour giving the difference; when that neighbour cannot, the
// drop is infeasible and changes nothing.
//
// Pure: every function returns a fresh tree and never mutates its input.
// Bundled into dock-engine.mjs, so it imports only ./layout.js and ./geometry.js.
import { axisOf, fractionsOf, itemFractions, layoutSizes, minimumSize } from './geometry.js';
import type { Axis, Rect } from './geometry.js';
import { isTile, mintTileId, normalize, tileIdsOf } from './layout.js';
import type { Direction, Layout, Split, Tile } from './layout.js';

// --- the drop vocabulary (structurally the same as cases.ts) ------------------

export type Side = 'left' | 'right' | 'top' | 'bottom';

export type Source = { new: string } | { tab: string } | { tile: string };

export type Target =
  | { tile: string; zone: 'centre' }
  | { tile: string; zone: 'tab'; index: number }
  | { tile: string; edge: Side }
  | { border: { tile: string; side: Side } }
  | { outer: Side };

export interface DockPx {
  px: { width: number; height: number };
  minPx: number;
}

/** What planDrop and replaceKey return: the layout after, and whether it differs. */
export interface DropResult {
  layout: Layout | null;
  changed: boolean;
}

/** A removed tile's layout, the neighbour that took its space and the side of it the tile sat on. */
export interface Removal {
  layout: Layout | null;
  neighbour: string | null;
  side: Side | null;
}

type Path = number[];

// --- reading ------------------------------------------------------------------

/** The direction a side's split runs in ('left'/'right': a row). */
export function directionOf(side: Side): Direction {
  return side === 'left' || side === 'right' ? 'row' : 'column';
}

function isBefore(side: Side): boolean {
  return side === 'left' || side === 'top';
}

function nodeAt(layout: Layout, path: Path): Layout {
  let node = layout;
  for (const i of path) node = (node as Split).items[i]!;
  return node;
}

function pathOfTile(layout: Layout | null, id: string): Path | null {
  if (layout === null) return null;
  if (isTile(layout)) return layout.tile === id ? [] : null;
  for (let i = 0; i < layout.items.length; i++) {
    const sub = pathOfTile(layout.items[i]!, id);
    if (sub !== null) return [i, ...sub];
  }
  return null;
}

/** The tile with this id, or null. */
export function findTile(layout: Layout | null, id: string): Tile | null {
  const path = pathOfTile(layout, id);
  return path === null ? null : (nodeAt(layout!, path) as Tile);
}

/** The tile holding this key, or null. */
export function tileOf(layout: Layout | null, key: string): Tile | null {
  if (layout === null) return null;
  if (isTile(layout)) return layout.tabs.includes(key) ? layout : null;
  for (const item of layout.items) {
    const found = tileOf(item, key);
    if (found !== null) return found;
  }
  return null;
}

/** Replace the node at `path` with `node` (a fresh spine; untouched subtrees are shared). */
function replaceAt(layout: Layout, path: Path, node: Layout): Layout {
  if (path.length === 0) return node;
  const split = layout as Split;
  const [head, ...rest] = path;
  const items = split.items.map((item, i) => (i === head ? replaceAt(item, rest, node) : item));
  return { ...split, items };
}

function withSize<T extends Layout>(node: T, size: number | undefined): T {
  const copy = { ...node };
  if (size === undefined) delete copy.size;
  else copy.size = size;
  return copy;
}

/** The rect of the node at `path` laid out on the dock, as geometry() lays it out. */
function rectAt(layout: Layout, path: Path, dock: DockPx): Rect {
  let rect: Rect = { x: 0, y: 0, width: dock.px.width, height: dock.px.height };
  let node = layout;
  for (const i of path) {
    const split = node as Split;
    const sizes = itemPx(split, rect, dock.minPx);
    const offset = sizes.slice(0, i).reduce((a, b) => a + b, 0);
    rect =
      split.split === 'row'
        ? { x: rect.x + offset, y: rect.y, width: sizes[i]!, height: rect.height }
        : { x: rect.x, y: rect.y + offset, width: rect.width, height: sizes[i]! };
    node = split.items[i]!;
  }
  return rect;
}

function itemPx(split: Split, rect: Rect, minPx: number): number[] {
  const axis = axisOf(split.split);
  const minimums = split.items.map((item) => minimumSize(item, axis, minPx));
  return layoutSizes(
    itemFractions(split.items),
    axis === 'width' ? rect.width : rect.height,
    minimums
  );
}

function spanOf(rect: Rect, axis: Axis): number {
  return axis === 'width' ? rect.width : rect.height;
}

/** Set a split's item sizes from fractions (rounded to 4 decimals, summing to 1). */
function sized(split: Split, fractions: number[]): Split {
  const rounded = fractionsOf(fractions);
  return { ...split, items: split.items.map((item, i) => withSize(item, rounded[i])) };
}

// --- primitives -------------------------------------------------------------

/**
 * Remove a tile per the C-LAYOUT leave rule: its space goes to the adjacent
 * sibling subtree holding its `from` tile, else the previous sibling, else the
 * next. The result is normalised (a one-item split collapses, a stale `from` is
 * cleared). `neighbour` is the `from` tile when that subtree grew, else the
 * adjacent tile of the sibling that did; `side` is the side of it the tile sat on.
 */
export function removeTile(layout: Layout | null, id: string): Removal {
  const path = pathOfTile(layout, id);
  if (layout === null || path === null) return { layout, neighbour: null, side: null };
  if (path.length === 0) return { layout: null, neighbour: null, side: null };
  const parentPath = path.slice(0, -1);
  const index = path[path.length - 1]!;
  const parent = nodeAt(layout, parentPath) as Split;
  const leaving = parent.items[index] as Tile;
  const from = leaving.from;
  const holds = (j: number): boolean =>
    from !== undefined &&
    parent.items[j] !== undefined &&
    tileIdsOf(parent.items[j]!).includes(from);
  let donor: number;
  if (holds(index - 1)) donor = index - 1;
  else if (holds(index + 1)) donor = index + 1;
  else donor = index > 0 ? index - 1 : index + 1;

  const fractions = itemFractions(parent.items);
  fractions[donor] = fractions[donor]! + fractions[index]!;
  fractions.splice(index, 1);
  const items = parent.items.filter((_, j) => j !== index);
  const next = replaceAt(layout, parentPath, sized({ ...parent, items }, fractions));

  const donorIds = tileIdsOf(parent.items[donor]!);
  const neighbour =
    from !== undefined && donorIds.includes(from)
      ? from
      : donor < index
        ? donorIds[donorIds.length - 1]!
        : donorIds[0]!;
  const before = donor < index;
  const side: Side =
    parent.split === 'row' ? (before ? 'right' : 'left') : before ? 'bottom' : 'top';
  return { layout: normalize(next), neighbour, side };
}

/**
 * Remove a tab from the tile holding it. Its tile's last tab removes the tile
 * (removeTile); a showing tab hands `showing` to the next tab, else the previous.
 */
export function removeTab(layout: Layout | null, key: string): Removal {
  const tile = tileOf(layout, key);
  if (layout === null || tile === null) return { layout, neighbour: null, side: null };
  if (tile.tabs.length === 1) return removeTile(layout, tile.tile);
  const index = tile.tabs.indexOf(key);
  const tabs = tile.tabs.filter((k) => k !== key);
  const showing =
    tile.showing === key ? (tile.tabs[index + 1] ?? tile.tabs[index - 1]!) : tile.showing;
  const path = pathOfTile(layout, tile.tile)!;
  return {
    layout: replaceAt(layout, path, { ...tile, tabs, showing }),
    neighbour: null,
    side: null,
  };
}

/**
 * Split tile `id` at `edge`, putting `tile` on that side; the two halves share
 * the tile's span equally. Clamped to `dock.minPx`: when a half would be
 * smaller, both halves get minPx and the leave-rule neighbour of the target
 * (along that axis) gives the difference. Null when that is infeasible or the
 * tile is unknown. The caller sets `from` on `tile`.
 */
export function splitTile(
  layout: Layout,
  id: string,
  edge: Side,
  tile: Tile,
  dock: DockPx
): Layout | null {
  const path = pathOfTile(layout, id);
  if (path === null) return null;
  const direction = directionOf(edge);
  const axis = axisOf(direction);
  const target = nodeAt(layout, path) as Tile;
  const halves: Layout[] = isBefore(edge)
    ? [withSize(tile, 0.5), withSize(target, 0.5)]
    : [withSize(target, 0.5), withSize(tile, 0.5)];
  let out = replaceAt(layout, path, { split: direction, items: halves, ...sizeOf(target) });

  // Every split between the target and X runs across the axis, so X's span
  // along the axis is the target's.
  let x = path;
  while (x.length > 0 && (nodeAt(layout, x.slice(0, -1)) as Split).split !== direction)
    x = x.slice(0, -1);
  const span = spanOf(rectAt(layout, path, dock), axis);
  const need = 2 * dock.minPx;
  if (Math.round(span / 2) >= dock.minPx) return normalize(out);
  if (x.length === 0) return null;

  const parentPath = x.slice(0, -1);
  const index = x[x.length - 1]!;
  const parent = nodeAt(layout, parentPath) as Split;
  const holds = (j: number): boolean =>
    target.from !== undefined &&
    parent.items[j] !== undefined &&
    tileIdsOf(parent.items[j]!).includes(target.from);
  let donor: number;
  if (holds(index - 1)) donor = index - 1;
  else if (holds(index + 1)) donor = index + 1;
  else donor = index > 0 ? index - 1 : index + 1;
  const px = itemPx(parent, rectAt(layout, parentPath, dock), dock.minPx);
  const deficit = need - px[index]!;
  if (px[donor]! - deficit < minimumSize(parent.items[donor]!, axis, dock.minPx)) return null;
  px[index] = px[index]! + deficit;
  px[donor] = px[donor]! - deficit;
  const resized = sized(nodeAt(out, parentPath) as Split, px);
  out = replaceAt(out, parentPath, resized);
  return normalize(out);
}

function sizeOf(node: Layout): { size?: number } {
  return node.size === undefined ? {} : { size: node.size };
}

/** Where a border target resolves to: the split and the index the new item takes. */
export interface BorderSlot {
  path: Path;
  index: number;
}

/**
 * C-DROP border resolution: the nearest ancestor split along the side's axis
 * in which the tile's subtree has a sibling on that side. The new item goes
 * between those two siblings, at `index` of that split. Null at the dock edge.
 */
export function resolveBorder(
  layout: Layout,
  border: { tile: string; side: Side }
): BorderSlot | null {
  const path = pathOfTile(layout, border.tile);
  if (path === null) return null;
  const direction = directionOf(border.side);
  for (let depth = path.length - 1; depth >= 0; depth--) {
    const parentPath = path.slice(0, depth);
    const parent = nodeAt(layout, parentPath) as Split;
    if (parent.split !== direction) continue;
    const i = path[depth]!;
    if (isBefore(border.side) && i > 0) return { path: parentPath, index: i };
    if (!isBefore(border.side) && i + 1 < parent.items.length)
      return { path: parentPath, index: i + 1 };
  }
  return null;
}

/**
 * Insert `node` into the split at `slot.path` at `slot.index`, taking 1/n of
 * it (n items after); the others keep their sizes relative to each other.
 * `exclude` drops one direct item of that split first (an item moving within
 * the same split keeps the ratios of before the drag). Null when a tile would
 * get less than `dock.minPx` and its neighbour cannot give the difference.
 */
export function insertAtBorder(
  layout: Layout,
  slot: BorderSlot,
  node: Layout,
  dock: DockPx,
  exclude?: number
): Layout | null {
  const parent = nodeAt(layout, slot.path) as Split;
  let index = slot.index;
  const kept = parent.items.map((item, i) => ({ item, i })).filter(({ i }) => i !== exclude);
  if (exclude !== undefined && exclude < index) index -= 1;
  const n = kept.length + 1;
  const keptFractions = itemFractions(kept.map(({ item }) => item));
  const share = 1 / n;
  const items = kept.map(({ item }) => item);
  const fractions = keptFractions.map((f) => f * (1 - share));
  items.splice(index, 0, node);
  fractions.splice(index, 0, share);
  const span = spanOf(rectAt(layout, slot.path, dock), axisOf(parent.split));
  const clamped = clampShare(items, fractions, index, span, dock.minPx, axisOf(parent.split));
  if (clamped === null) return null;
  return normalize(replaceAt(layout, slot.path, sized({ ...parent, items }, clamped)));
}

/**
 * The item at `index` gets at least minPx, its previous (else next) sibling
 * giving the difference; null when any item would end below its minimum.
 */
function clampShare(
  items: Layout[],
  fractions: number[],
  index: number,
  span: number,
  minPx: number,
  axis: Axis
): number[] | null {
  const px = fractions.map((f) => f * span);
  if (px[index]! < minPx) {
    const donor = index > 0 ? index - 1 : index + 1;
    const deficit = minPx - px[index]!;
    px[index] = minPx;
    px[donor] = px[donor]! - deficit;
  }
  for (let i = 0; i < items.length; i++)
    if (px[i]! + 0.5 < minimumSize(items[i]!, axis, minPx)) return null;
  return px.map((p) => p / span);
}

/**
 * Put `node` along the dock's outer edge on `side`, spanning the full width or
 * height and taking one third of the dock along that axis; the rest share two
 * thirds in their current proportions. Null when that leaves the rest below
 * their minimums.
 */
export function insertOuter(
  layout: Layout | null,
  side: Side,
  node: Layout,
  dock: DockPx
): Layout | null {
  if (layout === null) return normalize(withSize(node, undefined));
  const direction = directionOf(side);
  const axis = axisOf(direction);
  const span = axis === 'width' ? dock.px.width : dock.px.height;
  const items: Layout[] =
    !isTile(layout) && layout.split === direction
      ? [...layout.items]
      : [withSize(layout, undefined)];
  const fractions = itemFractions(items).map((f) => f * (2 / 3));
  const index = isBefore(side) ? 0 : items.length;
  items.splice(index, 0, node);
  fractions.splice(index, 0, 1 / 3);
  const clamped = clampShare(items, fractions, index, span, dock.minPx, axis);
  if (clamped === null) return null;
  return normalize(sized({ split: direction, items }, clamped));
}

// --- planDrop ---------------------------------------------------------------

const MOVING = '\u0000moving';

function unchanged(layout: Layout | null): DropResult {
  return { layout, changed: false };
}

function result(before: Layout | null, after: Layout | null): DropResult {
  return sameLayout(before, after) ? unchanged(before) : { layout: after, changed: true };
}

function sameLayout(a: unknown, b: unknown): boolean {
  if (typeof a === 'number' && typeof b === 'number') return Math.abs(a - b) < 1e-6;
  if (a === null || b === null || typeof a !== 'object' || typeof b !== 'object') return a === b;
  if (Array.isArray(a) !== Array.isArray(b)) return false;
  const ka = Object.keys(a).filter((k) => (a as Record<string, unknown>)[k] !== undefined);
  const kb = Object.keys(b).filter((k) => (b as Record<string, unknown>)[k] !== undefined);
  if (ka.length !== kb.length) return false;
  return ka.every((k) =>
    sameLayout((a as Record<string, unknown>)[k], (b as Record<string, unknown>)[k])
  );
}

/** The dragged item, resolved against the layout. */
interface Moving {
  /** The tabs that move, and the one shown after. */
  tabs: string[];
  showing: string;
  /** The tile the item comes from (null: a new item). */
  home: Tile | null;
  /** True when the whole home tile leaves (a tile source, or its only tab). */
  wholeTile: boolean;
  /** True when the item keeps its tile id (a tile source). */
  keepsId: boolean;
}

function resolveSource(layout: Layout | null, source: Source): Moving | null {
  if ('tile' in source) {
    const home = findTile(layout, source.tile);
    if (home === null) return null;
    return { tabs: [...home.tabs], showing: home.showing, home, wholeTile: true, keepsId: true };
  }
  const key = 'tab' in source ? source.tab : source.new;
  const home = tileOf(layout, key);
  // A {new} key that is already open acts as {tab}.
  if (home === null)
    return 'tab' in source
      ? null
      : { tabs: [key], showing: key, home: null, wholeTile: false, keepsId: false };
  return { tabs: [key], showing: key, home, wholeTile: home.tabs.length === 1, keepsId: false };
}

/** The layout with the moving item taken out (rule 5 for a whole tile). */
function takeOut(layout: Layout | null, moving: Moving): Layout | null {
  if (moving.home === null) return layout;
  if (moving.wholeTile) return removeTile(layout, moving.home.tile).layout;
  return removeTab(layout, moving.tabs[0]!).layout;
}

function newTile(before: Layout | null, moving: Moving, id?: string): Tile {
  const tile =
    moving.keepsId && moving.home !== null ? moving.home.tile : (id ?? mintTileId(before));
  return { tile, tabs: [...moving.tabs], showing: moving.showing };
}

/**
 * Plan a drop of `source` on `target`. The result's `changed` is false for a
 * self-drop, an infeasible zone, or a source or target the layout does not hold.
 */
export function planDrop(
  layout: Layout | null,
  source: Source,
  target: Target,
  dock: DockPx
): DropResult {
  const moving = resolveSource(layout, source);
  if (moving === null) return unchanged(layout);

  if ('outer' in target) {
    const node = newTile(layout, moving);
    const rest = takeOut(layout, moving);
    const out = insertOuter(rest, target.outer, node, dock);
    return out === null ? unchanged(layout) : result(layout, out);
  }

  if (layout === null) return unchanged(layout);

  if ('border' in target) return dropOnBorder(layout, moving, target.border, dock);

  const into = findTile(layout, target.tile);
  if (into === null) return unchanged(layout);
  const self = moving.home !== null && moving.home.tile === into.tile;

  if ('edge' in target) {
    if (self && moving.wholeTile) return unchanged(layout);
    const node: Tile = { ...newTile(layout, moving), from: into.tile };
    const rest = takeOut(layout, moving);
    if (rest === null) return unchanged(layout);
    const out = splitTile(rest, into.tile, target.edge, node, dock);
    return out === null ? unchanged(layout) : result(layout, out);
  }

  // centre or tab bar (rules 1 and 6)
  if (self && moving.wholeTile) return unchanged(layout);
  let index = target.zone === 'tab' ? target.index : into.tabs.length;
  if (self) {
    // The index counts gaps with the dragged tab still in place.
    const at = into.tabs.indexOf(moving.tabs[0]!);
    if (index > at) index -= 1;
  }
  const rest = takeOut(layout, moving)!;
  const host = findTile(rest, into.tile)!;
  const tabs = [...host.tabs];
  tabs.splice(Math.max(0, Math.min(index, tabs.length)), 0, ...moving.tabs);
  const out = replaceAt(rest, pathOfTile(rest, into.tile)!, {
    ...host,
    tabs,
    showing: moving.showing,
  });
  return result(layout, normalize(out));
}

function dropOnBorder(
  layout: Layout,
  moving: Moving,
  border: { tile: string; side: Side },
  dock: DockPx
): DropResult {
  const slot = resolveBorder(layout, border);
  if (slot === null) return unchanged(layout);
  const parent = nodeAt(layout, slot.path) as Split;
  const leaving = moving.wholeTile ? moving.home!.tile : null;
  const isLeaving = (item: Layout | undefined): boolean =>
    leaving !== null && item !== undefined && isTile(item) && item.tile === leaving;
  // Self-drop: the item already sits on one side of that border.
  if (isLeaving(parent.items[slot.index - 1]) || isLeaving(parent.items[slot.index]))
    return unchanged(layout);

  const id = mintTileId(layout);
  // A multi-tab source loses its tab first; nothing changes size.
  let work: Layout = layout;
  if (moving.home !== null && !moving.wholeTile) work = removeTab(work, moving.tabs[0]!).layout!;
  const direct = parent.items.findIndex(isLeaving);
  const exclude = direct >= 0 ? direct : undefined;
  // A tile keeping its id is inserted under a placeholder until its old place is gone.
  const node = newTile(layout, moving, id);
  const placed = moving.keepsId && exclude === undefined ? { ...node, tile: MOVING } : node;
  let out = insertAtBorder(work, slot, placed, dock, exclude);
  if (out === null) return unchanged(layout);
  if (leaving !== null && exclude === undefined) {
    out = removeTile(out, leaving).layout!;
    if (placed.tile === MOVING) out = renameTile(out, MOVING, leaving);
  }
  return result(layout, out);
}

function renameTile(layout: Layout, from: string, to: string): Layout {
  const path = pathOfTile(layout, from)!;
  const tile = nodeAt(layout, path) as Tile;
  return replaceAt(layout, path, { ...tile, tile: to });
}

// --- replaceKey -------------------------------------------------------------

/**
 * The new key takes the old key's place in the same tile at the same index;
 * `showing` follows; no size changes (screen replace, back/forward).
 */
export function replaceKey(layout: Layout | null, oldKey: string, newKey: string): DropResult {
  const tile = tileOf(layout, oldKey);
  if (layout === null || tile === null || oldKey === newKey) return unchanged(layout);
  const tabs = tile.tabs.map((k) => (k === oldKey ? newKey : k));
  const showing = tile.showing === oldKey ? newKey : tile.showing;
  return {
    layout: replaceAt(layout, pathOfTile(layout, tile.tile)!, { ...tile, tabs, showing }),
    changed: true,
  };
}
