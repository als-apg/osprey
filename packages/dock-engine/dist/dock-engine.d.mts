// @als-apg/dock-engine: declarations of dock-engine.mjs.
// Generated from packages/dock-engine/ts/ by packages/dock-engine/scripts/build.mjs.
// Do not edit; rebuild with `node packages/dock-engine/scripts/build.mjs`
// (`npm run build:dock-engine`).
//#region layout
export type Direction = 'row' | 'column';
/** One tile of the dock: its tabs, in order, and the one it shows. */
export interface Tile<K = string> {
    tile: string;
    tabs: K[];
    showing: K;
    size?: number;
    /** The tile this one was split from by an edge drop. */
    from?: string;
}
export interface Split<K = string> {
    split: Direction;
    items: Layout<K>[];
    size?: number;
}
export type Layout<K = string> = Tile<K> | Split<K>;
/** A key of the collection form: a 0-based member index (CANOPY) or a panel id (OSPREY). */
export type CollectionKey = number | string;
export interface CollectionTile<K = CollectionKey> {
    tabs: K[];
    showing?: K;
    /** A positive weight among its siblings; absent: 1. */
    size?: number;
}
export interface CollectionSplit<K = CollectionKey> {
    split: Direction;
    items: CollectionLayout<K>[];
    size?: number;
}
export type CollectionLayout<K = CollectionKey> = CollectionTile<K> | CollectionSplit<K>;
/** A normalised collection layout: fractions for sizes and every `showing` filled in. */
export type NormalizedCollection<K = CollectionKey> = (CollectionTile<K> & {
    showing: K;
}) | {
    split: Direction;
    items: NormalizedCollection<K>[];
    size?: number;
};
export type LayoutForm = 'dock' | 'collection';
/**
 * The named reasons a layout is refused. The values are the cross-repo
 * vocabulary of the C-CASES `validate` cases: canopy and OSPREY validators
 * return the same strings.
 */
export declare const LAYOUT_REASONS: {
    readonly notLayout: "not_layout";
    readonly unknownField: "unknown_field";
    readonly badSplit: "bad_split";
    readonly badTileId: "bad_tile_id";
    readonly duplicateTile: "duplicate_tile";
    readonly badKey: "bad_key";
    readonly emptyTabs: "empty_tabs";
    readonly duplicateKey: "duplicate_key";
    readonly showingNotInTabs: "showing_not_in_tabs";
    readonly splitTooFewItems: "split_too_few_items";
    readonly badSize: "bad_size";
    readonly unknownFrom: "unknown_from";
    readonly oldSlotShape: "old_slot_shape";
    readonly oldKeyShape: "old_key_shape";
    readonly memberMissing: "member_missing";
};
export type LayoutReason = (typeof LAYOUT_REASONS)[keyof typeof LAYOUT_REASONS];
export type LayoutValidation = {
    valid: true;
} | {
    valid: false;
    reason: LayoutReason;
    where: string;
};
export interface ValidateOptions {
    /** `dock` (default) or `collection`. */
    form?: LayoutForm;
    /** Collection form only: the member count; keys must then be indexes, each placed once. */
    members?: number;
}
/** Sibling fractions must sum to 1 within this. */
export declare const SIZE_TOLERANCE = 0.001;
export declare function isTile<K>(node: Layout<K>): node is Tile<K>;
export declare function isTile<K>(node: CollectionLayout<K>): node is CollectionTile<K>;
/**
 * Check a layout against C-LAYOUT. The first broken rule wins, in tree order;
 * the result names it with a stable reason and the path where it broke.
 */
export declare function validateLayout(input: unknown, options?: ValidateOptions): LayoutValidation;
/**
 * The normalised form of a dock layout, as a fresh tree: same-direction splits
 * flattened, one-item splits collapsed, every size filled in and siblings
 * summing to 1, a `from` whose tile is gone cleared. The root has no size.
 */
export declare function normalize<K>(layout: Layout<K>): Layout<K>;
/**
 * A collection layout read into fractions: weights normalised among siblings
 * (a missing weight counts 1), every `showing` filled in (default `tabs[0]`),
 * and the same flattening and collapsing as `normalize`.
 */
export declare function normalizeCollection<K>(layout: CollectionLayout<K>): NormalizedCollection<K>;
/** Every tile id, in tree order. */
export declare function tileIdsOf<K>(layout: Layout<K> | null): string[];
/** Every key, in tree order (tiles in order, then their tabs in order). */
export declare function keysOf<K>(layout: Layout<K> | CollectionLayout<K> | null): K[];
/**
 * The id a new tile gets: `t<k>`, k = 1 + the largest numeric suffix among the
 * layout's `t<n>` ids (`t1` for an empty layout). Only the engine mints.
 */
export declare function mintTileId<K>(layout: Layout<K> | null): string;
//#endregion

//#region geometry
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
/** The axis a split of this direction lays its items along. */
export declare function axisOf(direction: Direction): Axis;
/**
 * The smallest px a subtree can take along `axis` under dockview's minimum
 * constraints, with `minPx` per tile.
 */
export declare function minimumSize(node: Layout<unknown>, axis: Axis, minPx: number): number;
/**
 * The fractions of a split's items: given sizes kept, missing sizes share the
 * remainder equally, then scaled to sum to 1 (the C-LAYOUT reading of sizes).
 */
export declare function itemFractions(items: readonly {
    size?: number;
}[]): number[];
/**
 * Lay `span` px out over items with these fractions and minimums, exactly as
 * dockview's splitview does: round and clamp each, then hand the difference
 * out from the last item to the first, each clamped to its minimum. When the
 * minimums exceed the span the items stay at their minimums (overflow).
 */
export declare function layoutSizes(fractions: readonly number[], span: number, minimums: readonly number[]): number[];
/**
 * px sizes (as dockview reports them for one split's items) -> C-LAYOUT
 * fractions. Each fraction is rounded to 4 decimals and the last absorbs the
 * rounding, so the result sums to exactly 1. A zero or empty total gives
 * equal shares.
 */
export declare function fractionsOf(pxSizes: readonly number[]): number[];
/**
 * Each tile's rect and each interior border of `layout` laid out on
 * `dockRect`, with dockview's `minPx` minimum on every tile.
 */
export declare function geometry(layout: Layout<unknown>, dockRect: Rect, minPx: number): Geometry;
//#endregion

//#region drop
export type Side = 'left' | 'right' | 'top' | 'bottom';
export type Source = {
    new: string;
} | {
    tab: string;
} | {
    tile: string;
};
export type Target = {
    tile: string;
    zone: 'centre';
} | {
    tile: string;
    zone: 'tab';
    index: number;
} | {
    tile: string;
    edge: Side;
} | {
    border: {
        tile: string;
        side: Side;
    };
} | {
    outer: Side;
};
export interface DockPx {
    px: {
        width: number;
        height: number;
    };
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
/** The direction a side's split runs in ('left'/'right': a row). */
export declare function directionOf(side: Side): Direction;
/** The tile with this id, or null. */
export declare function findTile(layout: Layout | null, id: string): Tile | null;
/** The tile holding this key, or null. */
export declare function tileOf(layout: Layout | null, key: string): Tile | null;
/**
 * Remove a tile per the C-LAYOUT leave rule: its space goes to the adjacent
 * sibling subtree holding its `from` tile, else the previous sibling, else the
 * next. The result is normalised (a one-item split collapses, a stale `from` is
 * cleared). `neighbour` is the `from` tile when that subtree grew, else the
 * adjacent tile of the sibling that did; `side` is the side of it the tile sat on.
 */
export declare function removeTile(layout: Layout | null, id: string): Removal;
/**
 * Remove a tab from the tile holding it. Its tile's last tab removes the tile
 * (removeTile); a showing tab hands `showing` to the next tab, else the previous.
 */
export declare function removeTab(layout: Layout | null, key: string): Removal;
/**
 * Split tile `id` at `edge`, putting `tile` on that side; the two halves share
 * the tile's span equally. Clamped to `dock.minPx`: when a half would be
 * smaller, both halves get minPx and the leave-rule neighbour of the target
 * (along that axis) gives the difference. Null when that is infeasible or the
 * tile is unknown. The caller sets `from` on `tile`.
 */
export declare function splitTile(layout: Layout, id: string, edge: Side, tile: Tile, dock: DockPx): Layout | null;
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
export declare function resolveBorder(layout: Layout, border: {
    tile: string;
    side: Side;
}): BorderSlot | null;
/**
 * Insert `node` into the split at `slot.path` at `slot.index`, taking 1/n of
 * it (n items after); the others keep their sizes relative to each other.
 * `exclude` drops one direct item of that split first (an item moving within
 * the same split keeps the ratios of before the drag). Null when a tile would
 * get less than `dock.minPx` and its neighbour cannot give the difference.
 */
export declare function insertAtBorder(layout: Layout, slot: BorderSlot, node: Layout, dock: DockPx, exclude?: number): Layout | null;
/**
 * Put `node` along the dock's outer edge on `side`, spanning the full width or
 * height and taking one third of the dock along that axis; the rest share two
 * thirds in their current proportions. Null when that leaves the rest below
 * their minimums.
 */
export declare function insertOuter(layout: Layout | null, side: Side, node: Layout, dock: DockPx): Layout | null;
/**
 * Plan a drop of `source` on `target`. The result's `changed` is false for a
 * self-drop, an infeasible zone, or a source or target the layout does not hold.
 */
export declare function planDrop(layout: Layout | null, source: Source, target: Target, dock: DockPx): DropResult;
/**
 * The new key takes the old key's place in the same tile at the same index;
 * `showing` follows; no size changes (screen replace, back/forward).
 */
export declare function replaceKey(layout: Layout | null, oldKey: string, newKey: string): DropResult;
//#endregion

//#region open
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
export declare const TOO_MANY_FOR_SIDE = "too_many_for_side";
/** Open one item by rule 7; `focus` names the tile that shows it. */
export declare function openItem(layout: Layout | null, key: string, ctx: OpenContext): OpenResult;
/**
 * Open a set of members `as` tabs, side by side, or by its collection layout.
 * Members in `ctx.skip` are left out and reported in `skipped`.
 */
export declare function openSet(layout: Layout | null, members: string[], as: 'tabs' | 'side' | 'layout', collection: CollectionLayout<string> | undefined, ctx: OpenContext): OpenResult;
//#endregion

//#region close
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
/** Close one tab (rule 8); its tile's last tab closes the tile. */
export declare function closeTab(layout: Layout | null, key: string): CloseResult;
/** Close a tile and every tab in it, by the leave rule. */
export declare function closeTile(layout: Layout | null, id: string): CloseResult;
/** Close every other tab of the key's tile (rule 15). */
export declare function closeOthers(layout: Layout | null, key: string): CloseResult;
/** Close the tabs right of the key in its tile (rule 15). */
export declare function closeRight(layout: Layout | null, key: string): CloseResult;
/** Close the whole dock; the record holds it. */
export declare function closeAll(layout: Layout | null): CloseResult;
/** Reopen a close record (rule 8), falling back to rule 7 through openItem. */
export declare function reopen(layout: Layout | null, record: CloseRecord, ctx: CloseContext): CloseResult;
//#endregion

//#region hitTest
/** Per tile, the bar and tab rects the dock measured (with one tile, the strip's). */
export type TabRects = Record<string, {
    barRect: Rect;
    tabRects: Rect[];
}>;
export interface Point {
    x: number;
    y: number;
}
/** The tile floor every dock enforces. */
export declare const MIN_PX = 40;
/** Smallest edge depth; larger tiles use EDGE_RATIO of their content. */
export declare const EDGE_PX = 80;
/** Edge depth as a share of the tile content along that axis. */
export declare const EDGE_RATIO = 0.31;
/** Depth of the dock-edge strips that mean an outer drop. */
export declare const STRIP_PX = 48;
/** Halving `span` px leaves each half at least MIN_PX. */
export declare function halvable(span: number): boolean;
/**
 * The Target a drop of `source` at `point` would mean, or null: off the dock
 * or a self-drop. Tile rects come from `geometry` on `dockRect` with the 40 px
 * floor; bars and tabs from the dock's `tabRects`.
 */
export declare function hitTest(layout: Layout, dockRect: Rect, tabRects: TabRects, point: Point, source: Source): Target | null;
//#endregion

//#region resultRect
/** The highlight for one drop (structurally identical to cases.ts). */
export interface ResultRect {
    rect: Rect;
    /** The tab insertion gap, for centre and tab targets; null otherwise. */
    marker: {
        x: number;
        y: number;
        height: number;
    } | null;
}
/**
 * The part of the target tile the dragged item lands in after dropping
 * `source` on `target`, with the tab insertion marker for centre and tab
 * targets, or the new strip of an outer drop; null when the drop would not
 * land or has no pointer zone.
 */
export declare function resultRect(layout: Layout, dockRect: Rect, tabRects: TabRects, source: Source, target: Target): ResultRect | null;
//#endregion
