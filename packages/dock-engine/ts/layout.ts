// C-LAYOUT: the layout tree shared by both docks, both agents, collections,
// presets and persistence.
//
//   Layout = Tile | Split
//   Tile   = { tile, tabs: [key, ...>=1], showing: key in tabs, size?, from? }
//   Split  = { split: "row" | "column", items: [Layout, ...>=2], size? }
//
// `size` is the fraction of the parent along the parent's axis; siblings sum
// to 1 (+-0.001) and missing sizes share the remainder equally. There is no
// depth cap. The collection form (collections and OSPREY presets) has the same
// shape without `tile` and `from`; there `size` is a positive weight and a
// missing `showing` means `tabs[0]`.
//
// Pure and dependency-free: this module is bundled into dock-engine.mjs, so it
// imports nothing, not even from its own package.

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
export type NormalizedCollection<K = CollectionKey> =
  | (CollectionTile<K> & { showing: K })
  | { split: Direction; items: NormalizedCollection<K>[]; size?: number };

export type LayoutForm = 'dock' | 'collection';

/**
 * The named reasons a layout is refused. The values are the cross-repo
 * vocabulary of the C-CASES `validate` cases: canopy and OSPREY validators
 * return the same strings.
 */
export const LAYOUT_REASONS = {
  notLayout: 'not_layout',
  unknownField: 'unknown_field',
  badSplit: 'bad_split',
  badTileId: 'bad_tile_id',
  duplicateTile: 'duplicate_tile',
  badKey: 'bad_key',
  emptyTabs: 'empty_tabs',
  duplicateKey: 'duplicate_key',
  showingNotInTabs: 'showing_not_in_tabs',
  splitTooFewItems: 'split_too_few_items',
  badSize: 'bad_size',
  unknownFrom: 'unknown_from',
  oldSlotShape: 'old_slot_shape',
  oldKeyShape: 'old_key_shape',
  memberMissing: 'member_missing',
} as const;

export type LayoutReason = (typeof LAYOUT_REASONS)[keyof typeof LAYOUT_REASONS];

export type LayoutValidation =
  | { valid: true }
  | { valid: false; reason: LayoutReason; where: string };

export interface ValidateOptions {
  /** `dock` (default) or `collection`. */
  form?: LayoutForm;
  /** Collection form only: the member count; keys must then be indexes, each placed once. */
  members?: number;
}

/** Sibling fractions must sum to 1 within this. */
export const SIZE_TOLERANCE = 0.001;

const SPLIT_FIELDS: ReadonlySet<string> = new Set(['split', 'items', 'size']);
const DOCK_TILE_FIELDS: ReadonlySet<string> = new Set(['tile', 'tabs', 'showing', 'size', 'from']);
const COLLECTION_TILE_FIELDS: ReadonlySet<string> = new Set(['tabs', 'showing', 'size']);
const TILE_ID = /^t(\d+)$/;

export function isTile<K>(node: Layout<K>): node is Tile<K>;
export function isTile<K>(node: CollectionLayout<K>): node is CollectionTile<K>;
export function isTile(node: object): boolean {
  return !Object.hasOwn(node, 'split');
}

// --- validation ---------------------------------------------------------------

class Refused extends Error {
  readonly reason: LayoutReason;
  readonly where: string;
  constructor(reason: LayoutReason, where: string) {
    super(`${where}: ${reason}`);
    this.reason = reason;
    this.where = where;
  }
}

function isPlainObject(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value);
}

/** The old leaf shapes are named before anything else, wherever they sit. */
function findOldShape(value: unknown, where: string): void {
  if (Array.isArray(value)) {
    value.forEach((item, i) => findOldShape(item, `${where}[${i}]`));
    return;
  }
  if (!isPlainObject(value)) return;
  if (Object.hasOwn(value, 'slot')) throw new Refused(LAYOUT_REASONS.oldSlotShape, where);
  if (Object.hasOwn(value, 'key') && !Object.hasOwn(value, 'tabs')) {
    throw new Refused(LAYOUT_REASONS.oldKeyShape, where);
  }
  if (Array.isArray(value.items)) findOldShape(value.items, `${where}.items`);
}

interface Walk {
  form: LayoutForm;
  members: number | undefined;
  keys: Set<CollectionKey>;
  tiles: Map<string, string>;
  froms: { from: string; tile: string; where: string }[];
}

function checkKey(key: unknown, walk: Walk, where: string): CollectionKey {
  if (walk.form === 'dock') {
    if (typeof key !== 'string' || key === '') throw new Refused(LAYOUT_REASONS.badKey, where);
    return key;
  }
  const isIndex = typeof key === 'number' && Number.isInteger(key) && key >= 0;
  if (walk.members !== undefined) {
    if (!isIndex || key >= walk.members) throw new Refused(LAYOUT_REASONS.badKey, where);
    return key;
  }
  if (isIndex || (typeof key === 'string' && key !== '')) return key;
  throw new Refused(LAYOUT_REASONS.badKey, where);
}

function checkSizeValue(size: unknown, form: LayoutForm, where: string): void {
  if (size === undefined) return;
  const ok =
    typeof size === 'number' &&
    Number.isFinite(size) &&
    size > 0 &&
    (form === 'collection' || size <= 1 + SIZE_TOLERANCE);
  if (!ok) throw new Refused(LAYOUT_REASONS.badSize, where);
}

/** Dock form: given fractions sum to 1, or leave a remainder for the missing ones. */
function checkSiblingFractions(items: Record<string, unknown>[], where: string): void {
  let given = 0;
  let missing = 0;
  for (const item of items) {
    if (typeof item.size === 'number') given += item.size;
    else missing += 1;
  }
  const ok = missing === 0 ? Math.abs(given - 1) <= SIZE_TOLERANCE : given < 1 - SIZE_TOLERANCE;
  if (!ok) throw new Refused(LAYOUT_REASONS.badSize, where);
}

function checkFields(
  node: Record<string, unknown>,
  allowed: ReadonlySet<string>,
  where: string
): void {
  const unknown = Object.keys(node)
    .filter((field) => !allowed.has(field))
    .sort();
  if (unknown.length > 0) throw new Refused(LAYOUT_REASONS.unknownField, `${where}.${unknown[0]}`);
}

function checkTile(node: Record<string, unknown>, walk: Walk, where: string): void {
  checkFields(node, walk.form === 'dock' ? DOCK_TILE_FIELDS : COLLECTION_TILE_FIELDS, where);
  let id = '';
  if (walk.form === 'dock') {
    if (typeof node.tile !== 'string' || node.tile === '') {
      throw new Refused(LAYOUT_REASONS.badTileId, `${where}.tile`);
    }
    id = node.tile;
    if (walk.tiles.has(id)) throw new Refused(LAYOUT_REASONS.duplicateTile, `${where}.tile`);
    walk.tiles.set(id, where);
  }
  if (!Array.isArray(node.tabs)) throw new Refused(LAYOUT_REASONS.emptyTabs, `${where}.tabs`);
  if (node.tabs.length === 0) throw new Refused(LAYOUT_REASONS.emptyTabs, `${where}.tabs`);
  const tabs: CollectionKey[] = [];
  node.tabs.forEach((raw: unknown, i) => {
    const at = `${where}.tabs[${i}]`;
    const key = checkKey(raw, walk, at);
    if (walk.keys.has(key)) throw new Refused(LAYOUT_REASONS.duplicateKey, at);
    walk.keys.add(key);
    tabs.push(key);
  });
  const showingRequired = walk.form === 'dock';
  if (node.showing !== undefined || showingRequired) {
    if (!tabs.includes(node.showing as CollectionKey)) {
      throw new Refused(LAYOUT_REASONS.showingNotInTabs, `${where}.showing`);
    }
  }
  checkSizeValue(node.size, walk.form, `${where}.size`);
  if (node.from !== undefined) {
    if (typeof node.from !== 'string' || node.from === id) {
      throw new Refused(LAYOUT_REASONS.unknownFrom, `${where}.from`);
    }
    walk.froms.push({ from: node.from, tile: id, where: `${where}.from` });
  }
}

function checkNode(value: unknown, walk: Walk, where: string): void {
  if (!isPlainObject(value)) throw new Refused(LAYOUT_REASONS.notLayout, where);
  const isSplit = Object.hasOwn(value, 'split') || Object.hasOwn(value, 'items');
  if (!isSplit) {
    const isTileShaped = Object.hasOwn(value, 'tabs') || Object.hasOwn(value, 'tile');
    if (!isTileShaped) throw new Refused(LAYOUT_REASONS.notLayout, where);
    checkTile(value, walk, where);
    return;
  }
  checkFields(value, SPLIT_FIELDS, where);
  if (value.split !== 'row' && value.split !== 'column') {
    throw new Refused(LAYOUT_REASONS.badSplit, `${where}.split`);
  }
  const items: unknown = value.items;
  if (!Array.isArray(items) || items.length < 2) {
    throw new Refused(LAYOUT_REASONS.splitTooFewItems, `${where}.items`);
  }
  checkSizeValue(value.size, walk.form, `${where}.size`);
  items.forEach((item: unknown, i) => checkNode(item, walk, `${where}.items[${i}]`));
  if (walk.form === 'dock') {
    checkSiblingFractions(items as Record<string, unknown>[], `${where}.items`);
  }
}

/**
 * Check a layout against C-LAYOUT. The first broken rule wins, in tree order;
 * the result names it with a stable reason and the path where it broke.
 */
export function validateLayout(input: unknown, options: ValidateOptions = {}): LayoutValidation {
  const walk: Walk = {
    form: options.form ?? 'dock',
    members: options.form === 'collection' ? options.members : undefined,
    keys: new Set(),
    tiles: new Map(),
    froms: [],
  };
  try {
    findOldShape(input, 'layout');
    checkNode(input, walk, 'layout');
    for (const { from, where } of walk.froms) {
      if (!walk.tiles.has(from)) throw new Refused(LAYOUT_REASONS.unknownFrom, where);
    }
    if (walk.members !== undefined) {
      for (let i = 0; i < walk.members; i += 1) {
        if (!walk.keys.has(i)) throw new Refused(LAYOUT_REASONS.memberMissing, 'layout');
      }
    }
  } catch (error) {
    if (error instanceof Refused) return { valid: false, reason: error.reason, where: error.where };
    throw error;
  }
  return { valid: true };
}

// --- normalisation ------------------------------------------------------------

interface AnyTile {
  tabs: unknown[];
  showing?: unknown;
  size?: number;
  tile?: string;
  from?: string;
}

interface AnySplit {
  split: Direction;
  items: AnyNode[];
  size?: number;
}

type AnyNode = AnyTile | AnySplit;

function isAnySplit(node: AnyNode): node is AnySplit {
  return Object.hasOwn(node, 'split');
}

/** Dock form: missing sizes share the remainder; collection form: missing weight is 1. Then sum to 1. */
function fractionsOf(items: AnyNode[], form: LayoutForm): number[] {
  const given = items.map((item) =>
    typeof item.size === 'number' && Number.isFinite(item.size) && item.size > 0
      ? item.size
      : undefined
  );
  const missing = given.filter((size) => size === undefined).length;
  let fill = 1;
  if (form === 'dock' && missing > 0) {
    const sum = given.reduce<number>((total, size) => total + (size ?? 0), 0);
    const remainder = 1 - sum;
    fill = remainder > 1e-9 ? remainder / missing : 1 / items.length;
  }
  const raw = given.map((size) => size ?? fill);
  const total = raw.reduce((sum, size) => sum + size, 0);
  return raw.map((size) => size / total);
}

function withSize<T extends AnyNode>(node: T, size: number | undefined): T {
  const copy = { ...node };
  if (size === undefined) delete copy.size;
  else copy.size = size;
  return copy;
}

function normalizeNode(node: AnyNode, form: LayoutForm, tiles: ReadonlySet<string>): AnyNode {
  if (!isAnySplit(node)) {
    const out: AnyTile = { ...node, tabs: [...node.tabs] };
    if (out.showing === undefined) out.showing = out.tabs[0];
    if (out.from !== undefined && (!tiles.has(out.from) || out.from === out.tile)) delete out.from;
    return out;
  }
  const fractions = fractionsOf(node.items, form);
  const items: AnyNode[] = [];
  node.items.forEach((item, i) => {
    const share = fractions[i] ?? 0;
    const child = normalizeNode(item, form, tiles);
    if (isAnySplit(child) && child.split === node.split) {
      for (const grandchild of child.items)
        items.push(withSize(grandchild, share * (grandchild.size ?? 0)));
    } else {
      items.push(withSize(child, share));
    }
  });
  if (items.length === 1 && items[0] !== undefined) return withSize(items[0], node.size);
  return withSize({ split: node.split, items }, node.size);
}

function collectTileIds(node: AnyNode, out: string[]): void {
  if (isAnySplit(node)) {
    for (const item of node.items) collectTileIds(item, out);
  } else if (typeof node.tile === 'string') {
    out.push(node.tile);
  }
}

function normalizeAny(layout: AnyNode, form: LayoutForm): AnyNode {
  const ids: string[] = [];
  collectTileIds(layout, ids);
  return withSize(normalizeNode(layout, form, new Set(ids)), undefined);
}

/**
 * The normalised form of a dock layout, as a fresh tree: same-direction splits
 * flattened, one-item splits collapsed, every size filled in and siblings
 * summing to 1, a `from` whose tile is gone cleared. The root has no size.
 */
export function normalize<K>(layout: Layout<K>): Layout<K> {
  return normalizeAny(layout as AnyNode, 'dock') as Layout<K>;
}

/**
 * A collection layout read into fractions: weights normalised among siblings
 * (a missing weight counts 1), every `showing` filled in (default `tabs[0]`),
 * and the same flattening and collapsing as `normalize`.
 */
export function normalizeCollection<K>(layout: CollectionLayout<K>): NormalizedCollection<K> {
  return normalizeAny(layout as AnyNode, 'collection') as NormalizedCollection<K>;
}

// --- reading ------------------------------------------------------------------

/** Every tile id, in tree order. */
export function tileIdsOf<K>(layout: Layout<K> | null): string[] {
  const ids: string[] = [];
  if (layout !== null) collectTileIds(layout as AnyNode, ids);
  return ids;
}

/** Every key, in tree order (tiles in order, then their tabs in order). */
export function keysOf<K>(layout: Layout<K> | CollectionLayout<K> | null): K[] {
  const keys: K[] = [];
  const visit = (node: AnyNode): void => {
    if (isAnySplit(node)) node.items.forEach(visit);
    else keys.push(...(node.tabs as K[]));
  };
  if (layout !== null) visit(layout as AnyNode);
  return keys;
}

/**
 * The id a new tile gets: `t<k>`, k = 1 + the largest numeric suffix among the
 * layout's `t<n>` ids (`t1` for an empty layout). Only the engine mints.
 */
export function mintTileId<K>(layout: Layout<K> | null): string {
  let largest = 0;
  for (const id of tileIdsOf(layout)) {
    const match = TILE_ID.exec(id);
    if (match?.[1] !== undefined) largest = Math.max(largest, Number(match[1]));
  }
  return `t${largest + 1}`;
}
