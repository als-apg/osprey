// C-CASES: the dock rule set as data, and the one runner every replay uses.
//
// The canonical cases live in ./cases/<kind>/<name>.json, one file per case,
// formatted by prettier (2-space, LF, trailing newline; lint-staged rewrites any
// *.json on commit, so that is the only form that survives) so the byte copies
// in OSPREY (the whole set) and canopy (validate/ and normalize/ only) stay
// identical and their git-blob pins stable. Rule changes change the cases first.
//
// Test-side only: this module reads the files with node:fs, so no engine module
// may import it and the dock-engine.mjs bundle entry must leave it out. The
// runner takes the engine functions as an injected object, so the same cases
// run against the model sources and against the built bundle.
//
// Conventions the cases pin (the engine tasks implement them):
// - `dock.px` is the dock's size in px; `minPx` is the 40 px floor.
// - A `{tab}` target index counts gaps on the bar as the pointer sees it, with
//   the dragged tab still in place; moving a tab forward within its own tile
//   lands at index - 1. The dropped tab is shown.
// - A `{tile}` source keeps its tile id wherever it lands (border, edge, outer);
//   a `{tab}` or `{new}` source that makes a tile gets a minted id. Edge splits
//   set `from` to the target tile; border and outer inserts set none.
// - Clamp: the result tiles get minPx and the target's neighbour under the
//   C-LAYOUT leave rule (its `from` tile's subtree, else previous, else next
//   sibling) gives the difference; when that neighbour would drop below minPx
//   the drop is infeasible (`changed:false`, hitTest null).
// - Closing the showing tab shows the next tab, else the previous one. Closing a
//   tile's last tab returns a tile record. A tile record names the neighbour that
//   took its space (the `from` tile when that subtree grew, else the adjacent
//   tile of the previous or next sibling) and the side of that neighbour it sat
//   on; `null` for the only tile. It also carries the tile's `share` of its
//   parent split (its fraction there, 4 decimals), absent for the only tile.
//   Group records list tab records in ascending pre-close index. closeAll
//   returns a `layout` record of the whole dock.
// - Reopen shows a reopened tab; a group reopen restores each tab at its index
//   and shows the one recorded as showing, else keeps the tile's showing tab. A
//   tile reopens with its recorded id when that id is free, as an edge split of
//   its recorded neighbour, and takes its recorded share of the split it lands
//   in from that neighbour only (no other tile moves), clamped so both keep
//   minPx. A tile record without a share halves the neighbour.
// - openSet `tabs` is one new tile split off the focused tile's right edge;
//   `side` shares the focused tile's span equally among it and the members, in
//   a row, else a near-square grid (cols = ceil(sqrt n), rows = ceil(n / cols),
//   filled row by row, the focused tile first); `layout` mints ids from t1.
// - Hit: every point of a tile is a target; no pointer zone makes a border
//   target (agents still send them). The header row (bar, and anything above
//   the content) is a tab target; a tab index is the count of tabs whose centre
//   lies left of the pointer. Next, the dock-edge strips, 48 px deep, are outer
//   targets: the dock's outermost left, right and bottom 48 px, then the first
//   48 px of content under a top-row tile's header row. Edge depth is max(80,
//   31 % of the tile's content along that axis), at most half, measured over
//   the content (tile rect minus its bar); left and right win at the corners;
//   an axis too short to halve into two minPx tiles has no edges. Zones do not
//   depend on the tile the drag started in (no shared-edge rule). An edge
//   result rect is that half of the hovered tile, bar included, as the split
//   lays it out; an outer result rect is the new full-span strip as the drop
//   lays it out. The result marker is the insertion gap: the left of the tab at
//   that index, else the right of the last tab, at the bar's y and height.

import { readFileSync, readdirSync } from 'node:fs';
import { createRequire } from 'node:module';
import { dirname, join } from 'node:path';
import {
  LAYOUT_REASONS,
  isTile,
  normalize,
  normalizeCollection,
  tileIdsOf,
  validateLayout,
} from './layout.js';
import type {
  CollectionLayout,
  Layout,
  LayoutValidation,
  NormalizedCollection,
  ValidateOptions,
} from './layout.js';

// --- the drop vocabulary (C-DROP) ---------------------------------------------

export type Side = 'left' | 'right' | 'top' | 'bottom';

export type Source = { new: string } | { tab: string } | { tile: string };

export type Target =
  | { tile: string; zone: 'centre' }
  | { tile: string; zone: 'tab'; index: number }
  | { tile: string; edge: Side }
  | { border: { tile: string; side: Side } }
  | { outer: Side };

export interface Rect {
  x: number;
  y: number;
  width: number;
  height: number;
}

export interface DockPx {
  px: { width: number; height: number };
  minPx: number;
}

/** Per tile, the bar and tab rects the dock measured (with one tile, the strip's). */
export type TabRects = Record<string, { barRect: Rect; tabRects: Rect[] }>;

export interface ResultRect {
  rect: Rect;
  /** The tab insertion gap, for centre and tab targets; null otherwise. */
  marker: { x: number; y: number; height: number } | null;
}

// --- reopen records -----------------------------------------------------------

export interface TabRecord {
  kind: 'tab';
  key: string;
  tile: string;
  index: number;
  showing: boolean;
}

export interface TileRecord {
  kind: 'tile';
  tile: string;
  tabs: string[];
  showing: string;
  neighbour: string | null;
  side: Side | null;
  share?: number;
}

export interface TabsGroupRecord {
  kind: 'tabs';
  records: TabRecord[];
}

export interface LayoutGroupRecord {
  kind: 'layout';
  layout: Layout | null;
}

export type CloseRecord = TabRecord | TileRecord | TabsGroupRecord | LayoutGroupRecord;

// --- operations ---------------------------------------------------------------

export interface OpenContext {
  focusedTile: string | null;
  dock: DockPx;
  /** As data: the Target the dock's pickTarget returns (null: it declines). */
  pickTarget?: Target | null;
  caller?: 'agent' | 'person';
}

export type Op =
  | { fn: 'planDrop'; source: Source; target: Target; dock: DockPx }
  | { fn: 'replaceKey'; oldKey: string; newKey: string }
  | { fn: 'closeTab'; key: string }
  | { fn: 'closeTile'; tile: string }
  | { fn: 'closeOthers'; key: string }
  | { fn: 'closeRight'; key: string }
  | { fn: 'closeAll' }
  | { fn: 'reopen'; record: CloseRecord; context: OpenContext }
  | { fn: 'openItem'; key: string; context: OpenContext }
  | {
      fn: 'openSet';
      members: string[];
      as: 'tabs' | 'side' | 'layout';
      layout?: CollectionLayout<string>;
      /** Members the dock cannot open (missing, unavailable, over the tab cap). */
      skip?: string[];
      context: OpenContext;
    };

export type OpKind = 'drop' | 'replace-key' | 'close' | 'reopen' | 'open' | 'open-set';
export type CaseKind = OpKind | 'hit' | 'validate' | 'normalize';

export const CASE_KINDS: readonly CaseKind[] = [
  'drop',
  'replace-key',
  'close',
  'reopen',
  'open',
  'open-set',
  'hit',
  'validate',
  'normalize',
];

/** The functions each op kind may call. */
export const OP_FNS: Record<OpKind, readonly Op['fn'][]> = {
  drop: ['planDrop'],
  'replace-key': ['replaceKey'],
  close: ['closeTab', 'closeTile', 'closeOthers', 'closeRight', 'closeAll'],
  reopen: ['reopen'],
  open: ['openItem'],
  'open-set': ['openSet'],
};

/** What a case may name as its rule: a success task, a numbered rule, or a contract ruling. */
export const RULES: readonly string[] = [
  'T1',
  'T2',
  'T3',
  'T4',
  ...Array.from({ length: 17 }, (_, i) => `rule-${i + 1}`),
  'min-size',
  'no-depth-cap',
  'hit-precedence',
  'C-LAYOUT',
  'C-DROP',
];

interface Named {
  name: string;
  rule: string;
}

export interface OpCase extends Named {
  before: Layout | null;
  op: Op;
  after: Layout | null | 'unchanged';
  /** close: the reopen record (null when nothing closed). */
  record?: CloseRecord | null;
  /** open: the tile that gets focus. */
  focus?: string;
  /** openSet: a refusal, a notice, the skipped members. */
  refusal?: string;
  notice?: string;
  skipped?: string[];
}

export interface HitCase extends Named {
  layout: Layout;
  dockRect: Rect;
  tabRects: TabRects;
  point: { x: number; y: number };
  source: Source;
  target: Target | null;
  resultRect: ResultRect | null;
}

export type ValidateCase = Named & {
  input: unknown;
  options?: ValidateOptions;
  where?: string;
} & ({ valid: true } | { reason: string });

export interface NormalizeCase extends Named {
  form?: 'dock' | 'collection';
  input: unknown;
  output: unknown;
}

export interface CaseByKind {
  drop: OpCase;
  'replace-key': OpCase;
  close: OpCase;
  reopen: OpCase;
  open: OpCase;
  'open-set': OpCase;
  hit: HitCase;
  validate: ValidateCase;
  normalize: NormalizeCase;
}

export type AnyCase = CaseByKind[CaseKind];

// --- loading ------------------------------------------------------------------

/**
 * The canonical case directory. Resolved through one anchor case because under
 * vitest's jsdom environment import.meta.url is not a file: URL.
 */
export const CASES_DIR: string = dirname(dirname(createRequire(import.meta.url).resolve('../cases/drop/t1-new-outer-bottom.json')));

export interface LoadedCase<K extends CaseKind = CaseKind> {
  kind: K;
  file: string;
  bytes: string;
  data: CaseByKind[K];
}

/** Every case of one kind, sorted by file name. */
export function loadCases<K extends CaseKind>(kind: K, dir: string = CASES_DIR): LoadedCase<K>[] {
  const kindDir = join(dir, kind);
  return readdirSync(kindDir)
    .filter((file) => file.endsWith('.json'))
    .sort()
    .map((file) => {
      const bytes = readFileSync(join(kindDir, file), 'utf8');
      return { kind, file, bytes, data: JSON.parse(bytes) as CaseByKind[K] };
    });
}

// --- comparison ---------------------------------------------------------------

/** Layout sizes compare at this. */
export const SIZE_EPSILON = 0.005;
/** Rects compare at this many px. */
export const RECT_EPSILON = 2;

/**
 * Structural difference of two JSON values: numbers within `epsilon`, every
 * other value exactly, object key sets equal. Returns one line per mismatch.
 */
export function diffApprox(
  actual: unknown,
  expected: unknown,
  epsilon: number,
  path = '$'
): string[] {
  if (typeof expected === 'number' && typeof actual === 'number') {
    return Math.abs(actual - expected) <= epsilon ? [] : [`${path}: ${actual} != ${expected}`];
  }
  if (Array.isArray(expected)) {
    if (!Array.isArray(actual))
      return [`${path}: expected an array, got ${JSON.stringify(actual)}`];
    if (actual.length !== expected.length) {
      return [`${path}: length ${actual.length} != ${expected.length}`];
    }
    return expected.flatMap((item, i) => diffApprox(actual[i], item, epsilon, `${path}[${i}]`));
  }
  if (typeof expected === 'object' && expected !== null) {
    if (typeof actual !== 'object' || actual === null || Array.isArray(actual)) {
      return [`${path}: expected an object, got ${JSON.stringify(actual)}`];
    }
    const a = actual as Record<string, unknown>;
    const e = expected as Record<string, unknown>;
    const keys = new Set([...Object.keys(a), ...Object.keys(e)]);
    const out: string[] = [];
    for (const key of [...keys].sort()) {
      if (!(key in e)) out.push(`${path}.${key}: unexpected (${JSON.stringify(a[key])})`);
      else if (!(key in a)) out.push(`${path}.${key}: missing`);
      else out.push(...diffApprox(a[key], e[key], epsilon, `${path}.${key}`));
    }
    return out;
  }
  return Object.is(actual, expected)
    ? []
    : [`${path}: ${JSON.stringify(actual)} != ${JSON.stringify(expected)}`];
}

/** Layouts (or null for an empty dock) at the size tolerance. */
export function diffLayout(actual: unknown, expected: Layout | null, path = '$.layout'): string[] {
  return diffApprox(actual, expected, SIZE_EPSILON, path);
}

/** Rects (and result rects) at the px tolerance. */
export function diffRect(actual: unknown, expected: unknown, path = '$.resultRect'): string[] {
  return diffApprox(actual, expected, RECT_EPSILON, path);
}

// --- the runner ---------------------------------------------------------------

/** An operation's result: the new layout and whether anything changed. */
export interface OpResult {
  layout: Layout | null;
  changed: boolean;
  record?: CloseRecord | null;
  focus?: string | null;
  refusal?: string;
  notice?: string;
  skipped?: string[];
}

type Fn<A extends unknown[], R> = (...args: A) => R;

/**
 * The engine under test, injected: the model sources, or the built
 * dock-engine.mjs. A case whose function is absent fails with a named line.
 */
export interface Engine {
  planDrop?: Fn<[Layout | null, Source, Target, DockPx], OpResult>;
  replaceKey?: Fn<[Layout | null, string, string], OpResult | Layout | null>;
  closeTab?: Fn<[Layout | null, string], OpResult>;
  closeTile?: Fn<[Layout | null, string], OpResult>;
  closeOthers?: Fn<[Layout | null, string], OpResult>;
  closeRight?: Fn<[Layout | null, string], OpResult>;
  closeAll?: Fn<[Layout | null], OpResult>;
  reopen?: Fn<[Layout | null, CloseRecord, EngineContext], OpResult>;
  openItem?: Fn<[Layout | null, string, EngineContext], OpResult>;
  openSet?: Fn<
    [
      Layout | null,
      string[],
      'tabs' | 'side' | 'layout',
      CollectionLayout<string> | undefined,
      EngineContext,
    ],
    OpResult
  >;
  hitTest?: Fn<[Layout, Rect, TabRects, { x: number; y: number }, Source], Target | null>;
  resultRect?: Fn<[Layout, Rect, TabRects, Source, Target], ResultRect | null>;
  validateLayout?: Fn<[unknown, ValidateOptions?], LayoutValidation>;
  normalize?: Fn<[Layout], Layout>;
  normalizeCollection?: Fn<
    [CollectionLayout<string | number>],
    NormalizedCollection<string | number>
  >;
}

/** The context an engine receives: the case's data, with pickTarget as a function. */
export interface EngineContext {
  focusedTile: string | null;
  dock: DockPx;
  pickTarget?: (layout: Layout | null, key: string) => Target | null;
  caller?: 'agent' | 'person';
  skip?: string[];
}

function engineContext(context: OpContext, skip?: string[]): EngineContext {
  const out: EngineContext = { focusedTile: context.focusedTile, dock: context.dock };
  if (context.pickTarget !== undefined) {
    const target = context.pickTarget;
    out.pickTarget = () => target;
  }
  if (context.caller !== undefined) out.caller = context.caller;
  if (skip !== undefined) out.skip = skip;
  return out;
}

type OpContext = OpenContext;

function missing(fn: string): string[] {
  return [`engine has no ${fn}`];
}

function callOp(op: Op, before: Layout | null, engine: Engine): OpResult | string[] {
  switch (op.fn) {
    case 'planDrop':
      return engine.planDrop
        ? engine.planDrop(before, op.source, op.target, op.dock)
        : missing(op.fn);
    case 'replaceKey': {
      if (!engine.replaceKey) return missing(op.fn);
      const result = engine.replaceKey(before, op.oldKey, op.newKey);
      if (result !== null && typeof result === 'object' && 'changed' in result) return result;
      const layout = result as Layout | null;
      return { layout, changed: diffLayout(layout, before).length > 0 };
    }
    case 'closeTab':
      return engine.closeTab ? engine.closeTab(before, op.key) : missing(op.fn);
    case 'closeTile':
      return engine.closeTile ? engine.closeTile(before, op.tile) : missing(op.fn);
    case 'closeOthers':
      return engine.closeOthers ? engine.closeOthers(before, op.key) : missing(op.fn);
    case 'closeRight':
      return engine.closeRight ? engine.closeRight(before, op.key) : missing(op.fn);
    case 'closeAll':
      return engine.closeAll ? engine.closeAll(before) : missing(op.fn);
    case 'reopen':
      return engine.reopen
        ? engine.reopen(before, op.record, engineContext(op.context))
        : missing(op.fn);
    case 'openItem':
      return engine.openItem
        ? engine.openItem(before, op.key, engineContext(op.context))
        : missing(op.fn);
    case 'openSet':
      return engine.openSet
        ? engine.openSet(before, op.members, op.as, op.layout, engineContext(op.context, op.skip))
        : missing(op.fn);
  }
}

/** Replay one operation case; [] when the engine matches it. */
export function runOpCase(c: OpCase, engine: Engine): string[] {
  const result = callOp(c.op, c.before, engine);
  if (Array.isArray(result)) return result;
  const out: string[] = [];
  if (c.after === 'unchanged') {
    if (result.changed) out.push('$.changed: true, expected false');
    out.push(...diffLayout(result.layout, c.before));
  } else {
    if (!result.changed) out.push('$.changed: false, expected true');
    out.push(...diffLayout(result.layout, c.after));
  }
  if (c.record !== undefined)
    out.push(...diffApprox(result.record ?? null, c.record, SIZE_EPSILON, '$.record'));
  if (c.focus !== undefined) out.push(...diffApprox(result.focus ?? null, c.focus, 0, '$.focus'));
  if (c.refusal !== undefined)
    out.push(...diffApprox(result.refusal ?? null, c.refusal, 0, '$.refusal'));
  if (c.notice !== undefined)
    out.push(...diffApprox(result.notice ?? null, c.notice, 0, '$.notice'));
  if (c.skipped !== undefined)
    out.push(...diffApprox(result.skipped ?? null, c.skipped, 0, '$.skipped'));
  return out;
}

/** Replay a hit case's target selection. */
export function runHitTarget(c: HitCase, engine: Engine): string[] {
  if (!engine.hitTest) return missing('hitTest');
  const target = engine.hitTest(c.layout, c.dockRect, c.tabRects, c.point, c.source);
  return diffApprox(target, c.target, 0, '$.target');
}

/** Replay a hit case's result rect (the highlight). Cases with no target have none. */
export function runHitResultRect(c: HitCase, engine: Engine): string[] {
  if (c.target === null) return c.resultRect === null ? [] : ['$.resultRect: set without a target'];
  if (!engine.resultRect) return missing('resultRect');
  const rect = engine.resultRect(c.layout, c.dockRect, c.tabRects, c.source, c.target);
  return diffRect(rect, c.resultRect);
}

/** Replay a validate case; the reason and, when present, the where path are compared. */
export function runValidateCase(c: ValidateCase, engine: Engine): string[] {
  if (!engine.validateLayout) return missing('validateLayout');
  const result = engine.validateLayout(c.input, c.options);
  const got = result.valid ? 'valid' : result.reason;
  const want = 'valid' in c ? 'valid' : c.reason;
  if (got !== want) return [`$.reason: ${got} != ${want}`];
  if (result.valid || c.where === undefined) return [];
  return result.where === c.where ? [] : [`$.where: ${result.where} != ${c.where}`];
}

/** Replay a normalize case. */
export function runNormalizeCase(c: NormalizeCase, engine: Engine): string[] {
  if (c.form === 'collection') {
    if (!engine.normalizeCollection) return missing('normalizeCollection');
    const output = engine.normalizeCollection(c.input as CollectionLayout<string | number>);
    return diffApprox(output, c.output, SIZE_EPSILON, '$.output');
  }
  if (!engine.normalize) return missing('normalize');
  return diffApprox(engine.normalize(c.input as Layout), c.output, SIZE_EPSILON, '$.output');
}

/** Replay any case against the engine. Hit cases check target and result rect. */
export function runCase(kind: CaseKind, c: AnyCase, engine: Engine): string[] {
  switch (kind) {
    case 'hit':
      return [...runHitTarget(c as HitCase, engine), ...runHitResultRect(c as HitCase, engine)];
    case 'validate':
      return runValidateCase(c as ValidateCase, engine);
    case 'normalize':
      return runNormalizeCase(c as NormalizeCase, engine);
    default:
      return runOpCase(c as OpCase, engine);
  }
}

// --- well-formedness ----------------------------------------------------------

const SIDES: ReadonlySet<string> = new Set(['left', 'right', 'top', 'bottom']);
const REASONS: ReadonlySet<string> = new Set(Object.values(LAYOUT_REASONS));

function isObject(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value);
}

function exactKeys(value: Record<string, unknown>, keys: string[]): boolean {
  const own = Object.keys(value).sort();
  return own.length === keys.length && [...keys].sort().every((key, i) => own[i] === key);
}

function isKey(value: unknown): value is string {
  return typeof value === 'string' && value !== '';
}

/** Problems with a Source; [] when well formed. */
export function checkSource(value: unknown, path: string): string[] {
  if (!isObject(value)) return [`${path}: not an object`];
  for (const field of ['new', 'tab', 'tile']) {
    if (exactKeys(value, [field]))
      return isKey(value[field]) ? [] : [`${path}.${field}: not a key`];
  }
  return [`${path}: not a Source`];
}

/** Problems with a Target; [] when well formed. */
export function checkTarget(value: unknown, path: string): string[] {
  if (!isObject(value)) return [`${path}: not an object`];
  if (exactKeys(value, ['outer']))
    return SIDES.has(value.outer as string) ? [] : [`${path}.outer: bad side`];
  if (exactKeys(value, ['border'])) {
    const border = value.border;
    if (!isObject(border) || !exactKeys(border, ['tile', 'side']))
      return [`${path}.border: bad shape`];
    return isKey(border.tile) && SIDES.has(border.side as string)
      ? []
      : [`${path}.border: bad tile or side`];
  }
  if (!isKey(value.tile)) return [`${path}: not a Target`];
  if (exactKeys(value, ['tile', 'edge']))
    return SIDES.has(value.edge as string) ? [] : [`${path}.edge: bad side`];
  if (exactKeys(value, ['tile', 'zone']) && value.zone === 'centre') return [];
  if (exactKeys(value, ['tile', 'zone', 'index']) && value.zone === 'tab') {
    return Number.isInteger(value.index) && (value.index as number) >= 0
      ? []
      : [`${path}.index: bad index`];
  }
  return [`${path}: not a Target`];
}

/** The tile a Target names, if any. */
export function targetTile(target: Target): string | null {
  if ('outer' in target) return null;
  if ('border' in target) return target.border.tile;
  return target.tile;
}

/** Problems with a dock-form layout (or null): valid, and already normalised. */
export function checkLayout(value: unknown, path: string): string[] {
  if (value === null) return [];
  const result = validateLayout(value);
  if (!result.valid) return [`${path}: ${result.reason} at ${result.where}`];
  const drift = diffApprox(normalize(value as Layout), value, SIZE_EPSILON, path);
  return drift.length === 0 ? [] : [`${path}: not normalised (${drift[0]})`];
}

function checkRect(value: unknown, path: string): string[] {
  if (!isObject(value) || !exactKeys(value, ['x', 'y', 'width', 'height']))
    return [`${path}: not a Rect`];
  const ok = ['x', 'y', 'width', 'height'].every((k) => typeof value[k] === 'number');
  return ok && (value.width as number) > 0 && (value.height as number) > 0
    ? []
    : [`${path}: bad Rect`];
}

function checkDockPx(value: unknown, path: string): string[] {
  if (!isObject(value) || !exactKeys(value, ['px', 'minPx'])) return [`${path}: not a DockPx`];
  const px = value.px;
  if (!isObject(px) || !exactKeys(px, ['width', 'height'])) return [`${path}.px: bad`];
  return typeof px.width === 'number' && typeof px.height === 'number' && value.minPx === 40
    ? []
    : [`${path}: bad px or minPx`];
}

function checkContext(value: unknown, path: string, layout: Layout | null): string[] {
  if (!isObject(value)) return [`${path}: not an object`];
  const out = checkDockPx(value.dock, `${path}.dock`);
  const focused = value.focusedTile;
  if (focused !== null && !(isKey(focused) && tileIdsOf(layout).includes(focused))) {
    out.push(`${path}.focusedTile: not a tile of before`);
  }
  if (value.pickTarget !== undefined && value.pickTarget !== null) {
    out.push(...checkTarget(value.pickTarget, `${path}.pickTarget`));
  }
  if (value.caller !== undefined && value.caller !== 'agent' && value.caller !== 'person') {
    out.push(`${path}.caller: bad`);
  }
  return out;
}

function checkOp(kind: OpKind, c: OpCase): string[] {
  const op = c.op as unknown;
  if (!isObject(op)) return ['$.op: not an object'];
  const fn = op.fn as Op['fn'];
  if (!OP_FNS[kind].includes(fn)) return [`$.op.fn: ${String(fn)} is not a ${kind} function`];
  const out: string[] = [];
  const tiles = tileIdsOf(c.before);
  const o = c.op;
  switch (o.fn) {
    case 'planDrop': {
      out.push(...checkSource(o.source, '$.op.source'), ...checkTarget(o.target, '$.op.target'));
      out.push(...checkDockPx(o.dock, '$.op.dock'));
      const named = out.length === 0 ? targetTile(o.target) : null;
      if (named !== null && !tiles.includes(named))
        out.push('$.op.target: names a tile not in before');
      if ('tile' in o.source && !tiles.includes(o.source.tile))
        out.push('$.op.source: tile not in before');
      break;
    }
    case 'closeTile':
      break;
    case 'reopen':
      out.push(...checkContext(o.context, '$.op.context', c.before));
      if (!isObject(o.record)) out.push('$.op.record: not an object');
      break;
    case 'openItem':
      out.push(...checkContext(o.context, '$.op.context', c.before));
      break;
    case 'openSet':
      out.push(...checkContext(o.context, '$.op.context', c.before));
      if (!['tabs', 'side', 'layout'].includes(o.as)) out.push('$.op.as: bad');
      if (o.as === 'layout') {
        const given = validateLayout(o.layout, { form: 'collection' });
        if (!given.valid) out.push(`$.op.layout: ${given.reason}`);
      }
      break;
    default:
      break;
  }
  return out;
}

/** Problems with one loaded case; [] when it is well formed and names a rule. */
export function checkCase(loaded: LoadedCase): string[] {
  const { kind, file, bytes, data } = loaded;
  const out: string[] = [];
  const c = data as unknown as Record<string, unknown>;
  if (!isObject(c)) return ['not an object'];
  if (c.name !== file.replace(/\.json$/, ''))
    out.push(`$.name: ${String(c.name)} is not the file name`);
  if (!/^[a-z0-9]+(-[a-z0-9]+)*$/.test(String(c.name))) out.push('$.name: not kebab-case');
  if (typeof c.rule !== 'string' || !RULES.includes(c.rule))
    out.push(`$.rule: ${String(c.rule)} names no rule`);
  const layoutOk =
    bytes.endsWith('}\n') &&
    !bytes.includes('\r') &&
    !bytes.includes('\t') &&
    bytes.split('\n').every((line) => /^( {2})*\S|^$/.test(line));
  if (!layoutOk) out.push('bytes: not the committed JSON form');

  if (kind === 'hit') {
    const h = data as HitCase;
    const allowed = [
      'name',
      'rule',
      'layout',
      'dockRect',
      'tabRects',
      'point',
      'source',
      'target',
      'resultRect',
    ];
    if (!exactKeys(c, allowed)) out.push('$: hit case fields');
    out.push(...checkLayout(h.layout, '$.layout'), ...checkRect(h.dockRect, '$.dockRect'));
    if (h.layout === null) out.push('$.layout: a hit needs a layout');
    const tiles = tileIdsOf(h.layout);
    if (
      !isObject(h.tabRects) ||
      Object.keys(h.tabRects).sort().join() !== [...tiles].sort().join()
    ) {
      out.push('$.tabRects: not one entry per tile');
    } else {
      for (const [id, bar] of Object.entries(h.tabRects)) {
        out.push(...checkRect(bar.barRect, `$.tabRects.${id}.barRect`));
        const keys = tiles.includes(id) ? tabCount(h.layout, id) : -1;
        if (!Array.isArray(bar.tabRects) || bar.tabRects.length !== keys) {
          out.push(`$.tabRects.${id}.tabRects: not one rect per tab`);
        } else
          bar.tabRects.forEach((r, i) =>
            out.push(...checkRect(r, `$.tabRects.${id}.tabRects[${i}]`))
          );
      }
    }
    if (!isObject(h.point) || typeof h.point.x !== 'number' || typeof h.point.y !== 'number') {
      out.push('$.point: bad');
    }
    out.push(...checkSource(h.source, '$.source'));
    if (h.target === null) {
      if (h.resultRect !== null) out.push('$.resultRect: must be null without a target');
    } else {
      out.push(...checkTarget(h.target, '$.target'));
      const named = targetTile(h.target);
      if (named !== null && !tiles.includes(named))
        out.push('$.target: names a tile not in the layout');
      const r = h.resultRect;
      if (!isObject(r) || !exactKeys(r, ['rect', 'marker'])) out.push('$.resultRect: bad shape');
      else {
        out.push(...checkRect(r.rect, '$.resultRect.rect'));
        const wantsMarker = 'zone' in h.target;
        if (wantsMarker !== (r.marker !== null))
          out.push('$.resultRect.marker: only centre and tab targets have one');
      }
    }
    return out;
  }

  if (kind === 'validate') {
    const v = data as ValidateCase;
    const outcome = 'valid' in v ? v.valid === true : REASONS.has((v as { reason: string }).reason);
    if (!outcome) out.push('$: needs valid:true or a LAYOUT_REASONS reason');
    if ('valid' in v && 'reason' in v) out.push('$: both valid and reason');
    if ('reason' in v && typeof c.where !== 'string') out.push('$: reason without a where');
    const allowed = new Set(['name', 'rule', 'input', 'options', 'valid', 'reason', 'where']);
    if (Object.keys(c).some((key) => !allowed.has(key))) out.push('$: validate case fields');
    return out;
  }

  if (kind === 'normalize') {
    const n = data as NormalizeCase;
    if (!('input' in n) || !('output' in n)) out.push('$: needs input and output');
    if (n.form !== undefined && n.form !== 'dock' && n.form !== 'collection')
      out.push('$.form: bad');
    if (n.form === 'collection') {
      const check = validateLayout(n.output, { form: 'collection' });
      if (!check.valid) out.push(`$.output: ${check.reason}`);
    } else {
      out.push(...checkLayout(n.output, '$.output'));
    }
    return out;
  }

  const o = data as OpCase;
  const allowed = new Set([
    'name',
    'rule',
    'before',
    'op',
    'after',
    'record',
    'focus',
    'refusal',
    'notice',
    'skipped',
  ]);
  if (Object.keys(c).some((key) => !allowed.has(key))) out.push('$: op case fields');
  if (!('before' in c) || !('after' in c)) out.push('$: needs before and after');
  out.push(...checkLayout(o.before, '$.before'));
  if (o.after !== 'unchanged') out.push(...checkLayout(o.after, '$.after'));
  if (
    o.after === 'unchanged' &&
    o.refusal === undefined &&
    kind === 'open-set' &&
    o.notice === undefined
  ) {
    out.push('$.after: an unchanged openSet names its refusal');
  }
  if (kind === 'close' && !('record' in c)) out.push('$.record: a close case names its record');
  out.push(...checkOp(kind as OpKind, o));
  return out;
}

function tabCount(layout: Layout, id: string): number {
  if (isTile(layout)) return layout.tile === id ? layout.tabs.length : -1;
  for (const item of layout.items) {
    const n = tabCount(item, id);
    if (n >= 0) return n;
  }
  return -1;
}

/** The model's own C-LAYOUT functions, as an Engine fragment. */
export const layoutEngine: Engine = {
  validateLayout,
  normalize,
  normalizeCollection: normalizeCollection as Engine['normalizeCollection'],
};
