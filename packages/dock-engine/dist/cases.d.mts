// @als-apg/dock-engine: declarations of cases.mjs.
// Generated from packages/dock-engine/ts/ by packages/dock-engine/scripts/build.mjs.
// Do not edit; rebuild with `node packages/dock-engine/scripts/build.mjs`
// (`npm run build:dock-engine`).
import type { CollectionLayout, Layout, LayoutValidation, NormalizedCollection, ValidateOptions } from './dock-engine.mjs';
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
export interface Rect {
    x: number;
    y: number;
    width: number;
    height: number;
}
export interface DockPx {
    px: {
        width: number;
        height: number;
    };
    minPx: number;
}
/** Per tile, the bar and tab rects the dock measured (with one tile, the strip's). */
export type TabRects = Record<string, {
    barRect: Rect;
    tabRects: Rect[];
}>;
export interface ResultRect {
    rect: Rect;
    /** The tab insertion gap, for centre and tab targets; null otherwise. */
    marker: {
        x: number;
        y: number;
        height: number;
    } | null;
}
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
export interface OpenContext {
    focusedTile: string | null;
    dock: DockPx;
    /** As data: the Target the dock's pickTarget returns (null: it declines). */
    pickTarget?: Target | null;
    caller?: 'agent' | 'person';
}
export type Op = {
    fn: 'planDrop';
    source: Source;
    target: Target;
    dock: DockPx;
} | {
    fn: 'replaceKey';
    oldKey: string;
    newKey: string;
} | {
    fn: 'closeTab';
    key: string;
} | {
    fn: 'closeTile';
    tile: string;
} | {
    fn: 'closeOthers';
    key: string;
} | {
    fn: 'closeRight';
    key: string;
} | {
    fn: 'closeAll';
} | {
    fn: 'reopen';
    record: CloseRecord;
    context: OpenContext;
} | {
    fn: 'openItem';
    key: string;
    context: OpenContext;
} | {
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
export declare const CASE_KINDS: readonly CaseKind[];
/** The functions each op kind may call. */
export declare const OP_FNS: Record<OpKind, readonly Op['fn'][]>;
/** What a case may name as its rule: a success task, a numbered rule, or a contract ruling. */
export declare const RULES: readonly string[];
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
    point: {
        x: number;
        y: number;
    };
    source: Source;
    target: Target | null;
    resultRect: ResultRect | null;
}
export type ValidateCase = Named & {
    input: unknown;
    options?: ValidateOptions;
    where?: string;
} & ({
    valid: true;
} | {
    reason: string;
});
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
/**
 * The canonical case directory. Resolved through one anchor case because under
 * vitest's jsdom environment import.meta.url is not a file: URL.
 */
export declare const CASES_DIR: string;
export interface LoadedCase<K extends CaseKind = CaseKind> {
    kind: K;
    file: string;
    bytes: string;
    data: CaseByKind[K];
}
/** Every case of one kind, sorted by file name. */
export declare function loadCases<K extends CaseKind>(kind: K, dir?: string): LoadedCase<K>[];
/** Layout sizes compare at this. */
export declare const SIZE_EPSILON = 0.005;
/** Rects compare at this many px. */
export declare const RECT_EPSILON = 2;
/**
 * Structural difference of two JSON values: numbers within `epsilon`, every
 * other value exactly, object key sets equal. Returns one line per mismatch.
 */
export declare function diffApprox(actual: unknown, expected: unknown, epsilon: number, path?: string): string[];
/** Layouts (or null for an empty dock) at the size tolerance. */
export declare function diffLayout(actual: unknown, expected: Layout | null, path?: string): string[];
/** Rects (and result rects) at the px tolerance. */
export declare function diffRect(actual: unknown, expected: unknown, path?: string): string[];
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
    openSet?: Fn<[
        Layout | null,
        string[],
        'tabs' | 'side' | 'layout',
        CollectionLayout<string> | undefined,
        EngineContext
    ], OpResult>;
    hitTest?: Fn<[Layout, Rect, TabRects, {
        x: number;
        y: number;
    }, Source], Target | null>;
    resultRect?: Fn<[Layout, Rect, TabRects, Source, Target], ResultRect | null>;
    validateLayout?: Fn<[unknown, ValidateOptions?], LayoutValidation>;
    normalize?: Fn<[Layout], Layout>;
    normalizeCollection?: Fn<[
        CollectionLayout<string | number>
    ], NormalizedCollection<string | number>>;
}
/** The context an engine receives: the case's data, with pickTarget as a function. */
export interface EngineContext {
    focusedTile: string | null;
    dock: DockPx;
    pickTarget?: (layout: Layout | null, key: string) => Target | null;
    caller?: 'agent' | 'person';
    skip?: string[];
}
/** Replay one operation case; [] when the engine matches it. */
export declare function runOpCase(c: OpCase, engine: Engine): string[];
/** Replay a hit case's target selection. */
export declare function runHitTarget(c: HitCase, engine: Engine): string[];
/** Replay a hit case's result rect (the highlight). Cases with no target have none. */
export declare function runHitResultRect(c: HitCase, engine: Engine): string[];
/** Replay a validate case; the reason and, when present, the where path are compared. */
export declare function runValidateCase(c: ValidateCase, engine: Engine): string[];
/** Replay a normalize case. */
export declare function runNormalizeCase(c: NormalizeCase, engine: Engine): string[];
/** Replay any case against the engine. Hit cases check target and result rect. */
export declare function runCase(kind: CaseKind, c: AnyCase, engine: Engine): string[];
/** Problems with a Source; [] when well formed. */
export declare function checkSource(value: unknown, path: string): string[];
/** Problems with a Target; [] when well formed. */
export declare function checkTarget(value: unknown, path: string): string[];
/** The tile a Target names, if any. */
export declare function targetTile(target: Target): string | null;
/** Problems with a dock-form layout (or null): valid, and already normalised. */
export declare function checkLayout(value: unknown, path: string): string[];
/** Problems with one loaded case; [] when it is well formed and names a rule. */
export declare function checkCase(loaded: LoadedCase): string[];
/** The model's own C-LAYOUT functions, as an Engine fragment. */
export declare const layoutEngine: Engine;
export {};
