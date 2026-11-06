// @ts-check
/* OSPREY Web Terminal — Dock Layout Holder
 *
 * The holder is the dock's single source of truth: the current dock layout
 * (the engine's dock form — tiles keyed by bare panel ids plus `terminal`,
 * each tile holding ONE panel, OSPREY's rule in dock-policy.js), the focused
 * tile, the last tile that showed a service, the panels living in their own
 * pop-out windows, the close records an undo reopens, the server `rev` last
 * acknowledged and whether the layout changed since. Its writers are:
 *
 *   - `run(op)`        an engine operation (open, close, move, ...); a result
 *                      giving a tile several panels is refused by the policy's
 *                      name and changes nothing;
 *   - `restore(l)`     a saved dock (validated, normalised, no undo record); a
 *                      tile saved with several panels keeps the one it showed;
 *   - `show(tile, k)`  a person focused a tile: dockview already shows it, so
 *                      the holder records it and does NOT re-apply;
 *   - `resized(l, px)` a sash gesture or a dock size change: sizes only, from a
 *                      tree of the same shape, never re-applied;
 *   - `setPoppedOut()` a panel left for, or came back from, its own window.
 *
 * This is the only module that calls dockview mutators (a repository test,
 * test_dock_mutator_guard.py, keeps it so). Applying a layout is one
 * `api.fromJSON(toSerialized(layout, px))`:
 *
 *   - group id = tile id, a group's one view = the tile's panel, active = showing;
 *   - the `iframe:` prefix of service panels is added here and stripped here
 *     (`fromSerialized`), nowhere else; `terminal` keeps its own id;
 *   - every serialized panel carries the 40 px minimums (dockview reads them off
 *     a group's active panel; constraints set after fromJSON come too late);
 *   - it runs inside the guard the page hands in (dock-sync's echo guard), so
 *     the rebuild's focus changes are never reported as a person's click;
 *   - it is held while a drag is live and runs once, a macrotask after the drag
 *     ends, so dockview's drag source still exists for its own `dragend`;
 *   - keyboard focus is saved and restored across it: the terminal's xterm (its
 *     subtree is moved, never rebuilt — createComponent hands dockview the same
 *     adopted node) and a focused header bar, re-found by tile and key.
 *
 * This module imports nothing but the engine and the policy (which imports
 * the engine alone), so the pure repair (dock-reconcile.js) and the placement
 * rules (dock-open.js) can read its vocabulary without pulling the page in.
 */

import {
  MIN_PX,
  SIZE_TOLERANCE,
  axisOf,
  findTile,
  fractionsOf,
  isTile,
  itemFractions,
  layoutSizes,
  minimumSize,
  normalize,
  reopen,
  validateLayout,
} from '/static/vendor/dock-engine.mjs';
import { ONE_PANEL_PER_TILE, crowdedTile, onePanelPerTile } from './dock-policy.js';

/** @typedef {import('/static/vendor/dock-engine.mjs').Layout} Layout */
/** @typedef {import('/static/vendor/dock-engine.mjs').Tile} Tile */
/** @typedef {import('/static/vendor/dock-engine.mjs').DockPx} DockPx */
/** @typedef {import('/static/vendor/dock-engine.mjs').CloseRecord} CloseRecord */
/** @typedef {{ width: number, height: number }} DockSize */
/** @typedef {Record<string, any>} PanelState  a SerializedDockview panel entry */

/**
 * What an engine operation returns: the layout after, whether it differs, and
 * optionally the tile to focus and a close record.
 * @typedef {object} OpResult
 * @property {Layout|null} layout
 * @property {boolean} changed
 * @property {string|null} [focus]
 * @property {CloseRecord|null} [record]
 */

/**
 * The context an operation runs in: the holder's focus facts and the dock's px.
 * @typedef {object} OpContext
 * @property {string|null} focusedTile
 * @property {string|null} lastServiceTile
 * @property {DockPx} dock
 */

/**
 * What reopening a stored panel in a new session needs beyond its key: the
 * title and params its dockview panel is built with.
 * @typedef {object} PanelRecord
 * @property {string} [title]
 * @property {any} [params]
 */

/** The terminal/chat card's layout key, which is also its dockview id (never prefixed). */
export const TERMINAL_KEY = 'terminal';
/** Dock panel title of the Agent pane. A real title makes the pane a first-class
 *  tile; the bar it adopts (dock-tab.js) is the visible label. Stored layouts
 *  carrying an older title (`SESSION`, or the empty string from before the pane
 *  was titled) are healed on the next apply, which writes this one. */
export const TERMINAL_TITLE = 'Agent';
/** dockview panel-id namespace of the service placeholders, kept distinct from
 *  the terminal's own id and from panel-manager's rail ids. A dock layout key
 *  never carries it. */
export const PLACEHOLDER_PREFIX = 'iframe:';
/** dockview component of every service placeholder. dock-workspace.js's
 *  createComponent does not know it, so it returns its neutral fallback element:
 *  the empty content node dock-iframe.js lays its iframe over. */
export const PLACEHOLDER_COMPONENT = 'dock-iframe-placeholder';
/** Attribute a header bar carries its bare key on; focus is re-found by it. */
export const TAB_KEY_ATTRIBUTE = 'data-panel-key';
/** Close records kept for undo. */
const CLOSED_CAP = 50;
/** Macrotask turns a focus restore waits for a rebuilt bar to appear. */
const RESTORE_TURNS = 10;

const HORIZONTAL = 'HORIZONTAL';
const VERTICAL = 'VERTICAL';

// --- keys <-> dockview ids --------------------------------------------------------

/**
 * The dockview panel id for a layout key.
 * @param {string} key
 * @returns {string}
 */
export function panelIdOf(key) {
  return key === TERMINAL_KEY ? TERMINAL_KEY : PLACEHOLDER_PREFIX + key;
}

/**
 * The layout key for a dockview panel id: the terminal's id is its own key, a
 * service placeholder's key is its id without {@link PLACEHOLDER_PREFIX}. Null
 * for anything else (a non-string, or an id this dock never builds).
 * @param {unknown} id
 * @returns {string | null}
 */
export function keyOfPanelId(id) {
  if (id === TERMINAL_KEY) return TERMINAL_KEY;
  return typeof id === 'string' && id.startsWith(PLACEHOLDER_PREFIX)
    ? id.slice(PLACEHOLDER_PREFIX.length)
    : null;
}

// --- serializer --------------------------------------------------------------------

/**
 * @typedef {object} SerializeOptions
 * @property {Readonly<Record<string, PanelState>>} [panels]  stored entries by dockview id
 *   (as `api.toJSON().panels`); kept for every key they cover.
 * @property {(key: string) => PanelRecord | undefined} [newPanel]  state merged over
 *   the default entry of a key with no stored entry.
 * @property {string|null} [activeTile]  the group to make active; ignored when absent.
 */

/**
 * @param {string} key
 * @param {SerializeOptions} options
 * @returns {PanelState}
 */
function panelEntry(key, options) {
  const id = panelIdOf(key);
  const stored = options.panels?.[id];
  const base = stored
    ? { ...stored }
    : key === TERMINAL_KEY
      ? { contentComponent: TERMINAL_KEY, title: TERMINAL_TITLE, ...options.newPanel?.(key) }
      : { contentComponent: PLACEHOLDER_COMPONENT, title: key, ...options.newPanel?.(key) };
  // The Agent pane's title is the page's own, whatever an older layout stored.
  if (key === TERMINAL_KEY) base.title = TERMINAL_TITLE;
  return { ...base, id, minimumWidth: MIN_PX, minimumHeight: MIN_PX };
}

/**
 * @param {Tile} tile
 * @param {number} size
 */
function leafOf(tile, size) {
  return {
    type: 'leaf',
    data: { views: tile.tabs.map(panelIdOf), activeView: panelIdOf(tile.showing), id: tile.tile },
    size,
  };
}

/**
 * A node laid out over `size` px along its parent's axis; its own items share
 * `rect` along its split axis exactly as the engine's `geometry` does.
 * @param {Layout} node
 * @param {number} size
 * @param {DockSize} rect
 * @returns {any}
 */
function nodeOf(node, size, rect) {
  if (isTile(node)) return leafOf(node, size);
  const axis = axisOf(node.split);
  const span = axis === 'width' ? rect.width : rect.height;
  const minimums = node.items.map((item) => minimumSize(item, axis, MIN_PX));
  const sizes = layoutSizes(itemFractions(node.items), span, minimums);
  const data = node.items.map((item, i) => {
    const px = sizes[i];
    return nodeOf(item, px, axis === 'width' ? { ...rect, width: px } : { ...rect, height: px });
  });
  return { type: 'branch', data, size };
}

/**
 * Every tile of a layout, in tree order.
 * @param {Layout|null} layout
 * @returns {Tile[]}
 */
export function tilesOf(layout) {
  if (layout === null) return [];
  if (isTile(layout)) return [layout];
  return layout.items.flatMap(tilesOf);
}

/**
 * The SerializedDockview for a dock layout on a dock of `dockPx`; `null` is the
 * empty dock. The input is normalised first and never mutated.
 * @param {Layout|null} layout
 * @param {DockSize} dockPx
 * @param {SerializeOptions} [options]
 * @returns {any}
 */
export function toSerialized(layout, dockPx, options = {}) {
  const width = Math.max(0, Math.round(dockPx.width));
  const height = Math.max(0, Math.round(dockPx.height));
  const tree = layout === null ? null : normalize(layout);
  let root;
  let orientation = HORIZONTAL;
  if (tree === null) {
    root = { type: 'branch', data: [] };
  } else if (isTile(tree)) {
    // dockview's fromJSON throws unless the root is a branch.
    root = { type: 'branch', data: [leafOf(tree, width)], size: height };
  } else {
    orientation = tree.split === 'row' ? HORIZONTAL : VERTICAL;
    // The root branch's `size` is its extent across its own axis.
    root = nodeOf(tree, tree.split === 'row' ? height : width, { width, height });
  }
  const tiles = tilesOf(tree);
  /** @type {Record<string, PanelState>} */
  const panels = {};
  for (const tile of tiles)
    for (const key of tile.tabs) panels[panelIdOf(key)] = panelEntry(key, options);
  /** @type {any} */
  const serialized = { grid: { root, width, height, orientation }, panels };
  const active = options.activeTile;
  if (active != null && tiles.some((t) => t.tile === active)) serialized.activeGroup = active;
  return serialized;
}

/**
 * The dock layout a SerializedDockview describes (as `api.toJSON()` returns it,
 * or as a layout stored by the dockview-era shell): groups become tiles keyed
 * by stripped ids, px sizes become fractions. `null` for an empty or unreadable
 * grid. Reads sizes back after a sash drag, and the whole tree after a change
 * dockview made on its own.
 * @param {any} serialized
 * @returns {Layout|null}
 */
export function fromSerialized(serialized) {
  const grid = serialized?.grid;
  if (!grid || !grid.root) return null;
  /**
   * @param {any} node
   * @param {boolean} horizontal  whether this node lays its items out along width
   * @returns {Layout|null}
   */
  const read = (node, horizontal) => {
    if (node?.type === 'leaf') {
      const data = node.data ?? {};
      /** @type {string[]} */
      const tabs = [];
      for (const view of Array.isArray(data.views) ? data.views : []) {
        const key = keyOfPanelId(view);
        if (key !== null && !tabs.includes(key)) tabs.push(key);
      }
      if (tabs.length === 0 || typeof data.id !== 'string' || !data.id) return null;
      const showing = keyOfPanelId(data.activeView) ?? tabs[0];
      return { tile: data.id, tabs, showing: tabs.includes(showing) ? showing : tabs[0] };
    }
    if (node?.type !== 'branch' || !Array.isArray(node.data)) return null;
    const children = node.data.map((/** @type {any} */ child) => ({
      layout: read(child, !horizontal),
      size: Number(child?.size) || 0,
    }));
    const kept = children.filter((/** @type {{layout: Layout|null}} */ c) => c.layout !== null);
    if (kept.length === 0) return null;
    if (kept.length === 1) return /** @type {Layout} */ (kept[0].layout);
    const total = kept.reduce(
      (/** @type {number} */ sum, /** @type {{size: number}} */ c) => sum + c.size,
      0
    );
    const fractions =
      total > 0 ? fractionsOf(kept.map((/** @type {{size: number}} */ c) => c.size)) : null;
    const items = kept.map((/** @type {{layout: Layout}} */ c, /** @type {number} */ i) =>
      fractions ? { ...c.layout, size: fractions[i] } : { ...c.layout }
    );
    return /** @type {Layout} */ ({ split: horizontal ? 'row' : 'column', items });
  };
  let layout;
  try {
    layout = read(grid.root, grid.orientation !== VERTICAL);
  } catch {
    return null;
  }
  if (layout === null) return null;
  // A tile id used twice (a hand-corrupted store) is no layout at all.
  const seen = new Set();
  for (const tile of tilesOf(layout)) {
    if (seen.has(tile.tile)) return null;
    seen.add(tile.tile);
  }
  // The root carries no size: read() sets sizes only on a split's items.
  return normalize(layout);
}

/**
 * Whether a value is a dockview serialization (`{grid: {root}, panels}`), the
 * shape the dockview-era shell stored and `api.toJSON()` returns.
 * @param {unknown} value
 * @returns {boolean}
 */
export function isSerializedDockview(value) {
  const v = /** @type {any} */ (value);
  return (
    !!v &&
    typeof v === 'object' &&
    !!v.grid &&
    typeof v.grid === 'object' &&
    !!v.grid.root &&
    typeof v.grid.root === 'object' &&
    !!v.panels &&
    typeof v.panels === 'object'
  );
}

// --- focus ---------------------------------------------------------------------------

/**
 * @typedef {object} SavedFocus
 * @property {HTMLElement} element
 * @property {{ key: string, tile: string|null } | null} bar  set for a header bar
 */

/**
 * @param {any} api
 * @param {Element} target
 * @returns {string|null}
 */
function groupHolding(api, target) {
  for (const group of api.groups ?? []) if (group.element?.contains(target)) return group.id;
  return null;
}

/**
 * @param {any} api
 * @returns {SavedFocus|null}
 */
function saveFocus(api) {
  const active = typeof document === 'undefined' ? null : document.activeElement;
  if (!(active instanceof HTMLElement) || active === document.body) return null;
  const barElement = active.closest(`[${TAB_KEY_ATTRIBUTE}]`);
  const tile = barElement ? groupHolding(api, barElement) : null;
  const key = barElement && tile !== null ? barElement.getAttribute(TAB_KEY_ATTRIBUTE) : null;
  return { element: active, bar: key !== null ? { key, tile } : null };
}

/**
 * Where a saved bar focus goes after a rebuild: the bar of the tile now
 * holding the key, else the bar of the tile it was in.
 * @param {{ key: string, tile: string|null }} bar
 * @param {Layout|null} layout
 * @returns {{ tile: string, key: string } | null}
 */
function focusTarget(bar, layout) {
  const holder = tilesOf(layout).find((t) => t.tabs.includes(bar.key));
  if (holder) return { tile: holder.tile, key: bar.key };
  const was = bar.tile === null ? null : findTile(layout, bar.tile);
  return was ? { tile: was.tile, key: was.showing } : null;
}

/**
 * @param {any} api
 * @param {string} tile
 * @param {string} key
 * @returns {HTMLElement|null}
 */
function barIn(api, tile, key) {
  const group = (api.groups ?? []).find((/** @type {any} */ g) => g.id === tile);
  if (!group) return null;
  for (const el of group.element.querySelectorAll(`[${TAB_KEY_ATTRIBUTE}]`)) {
    if (el.getAttribute(TAB_KEY_ATTRIBUTE) === key && el instanceof HTMLElement) return el;
  }
  return null;
}

/**
 * Put focus back after a rebuild. Resolves to whether it was restored.
 * @param {any} api
 * @param {SavedFocus} saved
 * @param {Layout|null} layout
 * @returns {Promise<boolean>}
 */
function restoreFocus(api, saved, layout) {
  if (!saved.bar) {
    // The xterm or overlay content: it is moved, not rebuilt, so it only needs
    // focus back if the move took it.
    if (!saved.element.isConnected) return Promise.resolve(false);
    if (document.activeElement !== saved.element) saved.element.focus({ preventScroll: true });
    return Promise.resolve(document.activeElement === saved.element);
  }
  const target = focusTarget(saved.bar, layout);
  if (!target) return Promise.resolve(false);
  const attempt = () => {
    const bar = barIn(api, target.tile, target.key);
    if (!bar) return false;
    bar.focus({ preventScroll: true });
    return true;
  };
  if (attempt()) return Promise.resolve(true);
  return new Promise((resolve) => {
    let turns = 0;
    const retry = () => {
      // Someone else moved focus meanwhile: leave it there.
      const current = document.activeElement;
      if (current && current !== document.body && current.isConnected) return resolve(false);
      if (attempt()) return resolve(true);
      if (++turns >= RESTORE_TURNS) return resolve(false);
      setTimeout(retry, 0);
    };
    queueMicrotask(retry);
  });
}

// --- shape helpers -------------------------------------------------------------------

/**
 * Whether two layouts are the same tree up to sizes.
 * @param {Layout|null} a
 * @param {Layout|null} b
 * @returns {boolean}
 */
export function sameShape(a, b) {
  if (a === null || b === null) return a === b;
  if (isTile(a) || isTile(b)) {
    return (
      isTile(a) &&
      isTile(b) &&
      a.tile === b.tile &&
      a.tabs.length === b.tabs.length &&
      a.tabs.every((key, i) => b.tabs[i] === key)
    );
  }
  return (
    a.split === b.split &&
    a.items.length === b.items.length &&
    a.items.every((item, i) => sameShape(item, b.items[i]))
  );
}

/**
 * Whether two same-shape layouts carry the same sizes, to within the engine's
 * fraction tolerance.
 * @param {Layout} a
 * @param {Layout} b
 * @returns {boolean}
 */
export function sameSizes(a, b) {
  if (Math.abs((a.size ?? 0) - (b.size ?? 0)) > SIZE_TOLERANCE) return false;
  if (isTile(a) || isTile(b)) return true;
  return a.items.every((item, i) => sameSizes(item, /** @type {any} */ (b).items[i]));
}

/**
 * `held` with every `size` taken from the same-shape `sized`.
 * @param {Layout} held
 * @param {Layout} sized
 * @returns {Layout}
 */
function withSizes(held, sized) {
  /** @type {any} */
  const next = isTile(held)
    ? { ...held, tabs: [...held.tabs] }
    : {
        ...held,
        items: held.items.map((item, i) => withSizes(item, /** @type {any} */ (sized).items[i])),
      };
  if (sized.size === undefined) delete next.size;
  else next.size = sized.size;
  return next;
}

// --- the holder ----------------------------------------------------------------------

/**
 * @typedef {object} HolderDeps
 * @property {any} [api]  the live DockviewApi; without one the holder only keeps state
 * @property {<T>(fn: () => T) => T} [guard]  wraps fromJSON (the page passes dock-sync's
 *   echo guard); runs the function bare by default
 * @property {(key: string) => PanelRecord | undefined} [newPanel]  extra state for a key
 *   the holder holds no record for (a restored store's panel records)
 * @property {DockSize} [dockPx]  the dock's px; dockview's own width and height by default
 */

/**
 * Create a layout holder bound to `deps.api`.
 * @param {HolderDeps} [deps]
 */
export function createLayoutHolder(deps = {}) {
  const { api = null, newPanel } = deps;
  const guard = deps.guard ?? ((/** @type {() => any} */ fn) => fn());
  /** @type {Layout|null} */
  let layout = null;
  /** @type {string|null} */
  let focusedTile = null;
  /** @type {string|null} */
  let lastServiceTile = null;
  /** @type {Set<string>} */
  const popouts = new Set();
  /** @type {Map<string, PanelRecord>} */
  const records = new Map();
  /** @type {CloseRecord[]} */
  const closed = [];
  /** @type {number|null} */
  let ackedRev = null;
  let dirty = false;
  /** @type {DockSize|null} */
  let dockPx = deps.dockPx ? { ...deps.dockPx } : null;
  let dragLive = false;
  let pending = false;
  let applying = false;

  /** @returns {DockSize} */
  const px = () => dockPx ?? { width: api?.width ?? 0, height: api?.height ?? 0 };
  /** @returns {DockPx} */
  const dock = () => ({ px: px(), minPx: MIN_PX });

  /** Keep the focused and last-service tiles pointing at live tiles. */
  const track = () => {
    if (focusedTile !== null && !findTile(layout, focusedTile)) focusedTile = null;
    const focused = focusedTile === null ? null : findTile(layout, focusedTile);
    if (focused && focused.showing !== TERMINAL_KEY) lastServiceTile = focused.tile;
    const last = lastServiceTile === null ? null : findTile(layout, lastServiceTile);
    if (!last || last.tabs.every((key) => key === TERMINAL_KEY)) lastServiceTile = null;
  };

  /** @param {string} key @returns {PanelRecord | undefined} */
  const recordOf = (key) => records.get(key) ?? newPanel?.(key);

  /** @returns {Promise<boolean>} */
  const applyNow = () => {
    if (!api) return Promise.resolve(false);
    const saved = saveFocus(api);
    /** @type {Record<string, PanelState>} */
    let panels = {};
    try {
      panels = api.toJSON?.().panels ?? {};
    } catch {
      /* an unreadable dock keeps no entries beyond the records */
    }
    const serialized = toSerialized(layout, px(), {
      panels,
      newPanel: recordOf,
      activeTile: focusedTile,
    });
    applying = true;
    try {
      guard(() => api.fromJSON(serialized));
    } catch (err) {
      console.error('dock: fromJSON failed', err);
      return Promise.resolve(false);
    } finally {
      applying = false;
    }
    return saved ? restoreFocus(api, saved, layout) : Promise.resolve(false);
  };

  const holder = {
    /** The current dock layout (do not mutate). */
    get layout() {
      return layout;
    },
    get focusedTile() {
      return focusedTile;
    },
    get lastServiceTile() {
      return lastServiceTile;
    },
    /** The panels living in their own pop-out windows, sorted. */
    get popouts() {
      return [...popouts].sort();
    },
    /** Close records, oldest first (do not mutate). */
    get closed() {
      return /** @type {readonly CloseRecord[]} */ (closed);
    },
    get ackedRev() {
      return ackedRev;
    },
    get dirty() {
      return dirty;
    },
    get dragLive() {
      return dragLive;
    },
    /** Whether the holder is inside its own `fromJSON`: the active-panel
     *  changes dockview fires while it rebuilds are not a person's. */
    get applying() {
      return applying;
    },
    /** The engine's dock argument for the current px. */
    dock,
    /** @returns {OpContext} the context an operation takes */
    context() {
      return { focusedTile, lastServiceTile, dock: dock() };
    },

    /**
     * Run an engine operation on the held layout; a changed result is kept and
     * applied, a record is kept for undo, `focus` moves the focused tile. A
     * result giving a tile several panels breaks the policy: it is refused by
     * name (`refusal: 'one_panel_per_tile'`) and nothing changes.
     * @param {(layout: Layout|null, ctx: OpContext) => OpResult} op
     * @returns {OpResult & {refusal?: string}}
     */
    run(op) {
      const result = op(layout, holder.context());
      if (!result) return result;
      if (result.changed && crowdedTile(result.layout) !== null) {
        return { layout, changed: false, refusal: ONE_PANEL_PER_TILE };
      }
      if (result.record) {
        closed.push(result.record);
        if (closed.length > CLOSED_CAP) closed.shift();
      }
      if (result.changed) {
        layout = result.layout;
        dirty = true;
      }
      if (result.focus !== undefined) focusedTile = result.focus;
      track();
      if (result.changed) void holder.apply();
      return result;
    },

    /**
     * Reopen the newest close record (the engine's reopen); false when there is none.
     * @returns {boolean}
     */
    reopenLast() {
      const record = closed.pop();
      if (!record) return false;
      return holder.run((current, ctx) => reopen(current, record, ctx)).changed;
    },

    /**
     * Replace the layout with a saved dock: validated and normalised, no undo
     * record. A tile saved with several panels (a dock stored by the release
     * that had tabs) keeps the one it showed; the others leave the dock and
     * keep their sidebar rows. False (and nothing changes) for an invalid
     * layout.
     * @param {unknown} input
     * @returns {boolean}
     */
    restore(input) {
      if (input !== null && !validateLayout(input, { form: 'dock' }).valid) return false;
      const saved = input === null ? null : structuredClone(/** @type {Layout} */ (input));
      const kept = onePanelPerTile(saved).layout;
      layout = kept === null ? null : normalize(kept);
      dirty = true;
      track();
      void holder.apply();
      return true;
    },

    /**
     * A person focused `tile`, which shows `key` (dockview already shows it):
     * record the focus, without re-applying. False for a tile the dock does
     * not hold, or one not showing `key`.
     * @param {string} tile
     * @param {string} key
     * @returns {boolean}
     */
    show(tile, key) {
      const found = findTile(layout, tile);
      if (!layout || !found || !found.tabs.includes(key)) return false;
      focusedTile = tile;
      track();
      return true;
    },

    /**
     * Sizes from a sash gesture or a dock size change. `next` (when given) must
     * be the held tree up to sizes; only its sizes are taken. Never re-applies.
     * @param {Layout|null} next
     * @param {DockSize} [size]
     * @returns {boolean}
     */
    resized(next, size) {
      if (size) dockPx = { width: size.width, height: size.height };
      if (next === null || layout === null) return size !== undefined;
      if (!sameShape(layout, next)) return false;
      const sized = withSizes(layout, next);
      // A read-back of the tree the holder itself just applied carries the
      // same sizes to within dockview's rounding; that is not a change.
      if (!sameSizes(layout, sized)) dirty = true;
      layout = sized;
      return true;
    },

    /**
     * A panel left for its own window (`out`), or came back. The layout the
     * agent reads names the pop-outs beside the tiles, so either bumps it.
     * @param {string} key
     * @param {boolean} out
     * @returns {boolean} whether anything changed
     */
    setPoppedOut(key, out) {
      if (out ? popouts.has(key) : !popouts.has(key)) return false;
      if (out) popouts.add(key);
      else popouts.delete(key);
      dirty = true;
      return true;
    },

    /**
     * Remember the title and params a key's panel is built with when it next
     * enters the dock. Nothing is applied: the record is read at serialization.
     * @param {string} key
     * @param {PanelRecord} record
     */
    setPanelRecord(key, record) {
      records.set(key, { ...records.get(key), ...record });
    },

    /** @param {string} key @returns {PanelRecord | undefined} */
    panelRecord(key) {
      return recordOf(key);
    },

    /**
     * Apply the held layout to dockview now, or once the live drag ends.
     * Resolves to whether keyboard focus was restored.
     * @returns {Promise<boolean>}
     */
    apply() {
      if (dragLive) {
        pending = true;
        return Promise.resolve(false);
      }
      pending = false;
      return applyNow();
    },

    /**
     * Mark a drag live or over. While live, applies are held; when it ends a
     * held apply runs one macrotask later, after dockview's own `dragend`.
     * @param {boolean} live
     */
    setDragLive(live) {
      dragLive = live;
      if (live || !pending) return;
      setTimeout(() => {
        if (!dragLive && pending) void holder.apply();
      }, 0);
    },

    /**
     * The server acknowledged the layout at `rev`.
     * @param {number} rev
     */
    ack(rev) {
      ackedRev = rev;
      dirty = false;
    },
  };
  return holder;
}

/** @typedef {ReturnType<typeof createLayoutHolder>} LayoutHolder */

/** @type {LayoutHolder|null} */
let current = null;

/**
 * Create the page's holder (replacing any earlier one) and return it.
 * @param {HolderDeps} deps
 * @returns {LayoutHolder}
 */
export function initLayoutHolder(deps) {
  current = createLayoutHolder(deps);
  return current;
}

/**
 * The page's holder, or null before initLayoutHolder.
 * @returns {LayoutHolder|null}
 */
export function getLayoutHolder() {
  return current;
}
