// @ts-check
/* OSPREY Web Terminal — Detached Panes
 *
 * The layout holder's record of the panes living in windows of their own
 * (dock-apply.js composes it; nothing else creates one). Each entry holds what
 * brings its pane back:
 *
 *   - `record`        the engine close record of the tile it left, or null for
 *                     a pane that left with no tile (it comes back by the open
 *                     rule);
 *   - `neighbourKey`  the panel its neighbour tile showed then. Tile ids are
 *                     re-minted as max+1, so a kept id can name an unrelated
 *                     tile later; the neighbour is found again BY KEY before
 *                     any re-dock ({@link resolved});
 *   - `panelRecord`   the title and params its panel is built with;
 *   - `page`          the root-relative page it was on;
 *   - `seq`           the order it left in.
 *
 * Detached state is never stored. `persisted(layout)` is the layout with every
 * detached pane docked again, on a copy, newest first, so each record meets
 * the layout it was taken from; that is the form localStorage and the session
 * layout keep, and a reload brings each pane back where it was (R7). Where a
 * pane lands is the injected {@link Placement} (dock-open.js's
 * `detachPlacement`), which this module does not import: like dock-apply.js it
 * imports nothing but the engine and the policy.
 */

import { closeTab, findTile, keysOf, tileOf } from '/static/vendor/dock-engine.mjs';
import { crowdedTile } from './dock-policy.js';

/** @typedef {import('/static/vendor/dock-engine.mjs').Layout} Layout */
/** @typedef {import('/static/vendor/dock-engine.mjs').CloseRecord} CloseRecord */
/** @typedef {import('./dock-apply.js').OpResult} OpResult */
/** @typedef {import('./dock-apply.js').OpContext} OpContext */
/** @typedef {import('./dock-apply.js').PanelRecord} PanelRecord */

/**
 * One pane in a window of its own: what brings it back.
 * @typedef {object} DetachedEntry
 * @property {CloseRecord|null} record  the engine record of the tile it left;
 *   null when it held none
 * @property {string|null} neighbourKey  the panel its neighbour showed then, or
 *   null once that neighbour is no longer known (a layout without the pane
 *   came in from outside)
 * @property {PanelRecord|undefined} panelRecord  the title and params its panel is built with
 * @property {string|null} page  the root-relative page it was on
 * @property {number} seq  the order it left in
 */

/**
 * Where a detached pane lands. Each rule takes the layout, the pane's key, its
 * entry (its `record.neighbour` already resolved from `neighbourKey`) and the
 * holder's context, and returns the accepted result, or null when there is
 * nowhere it may go. `reattach` is the live return (R6); `persist` places it in
 * the saved copy, and never at a docked panel's cost.
 * @typedef {object} Placement
 * @property {(layout: Layout|null, key: string, entry: DetachedEntry, ctx: OpContext) => OpResult|null} reattach
 * @property {(layout: Layout|null, key: string, entry: DetachedEntry, ctx: OpContext) => OpResult|null} persist
 */

/** A refusal (by name, changing nothing): the result docks a detached pane. */
export const DETACHED = 'detached';

/**
 * A panel record from a dockview panel entry: its title and params.
 * @param {any} entry  a SerializedDockview panel entry
 * @returns {PanelRecord | undefined}  undefined without an entry
 */
export function panelRecordOf(entry) {
  if (!entry) return undefined;
  return {
    ...(typeof entry.title === 'string' ? { title: entry.title } : {}),
    ...(entry.params != null ? { params: entry.params } : {}),
  };
}

/**
 * The panel the neighbour tile a close record names shows in `layout`, or null.
 * @param {Layout|null} layout
 * @param {CloseRecord} record
 * @returns {string|null}
 */
function neighbourKeyOf(layout, record) {
  if (record.kind !== 'tile' || record.neighbour === null) return null;
  return findTile(layout, record.neighbour)?.showing ?? null;
}

/**
 * `entry` with its record's neighbour found again by key in `layout`.
 * @param {Layout|null} layout
 * @param {DetachedEntry} entry
 * @returns {DetachedEntry}
 */
function resolved(layout, entry) {
  if (entry.record?.kind !== 'tile') return entry;
  const tile = entry.neighbourKey === null ? null : tileOf(layout, entry.neighbourKey);
  return { ...entry, record: { ...entry.record, neighbour: tile?.tile ?? null } };
}

/** @param {OpResult|null|undefined} result @returns {result is OpResult} */
const lands = (result) => !!result && result.changed && crowdedTile(result.layout) === null;

/**
 * The detached panes of one holder.
 * @param {Placement|null} placement  where a pane lands; without one a pane
 *   can leave but not come back
 */
export function createDetachedPanes(placement) {
  /** @type {Map<string, DetachedEntry>} */
  const entries = new Map();
  /** @type {Set<() => void>} */
  const listeners = new Set();
  let seq = 0;

  /** The queries the holder exposes as its own. */
  const queries = {
    /** @param {string} key @returns {boolean} whether the pane is in a window of its own */
    isDetached(key) {
      return entries.has(key);
    },
    /** @returns {string[]} the detached panes' keys, sorted */
    detachedKeys() {
      return [...entries.keys()].sort();
    },
    /**
     * A detached pane's entry (do not mutate), or undefined.
     * @param {string} key
     * @returns {Readonly<DetachedEntry> | undefined}
     */
    detachedEntry(key) {
      return entries.get(key);
    },
    /**
     * Hear every change of the detached set: a detach, a reattach, a forget.
     * @param {() => void} listener
     * @returns {() => void} unsubscribe
     */
    onDetachedChange(listener) {
      listeners.add(listener);
      return () => {
        listeners.delete(listener);
      };
    },
  };

  /** Tell every listener the detached set changed. */
  const changed = () => {
    for (const listener of [...listeners]) {
      try {
        listener();
      } catch (err) {
        console.error('dock: a detached-change listener failed', err);
      }
    }
  };

  return {
    queries,

    changed,

    /**
     * The pane `key` leaves `layout`: its tile closes (the engine's leave
     * rule), if it has one, and its entry is stored, before the caller applies
     * the result. Null, storing nothing, for a pane already detached.
     * @param {Layout|null} layout
     * @param {string} key
     * @param {PanelRecord|undefined} panelRecord
     * @param {string|null} page
     * @returns {OpResult|null}  the layout after, unannounced
     */
    leave(layout, key, panelRecord, page) {
      if (entries.has(key)) return null;
      if (tileOf(layout, key) === null) {
        const entry = { record: null, neighbourKey: null, panelRecord, page, seq: ++seq };
        entries.set(key, entry);
        return { layout, changed: false };
      }
      const result = closeTab(layout, key);
      if (!result.changed || !result.record) return null;
      const { record } = result;
      const neighbourKey = neighbourKeyOf(layout, record);
      entries.set(key, { record, neighbourKey, panelRecord, page, seq: ++seq });
      return result;
    },

    /**
     * Where the pane `key` lands in `layout` now; its entry is dropped when it
     * lands. Null, keeping the entry, when it is not detached, there is no
     * placement, or the placement finds no place.
     * @param {Layout|null} layout
     * @param {string} key
     * @param {OpContext} ctx
     * @returns {OpResult|null}  the layout after, unannounced
     */
    comeBack(layout, key, ctx) {
      const entry = entries.get(key);
      if (!entry || !placement) return null;
      const result = placement.reattach(layout, key, resolved(layout, entry), ctx);
      if (!lands(result)) return null;
      entries.delete(key);
      return result;
    },

    /**
     * Drop a pane's entry without docking it, and announce it.
     * @param {string} key
     * @returns {boolean} whether it was detached
     */
    forget(key) {
      if (!entries.delete(key)) return false;
      changed();
      return true;
    },

    /**
     * Record the page a detached pane is on, and announce a change of it.
     * @param {string} key
     * @param {string|null} page  null (a page the hub cannot read) changes nothing
     */
    setPage(key, page) {
      const entry = entries.get(key);
      if (!entry || page === null || entry.page === page) return;
      entry.page = page;
      changed();
    },

    /** Whether `layout` docks any detached pane. @param {Layout|null} layout */
    docks(layout) {
      return entries.size > 0 && keysOf(layout).some((key) => entries.has(key));
    },

    /**
     * Take every detached pane out of an incoming layout (a stored layout, a
     * session switch, Reset layout). A pane it holds is closed there, and its
     * entry takes that close as its way back; a layout without it says nothing
     * about where it was, so its old neighbour is forgotten and it comes back
     * by the open rule.
     * @param {Layout|null} incoming
     * @returns {Layout|null}
     */
    prune(incoming) {
      let next = incoming;
      for (const [key, entry] of entries) {
        if (tileOf(next, key) === null) {
          entry.neighbourKey = null;
          continue;
        }
        const result = closeTab(next, key);
        if (result.record) {
          entry.record = result.record;
          entry.neighbourKey = neighbourKeyOf(next, result.record);
        }
        next = result.layout;
      }
      return next;
    },

    /**
     * A copy of `layout` with every detached pane docked again by the
     * placement's `persist` rule, newest first. A pane with no place that
     * would not cost a docked panel stays out, and loads closed.
     * @param {Layout|null} layout
     * @param {OpContext} ctx
     * @returns {Layout|null}
     */
    persisted(layout, ctx) {
      let copy = layout === null ? null : structuredClone(layout);
      const newestFirst = [...entries].sort((a, b) => b[1].seq - a[1].seq);
      for (const [key, entry] of newestFirst) {
        const result = placement?.persist(copy, key, resolved(copy, entry), ctx) ?? null;
        if (lands(result)) copy = result.layout;
        else
          console.warn(`dock: no place for detached ${key} in the stored layout; it loads closed`);
      }
      return copy;
    },
  };
}
