// @ts-check
/* OSPREY Web Terminal — Panel Params
 *
 * The layout holder's record of each panel's own state: the dockview params
 * its panel is built with (dock-apply.js composes it; nothing else creates
 * one). The params belong to the tile, wherever it is:
 *
 *   - docked: on the live dockview panel, which `api.toJSON()` stores with it;
 *   - in a window of its own: in its detached entry's record (dock-detached.js);
 *   - between a restore or a reattach and the apply that builds it: pending,
 *     laid over the panel's stored entry and record for the serializer
 *     ({@link laidOver}), and spent by that apply.
 *
 * A restore with the saved dock's records ({@link restored}) gives every panel
 * the params stored for it, or none, so a session's layout brings its own
 * tile state even where the same panel is on screen already. The params never
 * enter the holder's panel records, so a panel closed and opened again has
 * none.
 *
 * Like dock-detached.js, it imports nothing from the page but the engine: the
 * dockview API, the key → dockview id mapping and the detached panes come in
 * as arguments.
 */

import { keysOf } from '/static/vendor/dock-engine.mjs';

/** @typedef {import('./dock-apply.js').PanelRecord} PanelRecord */
/** @typedef {import('./dock-apply.js').PanelState} PanelState */
/** @typedef {import('/static/vendor/dock-engine.mjs').Layout} Layout */
/** @typedef {ReturnType<typeof import('./dock-detached.js').createDetachedPanes>} DetachedPanes */
/** @typedef {Record<string, any>} Params */

/**
 * `params` with `patch` merged in as dockview merges them: a field set to
 * undefined is dropped.
 * @param {Params | undefined} params
 * @param {Params} patch
 * @returns {Params}
 */
function merged(params, patch) {
  const next = { ...params, ...patch };
  for (const [field, value] of Object.entries(patch)) if (value === undefined) delete next[field];
  return next;
}

/**
 * A panel entry or record with `params` in place of its own (none: none).
 * @template {{params?: any}} T
 * @param {T} entry
 * @param {Params | undefined} params
 * @returns {T}
 */
function withParams(entry, params) {
  const next = { ...entry };
  delete next.params;
  return params == null ? next : { ...next, params };
}

/**
 * The params of one holder's panels.
 * @param {object} deps
 * @param {any} deps.api  the live DockviewApi, or null
 * @param {(key: string) => string} deps.idOf  a key's dockview panel id
 * @param {DetachedPanes} deps.panes  the holder's detached panes
 */
export function createPanelParams({ api, idOf, panes }) {
  /** @type {Map<string, Params | undefined>} */
  const pending = new Map();

  return {
    /**
     * The serializer's stored entries and new-panel records, with the pending
     * params laid over them (dock-apply.js `toSerialized`).
     * @param {Record<string, PanelState>} stored  dockview's entries by id
     * @param {(key: string) => PanelRecord | undefined} newPanel
     */
    laidOver(stored, newPanel) {
      if (pending.size === 0) return { panels: stored, newPanel };
      const panels = { ...stored };
      for (const [key, params] of pending) {
        const id = idOf(key);
        if (panels[id]) panels[id] = withParams(panels[id], params);
      }
      return {
        panels,
        newPanel: (/** @type {string} */ key) =>
          pending.has(key) ? withParams(newPanel(key) ?? {}, pending.get(key)) : newPanel(key),
      };
    },

    /** The apply built the pending params. */
    spent() {
      pending.clear();
    },

    /**
     * A saved dock is restored: with its records, each of its panels gets the
     * params its record stores, or none; a detached one in its entry's record.
     * @param {Layout | null} layout  the restored layout
     * @param {Readonly<Record<string, PanelRecord>>} [panels]  the saved dock's records
     * @returns {Layout | null} `layout`
     */
    restored(layout, panels) {
      if (!panels) return layout;
      for (const key of keysOf(layout)) {
        const params = panels[key]?.params;
        if (panes.queries.isDetached(key)) panes.setParams(key, params);
        else pending.set(key, params);
      }
      return layout;
    },

    /**
     * A detached pane docks again: it is built once with the params it left
     * with, which stay out of the record kept for it.
     * @param {string} key
     * @param {PanelRecord} record  the record it left with
     * @returns {PanelRecord} that record without its params
     */
    returning(key, record) {
      pending.set(key, record.params);
      return withParams(record, undefined);
    },

    /**
     * A panel's params: a detached pane's record's, else what the next apply
     * builds it with, else the live panel's. Undefined when it has none, or
     * there is no panel for the key. Do not mutate.
     * @param {string} key
     * @returns {Params | undefined}
     */
    read(key) {
      if (panes.queries.isDetached(key))
        return panes.queries.detachedEntry(key)?.panelRecord?.params;
      if (pending.has(key)) return pending.get(key) ?? undefined;
      // dockview holds `{}` for a panel built with none, and stores none for it.
      const live = api?.getPanel?.(idOf(key))?.params;
      return live && Object.keys(live).length > 0 ? live : undefined;
    },

    /**
     * Merge `patch` into a panel's params. A docked panel takes it live,
     * which dockview announces as a layout change (so the stored form and the
     * session layout are written); a detached pane's record takes it, which
     * is announced as a change of the detached set.
     * @param {string} key
     * @param {Params} patch
     * @returns {boolean} false when there is no panel for the key
     */
    write(key, patch) {
      if (panes.queries.isDetached(key)) {
        panes.setParams(key, merged(panes.queries.detachedEntry(key)?.panelRecord?.params, patch));
        panes.changed();
        return true;
      }
      const waiting = pending.has(key);
      if (waiting) pending.set(key, merged(pending.get(key), patch));
      const panel = api?.getPanel?.(idOf(key));
      panel?.api.updateParameters(patch);
      return waiting || Boolean(panel);
    },
  };
}
