// @ts-check
/* OSPREY Web Terminal — Dock Layout Reconcile
 *
 * The pure repair run on a persisted dock layout before it is applied: a
 * dock layout (the engine's dock form — tiles, the panel each shows, split
 * fractions) is checked with the engine's `validateLayout`, every key the
 * dock cannot build any more is removed with the engine's `removeTab` (a
 * tile's last key closes the tile by the leave rule), a tile stored with
 * several panels — a dock saved by the release that had tabs — keeps the one
 * it was showing (dock-policy.js's `onePanelPerTile`; the others leave the
 * dock and keep their sidebar rows), and the result goes through the
 * engine's `normalize`.
 *
 * A layout the dockview-era shell stored — dockview's own serialization,
 * `{grid, panels}` — is read into the dock form first (dock-apply.js's
 * `fromSerialized`), so a browser that last ran that shell comes back with
 * its tiles.
 *
 * PURE by contract — nothing in this module reads or mutates live dock/DOM
 * state, and the argument is never mutated (it is deep-copied before the
 * repair). Registered keys absent from the layout are NOT added here.
 */

import { keysOf, normalize, removeTab, validateLayout } from '/static/vendor/dock-engine.mjs';
import { TERMINAL_KEY, fromSerialized, isSerializedDockview } from './dock-apply.js';
import { onePanelPerTile } from './dock-policy.js';

/** @typedef {import('/static/vendor/dock-engine.mjs').Layout} Layout */

/**
 * Repair a persisted dock layout against the keys the dock can build right now.
 *
 * @param {unknown} layout  A dock-form layout, a dockview serialization, or
 *   the raw JSON string of either. A string is parsed here; a parse failure
 *   counts as unusable input.
 * @param {Iterable<string>} registeredKeys  The keys the dock can (re)create:
 *   bare panel ids, as the holder's layout and `GET /api/panels` name them.
 *   `terminal` is always kept.
 * @returns {Layout|null}  A fresh, normalised layout with every unknown key
 *   closed and every tile holding one panel, or `null` when the input is
 *   unparseable, fails `validateLayout`, or has no registered key left — the
 *   only cases that warrant the default.
 */
export function reconcile(layout, registeredKeys) {
  const registered = registeredKeySet(registeredKeys);

  let data;
  try {
    data = typeof layout === 'string' ? JSON.parse(layout) : structuredClone(layout);
  } catch {
    return null;
  }
  if (isSerializedDockview(data)) data = fromSerialized(data);
  if (data == null || !validateLayout(data, { form: 'dock' }).valid) return null;

  /** @type {Layout|null} */
  let next = data;
  for (const key of keysOf(/** @type {Layout} */ (data))) {
    if (!registered.has(key)) next = removeTab(next, key).layout;
  }
  next = onePanelPerTile(next).layout;
  return next == null ? null : normalize(next);
}

/**
 * The registered set, always including the terminal; entries that are not
 * non-empty strings are skipped.
 * @param {Iterable<string>} list
 * @returns {Set<string>}
 */
function registeredKeySet(list) {
  const out = new Set([TERMINAL_KEY]);
  if (list == null || typeof list !== 'object') return out;
  for (const key of list) {
    if (typeof key === 'string' && key) out.add(key);
  }
  return out;
}
