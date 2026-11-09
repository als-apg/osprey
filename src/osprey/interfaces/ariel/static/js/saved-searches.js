// @ts-check
/**
 * ARIEL saved searches: the named records kept in browser storage only
 * (ruling e-0001 R3). No DOM.
 *
 * A record is `{name, query, mode, filter, filters}`: `filter` is the C-FILTER
 * the search sent when it was saved (`requestFilter`, so a search saved under
 * Mine carries the Mine name as its author) and `filters` the per-mode knobs.
 * Records written by P1 are migrated on every read (`migrateSaved`). Storage
 * may be blocked in a sandboxed iframe: nothing here throws.
 */

import { scopedStorageKey } from '/design-system/js/storage-scope.js';
import { migrateSaved } from './filter-state.js';

/**
 * @typedef {Object} SavedSearch
 * @property {string} name
 * @property {string} query
 * @property {string|null} mode
 * @property {Record<string, any>} filter - C-FILTER
 * @property {Record<string, any>} filters - per-mode knobs
 */

/** Base storage key; scoped by `scopedStorageKey`, shared by both frames. */
export const SAVED_KEY = 'ariel-saved-searches';

/** @returns {string|null} */
function readStored() {
  try {
    return localStorage.getItem(scopedStorageKey(SAVED_KEY));
  } catch {
    return null;
  }
}

/** @param {string} value */
function writeStored(value) {
  try {
    localStorage.setItem(scopedStorageKey(SAVED_KEY), value);
  } catch {
    // Storage blocked: the list lives for this page only.
  }
}

/**
 * Every saved search, in stored order. Malformed items and repeated names are
 * skipped; P1 records come back in the current shape.
 * @returns {SavedSearch[]}
 */
export function readSaved() {
  const raw = readStored();
  if (!raw) return [];
  /** @type {unknown} */
  let parsed;
  try {
    parsed = JSON.parse(raw);
  } catch {
    return [];
  }
  if (!Array.isArray(parsed)) return [];
  /** @type {SavedSearch[]} */
  const out = [];
  for (const item of parsed) {
    if (!item || typeof item !== 'object') continue;
    const name = typeof item.name === 'string' ? item.name.trim() : '';
    if (!name || out.some((s) => s.name === name)) continue;
    const migrated = migrateSaved({
      filter: item.filter && typeof item.filter === 'object' ? item.filter : {},
      filters:
        item.filters && typeof item.filters === 'object' && !Array.isArray(item.filters)
          ? item.filters
          : {},
    });
    out.push({
      name,
      query: typeof item.query === 'string' ? item.query : '',
      mode: typeof item.mode === 'string' ? item.mode : null,
      filter: migrated.filter,
      filters: { ...migrated.filters },
    });
  }
  return out;
}

/** @param {SavedSearch[]} list */
function writeSaved(list) {
  writeStored(JSON.stringify(list));
}

/**
 * @param {string} name
 * @returns {SavedSearch|undefined}
 */
export function findSaved(name) {
  return readSaved().find((s) => s.name === name);
}

/**
 * Store `record` under its (trimmed) name; a blank name is ignored. Saving an
 * existing name replaces that record in place. The record is copied.
 * @param {SavedSearch} record
 * @returns {boolean} whether it was stored
 */
export function saveSearch(record) {
  const name = typeof record?.name === 'string' ? record.name.trim() : '';
  if (!name) return false;
  /** @type {SavedSearch} */
  const entry = {
    name,
    query: typeof record.query === 'string' ? record.query : '',
    mode: typeof record.mode === 'string' ? record.mode : null,
    filter: structuredClone(record.filter ?? {}),
    filters: structuredClone(record.filters ?? {}),
  };
  const list = readSaved();
  const at = list.findIndex((s) => s.name === name);
  if (at >= 0) list[at] = entry;
  else list.push(entry);
  writeSaved(list);
  return true;
}

/**
 * Delete the saved search called `name`, if any.
 * @param {string} name
 */
export function removeSearch(name) {
  writeSaved(readSaved().filter((s) => s.name !== name));
}
