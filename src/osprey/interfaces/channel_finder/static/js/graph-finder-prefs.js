// @ts-check
/**
 * OSPREY Channel Finder — graph finder columns and view preferences.
 *
 * What an operator arranges in the graph finder and expects to find again on
 * the next visit: the result table's column layout (order, widths, hidden
 * columns), the row density, and which facets are folded. Each lives in
 * localStorage under a per-render key ({@link scopedStorageKey}), so two
 * deployments served from one origin never share an arrangement.
 *
 * Storage is a convenience, never a dependency: every read falls back to the
 * defaults and every write swallows a refused or full store, so a private
 * window or a locked-down browser gets a working finder that simply forgets.
 */

import { scopedStorageKey } from '/design-system/js/storage-scope.js';
import { defaultLayout, restoreLayout, serializeLayout } from '/design-system/js/table-columns.js';

/** @typedef {import('/design-system/js/table-columns.js').ColumnSpec} ColumnSpec */
/** @typedef {import('/design-system/js/table-columns.js').ColumnLayout} ColumnLayout */

/**
 * The result table's columns, in default order. `key` is also the server's
 * sort key for every data column; `chk` and `act` are chrome, pinned to the
 * edges. System is off by default — it is in every row the server sends, and
 * a facility that splits sections by system wants it one right-click away.
 * @type {ColumnSpec[]}
 */
export const COLUMN_SPECS = [
  { key: 'chk', label: 'Select', width: 30, pin: 'start' },
  { key: 'device', label: 'Device', width: 170, minWidth: 64 },
  { key: 'section', label: 'Sec', width: 72, minWidth: 44 },
  { key: 'address', label: 'Address', width: 220, minWidth: 80 },
  { key: 'dir', label: 'R/W', width: 56, minWidth: 44 },
  { key: 'signal', label: 'Signal', width: 130, minWidth: 56 },
  { key: 'system', label: 'System', width: 80, minWidth: 48, hidden: true },
  { key: 'description', label: 'Description', width: 320, minWidth: 80 },
  { key: 'act', label: 'Copy', width: 36, pin: 'end' },
];

export const COLUMNS_KEY = 'cf-graph-columns';
export const PREFS_KEY = 'cf-graph-prefs';
export const RAIL_KEY = 'cf-graph-rail';
export const INSPECTOR_KEY = 'cf-graph-inspector';

/**
 * @typedef {object} ViewPrefs
 * @property {'comfortable'|'compact'} density
 * @property {string[]} collapsed - facet keys whose bodies are folded
 */

/** @returns {ViewPrefs} */
function defaultPrefs() {
  return { density: 'comfortable', collapsed: [] };
}

/**
 * @param {string} base
 * @returns {string|null}
 */
function read(base) {
  try {
    return window.localStorage.getItem(scopedStorageKey(base));
  } catch {
    return null;
  }
}

/**
 * @param {string} base
 * @param {unknown} value
 * @returns {void}
 */
function write(base, value) {
  try {
    window.localStorage.setItem(scopedStorageKey(base), JSON.stringify(value));
  } catch {
    /* refused or full: the arrangement is simply not remembered */
  }
}

/** @returns {ColumnLayout} the stored layout, or the default one */
export function loadColumns() {
  const raw = read(COLUMNS_KEY);
  return raw === null ? defaultLayout(COLUMN_SPECS) : restoreLayout(COLUMN_SPECS, raw);
}

/** @param {ColumnLayout} layout @returns {void} */
export function saveColumns(layout) {
  write(COLUMNS_KEY, serializeLayout(layout));
}

/** @returns {ViewPrefs} the stored preferences, or the defaults */
export function loadPrefs() {
  const raw = read(PREFS_KEY);
  if (raw === null) return defaultPrefs();
  try {
    const parsed = JSON.parse(raw);
    return {
      density: parsed && parsed.density === 'compact' ? 'compact' : 'comfortable',
      collapsed: Array.isArray(parsed && parsed.collapsed)
        ? parsed.collapsed.filter((/** @type {unknown} */ k) => typeof k === 'string')
        : [],
    };
  } catch {
    return defaultPrefs();
  }
}

/** @param {ViewPrefs} prefs @returns {void} */
export function savePrefs(prefs) {
  write(PREFS_KEY, { density: prefs.density, collapsed: [...prefs.collapsed] });
}
