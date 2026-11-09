// @ts-check
/**
 * ARIEL panel filter state: the C-FILTER object the store keeps under `filter`,
 * and the helpers every writer of it goes through. No DOM, no imports.
 *
 * A filter is `{"<dim>": ["v", ...], ..., "time": {start, end}}`: OR within a
 * dimension, AND across dimensions. `time` is a window, not a facet, so it is
 * never a dimension name and every helper here keeps it. The store's `filters`
 * key holds only the per-mode tuning knobs; `splitFilters` / `joinFilters`
 * translate between the flat Filters & Options values and the two keys.
 */

/** @typedef {{start: string|null, end: string|null}} TimeWindow */

/**
 * @typedef {Object<string, any>} Filter
 * A C-FILTER object: dimension names map to value lists, `time` to a window.
 */

/** The reserved window key. */
export const TIME = 'time';

/** Flat Filters & Options names that are dimensions, with their filter key. */
const FLAT_DIMENSIONS = /** @type {const} */ (['author', 'source_system']);

/** Flat Filters & Options names that are the time window's bounds. */
const FLAT_TIME = /** @type {const} */ ({ start_date: 'start', end_date: 'end' });

/**
 * A bound as the wire carries it: a non-blank string, else null.
 * @param {unknown} value
 * @returns {string|null}
 */
function bound(value) {
  if (typeof value !== 'string') return null;
  const s = value.trim();
  return s ? s : null;
}

/**
 * The time window of `raw`, or null when it sets neither bound.
 * @param {unknown} raw
 * @returns {TimeWindow|null}
 */
function normalizeTime(raw) {
  if (!raw || typeof raw !== 'object') return null;
  const t = /** @type {Record<string, unknown>} */ (raw);
  const start = bound(t.start);
  const end = bound(t.end);
  return start || end ? { start, end } : null;
}

/**
 * A dimension's values: non-blank strings, deduplicated ignoring case (the
 * first spelling wins), sorted ignoring case. A lone string counts as a list
 * of one.
 * @param {unknown} raw
 * @returns {string[]}
 */
function normalizeValues(raw) {
  const list = Array.isArray(raw) ? raw : typeof raw === 'string' ? [raw] : [];
  /** @type {Map<string, string>} */
  const seen = new Map();
  for (const v of list) {
    if (typeof v !== 'string') continue;
    const s = v.trim();
    if (s && !seen.has(s.toLowerCase())) seen.set(s.toLowerCase(), s);
  }
  return [...seen.entries()].sort(([a], [b]) => (a < b ? -1 : a > b ? 1 : 0)).map(([, s]) => s);
}

/**
 * The canonical form of a filter: sorted keys; sorted, deduplicated values;
 * empty dimensions and an unset time window dropped.
 * @param {unknown} f
 * @returns {Filter}
 */
export function normalizeFilter(f) {
  /** @type {Filter} */
  const out = {};
  if (!f || typeof f !== 'object' || Array.isArray(f)) return out;
  const raw = /** @type {Record<string, unknown>} */ (f);
  for (const key of Object.keys(raw).sort()) {
    if (key === TIME) {
      const time = normalizeTime(raw[key]);
      if (time) out[TIME] = time;
      continue;
    }
    const values = normalizeValues(raw[key]);
    if (values.length) out[key] = values;
  }
  return out;
}

/**
 * The filter a request in `state` sends: the store's `filter` plus the mailbox
 * scope. Under Mine the author dimension is replaced by the Mine name; Drafts
 * makes no request and returns null.
 * @param {{filter?: unknown, mailbox?: unknown, mineAuthor?: string|null}} state
 * @returns {Filter|null}
 */
export function requestFilter(state) {
  if (state.mailbox === 'drafts') return null;
  const filter = normalizeFilter(state.filter);
  if (state.mailbox === 'mine' && state.mineAuthor) {
    return normalizeFilter({ ...filter, author: [state.mineAuthor] });
  }
  return filter;
}

/**
 * Drop the dimensions the deployment does not declare. `time` is always kept.
 * With no capabilities (null dimensions) the filter is returned unchanged and
 * the API's 422 reports any problem.
 * @param {unknown} f
 * @param {Array<{name: string}>|null|undefined} dimensions - capabilities.filter_dimensions
 * @returns {{filter: Filter, dropped: string[]}}
 */
export function sanitizeFilter(f, dimensions) {
  if (!dimensions) return { filter: /** @type {Filter} */ (f ?? {}), dropped: [] };
  const known = new Set(dimensions.map((d) => d.name));
  const filter = normalizeFilter(f);
  /** @type {string[]} */
  const dropped = [];
  for (const key of Object.keys(filter)) {
    if (key !== TIME && !known.has(key)) {
      dropped.push(key);
      delete filter[key];
    }
  }
  return { filter, dropped };
}

/**
 * Select `value` in `dim` when it is not selected (ignoring case), else
 * deselect it.
 * @param {unknown} filter
 * @param {string} dim
 * @param {string} value
 * @returns {Filter}
 */
export function toggleValue(filter, dim, value) {
  const f = normalizeFilter(filter);
  const current = /** @type {string[]} */ (f[dim] ?? []);
  const lower = value.trim().toLowerCase();
  const next = current.some((v) => v.toLowerCase() === lower)
    ? current.filter((v) => v.toLowerCase() !== lower)
    : [...current, value];
  return normalizeFilter({ ...f, [dim]: next });
}

/**
 * Replace each key of `partial` in `filter`. A key whose value is null, an
 * empty list or an unset window clears that key.
 * @param {unknown} filter
 * @param {Record<string, unknown>} partial
 * @returns {Filter}
 */
export function mergeFilter(filter, partial) {
  return normalizeFilter({ ...normalizeFilter(filter), ...(partial ?? {}) });
}

/**
 * Split flat Filters & Options values into the filter and the knobs:
 * `author` and `source_system` become dimensions, `start_date`/`end_date` the
 * time window, and everything else stays a knob.
 * @param {Record<string, unknown>|null|undefined} values
 * @returns {{filter: Filter, knobs: Record<string, any>}}
 */
export function splitFilters(values) {
  /** @type {Record<string, unknown>} */
  const raw = {};
  /** @type {Record<string, any>} */
  const knobs = {};
  /** @type {Record<string, unknown>} */
  const time = {};
  for (const [key, value] of Object.entries(values ?? {})) {
    if (/** @type {readonly string[]} */ (FLAT_DIMENSIONS).includes(key)) raw[key] = value;
    else if (key in FLAT_TIME) time[FLAT_TIME[/** @type {keyof typeof FLAT_TIME} */ (key)]] = value;
    else knobs[key] = value;
  }
  return { filter: normalizeFilter({ ...raw, [TIME]: time }), knobs };
}

/**
 * Join a filter and the knobs back into flat Filters & Options values, the
 * inverse of `splitFilters` for the names it maps. A one-value dimension joins
 * as a string, several values as a list; other dimensions are left out.
 * @param {unknown} filter
 * @param {Record<string, any>|null|undefined} knobs
 * @returns {Record<string, any>}
 */
export function joinFilters(filter, knobs) {
  const f = normalizeFilter(filter);
  /** @type {Record<string, any>} */
  const values = { ...(knobs ?? {}) };
  for (const dim of FLAT_DIMENSIONS) {
    const list = f[dim];
    if (list) values[dim] = list.length === 1 ? list[0] : list;
  }
  const time = /** @type {TimeWindow|undefined} */ (f[TIME]);
  if (time?.start) values.start_date = time.start;
  if (time?.end) values.end_date = time.end;
  return values;
}

/**
 * Bring a saved-search record to the current shape `{name, query, mode,
 * filter, filters}`. A P1 record kept `author`/`source_system` strings and
 * `start_date`/`end_date` among its `filters`; those move into `filter`. A
 * record's own `filter` wins over what is moved.
 * @param {Record<string, any>} record
 * @returns {{name: string, query: string, mode: string|null, filter: Filter, filters: Record<string, any>}}
 */
export function migrateSaved(record) {
  const { filter: moved, knobs } = splitFilters(record?.filters);
  const own = normalizeFilter(record?.filter);
  return {
    ...record,
    name: record?.name,
    query: record?.query ?? '',
    mode: record?.mode ?? null,
    filter: normalizeFilter({ ...moved, ...own }),
    filters: knobs,
  };
}
