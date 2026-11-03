// @ts-check
/**
 * ARIEL query operators: lift `dim:value` tokens out of search text into the
 * C-FILTER object. No DOM, no store.
 *
 * `author:bob` and `author:"bob beam"` work for every dimension name the
 * deployment declares (`capabilities.filter_dimensions`, matched ignoring
 * case); `date:YYYY-MM-DD` and `date:YYYY-MM` set the `time` window (one day,
 * or the first to the last day of the month). A token whose prefix is not a
 * dimension stays in the query and is reported in `unknown`.
 *
 * A token is **complete** when whitespace follows it or it is a closed quote.
 * Without `final`, a trailing incomplete token whose prefix is a dimension is
 * stripped from the query but not applied (the tile bar reports partial text);
 * with `final` (Enter, or the tile bar's settle point) the end of the text
 * completes it.
 */

import { normalizeFilter, TIME } from './filter-state.js';

/** @typedef {import('./filter-state.js').Filter} Filter */

/** The time-window operator. */
export const DATE_OPERATOR = 'date';

const PREFIX = /^([A-Za-z_][\w-]*):/;
const DAY = /^(\d{4})-(\d{2})-(\d{2})$/;
const MONTH = /^(\d{4})-(\d{2})$/;

/**
 * @typedef {Object} Token
 * @property {string} raw - The token's text as typed
 * @property {string|null} prefix - The `name` of `name:value`, else null
 * @property {string} value - The value, without quotes
 * @property {boolean} complete - Followed by whitespace, or a closed quote
 */

/**
 * Split `text` into whitespace-separated tokens; a `name:"..."` value may hold
 * whitespace.
 * @param {string} text
 * @returns {Token[]}
 */
function tokenize(text) {
  /** @type {Token[]} */
  const tokens = [];
  let i = 0;
  while (i < text.length) {
    if (/\s/.test(text[i])) {
      i += 1;
      continue;
    }
    const start = i;
    const m = PREFIX.exec(text.slice(i));
    if (m && text[i + m[0].length] === '"') {
      const open = i + m[0].length;
      const close = text.indexOf('"', open + 1);
      const end = close === -1 ? text.length : close + 1;
      tokens.push({
        raw: text.slice(start, end),
        prefix: m[1],
        value: text.slice(open + 1, close === -1 ? text.length : close),
        complete: close !== -1,
      });
      i = end;
      continue;
    }
    while (i < text.length && !/\s/.test(text[i])) i += 1;
    tokens.push({
      raw: text.slice(start, i),
      prefix: m ? m[1] : null,
      value: m ? text.slice(start + m[0].length, i) : '',
      complete: i < text.length,
    });
  }
  return tokens;
}

/**
 * The time window a `date:` value names, or null when it names none.
 * @param {string} value
 * @returns {{start: string, end: string}|null}
 */
export function dateWindow(value) {
  const day = DAY.exec(value);
  if (day) {
    const [y, mo, d] = [Number(day[1]), Number(day[2]), Number(day[3])];
    const probe = new Date(Date.UTC(y, mo - 1, d));
    if (probe.getUTCMonth() !== mo - 1 || probe.getUTCDate() !== d) return null;
    return { start: value, end: value };
  }
  const month = MONTH.exec(value);
  if (month) {
    const mo = Number(month[2]);
    if (mo < 1 || mo > 12) return null;
    const last = new Date(Date.UTC(Number(month[1]), mo, 0)).getUTCDate();
    return { start: `${value}-01`, end: `${value}-${String(last).padStart(2, '0')}` };
  }
  return null;
}

/**
 * Lift the operator tokens of `text`.
 * @param {string} text
 * @param {Array<{name: string}>|null|undefined} dimensions - capabilities.filter_dimensions;
 *   null knows no dimension, so only `date:` lifts
 * @param {{final?: boolean}} [options] - `final`: the end of the text completes a token
 * @returns {{query: string, filters: Filter, unknown: string[]}}
 */
export function liftOperators(text, dimensions, { final = false } = {}) {
  /** @type {Map<string, string>} */
  const names = new Map((dimensions ?? []).map((d) => [d.name.toLowerCase(), d.name]));
  /** @type {string[]} */
  const kept = [];
  /** @type {Set<string>} */
  const unknown = new Set();
  /** @type {Record<string, any>} */
  const filters = {};
  for (const token of tokenize(text ?? '')) {
    const prefix = token.prefix?.toLowerCase() ?? null;
    const isDate = prefix === DATE_OPERATOR;
    const dim = prefix === null || isDate ? null : (names.get(prefix) ?? null);
    if (!isDate && !dim) {
      if (token.prefix) unknown.add(token.prefix);
      kept.push(token.raw);
      continue;
    }
    // A dimension token: stripped whether or not it applies.
    const value = token.value.trim();
    if (!(token.complete || final) || !value) continue;
    if (dim) {
      filters[dim] = [...(filters[dim] ?? []), value];
      continue;
    }
    const window = dateWindow(value);
    if (window) filters[TIME] = window;
    else kept.push(token.raw);
  }
  return { query: kept.join(' '), filters: normalizeFilter(filters), unknown: [...unknown] };
}

/**
 * `filter` with every lifted value added: dimension values are unioned, a
 * lifted window replaces the window. Idempotent.
 * @param {unknown} filter
 * @param {Filter} lifted
 * @returns {Filter}
 */
export function addLifted(filter, lifted) {
  const out = normalizeFilter(filter);
  for (const [key, value] of Object.entries(normalizeFilter(lifted))) {
    out[key] = key === TIME ? value : [...(out[key] ?? []), ...value];
  }
  return normalizeFilter(out);
}

/**
 * `filter` without the values `lifted` contributed: its dimension values
 * (ignoring case) and its window when the window is still the lifted one.
 * @param {unknown} filter
 * @param {Filter} lifted
 * @returns {Filter}
 */
export function removeLifted(filter, lifted) {
  const out = normalizeFilter(filter);
  for (const [key, value] of Object.entries(normalizeFilter(lifted))) {
    if (key === TIME) {
      if (sameWindow(out[TIME], value)) delete out[TIME];
      continue;
    }
    const drop = new Set(value.map((/** @type {string} */ v) => v.toLowerCase()));
    out[key] = (out[key] ?? []).filter((/** @type {string} */ v) => !drop.has(v.toLowerCase()));
  }
  return normalizeFilter(out);
}

/**
 * Whether every value `lifted` contributed is still in `filter`.
 * @param {unknown} filter
 * @param {Filter} lifted
 * @returns {boolean}
 */
export function holdsLifted(filter, lifted) {
  const f = normalizeFilter(filter);
  return Object.entries(normalizeFilter(lifted)).every(([key, value]) => {
    if (key === TIME) return sameWindow(f[TIME], value);
    const have = new Set((f[key] ?? []).map((/** @type {string} */ v) => v.toLowerCase()));
    return value.every((/** @type {string} */ v) => have.has(v.toLowerCase()));
  });
}

/**
 * @param {any} a
 * @param {any} b
 */
function sameWindow(a, b) {
  return !!a && !!b && a.start === b.start && a.end === b.end;
}

/**
 * The state patch that applies `lifted` on top of `state` after taking back
 * `previous` (an earlier contribution of the same writer): `filter` when it
 * changes, and `mailbox:'all'` when an author is lifted under Mine (FR10).
 * @param {{filter?: unknown, mailbox?: unknown}} state
 * @param {Filter} lifted
 * @param {Filter} [previous]
 * @returns {{filter?: Filter, mailbox?: 'all'}}
 */
export function liftedPatch(state, lifted, previous = {}) {
  /** @type {{filter?: Filter, mailbox?: 'all'}} */
  const patch = {};
  const current = normalizeFilter(state.filter);
  const filter = addLifted(removeLifted(current, previous), lifted);
  if (JSON.stringify(filter) !== JSON.stringify(current)) patch.filter = filter;
  if (lifted.author?.length && state.mailbox === 'mine') patch.mailbox = 'all';
  return patch;
}

/**
 * The standalone search box's Enter write: the stripped query, the lifted
 * values merged into the filter, and `rerank`.
 * @param {string|null|undefined} text
 * @param {{filter?: unknown, mailbox?: unknown}} state
 * @param {{filter_dimensions?: Array<{name: string}>|null}|null|undefined} capabilities
 * @param {'now'|'when-stable'} rerank
 * @returns {{query: string, rerank: 'now'|'when-stable', filter?: Filter, mailbox?: 'all'}}
 */
export function liftSearch(text, state, capabilities, rerank) {
  const { query, filters } = liftOperators(text ?? '', capabilities?.filter_dimensions, {
    final: true,
  });
  return { query, rerank, ...liftedPatch(state, filters) };
}
