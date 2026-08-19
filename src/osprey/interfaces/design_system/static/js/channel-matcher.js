// @ts-check
/**
 * Channel-catalog matcher for the plan-form channel typeahead.
 *
 * Pure, DOM-free ranking core (pure modules belong in the design system
 * because they carry no interface coupling): given a query and a catalog of
 * channel address strings it returns the ranked suggestion pool. Hand-written
 * ES module (mirrors theme-manager.js) with JSDoc types so it type-checks
 * under `tsc --noEmit --strict`.
 *
 * Pool rule: case-insensitive substring hits form the candidate pool when
 * there are at least MIN_POOL of them; below that threshold the pool widens
 * to every catalog entry that fuzzyMatch accepts (subsequence matching, so
 * "SRHCM" still finds "SR:HCM:..." with zero substring hits). Both modes
 * rank by fuzzyMatch score — a substring is a subsequence, so every
 * substring hit also fuzzy-matches — and `total` reports the honest pool
 * size while `results` is capped at RENDER_CAP entries.
 *
 * @module channel-matcher
 */

import { fuzzyMatch } from '/design-system/js/fuzzy.js';

/**
 * Minimum number of case-insensitive substring hits required before the
 * substring set alone is trusted as the candidate pool. Below this the
 * matcher falls through to subsequence (fuzzy) matching over the whole
 * catalog so terse queries like "SRHCM" still surface "SR:HCM:..." names.
 *
 * @type {number}
 */
export const MIN_POOL = 8;

/**
 * Maximum number of entries returned in `results`. The pool itself is not
 * truncated — `total` always reports the full pool size — but rendering
 * thousands of dropdown rows helps nobody, so the visible slice stops here.
 *
 * @type {number}
 */
export const RENDER_CAP = 50;

/**
 * @typedef {{ value: string, spans: Array<[number, number]> }} ChannelMatch
 */

/**
 * @typedef {{ results: ChannelMatch[], total: number }} ChannelMatchResult
 */

/**
 * Internal scored entry carried between pooling and ranking.
 *
 * @typedef {{ value: string, score: number, spans: Array<[number, number]> }} ScoredEntry
 */

/**
 * Rank scored entries by fuzzy score descending, breaking ties by value
 * ascending. The explicit tie-break exists because catalogs are full of
 * same-shaped names that score identically; without it the order would
 * depend on engine sort stability and tests could not pin the contract.
 *
 * @param {ScoredEntry[]} entries
 * @returns {ScoredEntry[]} the same array, sorted in place
 */
function rank(entries) {
  return entries.sort((a, b) => {
    if (b.score !== a.score) {
      return b.score - a.score;
    }
    return a.value < b.value ? -1 : a.value > b.value ? 1 : 0;
  });
}

/**
 * Match a query against a channel catalog and return the ranked suggestion
 * pool. See the module header for the pool rule. An empty or whitespace-only
 * query matches everything: the pool is the whole catalog (honest `total`)
 * and `results` is the first RENDER_CAP entries in catalog order, because
 * fuzzyMatch scores every candidate 0 there and catalog order is the only
 * meaningful ranking.
 *
 * @param {string} query - raw user input; not trimmed by the caller
 * @param {string[]} catalog - channel address strings (sorted upstream)
 * @returns {ChannelMatchResult}
 */
export function matchChannels(query, catalog) {
  const trimmed = query.trim();
  if (trimmed === '') {
    return {
      results: catalog.slice(0, RENDER_CAP).map((value) => ({ value, spans: [] })),
      total: catalog.length,
    };
  }

  // Trim for the substring test so stray surrounding whitespace does not
  // silently empty the substring set; fuzzyMatch tokenizes on whitespace and
  // is indifferent to it.
  const lowerQuery = trimmed.toLowerCase();
  /** @type {string[]} */
  const substringHits = [];
  for (const value of catalog) {
    if (value.toLowerCase().includes(lowerQuery)) {
      substringHits.push(value);
    }
  }

  const candidates = substringHits.length >= MIN_POOL ? substringHits : catalog;
  /** @type {ScoredEntry[]} */
  const pool = [];
  for (const value of candidates) {
    const hit = fuzzyMatch(query, value);
    if (hit !== null) {
      pool.push({ value, score: hit.score, spans: hit.spans });
    }
  }

  rank(pool);
  return {
    results: pool.slice(0, RENDER_CAP).map(({ value, spans }) => ({ value, spans })),
    total: pool.length,
  };
}
