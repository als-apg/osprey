// @ts-check
/* OSPREY Web Terminal — Fuzzy Match Scorer
 *
 * Pure, DOM-free scorer for the command palette. The query is tokenized on
 * whitespace and EVERY token must independently subsequence-match the candidate
 * (case-insensitive, in-order, gaps allowed) — an all-tokens-AND contract, so
 * "wrt vrf" matches "control_system.write_verification" via two separate tokens.
 * Each token scores boundary hits (start / after . _ - / camelCase) and
 * consecutive runs; all per-token scores sum. Matched character ranges from
 * every token are merged into sorted, non-overlapping [start, end) spans for
 * the UI to highlight. Empty query matches everything (score 0, no spans).
 */

/**
 * @typedef {{ score: number, spans: Array<[number, number]> }} FuzzyResult
 */

const BASE = 1; // every matched character is worth at least this
const BOUNDARY_BONUS = 10; // match at index 0 or right after a word boundary
const CONSECUTIVE_BONUS = 5; // match immediately following the previous match

/**
 * Is `candidate[idx]` a word-boundary start? True at index 0, right after a
 * `.`/`_`/`-` separator, or on a lowercase→uppercase (camelCase) transition.
 *
 * @param {string} candidate
 * @param {number} idx
 * @returns {boolean}
 */
function isBoundary(candidate, idx) {
  if (idx === 0) {
    return true;
  }
  const prev = candidate.charAt(idx - 1);
  if (prev === '.' || prev === '_' || prev === '-') {
    return true;
  }
  const cur = candidate.charAt(idx);
  return /[a-z]/.test(prev) && /[A-Z]/.test(cur);
}

/**
 * Greedily subsequence-match one lowercased token against a lowercased
 * candidate, returning the matched indices in order, or null if any token
 * character cannot be placed.
 *
 * @param {string} token - already lowercased, non-empty
 * @param {string} lower - already lowercased candidate
 * @returns {number[]|null}
 */
function matchToken(token, lower) {
  /** @type {number[]} */
  const indices = [];
  let from = 0;
  for (const ch of token) {
    const found = lower.indexOf(ch, from);
    if (found === -1) {
      return null;
    }
    indices.push(found);
    from = found + 1;
  }
  return indices;
}

/**
 * Score a token's matched indices: base per character, plus a boundary bonus
 * and a consecutive-run bonus. Uses the original (mixed-case) candidate so the
 * camelCase boundary rule can see the casing.
 *
 * @param {number[]} indices
 * @param {string} candidate
 * @returns {number}
 */
function scoreIndices(indices, candidate) {
  let score = 0;
  for (let p = 0; p < indices.length; p++) {
    const idx = indices[p];
    score += BASE;
    if (isBoundary(candidate, idx)) {
      score += BOUNDARY_BONUS;
    }
    if (p > 0 && idx === indices[p - 1] + 1) {
      score += CONSECUTIVE_BONUS;
    }
  }
  return score;
}

/**
 * Merge single-character ranges into sorted, non-overlapping half-open
 * [start, end) spans. Adjacent ranges (end === next start) merge.
 *
 * @param {Array<[number, number]>} ranges
 * @returns {Array<[number, number]>}
 */
function mergeRanges(ranges) {
  const sorted = ranges.slice().sort((a, b) => a[0] - b[0]);
  /** @type {Array<[number, number]>} */
  const merged = [];
  for (const [start, end] of sorted) {
    const last = merged[merged.length - 1];
    if (last && start <= last[1]) {
      last[1] = Math.max(last[1], end);
    } else {
      merged.push([start, end]);
    }
  }
  return merged;
}

/**
 * Fuzzy-match a whitespace-tokenized query against a candidate string.
 * Returns a score and highlight spans on match, or null if any token fails to
 * subsequence-match. An empty/whitespace-only query matches with score 0.
 *
 * @param {string} query
 * @param {string} candidate
 * @returns {FuzzyResult|null}
 */
export function fuzzyMatch(query, candidate) {
  const tokens = query.split(/\s+/).filter((t) => t.length > 0);
  if (tokens.length === 0) {
    return { score: 0, spans: [] };
  }

  const lower = candidate.toLowerCase();
  /** @type {Array<[number, number]>} */
  const ranges = [];
  let score = 0;

  for (const token of tokens) {
    const indices = matchToken(token.toLowerCase(), lower);
    if (indices === null) {
      return null;
    }
    score += scoreIndices(indices, candidate);
    for (const idx of indices) {
      ranges.push([idx, idx + 1]);
    }
  }

  return { score, spans: mergeRanges(ranges) };
}
