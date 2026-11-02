// @ts-check
/**
 * ARIEL Search driver
 *
 * Runs one search against /api/search and reports what happened through
 * callbacks; it owns no DOM and no chosen state. The caller (the list pane)
 * passes the query, mode and filters it holds and paints whatever it is told.
 *
 * A reranked search runs in two phases rather than one: reranking costs seconds
 * the operator would otherwise spend looking at a spinner, so the fast ranking
 * is fetched and reported first, then the reranked one replaces it. The second
 * response is a full result set drawn from a larger candidate pool, so it is
 * reported whole — membership, not just order, may differ.
 */

import { searchApi } from './api.js';
import { declaredParamNames, effectiveParam } from './advanced-options.js';

/**
 * @typedef {Object} SearchResults
 * @property {string[]} [search_modes_used]
 * @property {number} [execution_time_ms]
 * @property {number} total_results
 * @property {import('./components.js').Entry[]} [entries]
 * @property {import('./components.js').Diagnostic[]} [diagnostics]
 * @property {import('./components.js').ExpandedTerm[]} [expanded_terms]
 */

/** @typedef {'reranking'|'updated'|'fallback'|null} RerankStatus */

/**
 * @typedef {Object} SearchRequest
 * @property {string} query
 * @property {string|null} mode
 * @property {Object<string, *>} [filters] - Touched Filters & Options values, any mode
 * @property {string|null} [author] - Restricts to one author (the Mine mailbox); wins over a filter
 */

/**
 * @typedef {Object} SearchHooks
 * @property {import('./advanced-options.js').Capabilities|null} capabilities
 * @property {'now'|'when-stable'} [rerank] - When phase 2 runs (default 'now')
 * @property {(results: SearchResults) => void} onResults - Today's /api/search response
 * @property {(status: RerankStatus) => void} onStatus - Called before the onResults it describes
 */

/**
 * How long a `'when-stable'` search waits after phase 1 before it reranks. A
 * newer call inside this window supersedes it, so a tile-bar report that is
 * still being typed never pays for a rerank.
 */
export const RERANK_SETTLE_MS = 1000;

// Monotonic id of the newest search started. Every await in runSearch is
// followed by a check that its call still owns this id, so a newer call or
// cancelSearch() silences everything an older call has left to report.
let searchSeq = 0;

/**
 * The pending when-stable wait, if any. Superseding it releases the waiting
 * call at once instead of leaving it to time out.
 * @type {{timer: ReturnType<typeof setTimeout>, release: () => void}|null}
 */
let settle = null;

/** Release a pending when-stable wait (its call is stale and returns). */
function releaseSettle() {
  if (!settle) return;
  clearTimeout(settle.timer);
  settle.release();
  settle = null;
}

/**
 * Wait the when-stable window.
 * @returns {Promise<void>}
 */
function waitUntilStable() {
  return new Promise((resolve) => {
    const release = () => resolve();
    settle = {
      timer: setTimeout(() => {
        settle = null;
        resolve();
      }, RERANK_SETTLE_MS),
      release,
    };
  });
}

/**
 * Whether a successful response is really the backend's rerank fallback: the
 * hybrid search answered 200 with the fast ranking and a warning diagnostic
 * because the reranker was unavailable. From the outside this is
 * indistinguishable from a real rerank, so the status has to say so.
 * @param {SearchResults} results - A phase-2 response
 * @returns {boolean}
 */
function hasRerankFallbackWarning(results) {
  return (results.diagnostics ?? []).some((d) => d.level === 'warning' && d.category === 'rerank');
}

/**
 * The advanced params a search in `mode` sends: only names the mode or the
 * shared parameters declare, only non-null values, plus `author` when given.
 * @param {import('./advanced-options.js').Capabilities|null} capabilities
 * @param {string|null} mode
 * @param {Object<string, *>} filters
 * @param {string|null|undefined} author
 * @returns {Object<string, *>}
 */
function wireParams(capabilities, mode, filters, author) {
  /** @type {Object<string, *>} */
  const params = {};
  for (const name of declaredParamNames(capabilities, mode)) {
    const value = filters[name];
    if (value !== undefined && value !== null) params[name] = value;
  }
  if (author) params.author = author;
  return params;
}

/**
 * Run one search.
 *
 * Two-phase is decided from the passed mode, filters and capabilities: when
 * the mode's effective `rerank` is true, phase 1 sends `rerank:false` and
 * phase 2 `rerank:true`, each layered on a copy of the sent params. Any other
 * search sends one request with no `rerank` key added. With `rerank:'now'`
 * phase 2 starts as soon as phase 1 is reported; with `'when-stable'` only if
 * no newer call arrived within RERANK_SETTLE_MS.
 *
 * A newer call or cancelSearch() supersedes this one: nothing more is reported
 * and the promise resolves. A phase-2 failure is not a failed search — status
 * `'fallback'` and phase 1's results are reported again. A phase-1 failure of
 * the current search rejects the returned promise, so the caller can show its
 * error state.
 * @param {SearchRequest} request
 * @param {SearchHooks} hooks
 * @returns {Promise<void>}
 */
export async function runSearch(
  { query, mode, filters = {}, author = null },
  { capabilities, rerank = 'now', onResults, onStatus }
) {
  releaseSettle();
  const seq = ++searchSeq;
  const q = (query ?? '').trim();
  if (!q) return;

  const sent = wireParams(capabilities, mode, filters ?? {}, author);
  const maxResults = sent.max_results || 10;
  const request = { query: q, mode: mode ?? undefined, maxResults };
  const twoPhase = effectiveParam(capabilities, mode, filters, 'rerank') === true;

  /** @type {SearchResults} */
  let fastResults;
  try {
    if (!twoPhase) {
      const results = await searchApi.search({ ...request, advancedParams: sent });
      if (seq !== searchSeq) return;
      onStatus(null);
      onResults(results);
      return;
    }
    fastResults = await searchApi.search({
      ...request,
      advancedParams: { ...sent, rerank: false },
    });
  } catch (error) {
    if (seq !== searchSeq) return;
    throw error;
  }
  if (seq !== searchSeq) return;
  onStatus('reranking');
  onResults(fastResults);

  if (rerank === 'when-stable') {
    await waitUntilStable();
    if (seq !== searchSeq) return;
  }

  try {
    const reranked = await searchApi.search({
      ...request,
      advancedParams: { ...sent, rerank: true },
    });
    if (seq !== searchSeq) return;
    onStatus(hasRerankFallbackWarning(reranked) ? 'fallback' : 'updated');
    onResults(reranked);
  } catch (error) {
    if (seq !== searchSeq) return;
    console.error('Rerank phase failed:', error);
    onStatus('fallback');
    onResults(fastResults);
  }
}

/**
 * Abandon any search in flight: nothing it has left reports, and a pending
 * when-stable wait is released.
 */
export function cancelSearch() {
  searchSeq++;
  releaseSettle();
}
