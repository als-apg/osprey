// @ts-check
/**
 * ARIEL Search Module
 *
 * Search functionality and UI management.
 */

import { searchApi } from './api.js';
import {
  renderEntryCard,
  renderEntryCardSimple,
  renderAnswerBox,
  renderDiagnosticsBar,
  renderExpansionBanner,
  renderConfigBanner,
  renderLoading,
  renderEmptyState,
  renderErrorState,
  escapeHtml,
} from './components.js';
import {
  getCurrentMode,
  getAdvancedParams,
  getEffectiveParam,
  closeAdvancedPanel,
} from './advanced-options.js';
import { showEntry } from './entries-detail.js';

/**
 * Read the resolved UI mode (presentation axis, independent of the search
 * mode). mode-boot.js has already stamped data-ui-mode on <html>; anything
 * other than "simple" is treated as "expert".
 * @returns {'expert'|'simple'}
 */
function getUiMode() {
  return document.documentElement.getAttribute('data-ui-mode') === 'simple'
    ? 'simple'
    : 'expert';
}

/**
 * @typedef {Object} SearchResults
 * @property {string} [answer]
 * @property {string[]} [sources]
 * @property {string[]} [search_modes_used]
 * @property {number} [execution_time_ms]
 * @property {number} total_results
 * @property {import('./components.js').Entry[]} [entries]
 * @property {import('./components.js').Diagnostic[]} [diagnostics]
 * @property {import('./components.js').ExpandedTerm[]} [expanded_terms]
 */

// Search state
let currentQuery = '';
let isSearching = false;
/** @type {SearchResults|null} */
let lastResults = null;
// The search mode of the last render, kept so a live Expert<->Simple UI-mode
// flip can re-render the same results without re-running the query.
let lastSearchMode = 'keyword';
// Monotonic id of the newest search the module has started. A reranked search
// paints twice, and the second paint arrives long after the first; every render
// a search performs is guarded on still owning this id, so a late phase-2
// response from a superseded (or cleared) search can never repaint the view.
let searchSeq = 0;
/**
 * Where the current search sits on the two-phase rerank path, or null when the
 * search never had a second phase. Read by both renderers so any repaint — a
 * phase completing, or a live Expert<->Simple flip — shows the same line.
 * @type {'reranking'|'updated'|'fallback'|null}
 */
let rerankStatus = null;

/**
 * The one-line status each phase shows, per UI mode. Expert names the mechanism
 * because Expert users tune it; Simple says what changed in plain words. The
 * fallback wording is deliberately gentle and shared: a reranker that could not
 * load in time is the normal first-query experience after a sidecar restart,
 * not an error the operator has to act on.
 * @type {Record<'expert'|'simple', Record<'reranking'|'updated'|'fallback', string>>}
 */
const RERANK_STATUS_TEXT = {
  expert: {
    reranking: 'Search complete — reranking with LLM…',
    updated: 'Results updated after reranking',
    fallback: 'Could not improve the ranking — showing fast results',
  },
  simple: {
    reranking: 'Showing quick results — improving the order now…',
    updated: 'Order improved — best matches first',
    fallback: 'Could not improve the ranking — showing fast results',
  },
};

/**
 * The rerank status line, or '' when the search has no second phase.
 * @param {'expert'|'simple'} uiMode - Which wording to use
 * @returns {string} HTML string
 */
function renderRerankStatus(uiMode) {
  if (!rerankStatus) return '';
  const message = RERANK_STATUS_TEXT[uiMode][rerankStatus];
  return `
    <div class="results-modes" data-testid="rerank-status" role="status" aria-live="polite">
      ${escapeHtml(message)}
    </div>
  `;
}

/**
 * Whether a successful response is really the backend's rerank fallback: the
 * hybrid search answered 200 with the fast ranking and a warning diagnostic
 * because the reranker was unavailable. The status line has to say so — from
 * the outside this is indistinguishable from a real rerank.
 * @param {SearchResults} results - A phase-2 response
 * @returns {boolean}
 */
function hasRerankFallbackWarning(results) {
  return (results.diagnostics ?? []).some(
    d => d.level === 'warning' && d.category === 'rerank'
  );
}

/**
 * Wire up delegated click handling for entry cards and cited-source links.
 *
 * #search-results' innerHTML is replaced wholesale on every search, so the
 * listener is delegated on the stable results container (attached once, at
 * init) instead of bound to child elements that get discarded on the next
 * render.
 */
export function initSearchResultsDelegation() {
  const resultsContainer = document.getElementById('search-results');
  resultsContainer?.addEventListener('click', (e) => {
    const target = /** @type {HTMLElement} */ (e.target);
    const sourceLink = target.closest('a[data-entry-id]');
    if (sourceLink) {
      e.preventDefault();
      const entryId = /** @type {HTMLElement} */ (sourceLink).dataset.entryId;
      if (entryId) showEntry(entryId);
      return;
    }
    const card = target.closest('[data-entry-id]');
    if (card) {
      const entryId = /** @type {HTMLElement} */ (card).dataset.entryId;
      if (entryId) showEntry(entryId);
    }
  });
}

/**
 * Render the degraded-configuration banner reported by /api/capabilities.
 *
 * `configuration_invalid` means search cannot run at all: the banner
 * (`#config-invalid-banner`) is inserted as the first child of `#search-form`
 * and the search controls are disabled, so the form is present but unusable —
 * the button stays in the DOM rather than being wiped, because that is what
 * the operator (and the browser tests) look at to see search is blocked.
 * `configuration_warning` inserts the same markup as `#config-warning-banner`
 * and leaves the form fully usable. Anything else renders nothing.
 * @param {import('./advanced-options.js').Capabilities|null} [capabilities]
 */
function renderConfigStatus(capabilities) {
  const status = capabilities?.status;
  if (status !== 'configuration_invalid' && status !== 'configuration_warning') return;

  const form = document.getElementById('search-form');
  if (!form) return;

  const blocking = status === 'configuration_invalid';
  form.insertAdjacentHTML(
    'afterbegin',
    renderConfigBanner(capabilities?.config_errors ?? [], capabilities?.remedy ?? null, blocking)
  );

  if (!blocking) return;

  const searchBtn = /** @type {HTMLButtonElement|null} */ (document.getElementById('search-btn'));
  if (searchBtn) searchBtn.disabled = true;
  const searchInput = /** @type {HTMLInputElement|null} */ (document.getElementById('search-input'));
  if (searchInput) searchInput.disabled = true;
}

/**
 * Initialize search module.
 * @param {import('./advanced-options.js').Capabilities|null} [capabilities] -
 *   The /api/capabilities payload (null when the fetch failed), used only to
 *   surface a degraded configuration.
 */
export function initSearch(capabilities = null) {
  renderConfigStatus(capabilities);

  const searchInput = /** @type {HTMLInputElement|null} */ (document.getElementById('search-input'));
  const searchBtn = document.getElementById('search-btn');

  // Search input enter key
  searchInput?.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      performSearch();
    }
  });

  // Search button click
  searchBtn?.addEventListener('click', () => {
    performSearch();
  });

  // Focus search on page load
  searchInput?.focus();

  // Keyboard shortcut: / to focus search
  document.addEventListener('keydown', (e) => {
    if (e.key === '/' && document.activeElement?.tagName !== 'INPUT') {
      e.preventDefault();
      searchInput?.focus();
    }
    // Escape to clear search
    if (e.key === 'Escape' && document.activeElement === searchInput && searchInput) {
      searchInput.value = '';
      searchInput.blur();
    }
  });

  initSearchResultsDelegation();
}

/**
 * Perform a search.
 *
 * A reranked search runs in two phases rather than one: reranking costs seconds
 * the operator would otherwise spend looking at a spinner, so the fast ranking
 * is fetched and painted first, then the reranked one replaces it. The second
 * response is a full result set drawn from a larger candidate pool, so the view
 * is re-rendered wholesale — membership, not just order, may differ.
 *
 * Everything else — any mode that does not declare `rerank`, or declares it and
 * has it off — takes the single-request path unchanged, with no `rerank` key
 * added to the wire params.
 * @param {string|null} [query] - Optional query override
 */
export async function performSearch(query = null) {
  const searchInput = /** @type {HTMLInputElement|null} */ (document.getElementById('search-input'));
  const resultsContainer = document.getElementById('search-results');

  query = query || searchInput?.value?.trim();
  if (!query || isSearching) return;

  currentQuery = query;
  isSearching = true;
  const seq = ++searchSeq;
  rerankStatus = null;

  // The button stays disabled across BOTH phases: the search is still running
  // while phase-1 results are on screen, and a second search cannot start.
  const searchBtn = /** @type {HTMLButtonElement|null} */ (document.getElementById('search-btn'));
  if (searchBtn) searchBtn.disabled = true;

  // Collapse the filters & options panel
  closeAdvancedPanel();

  // Show loading state
  if (resultsContainer) {
    resultsContainer.innerHTML = renderLoading('Searching...');
  }

  // Get mode and advanced params from the unified capabilities-driven UI
  const mode = getCurrentMode();
  /** @type {Object<string, *>} */
  const advancedParams = getAdvancedParams();
  const maxResults = advancedParams.max_results || 10;
  // getEffectiveParam answers for the knob even when the operator never touched
  // it (getAdvancedParams deliberately omits untouched knobs), and returns
  // undefined for a mode that has no reranker at all.
  const twoPhase = getEffectiveParam('rerank') === true;

  try {
    if (!twoPhase) {
      const results = await searchApi.search({ query, mode, maxResults, advancedParams });
      if (seq !== searchSeq) return;
      lastResults = results;
      renderSearchResults(results, mode);
      return;
    }

    // Phase 1 — the fast ranking. Only the per-phase `rerank` key is layered on
    // top of the touched params, on a copy: the operator's panel is untouched.
    const fastResults = await searchApi.search({
      query,
      mode,
      maxResults,
      advancedParams: { ...advancedParams, rerank: false },
    });
    if (seq !== searchSeq) return;
    lastResults = fastResults;
    rerankStatus = 'reranking';
    renderSearchResults(fastResults, mode);

    // Phase 2 — the reranked ranking. Its own catch, because a failure here is
    // not a failed search: phase 1's results stay on screen behind a warning.
    try {
      const rerankedResults = await searchApi.search({
        query,
        mode,
        maxResults,
        advancedParams: { ...advancedParams, rerank: true },
      });
      if (seq !== searchSeq) return;
      lastResults = rerankedResults;
      rerankStatus = hasRerankFallbackWarning(rerankedResults) ? 'fallback' : 'updated';
      renderSearchResults(rerankedResults, mode);
    } catch (error) {
      console.error('Rerank phase failed:', error);
      if (seq !== searchSeq) return;
      rerankStatus = 'fallback';
      renderSearchResults(fastResults, mode);
    }
  } catch (error) {
    console.error('Search failed:', error);
    if (seq !== searchSeq) return;
    if (resultsContainer) {
      resultsContainer.innerHTML = renderErrorState('Search Failed', error);
    }
  } finally {
    // A superseded search must not hand the controls back: the search that took
    // its place still owns them.
    if (seq === searchSeq) {
      isSearching = false;
      if (searchBtn) searchBtn.disabled = false;
    }
  }
}

/**
 * Render search results.
 * @param {SearchResults} results - Search results from API
 * @param {string} mode - The search mode selected by the user (e.g. 'keyword', 'semantic')
 */
function renderSearchResults(results, mode = 'keyword') {
  const resultsContainer = document.getElementById('search-results');
  if (!resultsContainer) return;

  lastSearchMode = mode;

  if (getUiMode() === 'simple') {
    renderSearchResultsSimple(resultsContainer, results);
    return;
  }

  // Build results header
  const modesUsed = results.search_modes_used?.join(', ') || 'none';
  const execTime = results.execution_time_ms || 0;

  let html = '';

  // Answer box with mode label and tools used
  if (results.answer) {
    const toolsUsed = results.search_modes_used || [];
    html += renderAnswerBox(results.answer, results.sources, mode, toolsUsed);
  }

  // Diagnostics bar if issues detected
  if ((results.diagnostics?.length ?? 0) > 0) {
    html += renderDiagnosticsBar(results.diagnostics ?? []);
  }

  // Two-phase rerank status, immediately above the results header it describes
  html += renderRerankStatus('expert');

  // Results header
  html += `
    <div class="results-header">
      <span class="results-count">
        <strong>${results.total_results}</strong> results
        <span class="text-muted">(${execTime}ms)</span>
      </span>
      <span class="results-modes">
        Modes: ${escapeHtml(modesUsed)}
      </span>
    </div>
  `;

  // What the vocabulary expanded, Expert mode only (Simple deliberately omits
  // the search-mechanics chrome). Empty/absent expansions render nothing.
  html += renderExpansionBanner(results.expanded_terms);

  // Results list
  if ((results.entries?.length ?? 0) > 0) {
    const sourcesSet = results.sources?.length ? new Set(results.sources) : null;
    html += '<div class="results-list">';
    (results.entries ?? []).forEach(entry => {
      const isCited = sourcesSet ? sourcesSet.has(entry.entry_id) : false;
      html += renderEntryCard(entry, isCited);
    });
    html += '</div>';
  } else {
    html += renderEmptyState(
      'No Results Found',
      'Try adjusting your search terms or filters.'
    );
  }

  resultsContainer.innerHTML = html;
}

/**
 * Render search results for Simple mode (frame 1b): the plain-language answer
 * (when present), a friendly "N entries found — newest first" header, and
 * plain result cards. Deliberately omits the score/mode diagnostics chrome
 * that the Expert render carries.
 * @param {HTMLElement} container - The #search-results container
 * @param {SearchResults} results - Search results from API
 */
function renderSearchResultsSimple(container, results) {
  let html = '';

  if (results.answer) {
    html += renderAnswerBox(results.answer, results.sources);
  }

  // The one plain-language status line Simple mode gets. It is emitted before
  // the empty-results branch too: "still improving the order" is exactly as
  // true of an empty fast ranking as of a full one.
  html += renderRerankStatus('simple');

  const entries = results.entries ?? [];
  if (entries.length === 0) {
    html += renderEmptyState(
      'No entries found',
      'Try different words, or browse all entries.'
    );
    container.innerHTML = html;
    return;
  }

  const count = results.total_results;
  const noun = count === 1 ? 'entry' : 'entries';
  html += `
    <div class="results-header results-header-simple">
      <span class="results-count"><strong>${count}</strong> ${noun} found &mdash; newest first</span>
    </div>
  `;

  html += '<div class="results-list">';
  entries.forEach(entry => {
    html += renderEntryCardSimple(entry);
  });
  html += '</div>';

  container.innerHTML = html;
}

/**
 * Re-render the current results after a live Expert<->Simple UI-mode flip.
 * mode-boot.js / the app.js message listener has already stamped the new
 * data-ui-mode on <html>; this just repaints the existing results (if any)
 * into the layout that mode calls for. No-op when nothing has been searched.
 */
export function onUiModeChange() {
  if (lastResults) {
    renderSearchResults(lastResults, lastSearchMode);
  }
}

/**
 * Clear search results.
 *
 * Clearing also abandons any search still in flight — bumping the sequence makes
 * every render it has left stale, so a reranked search whose second phase lands
 * after the operator emptied the box cannot repaint the cleared view.
 */
export function clearSearch() {
  const searchInput = /** @type {HTMLInputElement|null} */ (document.getElementById('search-input'));
  const resultsContainer = document.getElementById('search-results');

  if (searchInput) searchInput.value = '';
  if (resultsContainer) resultsContainer.innerHTML = '';

  searchSeq++;
  isSearching = false;
  rerankStatus = null;
  currentQuery = '';
  lastResults = null;

  // Hand back the button the cancelled search was holding — but never the one
  // the configuration_invalid banner disabled, which owns it permanently.
  const searchBtn = /** @type {HTMLButtonElement|null} */ (document.getElementById('search-btn'));
  if (searchBtn?.disabled && !document.getElementById('config-invalid-banner')) {
    searchBtn.disabled = false;
  }
}

/**
 * Get current search state.
 * @returns {Object} Current state
 */
export function getSearchState() {
  return {
    query: currentQuery,
    mode: getCurrentMode(),
    isSearching,
    results: lastResults,
  };
}

export default {
  initSearch,
  performSearch,
  clearSearch,
  getSearchState,
};
