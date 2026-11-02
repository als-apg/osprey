// @ts-check
/**
 * ARIEL List Pane
 *
 * One list for browsing and searching, rendered into the container it is
 * given (`#ariel-list`). The pane head holds the standalone search box, the
 * mode chip with its Filters & Options popover, and the standalone compose
 * button; embedded, the search box and compose button belong to the host's
 * tile bar, so only the mode chip stays.
 *
 * What the list shows follows the store: it subscribes to `query`, `mode`,
 * `filters`, `mailbox` (and `mineAuthor` under Mine) and either pages through
 * `/api/entries` newest first with day separators (empty query) or hands the
 * query to the search driver with the state's `rerank` (any other query). The
 * mailbox is a scope: Mine passes its author to both; Drafts has no listing
 * this phase and shows its empty state whatever the query.
 *
 * The list writes only what it chooses: `query` (search-box Enter, with
 * `force`), `mode` and `filters` (the popover), each with `rerank:'now'`;
 * `sel` with `narrowView:'reader'` (row click); and `compose` (compose button).
 */

import { entriesApi } from './api.js';
import {
  renderListRow,
  renderDayGroupedRows,
  renderRerankStatus,
  renderDiagnosticsBar,
  renderExpansionBanner,
  renderConfigBanner,
  renderLoading,
  renderEmptyState,
  renderErrorState,
  escapeHtml,
} from './components.js';
import { runSearch, cancelSearch } from './search.js';
import {
  allModes,
  defaultMode,
  renderFilters,
  renderModeChoices,
  FALLBACK_CAPABILITIES,
} from './advanced-options.js';
import { getState, setState, subscribe } from './store.js';
import { isEmbedded } from '/design-system/js/frame-params.js';
import { icon } from '/design-system/js/icons.js';
import { togglePopover, closePopover } from '/design-system/js/popover.js';

/** @typedef {import('./store.js').State} State */
/** @typedef {import('./advanced-options.js').Capabilities} Capabilities */
/** @typedef {import('./search.js').SearchResults} SearchResults */
/** @typedef {import('./search.js').RerankStatus} RerankStatus */

/** Entries per `/api/entries` page. */
export const PAGE_SIZE = 20;

/** The Drafts mailbox's empty state: no draft listing in this phase. */
export const DRAFTS_EMPTY = {
  title: 'No Draft Listing Yet',
  text: "Agent drafts open from the agent's link.",
};

/** Store keys whose change re-runs the list. */
const LIST_KEYS = ['query', 'mode', 'filters', 'mailbox'];

/**
 * @typedef {Object} ListView
 * @property {HTMLElement} body - Where rows, banners and states are painted
 * @property {HTMLInputElement|null} input - The standalone search box
 * @property {HTMLButtonElement} chip - The mode chip
 * @property {HTMLElement} popover - The Filters & Options popover (body level)
 * @property {Capabilities|null} capabilities
 * @property {boolean} blocked - Search is dead (configuration_invalid)
 */

/** @type {ListView|null} */
let view = null;

/** @type {(() => void)|null} */
let unsubscribe = null;

// Monotonic id of the newest list run. A listing page or a search that no
// longer owns it paints nothing and leaves `aria-busy` alone.
let runSeq = 0;

/**
 * Listing paging state for the current empty-query run.
 * @type {{page: number, totalPages: number, author: string|null, lastDay: string|null, loading: boolean}|null}
 */
let paging = null;

// --- Pure helpers ---

/**
 * The mode a search runs in: the store's when set, else the deployment default.
 * @param {State} state
 * @param {Capabilities|null} capabilities
 * @returns {string}
 */
function effectiveMode(state, capabilities) {
  return state.mode ?? defaultMode(capabilities);
}

/**
 * The author the mailbox restricts to: the Mine name under Mine, else none.
 * @param {State} state
 * @returns {string|null}
 */
function scopeAuthor(state) {
  return state.mailbox === 'mine' ? state.mineAuthor : null;
}

// --- Painting ---

/**
 * @param {boolean} busy
 */
function setBusy(busy) {
  if (!view) return;
  view.body.setAttribute('aria-busy', String(busy));
}

/**
 * Paint one search response.
 * @param {SearchResults} results
 * @param {RerankStatus} status
 */
function paintResults(results, status) {
  if (!view) return;
  const selected = getState().sel;
  const modesUsed = results.search_modes_used?.join(', ') || 'none';
  const execTime = results.execution_time_ms || 0;

  let html = '';
  if ((results.diagnostics?.length ?? 0) > 0) {
    html += renderDiagnosticsBar(results.diagnostics ?? []);
  }
  html += renderRerankStatus(status);
  html += `
    <div class="results-header">
      <span class="results-count">
        <strong>${escapeHtml(String(results.total_results ?? 0))}</strong> results
        <span class="text-muted">(${escapeHtml(String(execTime))}ms)</span>
      </span>
      <span class="results-modes">Modes: ${escapeHtml(modesUsed)}</span>
    </div>
  `;
  html += renderExpansionBanner(results.expanded_terms);

  const entries = results.entries ?? [];
  if (entries.length > 0) {
    html += `<div class="ariel-rows" role="listbox" aria-label="Search results">${entries
      .map((e) => renderListRow(e, selected))
      .join('')}</div>`;
  } else {
    html += renderEmptyState('No Results Found', 'Try adjusting your search terms or filters.');
  }
  view.body.innerHTML = html;
}

/**
 * The "Load older" control at the end of a listing, when there is more.
 * @returns {string}
 */
function renderLoadOlder() {
  if (!paging || paging.page >= paging.totalPages) return '';
  return `
    <div class="ariel-load-older-row">
      <button type="button" class="btn btn-secondary btn-sm ariel-load-older" data-load-older>Load older entries</button>
    </div>
  `;
}

/**
 * Run the empty-query listing from its first page.
 * @param {number} seq
 * @param {string|null} author
 */
async function runListing(seq, author) {
  if (!view) return;
  paging = { page: 0, totalPages: 1, author, lastDay: null, loading: true };
  view.body.innerHTML = renderLoading('Loading entries...');
  setBusy(true);
  /** @type {any} */
  let result;
  try {
    result = await entriesApi.list({ page: 1, pageSize: PAGE_SIZE, author: author ?? undefined });
  } catch (error) {
    if (seq !== runSeq || !view) return;
    console.error('Failed to load entries:', error);
    view.body.innerHTML = renderErrorState('Failed to Load Entries', error);
    paging.loading = false;
    setBusy(false);
    return;
  }
  if (seq !== runSeq || !view) return;
  paging.loading = false;
  setBusy(false);

  const entries = result?.entries ?? [];
  if (entries.length === 0) {
    view.body.innerHTML = author
      ? renderEmptyState('No Entries', `No logbook entries by ${author} yet.`)
      : renderEmptyState('No Entries', 'No logbook entries found.');
    return;
  }
  paging.page = result.page || 1;
  paging.totalPages = result.total_pages || 1;
  const { html, lastDay } = renderDayGroupedRows(entries, null, getState().sel);
  paging.lastDay = lastDay;
  view.body.innerHTML = `<div class="ariel-rows" role="listbox" aria-label="Entries">${html}</div>${renderLoadOlder()}`;
}

/**
 * Append the next older page of the current listing.
 * @returns {Promise<void>}
 */
export async function loadOlder() {
  if (!view || !paging || paging.loading) return;
  if (paging.page >= paging.totalPages) return;
  const seq = runSeq;
  const current = paging;
  current.loading = true;
  const button = /** @type {HTMLButtonElement|null} */ (
    view.body.querySelector('[data-load-older]')
  );
  if (button) button.disabled = true;
  setBusy(true);
  /** @type {any} */
  let result;
  try {
    result = await entriesApi.list({
      page: current.page + 1,
      pageSize: PAGE_SIZE,
      author: current.author ?? undefined,
    });
  } catch (error) {
    if (seq !== runSeq || !view) return;
    console.error('Failed to load older entries:', error);
    current.loading = false;
    setBusy(false);
    if (button) button.disabled = false;
    const row = view.body.querySelector('.ariel-load-older-row');
    row?.querySelector('[data-load-older-error]')?.remove();
    row?.insertAdjacentHTML(
      'afterbegin',
      `<span class="text-error" data-load-older-error>${escapeHtml(
        error instanceof Error ? error.message : String(error)
      )}</span>`
    );
    return;
  }
  if (seq !== runSeq || !view) return;
  current.loading = false;
  setBusy(false);
  current.page = result?.page || current.page + 1;
  current.totalPages = result?.total_pages || current.totalPages;
  const rows = view.body.querySelector('.ariel-rows');
  const { html, lastDay } = renderDayGroupedRows(
    result?.entries ?? [],
    current.lastDay,
    getState().sel
  );
  current.lastDay = lastDay;
  rows?.insertAdjacentHTML('beforeend', html);
  view.body.querySelector('.ariel-load-older-row')?.remove();
  view.body.insertAdjacentHTML('beforeend', renderLoadOlder());
}

/**
 * Run a search for the state's query through the driver.
 * @param {number} seq
 * @param {State} state
 */
async function runQuery(seq, state) {
  if (!view) return;
  const capabilities = view.capabilities;
  paging = null;
  view.body.innerHTML = renderLoading('Searching...');
  setBusy(true);
  /** @type {RerankStatus} */
  let status = null;
  try {
    await runSearch(
      {
        query: state.query,
        mode: effectiveMode(state, capabilities),
        filters: state.filters,
        author: scopeAuthor(state),
      },
      {
        capabilities,
        rerank: state.rerank,
        onStatus: (s) => {
          status = s;
        },
        onResults: (results) => {
          if (seq !== runSeq) return;
          paintResults(results, status);
        },
      }
    );
  } catch (error) {
    if (seq !== runSeq || !view) return;
    console.error('Search failed:', error);
    view.body.innerHTML = renderErrorState('Search Failed', error);
  }
  if (seq === runSeq) setBusy(false);
}

/**
 * Re-run the list for the current state: Drafts' empty state, the listing, or
 * a search.
 */
function refresh() {
  if (!view) return;
  const seq = ++runSeq;
  const state = getState();
  /** @param {string} html - A still state: no request runs behind it */
  const still = (html) => {
    cancelSearch();
    paging = null;
    setBusy(false);
    if (view) view.body.innerHTML = html;
  };

  if (state.mailbox === 'drafts') {
    still(renderEmptyState(DRAFTS_EMPTY.title, DRAFTS_EMPTY.text));
    return;
  }
  if (state.mailbox === 'mine' && !state.mineAuthor) {
    still(
      renderEmptyState(
        'Mine Needs a Name',
        'Choose Mine in the rail and enter the name you create entries under.'
      )
    );
    return;
  }
  // While search is blocked (configuration_invalid) a query from the hash
  // cannot run; the newest-first listing is shown instead.
  if (!(state.query ?? '').trim() || view.blocked) {
    cancelSearch();
    runListing(seq, scopeAuthor(state));
    return;
  }
  runQuery(seq, state);
}

// --- Pane head ---

/**
 * The mode chip's text: the effective mode's label.
 */
function paintChip() {
  if (!view) return;
  const name = effectiveMode(getState(), view.capabilities);
  const mode = allModes(view.capabilities || FALLBACK_CAPABILITIES).find((m) => m.name === name);
  const label = view.chip.querySelector('.ariel-mode-chip-label');
  if (label) label.textContent = mode?.label ?? name;
  view.chip.dataset.mode = name;
}

/**
 * Render the popover's mode choices and Filters & Options from the store.
 * Under Mine the author is locked to the Mine name, because a Filters author
 * would override the request author.
 */
function paintPopover() {
  if (!view) return;
  const state = getState();
  const capabilities = view.capabilities;
  const mode = effectiveMode(state, capabilities);
  const modesRoot = /** @type {HTMLElement} */ (view.popover.querySelector('[data-mode-choices]'));
  const filtersRoot = /** @type {HTMLElement} */ (view.popover.querySelector('[data-filters]'));
  renderModeChoices(modesRoot, {
    capabilities,
    mode,
    onSelect: (chosen) => setState({ mode: chosen, rerank: 'now' }),
  });
  /** @type {Object<string, *>} */
  const locked = {};
  const author = scopeAuthor(state);
  if (author) locked.author = author;
  renderFilters(filtersRoot, {
    capabilities,
    mode,
    filters: state.filters,
    locked,
    onChange: (filters) => {
      ownFiltersWrite = true;
      try {
        setState({ filters, rerank: 'now' });
      } finally {
        ownFiltersWrite = false;
      }
    },
  });
}

// Set while the popover reports a filters change, so the store's echo does not
// re-render the controls the operator is typing in.
let ownFiltersWrite = false;

/**
 * The pane head: search box and compose button standalone, the mode chip always.
 * `paintChip` fills the chip's label.
 * @returns {{head: HTMLElement, input: HTMLInputElement|null, chip: HTMLButtonElement}}
 */
function buildHead() {
  const embedded = isEmbedded();
  const head = document.createElement('div');
  head.className = 'osprey-pane-head ariel-list-head';

  /** @type {HTMLInputElement|null} */
  let input = null;
  if (!embedded) {
    const search = document.createElement('label');
    search.className = 'osprey-search ariel-list-search';
    search.append(icon('search'));
    input = document.createElement('input');
    input.type = 'search';
    input.id = 'ariel-search-input';
    input.placeholder = 'Search entries';
    input.setAttribute('aria-label', 'Search entries');
    input.autocomplete = 'off';
    input.value = getState().query ?? '';
    search.append(input);
    head.append(search);
  }

  const chip = document.createElement('button');
  chip.type = 'button';
  chip.className = 'osprey-btn ariel-mode-chip';
  chip.id = 'ariel-mode-chip';
  chip.title = 'Search mode, filters and options';
  chip.setAttribute('aria-haspopup', 'dialog');
  const chipLabel = document.createElement('span');
  chipLabel.className = 'osprey-btn-label ariel-mode-chip-label';
  chip.append(chipLabel, icon('chevron-down'));
  head.append(chip);

  if (!embedded) {
    const compose = document.createElement('button');
    compose.type = 'button';
    compose.className = 'osprey-icon-btn ariel-compose-btn';
    compose.id = 'ariel-compose-btn';
    compose.setAttribute('aria-label', 'New entry');
    compose.title = 'New entry';
    compose.append(icon('plus'));
    compose.addEventListener('click', () => setState({ compose: { draft: null } }));
    head.append(compose);
  }
  return { head, input, chip };
}

/**
 * Replace the body-level Filters & Options popover with a fresh, hidden one.
 * @returns {HTMLElement}
 */
function buildFiltersPopover() {
  document.getElementById('ariel-filters-popover')?.remove();
  const popover = document.createElement('div');
  popover.className = 'osprey-popover ariel-filters-popover';
  popover.id = 'ariel-filters-popover';
  popover.setAttribute('role', 'dialog');
  popover.setAttribute('aria-label', 'Search mode, filters and options');
  popover.hidden = true;
  popover.innerHTML = `
    <div class="ariel-mode-choices" data-mode-choices></div>
    <div class="advanced-sections ariel-filters" data-filters></div>
  `;
  document.body.append(popover);
  return popover;
}

/**
 * Build the pane head, the body and the popover into `root`, and apply a
 * degraded configuration: its banner, and dead search controls when invalid.
 * @param {HTMLElement} root
 * @param {Capabilities|null} capabilities
 * @returns {ListView}
 */
function buildView(root, capabilities) {
  root.innerHTML = '';
  const { head, input, chip } = buildHead();

  const banners = document.createElement('div');
  banners.className = 'ariel-list-banners';

  const body = document.createElement('div');
  body.className = 'ariel-list-body';
  body.id = 'ariel-list-body';
  body.setAttribute('aria-busy', 'false');

  root.append(head, banners, body);

  const popover = buildFiltersPopover();

  const status = capabilities?.status;
  const blocked = status === 'configuration_invalid';
  if (blocked || status === 'configuration_warning') {
    banners.innerHTML = renderConfigBanner(
      capabilities?.config_errors ?? [],
      capabilities?.remedy ?? null,
      blocked
    );
  }
  if (blocked) {
    if (input) input.disabled = true;
    chip.disabled = true;
  }

  return { body, input, chip, popover, capabilities, blocked };
}

/**
 * Bind the pane's own listeners: search-box Enter, the chip, row clicks and
 * Load older (delegated on the stable body, whose content is replaced).
 * @param {ListView} v
 */
function bindView(v) {
  v.input?.addEventListener('keydown', (e) => {
    if (e.key !== 'Enter' || e.shiftKey || e.isComposing) return;
    e.preventDefault();
    const value = /** @type {HTMLInputElement} */ (e.currentTarget).value.trim();
    setState({ query: value, rerank: 'now' }, { force: true });
  });

  v.chip.addEventListener('click', () => {
    if (v.popover.hidden) paintPopover();
    togglePopover(v.chip, v.popover, { place: 'below-end' });
  });

  v.body.addEventListener('click', (e) => {
    const target = /** @type {HTMLElement|null} */ (e.target);
    if (!target?.closest) return;
    if (target.closest('[data-load-older]')) {
      loadOlder();
      return;
    }
    const row = /** @type {HTMLElement|null} */ (target.closest('[data-entry-id]'));
    if (row && v.body.contains(row)) {
      const id = row.dataset.entryId;
      if (id) setState({ sel: id, narrowView: 'reader' });
    }
  });
}

/**
 * Reflect the selection onto the rows already painted.
 * @param {string|null} sel
 */
function paintSelection(sel) {
  if (!view) return;
  const rows = /** @type {NodeListOf<HTMLElement>} */ (
    view.body.querySelectorAll('[data-entry-id]')
  );
  rows.forEach((row) => {
    row.setAttribute('aria-selected', String(sel !== null && row.dataset.entryId === sel));
  });
}

/**
 * The store subscriber.
 * @param {State} state
 * @param {State} _prev
 * @param {Set<string>} changed
 */
function onStoreChange(state, _prev, changed) {
  if (!view) return;
  if (changed.has('query') && view.input && view.input.value !== state.query) {
    view.input.value = state.query ?? '';
  }
  if (changed.has('mode')) paintChip();
  if (!view.popover.hidden) {
    const relevant =
      changed.has('mode') ||
      changed.has('mailbox') ||
      changed.has('mineAuthor') ||
      (changed.has('filters') && !ownFiltersWrite);
    if (relevant) paintPopover();
  }
  if (changed.has('sel')) paintSelection(state.sel);

  const rerun =
    LIST_KEYS.some((k) => changed.has(k)) ||
    (changed.has('mineAuthor') && state.mailbox === 'mine');
  if (!rerun) return;
  // A mode or filters change does not alter the newest-first listing.
  const onlySearchKnobs = !['query', 'mailbox', 'mineAuthor'].some((k) => changed.has(k));
  if (onlySearchKnobs && !(state.query ?? '').trim()) return;
  refresh();
}

/**
 * Render the list pane into `root` and follow the store.
 * @param {HTMLElement} root - The list pane root (`#ariel-list`)
 * @param {{capabilities?: Capabilities|null}} [options]
 */
export function initList(root, { capabilities = null } = {}) {
  unsubscribe?.();
  if (view) closePopover(view.popover);
  runSeq++;
  view = buildView(root, capabilities);
  bindView(view);
  paintChip();
  unsubscribe = subscribe(onStoreChange);
  refresh();
}

/**
 * Focus the standalone search box (the `/` shortcut). A no-op embedded, where
 * the box belongs to the host, and while search is blocked.
 */
export function focusSearch() {
  const input = view?.input;
  if (!input || input.disabled) return;
  input.focus();
  input.select();
}
