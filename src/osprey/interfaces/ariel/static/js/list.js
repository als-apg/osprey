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
 * `filters`, `filter`, `mailbox` (and `mineAuthor` under Mine) and either
 * browses `/api/entries` newest first with day separators (empty query) or
 * hands the query to the search driver with the state's `rerank` (any other
 * query), in ranked order. paging.js owns the pages: any of those changes
 * starts over, and "Load more" fetches the next one. Both apply the filter;
 * the mailbox is a scope: Mine sends its name as the exact author, Drafts
 * hands the body to drafts-list.js whatever the query. A
 * display-prefs.js change re-runs the list; `notice` shows until a filter change.
 *
 * The list writes only what it chooses: `query` (search-box Enter, with
 * `force`; its `dim:value` operators are lifted into `filter`, query-ops.js), `mode`, `filter` and `filters` (the popover, through
 * `splitFilters`/`joinFilters`), each with `rerank:'now'`;
 * `sel` with `narrowView:'reader'` and `att:null` (row click); and `compose` (compose button).
 */

import {
  renderListRow,
  renderDayGroupedRows,
  renderRerankStatus,
  renderDiagnosticsBar,
  diagnosticsLevel,
  diagnosticsText,
  renderExpansionBanner,
  renderConfigBanner,
  renderLoading,
  renderEmptyState,
  renderErrorState,
  escapeHtml,
} from './components.js';
import { cancelSearch, isAbort } from './search.js';
import {
  PAGE_SIZE,
  canLoadMore,
  loadMore,
  pageInfo,
  reset as resetPaging,
  startListing,
  startSearch,
  listingFilter,
  totalText,
} from './paging.js';
import { joinFilters, mergeFilter, requestFilter, splitFilters, TIME } from './filter-state.js';
import { liftSearch } from './query-ops.js';
import {
  allModes,
  defaultMode,
  renderFilters,
  renderModeChoices,
  FALLBACK_CAPABILITIES,
} from './advanced-options.js';
import { getState, setState, subscribe } from './store.js';
import { renderDrafts } from './drafts-list.js';
import { onStripPrefChange } from './display-prefs.js';
import { isEmbedded } from '/design-system/js/frame-params.js';
import { icon } from '/design-system/js/icons.js';
import { togglePopover, closePopover } from '/design-system/js/popover.js';

/** @typedef {import('./store.js').State} State */
/** @typedef {import('./advanced-options.js').Capabilities} Capabilities */
/** @typedef {import('./search.js').SearchResults} SearchResults */
/** @typedef {import('./search.js').RerankStatus} RerankStatus */

export { PAGE_SIZE };

/** Store keys whose change re-runs the list. */
const LIST_KEYS = ['query', 'mode', 'filters', 'filter', 'mailbox'];

/**
 * @typedef {Object} ListView
 * @property {HTMLElement} body - Where rows, banners and states are painted
 * @property {HTMLElement} notice - The store notice's line in the banners
 * @property {HTMLInputElement|null} input - The standalone search box
 * @property {HTMLButtonElement} chip - The mode chip
 * @property {HTMLElement} popover - The Filters & Options popover (body level)
 * @property {Capabilities|null} capabilities
 * @property {boolean} blocked - Search is dead (configuration_invalid)
 */

/** @type {ListView|null} */
let view = null;

/** @type {Array<() => void>} The store and display-prefs unsubscribers. */
let unsubscribe = [];

// Monotonic id of the newest list run. A listing page or a search that no
// longer owns it paints nothing and leaves `aria-busy` alone.
let runSeq = 0;

// The last day separator the browse listing painted, so the next page
// continues the day without repeating it.
/** @type {string|null} */
let lastDay = null;

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
 * The search request the list runs for `state`, without offset, page size or
 * view: list-activity.js sends the same one for its page-1 side fetch.
 * @param {State} state
 * @param {Capabilities|null} capabilities
 * @returns {import('./search.js').SearchRequest}
 */
export function searchRequest(state, capabilities) {
  return {
    query: state.query,
    mode: effectiveMode(state, capabilities),
    filters: state.filters,
    filter: requestFilter(state),
  };
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

  paintIssues(results.diagnostics ?? []);
  let html = '';
  html += renderRerankStatus(status);
  html += `
    <div class="results-header">
      <span class="results-count">
        <strong data-list-total>${escapeHtml(totalText())}</strong> results
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
      .join('')}</div>${renderLoadMore()}`;
  } else {
    html += renderEmptyState('No Results Found', 'Try adjusting your search terms or filters.');
  }
  view.body.innerHTML = html;
}

/**
 * The "Load more" control at the end of the list, while there is a next page;
 * disabled while a rerank is pending.
 * @returns {string}
 */
function renderLoadMore() {
  if (!canLoadMore()) return '';
  const info = pageInfo();
  const label = info.kind === 'browse' ? 'Load older entries' : 'Load more results';
  return `
    <div class="ariel-load-older-row">
      <button type="button" class="btn btn-secondary btn-sm ariel-load-older" data-load-more${
        info.held ? ' disabled' : ''
      }>${label}</button>
    </div>
  `;
}

/** Replace the "Load more" control with the current one. */
function repaintLoadMore() {
  if (!view) return;
  view.body.querySelector('.ariel-load-older-row')?.remove();
  view.body.insertAdjacentHTML('beforeend', renderLoadMore());
}

/**
 * Run the empty-query listing from its first page.
 * @param {number} seq
 * @param {State} state
 */
async function runListing(seq, state) {
  if (!view) return;
  paintIssues([]);
  lastDay = null;
  view.body.innerHTML = renderLoading('Loading entries...');
  setBusy(true);
  /** @type {{entries: any[], total: number|null}|null} */
  let result;
  try {
    result = await startListing(listingFilter(state));
  } catch (error) {
    if (seq !== runSeq || !view) return;
    console.error('Failed to load entries:', error);
    view.body.innerHTML = renderErrorState('Failed to Load Entries', error);
    setBusy(false);
    return;
  }
  if (!result || seq !== runSeq || !view) return;
  setBusy(false);

  const author = scopeAuthor(state);
  if (result.entries.length === 0) {
    const none = author ? `No logbook entries by ${author} yet. Names must match exactly.` : '';
    view.body.innerHTML = renderEmptyState('No Entries', none || 'No logbook entries found.');
    return;
  }
  const rows = renderDayGroupedRows(result.entries, null, getState().sel);
  lastDay = rows.lastDay;
  const count = `<strong data-list-total>${escapeHtml(totalText())}</strong> entries`;
  const header = `<div class="results-header ariel-list-header"><span>${count}</span></div>`;
  view.body.innerHTML = `${header}<div class="ariel-rows" role="listbox" aria-label="Entries">${
    rows.html
  }</div>${renderLoadMore()}`;
}

/**
 * Append the next page of the current run, from the "Load more" control.
 * @returns {Promise<void>}
 */
async function onLoadMore() {
  if (!view || !canLoadMore()) return;
  const info = pageInfo();
  if (info.held || info.loading) return;
  const seq = runSeq;
  const button = view.body.querySelector('[data-load-more]');
  button?.setAttribute('disabled', '');
  setBusy(true);
  /** @type {{entries: any[]}|null} */
  let page;
  try {
    page = await loadMore();
  } catch (error) {
    if (seq !== runSeq || !view) return;
    console.error('Failed to load more entries:', error);
    setBusy(false);
    repaintLoadMore();
    const message = escapeHtml(error instanceof Error ? error.message : String(error));
    view.body
      .querySelector('.ariel-load-older-row')
      ?.insertAdjacentHTML(
        'afterbegin',
        `<span class="text-error" data-load-more-error>${message}</span>`
      );
    return;
  }
  if (seq !== runSeq || !view) return;
  setBusy(false);
  const rows = view.body.querySelector('.ariel-rows');
  rows?.setAttribute('data-paged', ''); // past page 1: list-activity.js only pills
  const sel = getState().sel;
  if (page && info.kind === 'browse') {
    const more = renderDayGroupedRows(page.entries, lastDay, sel);
    lastDay = more.lastDay;
    rows?.insertAdjacentHTML('beforeend', more.html);
  } else if (page) {
    rows?.insertAdjacentHTML('beforeend', page.entries.map((e) => renderListRow(e, sel)).join(''));
    const total = view.body.querySelector('[data-list-total]');
    if (total) total.textContent = totalText();
  }
  repaintLoadMore();
}

/**
 * Run a search for the state's query through the driver.
 * @param {number} seq
 * @param {State} state
 */
async function runQuery(seq, state) {
  if (!view) return;
  const capabilities = view.capabilities;
  view.body.innerHTML = renderLoading('Searching...');
  setBusy(true);
  /** @type {RerankStatus} */
  let status = null;
  try {
    await startSearch(searchRequest(state, capabilities), {
      capabilities,
      rerank: state.rerank,
      onStatus: (s) => {
        status = s;
      },
      onResults: (results) => {
        if (seq !== runSeq) return;
        paintResults(results, status);
      },
    });
  } catch (error) {
    if (seq !== runSeq || !view || isAbort(error)) return;
    console.error('Search failed:', error);
    view.body.innerHTML = renderErrorState('Search Failed', error);
  }
  if (seq === runSeq) setBusy(false);
}

/**
 * Re-run the list for the current state: the Drafts rows, the listing, or a
 * search. The promise settles when the run has painted (or was superseded).
 * @returns {Promise<void>|undefined}
 */
export function refresh() {
  if (!view) return;
  const seq = ++runSeq;
  const state = getState();
  /** @param {string} html - A still state: no request runs behind it */
  const still = (html) => {
    cancelSearch();
    resetPaging();
    setBusy(false);
    if (view) view.body.innerHTML = html;
  };

  if (state.mailbox === 'drafts') {
    still('');
    if (view) renderDrafts(view.body, () => seq === runSeq);
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
    return runListing(seq, state);
  }
  return runQuery(seq, state);
}

// --- Pane head ---

/** The mode chip's own tooltip, before any search issue is appended. */
const CHIP_TITLE = 'Search mode, filters and options';

/**
 * A search's diagnostics, off the result list: a dot on the mode chip, the
 * issue text as the chip's tooltip and accessible description, and the full
 * list at the top of the chip's popover, which the keyboard reaches through
 * the chip. An empty list clears all three.
 * @param {import('./components.js').Diagnostic[]} diagnostics
 */
function paintIssues(diagnostics) {
  if (!view) return;
  const { chip, popover } = view;
  const issues = /** @type {HTMLElement|null} */ (popover.querySelector('[data-search-issues]'));
  chip.querySelector('.ariel-mode-chip-dot')?.remove();
  if (diagnostics.length === 0) {
    chip.title = CHIP_TITLE;
    chip.removeAttribute('aria-description');
    if (issues) {
      issues.hidden = true;
      issues.innerHTML = '';
    }
    return;
  }
  const text = diagnosticsText(diagnostics);
  const dot = document.createElement('span');
  dot.className = 'ariel-mode-chip-dot';
  dot.dataset.level = diagnosticsLevel(diagnostics);
  dot.setAttribute('aria-hidden', 'true');
  chip.append(dot);
  chip.title = `${CHIP_TITLE}\n${text}`;
  chip.setAttribute('aria-description', text);
  if (issues) {
    issues.innerHTML = renderDiagnosticsBar(diagnostics, { open: true });
    issues.hidden = false;
  }
}

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
 * The controls show `filter` and the knobs joined; a change splits them back,
 * replacing author, source system and time in `filter` (other dimensions
 * stay). Under Mine the author is locked to the Mine
 * name, because a Filters author would override the request author.
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
    filters: joinFilters(state.filter, state.filters),
    locked,
    onChange: (values) => {
      const { filter, knobs } = splitFilters(values);
      const own = { author: null, source_system: null, [TIME]: null, ...filter };
      ownFiltersWrite = true;
      try {
        setState({ filter: mergeFilter(getState().filter, own), filters: knobs, rerank: 'now' });
      } finally {
        ownFiltersWrite = false;
      }
    },
  });
}

// Set while the popover reports a change, so the store's echo does not
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
  chip.title = CHIP_TITLE;
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
    <div class="ariel-search-issues" data-search-issues hidden></div>
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

  const notice = document.createElement('div');
  notice.className = 'ariel-list-notice text-muted';
  notice.setAttribute('role', 'status');
  notice.dataset.listNotice = '';
  notice.hidden = true;

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
  // After the config banner, which stays the first child.
  banners.append(notice);
  if (blocked) {
    if (input) input.disabled = true;
    chip.disabled = true;
  }

  return { body, notice, input, chip, popover, capabilities, blocked };
}

/**
 * Bind the pane's own listeners: search-box Enter, the chip, row clicks and
 * Load more (delegated on the stable body, whose content is replaced).
 * @param {ListView} v
 */
function bindView(v) {
  v.input?.addEventListener('keydown', (e) => {
    if (e.key !== 'Enter' || e.shiftKey || e.isComposing) return;
    e.preventDefault();
    setState(liftSearch(v.input?.value, getState(), v.capabilities, 'now'), { force: true });
  });

  v.chip.addEventListener('click', () => {
    if (v.popover.hidden) paintPopover();
    togglePopover(v.chip, v.popover, { place: 'below-end' });
  });

  v.body.addEventListener('click', (e) => {
    const target = /** @type {HTMLElement|null} */ (e.target);
    if (!target?.closest) return;
    if (target.closest('[data-load-more]')) {
      onLoadMore();
      return;
    }
    const row = /** @type {HTMLElement|null} */ (target.closest('[data-entry-id]'));
    if (row && v.body.contains(row)) {
      const id = row.dataset.entryId;
      if (id) setState({ sel: id, att: null, narrowView: 'reader' });
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
 * Show the store's notice in the banners, or hide its line.
 * @param {string|null} notice
 */
function paintNotice(notice) {
  if (!view) return;
  view.notice.textContent = notice ?? '';
  view.notice.hidden = !notice;
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
      ((changed.has('filters') || changed.has('filter')) && !ownFiltersWrite);
    if (relevant) paintPopover();
  }
  if (changed.has('sel')) paintSelection(state.sel);
  if (changed.has('notice')) paintNotice(state.notice);
  // A notice lasts until the next filter change. The router writes a filter
  // and its notice in one patch, so that write keeps it.
  const filterChanged = changed.has('filter') || changed.has('filters');
  if (filterChanged && !changed.has('notice') && state.notice) {
    setState({ notice: null });
  }

  const rerun =
    LIST_KEYS.some((k) => changed.has(k)) ||
    (changed.has('mineAuthor') && state.mailbox === 'mine');
  if (!rerun) return;
  refresh();
}

/**
 * Render the list pane into `root` and follow the store.
 * @param {HTMLElement} root - The list pane root (`#ariel-list`)
 * @param {{capabilities?: Capabilities|null}} [options]
 */
export function initList(root, { capabilities = null } = {}) {
  unsubscribe.forEach((off) => off());
  if (view) closePopover(view.popover);
  runSeq++;
  view = buildView(root, capabilities);
  bindView(view);
  paintChip();
  paintNotice(getState().notice);
  unsubscribe = [subscribe(onStoreChange), onStripPrefChange(() => refresh())];
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
