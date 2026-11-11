// @ts-check
/**
 * OSPREY Channel Finder — All PVs
 *
 * The view of `#/allpvs?q=<words>&ioc=<name>` (Panel view contract): a search
 * of the current target's PV directory, one page of rows at a time. The route's
 * words and filters seed the search and the facets on mount.
 *
 *   toolbar   [sidebar toggle] [Curated | <directory> | Feedback] …… [N PVs] [search]
 *   sidebar   Origin · IOC · Record type, each value with its count
 *   body      <what was searched> [partial] …… [refresh] [show request]
 *             the rows, with their current values
 *             N selected · clear …… page · CSV · Plot · Send to assistant
 *
 * While mounted the view holds the toolbar's search field, its result count and
 * its sidebar toggle (panel-toolbar.js). The words it searches are
 * `state.scopeQuery`, shared with the curated view. Current values are read for
 * the rows shown, once per page and on Refresh, through the panel's values
 * route. The selection is a basket: it survives new words, filter changes and
 * paging until "clear" or until the panel leaves All PVs and its PV and IOC
 * pages; it lives on `state`. Plot and Send hand it to the host, in the order
 * it was selected, only inside one.
 */

import { renderState, clearState } from '/design-system/js/panel-state.js';
import { openInfoPopover } from '/design-system/js/info-popover.js';
import { icon } from '/design-system/js/icons.js';
import { copyText } from '/design-system/js/clipboard.js';
import { messageOf } from './utils.js';
import {
  PAGE_SIZE,
  actionsHtml,
  countHtml,
  facetsHtml,
  skeletonHtml,
  tableHtml,
} from './allpvs-render.js';
import { FACET_KEYS } from './route.js';
import { state } from './state.js';
import { attachSidebar, claimToolbar, searchInput, setResultCount } from './panel-toolbar.js';
import {
  createPlotCheck,
  directoryExportUrl,
  handoffAllowed,
  plotButton,
  plotUrl,
  readValues,
  searchDirectory,
} from './directory-api.js';

/** @typedef {import('./directory-api.js').DirectoryPage} DirectoryPage */
/** @typedef {import('./directory-api.js').DirectoryFilters} DirectoryFilters */
/** @typedef {import('./directory-api.js').ValueRow} ValueRow */
/** @typedef {import('/design-system/js/panel-sidebar.js').PanelSidebarApi} PanelSidebarApi */

/** @typedef {import('./route.js').FacetKey} FacetKey */
/** @typedef {import('./route.js').RouteParams} RouteParams */

const SEARCH_DEBOUNCE_MS = 200;
const SIDEBAR_KEY = 'cf-allpvs-filters';

/**
 * The mounted view's state; null while nothing is mounted.
 * @typedef {{
 *   root: HTMLElement,
 *   page: number,
 *   filters: Record<FacetKey, Set<string>>,
 *   answer: DirectoryPage|null,
 *   values: Map<string, ValueRow>|null,
 *   valuesReason: string,
 *   searchSeq: number,
 *   valuesSeq: number,
 *   canPlot: boolean,
 *   csvStatus: string,
 *   debounce: number|undefined,
 *   sidebar: PanelSidebarApi,
 *   release: () => void,
 *   controller: AbortController,
 * }} View
 */

/** @type {View|null} */
let view = null;

/**
 * @param {HTMLElement} container
 * @param {RouteParams} [params]
 */
export function mountAllPvs(container, params = {}) {
  unmountAllPvs();
  if (params.q !== undefined) state.scopeQuery = params.q;

  container.innerHTML = skeletonHtml();
  const root = /** @type {HTMLElement} */ (container.querySelector('.allpvs'));
  root.querySelector('[data-action="refresh"]')?.append(icon('refresh'));
  root.querySelector('[data-action="copy-request"]')?.append(icon('copy'));

  const release = claimToolbar();
  const input = searchInput();
  if (input) input.value = state.scopeQuery;

  const sidebar = attachSidebar(
    /** @type {HTMLElement} */ (root.querySelector('.allpvs-facets')),
    root.querySelector('.osprey-splitter'),
    SIDEBAR_KEY
  );

  const controller = new AbortController();
  const { signal } = controller;
  view = {
    root,
    page: 0,
    filters: facetRecord((key) => new Set(params[key])),
    answer: null,
    values: null,
    valuesReason: '',
    searchSeq: 0,
    valuesSeq: 0,
    canPlot: false,
    csvStatus: '',
    debounce: undefined,
    sidebar,
    release,
    controller,
  };

  input?.addEventListener('input', () => onWords(input.value), { signal });
  root.addEventListener('click', onClick, { signal });
  root.addEventListener('change', onChange, { signal });

  const mounted = view;
  void createPlotCheck()().then((ok) => {
    if (view !== mounted) return;
    mounted.canPlot = ok;
    renderActions();
  });

  void runSearch();
}

export function unmountAllPvs() {
  if (!view) return;
  const { controller, debounce, sidebar, release } = view;
  view = null;
  window.clearTimeout(debounce);
  controller.abort();
  sidebar.destroy();
  release();
}

// ---- Searching ----

/**
 * The words typed in the toolbar: shared at once, searched after a pause, over
 * a fresh result set.
 * @param {string} words
 */
function onWords(words) {
  if (!view) return;
  state.scopeQuery = words;
  const current = view;
  window.clearTimeout(current.debounce);
  current.debounce = window.setTimeout(() => {
    for (const set of Object.values(current.filters)) set.clear();
    restart();
  }, SEARCH_DEBOUNCE_MS);
}

/** Search again from the first page. */
function restart() {
  if (!view) return;
  view.page = 0;
  void runSearch();
}

/**
 * One entry per route filter key.
 * @template T
 * @param {(key: FacetKey) => T} make
 * @returns {Record<FacetKey, T>}
 */
function facetRecord(make) {
  return /** @type {Record<FacetKey, T>} */ (
    Object.fromEntries(FACET_KEYS.map((key) => [key, make(key)]))
  );
}

/** @returns {DirectoryFilters} the current search's filters */
function currentFilters() {
  const { filters, page } = /** @type {View} */ (view);
  return {
    q: state.scopeQuery.trim(),
    ...facetRecord((key) => [...filters[key]]),
    page,
    page_size: PAGE_SIZE,
  };
}

async function runSearch() {
  if (!view) return;
  const current = view;
  const seq = ++current.searchSeq;
  current.valuesSeq++;
  if (!current.answer) {
    renderState(body(), { icon: 'search', message: 'Searching' });
  }
  /** @type {DirectoryPage} */
  let answer;
  try {
    answer = await searchDirectory(currentFilters());
  } catch (err) {
    if (view !== current || seq !== current.searchSeq) return;
    current.answer = null;
    showFailure(`Directory did not answer: ${messageOf(err)}`, true);
    return;
  }
  if (view !== current || seq !== current.searchSeq) return;
  current.answer = answer;
  current.values = null;
  current.valuesReason = '';
  state.emit('directoryLabel', answer.source.label);
  render();
  if (answer.rows.length) void readPageValues();
}

/** Read the current values of the rows shown. */
async function readPageValues() {
  if (!view?.answer) return;
  const current = view;
  const seq = ++current.valuesSeq;
  const names = current.answer?.rows.map((row) => row.name) ?? [];
  /** @type {ValueRow[]} */
  let rows;
  try {
    rows = await readValues(names);
  } catch (err) {
    if (view !== current || seq !== current.valuesSeq) return;
    current.values = new Map();
    current.valuesReason = messageOf(err);
    renderTable();
    return;
  }
  if (view !== current || seq !== current.valuesSeq) return;
  current.values = new Map(rows.map((row) => [row.name, row]));
  renderTable();
}

// ---- Rendering ----

/** @returns {HTMLElement} */
function body() {
  return /** @type {HTMLElement} */ (view?.root.querySelector('.allpvs-body'));
}

/** @param {string} selector @returns {HTMLElement} */
function part(selector) {
  return /** @type {HTMLElement} */ (view?.root.querySelector(selector));
}

/**
 * The centred state in place of the rows, with Retry when one could help.
 * @param {string} message
 * @param {boolean} retry
 */
function showFailure(message, retry) {
  setResultCount('');
  renderHead();
  renderFacets();
  renderActions();
  renderState(body(), {
    icon: 'database',
    message,
    ...(retry ? { action: { label: 'Retry', onClick: () => void runSearch() } } : {}),
  });
}

function render() {
  const answer = view?.answer;
  if (!answer) return;
  setResultCount(answer.outcome === 'found' ? countHtml(answer.total) : '');
  if (answer.outcome === 'unavailable') {
    const reason = answer.reason || 'no answer';
    showFailure(`${answer.source.label}: ${reason}`, answer.source.service !== 'none');
    return;
  }
  renderHead();
  renderFacets();
  renderActions();
  if (answer.outcome === 'no_match' || answer.rows.length === 0) {
    renderState(body(), {
      icon: 'search',
      message: answer.searched ? `No match: ${answer.searched}` : 'No match',
    });
    return;
  }
  renderTable();
}

/** The section header: what was searched, partial, Refresh and the request. */
function renderHead() {
  if (!view) return;
  const answer = view.answer;
  part('.allpvs-searched').textContent = answer?.searched ?? '';

  const partial = part('.allpvs-partial');
  partial.replaceChildren();
  if (answer?.partial.length) {
    const chip = document.createElement('button');
    chip.type = 'button';
    chip.className = 'osprey-chip osprey-chip--warn';
    chip.dataset.action = 'partial';
    chip.textContent = 'partial';
    partial.append(chip);
  }

  const refresh = /** @type {HTMLButtonElement} */ (part('[data-action="refresh"]'));
  refresh.disabled = !answer?.rows.length;

  const request = answer?.request ?? '';
  part('[data-action="request"]').hidden = request === '';
  part('.allpvs-request code').textContent = request;
  if (request === '') showRequest(false);
}

/** @param {boolean} shown */
function showRequest(shown) {
  part('.allpvs-request').hidden = !shown;
  part('[data-action="request"]').textContent = shown ? 'hide request' : 'show request';
}

/** The filters, in the sidebar. */
function renderFacets() {
  if (!view) return;
  part('.allpvs-facets').innerHTML = facetsHtml(view.answer, view.filters);
}

/** The rows of the current page. */
function renderTable() {
  if (!view?.answer?.rows.length || view.answer.outcome === 'unavailable') return;
  const { answer, values, valuesReason } = view;
  const selected = state.selection;
  const host = body();
  clearState(host);
  host.innerHTML = tableHtml(answer, selected, values, valuesReason);
  const head = /** @type {HTMLInputElement} */ (host.querySelector('[data-action="select-page"]'));
  head.indeterminate = !head.checked && answer.rows.some((row) => selected.has(row.name));
}

/** The selected names, in selection order. */
function selectedNames() {
  return [...state.selection];
}

/** The pager, the selection and the actions on it. */
function renderActions() {
  if (!view) return;
  const { answer, page, canPlot, csvStatus } = view;
  const selected = state.selection;
  part('.allpvs-actions').innerHTML = actionsHtml({
    answer,
    page,
    selected: selected.size,
    csvStatus,
    plot: canPlot ? plotButton(selected.size) : null,
    send: handoffAllowed(),
  });
}

// ---- Actions ----

/** @param {Event} event */
function onClick(event) {
  if (!view) return;
  const target = event.target instanceof Element ? event.target : null;
  if (!target) return;
  const facet = /** @type {HTMLElement|null} */ (target.closest('[data-facet]'));
  if (facet) {
    toggleFacet(/** @type {FacetKey} */ (facet.dataset.facet), facet.dataset.value ?? '');
    return;
  }
  const action = /** @type {HTMLElement|null} */ (target.closest('[data-action]'))?.dataset.action;
  switch (action) {
    case 'request':
      showRequest(part('.allpvs-request').hidden === true);
      return;
    case 'copy-request':
      void copyText(view.answer?.request ?? '');
      return;
    case 'partial':
      openInfoPopover(/** @type {HTMLElement} */ (target.closest('[data-action]')), {
        title: 'Partial answer',
        rows: (view.answer?.partial ?? []).map(({ part: name, reason }) => ({
          label: name,
          value: reason,
        })),
      });
      return;
    case 'refresh':
      view.values = null;
      renderTable();
      void readPageValues();
      return;
    case 'prev':
    case 'next':
      view.page += action === 'next' ? 1 : -1;
      void runSearch();
      return;
    case 'clear':
      state.selection.clear();
      renderTable();
      renderActions();
      return;
    case 'csv':
      void exportCsv();
      return;
    case 'plot':
      handOff({ type: 'osprey:navigate', panel: 'canopy', url: plotUrl(selectedNames()) });
      return;
    case 'send':
      handOff({ type: 'osprey-paste-to-terminal', text: selectedNames().join(' ') });
      return;
    default:
  }
  // A click on a row, away from its links and its box, selects it.
  const row = /** @type {HTMLElement|null} */ (target.closest('tbody tr'));
  if (row && !target.closest('a, input')) {
    toggleRow(row.dataset.name ?? '');
  }
}

/** @param {Event} event */
function onChange(event) {
  if (!view?.answer) return;
  const box = event.target instanceof HTMLInputElement ? event.target : null;
  if (!box) return;
  if (box.dataset.action === 'select-page') {
    for (const row of view.answer.rows) {
      if (box.checked) state.selection.add(row.name);
      else state.selection.delete(row.name);
    }
    renderTable();
    renderActions();
    return;
  }
  const name = /** @type {HTMLElement|null} */ (box.closest('tbody tr'))?.dataset.name;
  if (name) toggleRow(name);
}

/**
 * Select or deselect a row of the current page; a selected row joins the end
 * of the selection.
 * @param {string} name
 */
function toggleRow(name) {
  if (!view?.answer) return;
  if (state.selection.has(name)) {
    state.selection.delete(name);
  } else {
    if (!view.answer.rows.some((row) => row.name === name)) return;
    state.selection.add(name);
  }
  view.csvStatus = '';
  renderTable();
  renderActions();
}

/**
 * @param {FacetKey} key
 * @param {string} value
 */
function toggleFacet(key, value) {
  if (!view || !Object.hasOwn(view.filters, key)) return;
  const set = view.filters[key];
  if (set.has(value)) set.delete(value);
  else set.add(value);
  restart();
}

/**
 * Post a hand-off to the host, addressed to this page's own origin.
 * @param {{ type: string, [key: string]: unknown }} message
 */
function handOff(message) {
  if (!view || !state.selection.size || !handoffAllowed()) return;
  if (message.type === 'osprey:navigate' && !message.url) return;
  window.parent.postMessage(message, window.location.origin);
}

/** The search as CSV, downloaded, or the export's reason beside the actions. */
async function exportCsv() {
  if (!view) return;
  const current = view;
  const words = state.scopeQuery.trim();
  current.csvStatus = '';
  /** @type {Response} */
  let resp;
  try {
    resp = await fetch(directoryExportUrl(currentFilters()));
  } catch {
    if (view !== current) return;
    current.csvStatus = 'CSV: export did not answer';
    renderActions();
    return;
  }
  if (view !== current) return;
  if (!resp.ok) {
    const text = await resp.text();
    /** @type {unknown} */
    let detail = null;
    try {
      detail = JSON.parse(text)?.detail;
    } catch {
      detail = null;
    }
    current.csvStatus =
      typeof detail === 'string' ? `CSV: ${detail}` : `CSV: export did not answer (${resp.status})`;
    renderActions();
    return;
  }
  const text = await resp.text();
  const url = URL.createObjectURL(new Blob([text], { type: 'text/csv' }));
  const link = document.createElement('a');
  link.href = url;
  link.download = `${words.replace(/[^\w.-]+/g, '_') || 'pvs'}.csv`;
  link.hidden = true;
  document.body.append(link);
  link.click();
  link.remove();
  // Revoked once the browser has taken the download, not before.
  setTimeout(() => URL.revokeObjectURL(url), 1000);
  renderActions();
}
