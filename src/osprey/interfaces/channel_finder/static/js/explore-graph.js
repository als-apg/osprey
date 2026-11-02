// @ts-check
/**
 * OSPREY Channel Finder — Graph Explore (the facility finder).
 *
 * The graph paradigm has no channel database to browse, so Explore is a
 * search: a toolbar with the query box, a facet rail built from the store's
 * own vocabulary, a page of matching channels, and a device inspector for the
 * row an operator drills into. It is laid out like the fleet's other panels —
 * full height, the rail and the inspector as panes behind the shared splitter,
 * a right-click menu on every surface that has verbs — and the store the
 * answers came from is named in the corpus strip below it (stats-badges.js).
 *
 * Three states, one panel: the finder, an informational pane for a store that
 * is reachable but unseeded, and the same pane carrying the store's own remedy
 * when a read failed. None of them is styled as an error.
 *
 * This module owns the finder's state, its network lifecycle and its DOM
 * regions. Around it, each piece is pure or nearly so:
 *
 *   graph-finder-state.js   what the operator asks for (query, facets, page, sort, selection)
 *   graph-finder-render.js  every string drawn
 *   graph-finder-menus.js   which verbs each right-click menu offers
 *   graph-finder-verbs.js   the verbs, and the hub tile-bar contribution
 *   graph-finder-events.js  which input reaches which verb
 *   graph-finder-prefs.js   the column layout and view preferences kept per render
 *   graph-finder-data.js    reading an endpoint, and the class forest
 *
 * The verbs and the event handlers reach this module only through the
 * {@link FinderHost} built in {@link mountGraph}: a narrow, live view of the
 * state and the redraws, so none of them holds state of its own.
 *
 * Every request kind carries its own abort controller and its own generation
 * counter. A reply is applied only when it is still the newest of its kind and
 * the panel it was asked for is still mounted, so a slow answer to an abandoned
 * query can never overwrite a faster answer to the current one.
 *
 * Embedded in the web terminal the toolbar's search box and ⋯ menu move into
 * the hub's tile bar (graph-finder-verbs.js contributeHeader →
 * state.setViewHeader → app.js), so the panel body starts at the filters like
 * every other tile; only standalone keeps the in-body toolbar.
 */

import { state } from './state.js';
import { esc } from './utils.js';
import { refreshStatsBadges } from './stats-badges.js';
import { showToast } from './app.js';
import { copyText as writeClipboard } from '/design-system/js/clipboard.js';
import { isEmbedded } from '/design-system/js/frame-params.js';
import { initSplitter } from '/design-system/js/splitter.js';
import { scopedStorageKey } from '/design-system/js/storage-scope.js';
import {
  attachColumnGestures,
  moveColumn,
  resizeColumn,
  visibleColumns,
} from '/design-system/js/table-columns.js';
import {
  activeChips,
  clampPage,
  copyText,
  createFinderState,
  toSearchParams,
} from './graph-finder-state.js';
import {
  chipsHtml,
  deviceCardErrorHtml,
  deviceCardHtml,
  facetRailHtml,
  fmt,
  footerHtml,
  resultsHtml,
  FINDER_IDS,
  skeletonHtml,
} from './graph-finder-render.js';
import { buildForest, classNameIn, fetchJson, stringList } from './graph-finder-data.js';
import { contributeHeader, createActions } from './graph-finder-verbs.js';
import { bindFinderEvents } from './graph-finder-events.js';
import {
  COLUMN_SPECS,
  INSPECTOR_KEY,
  RAIL_KEY,
  loadColumns,
  loadPrefs,
  saveColumns,
} from './graph-finder-prefs.js';

/** @typedef {import('./graph-finder-state.js').FinderState} FinderState */
/** @typedef {import('./graph-finder-data.js').ClassNode} ClassNode */
/** @typedef {import('./graph-finder-data.js').Reply} Reply */
/** @typedef {import('./graph-finder-verbs.js').FinderHost} FinderHost */
/** @typedef {import('/design-system/js/splitter.js').SplitterApi} SplitterApi */

/** One request kind's in-flight controller and its generation counter. */
/** @typedef {{controller: AbortController|null, generation: number}} Flight */

const ONTOLOGY_PATH = '/api/graph/ontology';
const SEARCH_PATH = '/api/graph/search';
const DEVICE_PATH = '/api/graph/device';

/** Splitters only drive the side-by-side layout; a narrow viewport stacks. */
const WIDE_QUERY = '(min-width: 769px)';

const LOADING_HTML =
  '<div class="loading-center"><div class="loading-spinner"></div> Searching the facility graph&hellip;</div>';
const EMPTY_TITLE = 'The graph store is reachable, but holds no facility corpus yet.';
const TRUNCATED_TEXT =
  'The store holds more values than this query returned, so the rail below is a ' +
  'partial view of its vocabulary.';

/** The element the panel is mounted into, or null when nothing is mounted. */
/** @type {HTMLElement|null} */
let paneEl = null;

/** What the operator is currently asking for. */
/** @type {FinderState} */
let finder = createFinderState();

/** The store's device taxonomy, as the rail draws it. */
/** @type {{tree: ClassNode[], truncated: boolean}} */
let ontology = { tree: [], truncated: false };

/** The newest search payload, or null before the first one lands. */
/** @type {any} */
let results = null;

/** The inspector's markup, or '' when no device is open. */
let cardHtml = '';

/** The device the inspector shows: its URI and its payload (null on a miss). */
/** @type {{uri: string, data: any}|null} */
let card = null;

/** Whether an assistant session is hosting the panel and can be sent to. */
let embedded = false;

/** The column layout and view preferences the operator arranged. */
let layout = loadColumns();
let prefs = loadPrefs();

/** The filter rail's splitter, for the verbs that hide and show it. */
/** @type {SplitterApi|null} */
let railSplit = null;

/** Teardowns: the listeners of the mount, and what the skeleton wires. */
/** @type {(() => void)|null} */
let unbindEvents = null;
/** @type {Array<() => void>} */
let skeletonTeardown = [];

/**
 * One flight per request kind. A new request of a kind aborts the previous one
 * and takes the next generation; a reply from an older generation is dropped.
 * @type {Record<'ontology'|'search'|'device', Flight>}
 */
const flights = {
  ontology: { controller: null, generation: 0 },
  search: { controller: null, generation: 0 },
  device: { controller: null, generation: 0 },
};

/** The mount as the verbs and handlers see it. @type {FinderHost} */
const host = {
  finder: () => finder,
  results: () => results,
  card: () => card,
  layout: () => layout,
  setLayout: (next) => {
    layout = next;
  },
  prefs: () => prefs,
  embedded: () => embedded,
  railSplit: () => railSplit,
  pane: () => paneEl,
  part,
  searchInputId: FINDER_IDS.search,
  runSearch,
  renderTable,
  renderRail,
  commitColumns,
  toggleDevice,
  openDevice,
  closeCard,
  copy: copyAndSay,
  send: sendAddresses,
  copySelection: () => {
    if (finder.selected.size) void copyAndSay(copyText(finder), addresses(finder.selected.size));
  },
  turnPage,
};

// ---------------------------------------------------------------------------
// Mount / unmount
// ---------------------------------------------------------------------------

/**
 * Render the finder into `content` and load the ontology and the first page.
 *
 * @param {HTMLElement} content - The pane the explore dispatcher owns.
 * @returns {Promise<void>} Resolves once the first render has settled.
 */
export async function mountGraph(content) {
  // A previous mount may still hold a device fetch or a pending debounce.
  // Unmount is a no-op when nothing is mounted, and it clears both.
  unmountGraph();
  paneEl = content;
  finder = createFinderState();
  ontology = { tree: [], truncated: false };
  layout = loadColumns();
  prefs = loadPrefs();
  embedded = isEmbedded() && window.parent !== window;
  content.innerHTML = `
    <div class="graph-explorer${prefs.density === 'compact' ? ' compact' : ''}" data-pipeline="graph">
      <div class="graph-explorer-body">${LOADING_HTML}</div>
    </div>`;
  unbindEvents = bindFinderEvents(content, host);
  contributeHeader(host);
  await load();
}

/**
 * Tear the panel down: abandon every in-flight request and clear the pane.
 * Safe to call when nothing is mounted.
 * @returns {void}
 */
export function unmountGraph() {
  for (const kind of /** @type {const} */ (['ontology', 'search', 'device'])) abort(kind);
  teardownSkeleton();
  unbindEvents?.();
  unbindEvents = null;
  if (paneEl) {
    paneEl.innerHTML = '';
    if (state.viewHeader) state.setViewHeader(null);
  }
  paneEl = null;
  results = null;
  cardHtml = '';
  card = null;
}

/**
 * Abort the in-flight request of one kind, if there is one, and retire its
 * generation so a reply already on its way is discarded.
 *
 * @param {'ontology'|'search'|'device'} kind - Which request to abandon.
 * @returns {void}
 */
function abort(kind) {
  const flight = flights[kind];
  flight.controller?.abort();
  flight.controller = null;
  flight.generation += 1;
}

/**
 * Run one request of `kind`, superseding whatever else that kind had running.
 *
 * @param {'ontology'|'search'|'device'} kind - Which request this is.
 * @param {string} url - The url to read.
 * @returns {Promise<Reply>} The reply, or null when it no longer applies.
 */
async function request(kind, url) {
  abort(kind);
  const flight = flights[kind];
  const mine = flight.generation;
  const pane = paneEl;
  const controller = new AbortController();
  flight.controller = controller;

  const reply = await fetchJson(url, controller.signal);

  // A newer request of this kind, or an unmount, happened while this one was
  // in flight.
  if (mine !== flight.generation || paneEl === null || paneEl !== pane) return null;
  flight.controller = null;
  return reply;
}

/** @returns {string} The search url for the current state. */
function searchUrl() {
  return `${SEARCH_PATH}?${toSearchParams(finder).toString()}`;
}

// ---------------------------------------------------------------------------
// Loading
// ---------------------------------------------------------------------------

/**
 * Load the taxonomy and the first page together, and render whichever of the
 * three states they imply. The two reads are independent, so they are asked
 * for at once rather than one after the other.
 *
 * @returns {Promise<void>} Resolves once the body has been rendered.
 */
async function load() {
  const body = bodyEl();
  if (!body) return;
  teardownSkeleton();
  body.innerHTML = LOADING_HTML;

  const [ontReply, searchReply] = await Promise.all([
    request('ontology', ONTOLOGY_PATH),
    request('search', searchUrl()),
  ]);
  if (ontReply === null || searchReply === null) return;

  // Either read failing leaves the panel with nothing honest to draw, so the
  // whole panel reports it and offers the retry.
  if (!ontReply.ok) return renderInfo(ontReply.detail, ontReply.suggestions);
  if (!searchReply.ok) return renderInfo(searchReply.detail, searchReply.suggestions);

  ontology = {
    tree: buildForest(ontReply.data && ontReply.data.classes),
    truncated: ontReply.data && ontReply.data.truncated === true,
  };
  if (ontReply.data && ontReply.data.empty === true && searchReply.data.empty !== true) {
    return renderInfo(EMPTY_TITLE, stringList(ontReply.data.suggestions));
  }
  await applySearch(searchReply.data);
}

/**
 * Re-run the search for the current state and draw the answer.
 * @returns {Promise<void>} Resolves once the answer has been rendered.
 */
async function runSearch() {
  const reply = await request('search', searchUrl());
  if (reply === null) return;
  if (!reply.ok) return renderInfo(reply.detail, reply.suggestions);
  await applySearch(reply.data);
}

/**
 * Take a search payload as the current result, clamping a page the filters
 * have narrowed away. A clamped page addresses a different set of rows than
 * the one just fetched, so it is fetched again rather than drawn.
 *
 * @param {any} data - The search payload.
 * @returns {Promise<void>} Resolves once the answer has been rendered.
 */
async function applySearch(data) {
  if (data && data.empty === true) {
    results = null;
    return renderInfo(EMPTY_TITLE, stringList(data.suggestions));
  }
  results = data;
  if (clampPage(finder, Number(data && data.pages) || 0)) {
    await runSearch();
    return;
  }
  renderFinder();
}

// ---------------------------------------------------------------------------
// Regions
// ---------------------------------------------------------------------------

/** @returns {Element|null} The panel body, or null when nothing is mounted. */
function bodyEl() {
  return paneEl ? paneEl.querySelector('.graph-explorer-body') : null;
}

/**
 * One element of the mounted panel, by id.
 * @param {string} id - The element id.
 * @returns {HTMLElement|null} The element, or null when the finder is not drawn.
 */
function part(id) {
  return paneEl ? /** @type {HTMLElement|null} */ (paneEl.querySelector(`#${id}`)) : null;
}

/**
 * Wire what the skeleton carries: the two splitters and the column gestures.
 * Called each time the skeleton is (re)written, after any previous wiring is
 * torn down.
 * @returns {void}
 */
function wireSkeleton() {
  teardownSkeleton();
  const wide = () => window.matchMedia?.(WIDE_QUERY).matches ?? true;
  const rail = initSplitter({
    handle: part(FINDER_IDS.railSplit),
    pane: part(FINDER_IDS.rail),
    storageKey: scopedStorageKey(RAIL_KEY),
    min: 160,
    max: 480,
    collapsedSize: 0,
    isEnabled: wide,
    onCommit: () => contributeHeader(host),
  });
  const inspector = initSplitter({
    handle: part(FINDER_IDS.cardSplit),
    pane: part(FINDER_IDS.card),
    storageKey: scopedStorageKey(INSPECTOR_KEY),
    anchor: 'end',
    min: 260,
    max: () => Math.max(320, Math.round((paneEl?.clientWidth || 1200) * 0.6)),
    isEnabled: wide,
  });
  if (wide()) {
    rail.restoreSize();
    inspector.restoreSize();
  }
  railSplit = rail;

  const table = part(FINDER_IDS.table);
  const detach = table
    ? attachColumnGestures(table, {
        onResizing: (key, width) => {
          resizeColumn(COLUMN_SPECS, layout, key, width);
          applyWidths();
        },
        onResize: (key, width) => {
          resizeColumn(COLUMN_SPECS, layout, key, width);
          saveColumns(layout);
          applyWidths();
        },
        onMove: (key, target, before) => {
          if (moveColumn(layout, key, target, before)) commitColumns();
        },
        onAutosize: (key) => createActions(host).autosize(key),
      })
    : () => {};
  skeletonTeardown = [() => rail.destroy(), () => inspector.destroy(), detach];
}

/** Undo {@link wireSkeleton}. Safe when nothing is wired. @returns {void} */
function teardownSkeleton() {
  for (const fn of skeletonTeardown) fn();
  skeletonTeardown = [];
  railSplit = null;
}

// ---------------------------------------------------------------------------
// Rendering
// ---------------------------------------------------------------------------

/**
 * Draw the finder for the current state and the newest search payload.
 *
 * The skeleton is written once and then only its regions are replaced, so the
 * query box keeps its focus and its caret while results stream in behind it.
 * @returns {void}
 */
function renderFinder() {
  const body = bodyEl();
  if (!body || !results) return;

  if (!part(FINDER_IDS.table)) {
    body.innerHTML = skeletonHtml(prefs.density);
    const input = /** @type {HTMLInputElement|null} */ (part(FINDER_IDS.search));
    if (input) input.value = finder.q;
    wireSkeleton();
  }

  renderRail();

  const chips = part(FINDER_IDS.chips);
  if (chips) {
    // A class chip is named by its class, not its URI.
    const list = activeChips(finder).map((chip) =>
      chip.key.startsWith('cls:')
        ? { ...chip, label: `Device class: ${classNameIn(ontology.tree, chip.key.slice(4))}` }
        : chip
    );
    chips.innerHTML =
      chipsHtml(list) +
      (list.length > 1
        ? '<button type="button" class="clear-all-filters" data-clear-all>Clear all</button>'
        : '');
  }

  const count = part(FINDER_IDS.count);
  if (count) count.innerHTML = `<strong>${fmt(results.total)}</strong> channels`;

  const caution = part(FINDER_IDS.caution);
  if (caution) {
    caution.innerHTML =
      ontology.truncated || results.truncated === true
        ? `<div class="graph-truncated">${esc(TRUNCATED_TEXT)}</div>`
        : '';
  }

  renderCard();
  renderTable();
  contributeHeader(host);
}

/** Redraw the facet rail. @returns {void} */
function renderRail() {
  const rail = part(FINDER_IDS.rail);
  if (rail && results)
    rail.innerHTML = facetRailHtml(results.facets || {}, ontology.tree, finder, prefs.collapsed);
}

/**
 * Redraw the rows and the footer. Called on its own after a selection or a
 * column change, which moves no other part of the page. The row holding the
 * table's Tab stop keeps it, and keeps focus if it had it, so a keyboard
 * selection does not throw the operator out of the table.
 * @returns {void}
 */
function renderTable() {
  const table = part(FINDER_IDS.table);
  if (!table || !results) return;
  const active = document.activeElement;
  const hadFocus = active instanceof HTMLTableRowElement && table.contains(active);
  const stop =
    table.querySelector('tbody tr[data-pv][tabindex="0"]')?.getAttribute('data-pv') ?? null;
  table.innerHTML =
    resultsHtml(results.rows || [], finder, visibleColumns(COLUMN_SPECS, layout), stop) +
    footerHtml(
      results.total,
      results.devices,
      finder.page,
      results.pages,
      finder.selected.size,
      embedded
    );
  markOpenRows();
  if (hadFocus) {
    /** @type {HTMLElement|null} */ (
      table.querySelector('tbody tr[data-pv][tabindex="0"]')
    )?.focus();
  }
}

/**
 * Write the current widths onto the drawn table without redrawing it — the
 * per-frame half of an edge drag.
 * @returns {void}
 */
function applyWidths() {
  const table = /** @type {HTMLTableElement|null} */ (
    paneEl?.querySelector('.result-table') ?? null
  );
  if (!table) return;
  let floor = 0;
  for (const col of visibleColumns(COLUMN_SPECS, layout)) {
    floor += col.width;
    const el = table.querySelector(`col[data-col="${col.key}"]`);
    if (el instanceof HTMLElement && !el.hasAttribute('data-fill'))
      el.style.width = `${col.width}px`;
  }
  table.style.minWidth = `${floor}px`;
}

/** Persist the layout and redraw the table. @returns {void} */
function commitColumns() {
  saveColumns(layout);
  renderTable();
}

/** Mark the rows of the device the inspector shows. @returns {void} */
function markOpenRows() {
  for (const row of paneEl?.querySelectorAll('.result-table tbody tr[data-uri]') ?? []) {
    row.classList.toggle('open', card !== null && row.getAttribute('data-uri') === card.uri);
  }
}

/** Redraw the inspector region from `cardHtml`. @returns {void} */
function renderCard() {
  const region = part(FINDER_IDS.card);
  if (!region) return;
  region.innerHTML = cardHtml;
  region.hidden = cardHtml === '';
  const split = part(FINDER_IDS.cardSplit);
  if (split) split.hidden = cardHtml === '';
  markOpenRows();
}

/**
 * Render the informational pane the unseeded and the unreachable store share:
 * a headline and one line per remedy, with a Retry that re-asks the ontology,
 * the search and the header statistics — the reads that fail together when the
 * store is down, so they recover together.
 *
 * @param {string} detail - What happened, in the store's own words.
 * @param {string[]} suggestions - One remedy per line.
 * @returns {void}
 */
function renderInfo(detail, suggestions) {
  const body = bodyEl();
  if (!body) return;
  teardownSkeleton();
  body.innerHTML = '';
  const pane = document.createElement('div');
  pane.className = 'explore-unknown explore-unknown--info';

  const title = document.createElement('div');
  title.className = 'explore-unknown-title';
  title.textContent = detail;

  const text = document.createElement('div');
  text.className = 'explore-unknown-body';
  for (const suggestion of suggestions) {
    const line = document.createElement('div');
    line.textContent = suggestion;
    text.appendChild(line);
  }

  const retry = document.createElement('button');
  retry.className = 'btn btn-secondary btn-sm';
  retry.id = 'graph-retry';
  retry.type = 'button';
  retry.textContent = 'Retry';
  retry.addEventListener('click', () => {
    void refreshStatsBadges();
    void load();
  });
  text.appendChild(retry);

  pane.append(title, text);
  body.appendChild(pane);
}

// ---------------------------------------------------------------------------
// Effects the verbs ask for
// ---------------------------------------------------------------------------

/**
 * Pluralise the word an address count is reported with.
 * @param {number} n - How many addresses.
 * @returns {string} `"1 address"` or `"N addresses"`.
 */
function addresses(n) {
  return `${fmt(n)} ${n === 1 ? 'address' : 'addresses'}`;
}

/**
 * Step the pager, refusing a step past either end.
 * @param {string|null} direction - `prev` or `next`.
 * @returns {void}
 */
function turnPage(direction) {
  const pages = Math.max(1, Number(results && results.pages) || 1);
  const next = direction === 'prev' ? finder.page - 1 : finder.page + 1;
  if (next < 1 || next > pages) return;
  finder.page = next;
  void runSearch();
}

/**
 * Copy text and say what was copied.
 * @param {string} text - What goes on the clipboard.
 * @param {string} what - How the toast names it.
 * @returns {Promise<void>} Resolves once the toast has been raised.
 */
async function copyAndSay(text, what) {
  if (!text) return;
  const ok = await writeClipboard(text);
  if (ok) showToast(`Copied ${what}`, 'success');
  else showToast('Copy failed — select the text and copy manually', 'error');
}

/**
 * Post addresses to the assistant prompt, space-joined on one line.
 *
 * The message is addressed to this page's own origin rather than to `*`, so a
 * host on another origin cannot be handed the operator's selection.
 * @param {string[]} pvs - The addresses to send.
 * @returns {void}
 */
function sendAddresses(pvs) {
  if (pvs.length === 0 || !embedded) return;
  window.parent.postMessage(
    { type: 'osprey-paste-to-terminal', text: pvs.join(' ') },
    window.location.origin
  );
  showToast(`Posted ${addresses(pvs.length)} to the prompt`, 'success');
}

/**
 * Open a device in the inspector, or close it when it is the one shown.
 * @param {string} uri
 * @returns {Promise<void>}
 */
async function toggleDevice(uri) {
  if (card && card.uri === uri) {
    closeCard();
    return;
  }
  await openDevice(uri);
}

/**
 * Open the inspector for one device, replacing whatever it shows.
 *
 * A URI the store no longer holds draws the error card and leaves the finder
 * standing beside it. A read the store could not serve at all goes to the
 * panel's informational pane instead, which is the only place with room for
 * the remedy such a failure carries.
 *
 * @param {string} uri - The device URI, as a row carries it.
 * @returns {Promise<void>} Resolves once the inspector has been drawn.
 */
async function openDevice(uri) {
  if (!uri) return;
  const reply = await request('device', `${DEVICE_PATH}?uri=${encodeURIComponent(uri)}`);
  if (reply === null) return;
  if (reply.ok) {
    card = { uri, data: reply.data };
    cardHtml = deviceCardHtml(reply.data, { embedded });
  } else if (reply.errorType === 'not_found') {
    card = { uri, data: null };
    cardHtml = deviceCardErrorHtml(reply.detail);
  } else {
    return renderInfo(reply.detail, reply.suggestions);
  }
  renderCard();
}

/** Close the inspector and abandon a lookup still in flight. @returns {void} */
function closeCard() {
  abort('device');
  card = null;
  cardHtml = '';
  renderCard();
}
