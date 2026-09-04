// @ts-check
/**
 * OSPREY Channel Finder — Graph Explore (the facility finder).
 *
 * The graph paradigm has no channel database to browse, so Explore is a
 * search: a query box, a facet rail built from the store's own vocabulary, a
 * page of matching channels, and a device card for the row an operator drills
 * into. The panel also names its provenance — the store the answers came from
 * and the tools the assistant queries it with — so a stale corpus can be told
 * from a live one at a glance.
 *
 * Three states, one panel: the finder, an informational pane for a store that
 * is reachable but unseeded, and the same pane carrying the store's own remedy
 * when a read failed. None of them is styled as an error.
 *
 * This module is the only owner of the DOM and of the network in graph mode.
 * The state it searches with is a plain object from graph-finder-state.js and
 * every string it draws comes from graph-finder-render.js; both are pure, so
 * what is left here is the fetch lifecycle, the event delegation, and the one
 * shape the server sends that the render module does not take directly (see
 * {@link buildForest}).
 *
 * The endpoints are read with `fetch` rather than `api.js`'s `fetchJSON`: the
 * graph routes carry `error_type` and `suggestions` beside `detail`, and
 * `fetchJSON` keeps only `detail`.
 *
 * Every request kind carries its own abort controller and its own generation
 * counter. A reply is applied only when it is still the newest of its kind and
 * the panel it was asked for is still mounted, so a slow answer to an abandoned
 * query can never overwrite a faster answer to the current one.
 */

import { state } from './state.js';
import { esc, messageOf } from './utils.js';
import { refreshStatsBadges } from './stats-badges.js';
import { showToast } from './app.js';
import { copyText as writeClipboard } from '/design-system/js/clipboard.js';
import { isEmbedded } from '/design-system/js/frame-params.js';
import {
  activeChips,
  clampPage,
  clearSelection,
  copyText,
  createFinderState,
  removeChip,
  sendText,
  setQuery,
  toggleFacet,
  togglePageSelection,
  toggleSelection,
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
} from './graph-finder-render.js';

/** @typedef {import('./graph-finder-state.js').FinderState} FinderState */
/** @typedef {import('./graph-finder-state.js').Facet} Facet */

/**
 * One class of the store's taxonomy while the forest is being built. `parents`
 * is what the store declared; `parent` is the one it is drawn under.
 * @typedef {object} ClassNode
 * @property {string} uri
 * @property {string} name
 * @property {string[]} parents
 * @property {string|null} parent
 * @property {boolean} abstract
 * @property {ClassNode[]} children
 */

/**
 * A settled request: the payload, a failure carrying the store's own remedy, or
 * `null` when the request was abandoned because the view moved on.
 * @typedef {{ok: true, data: any}
 *   | {ok: false, status: number, detail: string, errorType: string, suggestions: string[]}
 *   | null} Reply
 */

/** One request kind's in-flight controller and its generation counter. */
/** @typedef {{controller: AbortController|null, generation: number}} Flight */

const ONTOLOGY_PATH = '/api/graph/ontology';
const SEARCH_PATH = '/api/graph/search';
const DEVICE_PATH = '/api/graph/device';

/** How long typing settles before the query is sent, in ms. */
const SEARCH_DEBOUNCE_MS = 200;

const SEARCH_INPUT_ID = 'graph-finder-q';
const COUNT_ID = 'graph-finder-count';
const CHIPS_ID = 'graph-finder-chips';
const RAIL_ID = 'graph-finder-rail';
const CAUTION_ID = 'graph-finder-caution';
const CARD_ID = 'graph-finder-card';
const TABLE_ID = 'graph-finder-table';

const LOADING_HTML =
  '<div class="loading-center"><div class="loading-spinner"></div> Searching the facility graph&hellip;</div>';
const EMPTY_TITLE = 'The graph store is reachable, but holds no facility corpus yet.';
const PANEL_TITLE = 'Facility graph';
const PANEL_SUBTITLE = 'Search the graph store for devices and the channels bound to them.';
const SEARCH_PLACEHOLDER = 'Search devices, addresses and descriptions';
const TRUNCATED_TEXT =
  'The store holds more values than this query returned, so the rail below is a '
  + 'partial view of its vocabulary.';

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

/** The device card's markup, or '' when no card is open. */
let cardHtml = '';

/** Whether an assistant session is hosting the panel and can be sent to. */
let embedded = false;

/** Pending debounce for the query box. */
/** @type {number|undefined} */
let debounceTimer;

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
  embedded = isEmbedded() && window.parent !== window;
  content.innerHTML = shellHtml();
  bindEvents(content);
  await load();
}

/**
 * Tear the panel down: abandon every in-flight request and clear the pane.
 * Safe to call when nothing is mounted.
 * @returns {void}
 */
export function unmountGraph() {
  for (const kind of /** @type {const} */ (['ontology', 'search', 'device'])) abort(kind);
  window.clearTimeout(debounceTimer);
  debounceTimer = undefined;
  if (paneEl) paneEl.innerHTML = '';
  paneEl = null;
  results = null;
  cardHtml = '';
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

// ---------------------------------------------------------------------------
// Requests
// ---------------------------------------------------------------------------

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

/**
 * Read one graph endpoint, keeping the remedy fields a failure carries.
 *
 * @param {string} url - The url to read.
 * @param {AbortSignal} signal - Abort signal for the request.
 * @returns {Promise<Reply>} The payload, a failure, or null if aborted.
 */
async function fetchJson(url, signal) {
  try {
    const resp = await fetch(url, { signal });
    const body = await resp.json().catch(() => ({}));
    if (!resp.ok) {
      return {
        ok: false,
        status: Number(resp.status) || 0,
        detail: String(body.detail || resp.statusText || 'The graph store did not answer.'),
        errorType: String(body.error_type || ''),
        suggestions: stringList(body.suggestions),
      };
    }
    return { ok: true, data: body };
  } catch (e) {
    if (e instanceof Error && e.name === 'AbortError') return null;
    return { ok: false, status: 0, detail: messageOf(e), errorType: '', suggestions: [] };
  }
}

/**
 * Coerce an untyped payload field into a list of display strings.
 * @param {unknown} value - The field as the server sent it.
 * @returns {string[]} The strings it held, or an empty list.
 */
function stringList(value) {
  return Array.isArray(value) ? value.map((item) => String(item)) : [];
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
// Panel shell
// ---------------------------------------------------------------------------

/**
 * The panel chrome: title, provenance badge, subtitle with the tool chips, and
 * the body every later render replaces.
 * @returns {string} Markup for the panel.
 */
function shellHtml() {
  return `
    <div class="graph-panel" data-pipeline="graph">
      <div class="graph-panel-head">
        <div class="graph-panel-title">${esc(PANEL_TITLE)}</div>
        ${storeBadgeHtml()}
        <div class="graph-panel-sub">${esc(PANEL_SUBTITLE)}${toolChipsHtml()}</div>
      </div>
      <div class="graph-panel-body">${LOADING_HTML}</div>
    </div>
  `;
}

/**
 * Name the store behind these answers. Both halves are optional: a store seeded
 * from a TTL file is named `file @ uri`, one seeded another way by its URI
 * alone. A missing half is left out rather than printed as an empty word.
 * @returns {string} Markup for the badge, or '' when nothing is known.
 */
function storeBadgeHtml() {
  const store = state.graphStore;
  const uri = store && store.uri ? store.uri : '';
  const file = store && store.ttl_filename ? store.ttl_filename : '';
  const label = file && uri ? `${file} @ ${uri}` : (file || uri);
  if (!label) return '';
  return `<div class="graph-store-badge" title="Graph store"><code>${esc(label)}</code></div>`;
}

/**
 * The tools the assistant reads this same store with, so the reader knows the
 * panel and the agent are looking at one corpus.
 * @returns {string} Markup for the chips, or '' when none are reported.
 */
function toolChipsHtml() {
  const tools = state.tools || [];
  if (tools.length === 0) return '';
  const chips = tools.map((tool) => `<span class="graph-tool-chip">${esc(tool)}</span>`).join(' ');
  return ` The assistant queries it with ${chips}`;
}

/** @returns {Element|null} The panel body, or null when nothing is mounted. */
function bodyEl() {
  return paneEl ? paneEl.querySelector('.graph-panel-body') : null;
}

/**
 * One element of the mounted panel, by id.
 * @param {string} id - The element id.
 * @returns {HTMLElement|null} The element, or null when the finder is not drawn.
 */
function part(id) {
  return paneEl ? /** @type {HTMLElement|null} */ (paneEl.querySelector(`#${id}`)) : null;
}

// ---------------------------------------------------------------------------
// Rendering
// ---------------------------------------------------------------------------

/**
 * The finder's skeleton. The query box lives here rather than in a re-rendered
 * region: replacing an input while it has focus would drop the caret mid-word.
 * @returns {string} Markup for the search row, the rail and the results column.
 */
function skeletonHtml() {
  return `
    <div class="finder-search">
      <input type="search" id="${SEARCH_INPUT_ID}" placeholder="${esc(SEARCH_PLACEHOLDER)}"
             aria-label="${esc(SEARCH_PLACEHOLDER)}" autocomplete="off" spellcheck="false">
      <span class="finder-count" id="${COUNT_ID}"></span>
    </div>
    <div class="active-filters" id="${CHIPS_ID}"></div>
    <div class="finder-layout">
      <div class="facet-rail" id="${RAIL_ID}"></div>
      <div class="finder-results">
        <div id="${CAUTION_ID}"></div><div id="${CARD_ID}"></div><div id="${TABLE_ID}"></div>
      </div>
    </div>
  `;
}

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

  if (!part(TABLE_ID)) {
    body.innerHTML = skeletonHtml();
    const input = /** @type {HTMLInputElement|null} */ (part(SEARCH_INPUT_ID));
    if (input) input.value = finder.q;
  }

  const rail = part(RAIL_ID);
  if (rail) rail.innerHTML = facetRailHtml(results.facets || {}, ontology.tree, finder);

  const chips = part(CHIPS_ID);
  if (chips) chips.innerHTML = chipsHtml(activeChips(finder));

  const count = part(COUNT_ID);
  if (count) count.innerHTML = `<strong>${fmt(results.total)}</strong> channels`;

  const caution = part(CAUTION_ID);
  if (caution) {
    caution.innerHTML = ontology.truncated || results.truncated === true
      ? `<div class="graph-truncated">${esc(TRUNCATED_TEXT)}</div>`
      : '';
  }

  renderCard();
  renderTable();
}

/**
 * Redraw the rows and the footer. Called on its own after a selection change,
 * which moves no other part of the page.
 * @returns {void}
 */
function renderTable() {
  const table = part(TABLE_ID);
  if (!table || !results) return;
  table.innerHTML = resultsHtml(results.rows || [], finder)
    + footerHtml(
      results.total,
      results.devices,
      finder.page,
      results.pages,
      finder.selected.size,
      embedded,
    );
}

/** Redraw the device card region from `cardHtml`. @returns {void} */
function renderCard() {
  const card = part(CARD_ID);
  if (card) card.innerHTML = cardHtml;
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

/**
 * Pluralise the word an address count is reported with.
 * @param {number} n - How many addresses.
 * @returns {string} `"1 address"` or `"N addresses"`.
 */
function addresses(n) {
  return `${fmt(n)} ${n === 1 ? 'address' : 'addresses'}`;
}

// ---------------------------------------------------------------------------
// Payload shapes the render module does not take directly
// ---------------------------------------------------------------------------

/**
 * Build the class forest the rail draws from the flat taxonomy the ontology
 * endpoint answers with.
 *
 * The endpoint sends one row per class naming its parents by URI; the rail
 * needs those rows nested. A class with several known parents is attached under
 * the parent whose URI sorts first, parents absent from the payload are
 * ignored, and a `parents` cycle is broken by dropping the link that closes it,
 * so a corpus with a cyclic taxonomy still draws rather than recursing forever.
 * Siblings are ordered by name, URI breaking a tie, so the rail is stable
 * across reloads.
 *
 * @param {unknown} classes - The `classes` block of an ontology payload.
 * @returns {ClassNode[]} The roots, each carrying its children.
 */
function buildForest(classes) {
  /** @type {Map<string, ClassNode>} */
  const nodes = new Map();
  for (const cls of Array.isArray(classes) ? classes : []) {
    const uri = cls && cls.uri != null ? String(cls.uri) : '';
    if (!uri || nodes.has(uri)) continue;
    const parents = Array.isArray(cls.parents) ? cls.parents.map(String) : [];
    const name = cls.name ? String(cls.name) : uri;
    const abstract = Boolean(cls.abstract);
    nodes.set(uri, { uri, name, parents, parent: null, abstract, children: [] });
  }

  for (const node of nodes.values()) {
    const known = node.parents.filter((uri) => uri !== node.uri && nodes.has(uri)).sort();
    node.parent = known.length > 0 ? known[0] : null;
  }

  // Break a cycle at a class that is ON it: the walk is only allowed to demote
  // a node it returns to, so a class that merely descends from a cycle keeps
  // its parent. Every cycle still breaks, because each of its members closes on
  // itself, and once the first one is demoted the rest walk out to a root.
  for (const node of nodes.values()) {
    const seen = new Set([node.uri]);
    let cursor = node.parent;
    while (cursor && !seen.has(cursor)) {
      seen.add(cursor);
      cursor = nodes.get(cursor)?.parent ?? null;
    }
    if (cursor === node.uri) node.parent = null;
  }

  /** @type {ClassNode[]} */
  const roots = [];
  for (const node of nodes.values()) {
    const parent = node.parent ? nodes.get(node.parent) : undefined;
    if (parent) parent.children.push(node);
    else roots.push(node);
  }

  /** @param {ClassNode[]} siblings @returns {void} */
  const order = (siblings) => {
    siblings.sort((a, b) => (a.name === b.name ? cmp(a.uri, b.uri) : cmp(a.name, b.name)));
    for (const node of siblings) order(node.children);
  };
  order(roots);
  return roots;
}

/**
 * Compare two strings for a stable sort.
 * @param {string} a - First string.
 * @param {string} b - Second string.
 * @returns {number} Negative, zero or positive per `Array.prototype.sort`.
 */
function cmp(a, b) {
  return a === b ? 0 : (a < b ? -1 : 1);
}

// ---------------------------------------------------------------------------
// Interaction
// ---------------------------------------------------------------------------

/**
 * Wire the panel's one click, change and input listener. They sit on the pane,
 * which outlives every re-render inside it, so nothing has to be re-bound.
 *
 * @param {HTMLElement} content - The mounted pane.
 * @returns {void}
 */
function bindEvents(content) {
  content.addEventListener('click', onClick);
  content.addEventListener('change', onChange);
  content.addEventListener('input', onInput);
}

/**
 * Typing settles for {@link SEARCH_DEBOUNCE_MS} before it becomes a request,
 * so a typed word costs one search rather than one per keystroke.
 * @param {Event} event - The input event.
 * @returns {void}
 */
function onInput(event) {
  const target = event.target;
  if (!(target instanceof HTMLInputElement) || target.id !== SEARCH_INPUT_ID) return;
  const value = target.value;
  window.clearTimeout(debounceTimer);
  debounceTimer = window.setTimeout(() => {
    setQuery(finder, value);
    void runSearch();
  }, SEARCH_DEBOUNCE_MS);
}

/**
 * Selection: one row, or every row on the page. Neither refetches — the
 * selection is the operator's, not the server's.
 * @param {Event} event - The change event.
 * @returns {void}
 */
function onChange(event) {
  const target = event.target;
  if (!(target instanceof HTMLInputElement)) return;

  if (target.hasAttribute('data-select-page')) {
    togglePageSelection(finder, pagePvs(), target.checked);
    renderTable();
    return;
  }

  const pv = target.getAttribute('data-pv');
  if (pv !== null && target.type === 'checkbox') {
    toggleSelection(finder, pv);
    renderTable();
  }
}

/**
 * Every click the finder answers, in the order the render module's contract
 * lists them.
 * @param {Event} event - The click event.
 * @returns {void}
 */
function onClick(event) {
  const target = event.target;
  if (!(target instanceof Element)) return;

  const facet = target.closest('button.facet-item[data-facet][data-value]');
  if (facet) {
    const name = /** @type {Facet} */ (facet.getAttribute('data-facet'));
    toggleFacet(finder, name, facet.getAttribute('data-value') || '');
    void runSearch();
    return;
  }

  const chip = target.closest('button.active-filter[data-chip]');
  if (chip) {
    if (removeChip(finder, chip.getAttribute('data-chip') || '')) void runSearch();
    return;
  }

  const copy = target.closest('button.copy-btn[data-copy]');
  if (copy) return void copyOne(copy.getAttribute('data-copy') || '');

  const dev = target.closest('button.dev[data-uri]');
  if (dev) return void openDevice(dev.getAttribute('data-uri') || '');

  const pager = target.closest('[data-page]');
  if (pager) return turnPage(pager.getAttribute('data-page'));

  const action = target.closest('[data-action]');
  if (action) runAction(action.getAttribute('data-action'));
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
 * Run one footer or card action.
 * @param {string|null} action - `copy`, `send`, `clear` or `close-card`.
 * @returns {void}
 */
function runAction(action) {
  if (action === 'copy') void copySelection();
  else if (action === 'send') sendSelection();
  else if (action === 'clear') {
    clearSelection(finder);
    renderTable();
  } else if (action === 'close-card') closeCard();
}

/** @returns {string[]} The addresses drawn on the current page. */
function pagePvs() {
  const rows = (results && Array.isArray(results.rows)) ? results.rows : [];
  return rows.map((/** @type {any} */ row) => String(row.fullPv ?? ''));
}

/**
 * Copy the whole selection, newline-joined — the form an editor or a script
 * takes.
 * @returns {Promise<void>} Resolves once the toast has been raised.
 */
async function copySelection() {
  const count = finder.selected.size;
  if (count === 0) return;
  const ok = await writeClipboard(copyText(finder));
  if (ok) showToast(`Copied ${addresses(count)}`, 'success');
  else showToast('Copy failed — select the table and copy manually', 'error');
}

/**
 * Copy one address from a row or from the device card.
 * @param {string} pv - The address to copy.
 * @returns {Promise<void>} Resolves once the toast has been raised.
 */
async function copyOne(pv) {
  if (!pv) return;
  const ok = await writeClipboard(pv);
  if (ok) showToast(`Copied ${pv}`, 'success');
  else showToast('Copy failed — select the address and copy manually', 'error');
}

/**
 * Post the selection to the assistant prompt, space-joined on one line.
 *
 * The message is addressed to this page's own origin rather than to `*`, so a
 * host on another origin cannot be handed the operator's selection.
 * @returns {void}
 */
function sendSelection() {
  const count = finder.selected.size;
  if (count === 0 || !embedded) return;
  window.parent.postMessage(
    { type: 'osprey-paste-to-terminal', text: sendText(finder) },
    window.location.origin,
  );
  showToast(`Posted ${addresses(count)} to the prompt`, 'success');
}

/**
 * Open the card for one device, replacing whatever card is open.
 *
 * A URI the store no longer holds draws the error card and leaves the finder
 * standing under it. A read the store could not serve at all goes to the
 * panel's informational pane instead, which is the only place with room for
 * the remedy such a failure carries.
 *
 * @param {string} uri - The device URI, as a row carries it.
 * @returns {Promise<void>} Resolves once the card has been drawn.
 */
async function openDevice(uri) {
  if (!uri) return;
  const reply = await request('device', `${DEVICE_PATH}?uri=${encodeURIComponent(uri)}`);
  if (reply === null) return;
  if (reply.ok) cardHtml = deviceCardHtml(reply.data);
  else if (reply.errorType === 'not_found') cardHtml = deviceCardErrorHtml(reply.detail);
  else return renderInfo(reply.detail, reply.suggestions);
  renderCard();
}

/** Close the device card and abandon a lookup still in flight. @returns {void} */
function closeCard() {
  abort('device');
  cardHtml = '';
  renderCard();
}
