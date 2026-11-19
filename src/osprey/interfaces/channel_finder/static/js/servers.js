// @ts-check
/**
 * OSPREY Channel Finder — IOCs
 *
 * The view of `#/servers` (Panel view contract): every IOC of the session's
 * control target with its state, from the fleet route alone. The route asks
 * the server fleet service for its last snapshot; opening or polling this view
 * reads no PV and queries no directory.
 *
 *   toolbar   [Curated | All PVs | IOCs | …] …………………… [N IOCs] [filter]
 *   header    snapshot taken <time> · <age> ago   read path   sources
 *   band      state tiles | finding tiles ………………… [Group by host] [refresh]
 *   table     IOC · Host · State · In state · Findings · Directory · Sources
 *
 * The rows are drawn in the served order; a tile and the toolbar's text only
 * remove rows. Every non-`ok` answer is a statement about the snapshot, with
 * no rows: none of them says an IOC is down. The view polls every 5 s while
 * mounted and the document is visible. Tile, text and grouping are view
 * state; the route carries no params.
 *
 * A part's markup is a function of the answer and the filters, never of the
 * clock: a part is rewritten only when its markup changed (keeping the focus
 * on the same control), and the ages are ticked in place.
 */

import { icon } from '/design-system/js/icons.js';
import { renderState } from '/design-system/js/panel-state.js';
import { fetchFleet } from './directory-api.js';
import {
  ageText,
  bandHtml,
  countHtml,
  filterRows,
  headerHtml,
  outcomeStatement,
  skeletonHtml,
  tableHtml,
} from './servers-render.js';
import { LABELS } from './labels.js';
import { claimToolbar, searchInput, setFilterReason, setResultCount } from './panel-toolbar.js';
import { messageOf } from './utils.js';

/** @typedef {import('./directory-api.js').FleetAnswer} FleetAnswer */
/** @typedef {import('./directory-api.js').FleetSnapshot} FleetSnapshot */
/** @typedef {import('./servers-render.js').TileFilter} TileFilter */

/** How often the fleet is read while the view is mounted and visible. */
export const POLL_MS = 5000;

/** The toolbar's sidebar toggle has nothing to show here. */
const noSidebar = () => `The ${LABELS.serversView} view has no sidebar`;

/**
 * The mounted view's state; null while nothing is mounted.
 * @typedef {{
 *   root: HTMLElement,
 *   snapshot: FleetSnapshot|null,
 *   tile: TileFilter|null,
 *   text: string,
 *   grouped: boolean,
 *   pollSeq: number,
 *   reading: number,
 *   timer: number|undefined,
 *   ticker: number|undefined,
 *   controller: AbortController,
 *   release: () => void,
 * }} View
 */

/** @type {View|null} */
let view = null;

/** The markup each part was last written with. @type {WeakMap<Element, string>} */
const written = new WeakMap();

/**
 * Write `html` into a part unless it already holds that markup; the focus,
 * when inside, moves to the same control in the new markup.
 * @param {HTMLElement} el
 * @param {string} html
 * @returns {boolean} whether the part was rewritten
 */
function writePart(el, html) {
  if (written.get(el) === html) return false;
  /** @type {string|null} */
  let selector = null;
  const active = document.activeElement;
  if (active instanceof HTMLElement && el.contains(active)) {
    const { tile, action } = active.dataset;
    const href = active.matches('a[href]') ? active.getAttribute('href') : null;
    if (tile !== undefined) selector = `[data-tile="${CSS.escape(tile)}"]`;
    else if (action !== undefined) selector = `[data-action="${CSS.escape(action)}"]`;
    else if (href !== null) selector = `a[href="${CSS.escape(href)}"]`;
  }
  el.innerHTML = html;
  written.set(el, html);
  if (selector) /** @type {HTMLElement|null} */ (el.querySelector(selector))?.focus();
  return true;
}

/**
 * Tick every age under `el`.
 * @param {ParentNode} el
 */
function tick(el) {
  const now = Date.now();
  for (const node of el.querySelectorAll('[data-age]')) {
    node.textContent = ageText(Number(/** @type {HTMLElement} */ (node).dataset.age), now);
  }
}

/** @param {HTMLElement} container */
export function mountServers(container) {
  unmountServers();
  container.innerHTML = skeletonHtml();
  const root = /** @type {HTMLElement} */ (container.querySelector('.servers'));
  const controller = new AbortController();
  const { signal } = controller;
  const release = claimToolbar();
  view = {
    root,
    snapshot: null,
    tile: null,
    text: '',
    grouped: false,
    pollSeq: 0,
    reading: 0,
    timer: undefined,
    ticker: undefined,
    controller,
    release,
  };
  root.addEventListener('click', onClick, { signal });
  searchInput()?.addEventListener(
    'input',
    (event) => {
      if (!view) return;
      view.text = /** @type {HTMLInputElement} */ (event.target).value;
      renderList();
    },
    { signal }
  );
  document.addEventListener(
    'visibilitychange',
    () => {
      if (!document.hidden) void poll();
    },
    { signal }
  );
  setFilterReason(noSidebar());

  renderState(part('.servers-state'), { icon: 'cpu', message: 'Reading the server fleet' });
  // A tick waits for a read still in flight: a newer read would drop it.
  view.timer = window.setInterval(() => {
    if (!document.hidden && !view?.reading) void poll();
  }, POLL_MS);
  view.ticker = window.setInterval(() => {
    if (view && !document.hidden) tick(view.root);
  }, 1000);
  void poll();
}

export function unmountServers() {
  if (!view) return;
  const { controller, timer, ticker, release } = view;
  view = null;
  window.clearInterval(timer);
  window.clearInterval(ticker);
  controller.abort();
  release();
  setFilterReason('');
  setResultCount('');
}

// ---- Reading ----

/** Read the fleet; a late answer for a view that is gone is dropped. */
async function poll() {
  if (!view) return;
  const current = view;
  const seq = ++current.pollSeq;
  /** @type {FleetAnswer} */
  let answer;
  current.reading += 1;
  try {
    answer = await fetchFleet(current.controller.signal);
  } catch (err) {
    if (view !== current || seq !== current.pollSeq) return;
    current.snapshot = null;
    showStatement(
      'The server fleet route did not answer',
      `${messageOf(err)}. This says nothing about whether any ${LABELS.server} runs.`
    );
    return;
  } finally {
    current.reading -= 1;
  }
  if (view !== current || seq !== current.pollSeq) return;
  if (answer?.outcome !== 'ok' || !answer.snapshot) {
    current.snapshot = null;
    const { message, detail } = outcomeStatement(answer ?? /** @type {any} */ ({}));
    showStatement(message, detail);
    return;
  }
  current.snapshot = answer.snapshot;
  render();
}

// ---- Rendering ----

/** @param {string} selector @returns {HTMLElement} */
function part(selector) {
  return /** @type {HTMLElement} */ (view?.root.querySelector(selector));
}

/**
 * A statement in place of the header, band and table, with Retry.
 * @param {string} message
 * @param {string} detail
 */
function showStatement(message, detail) {
  if (!view) return;
  setResultCount('');
  for (const selector of ['.servers-header', '.servers-band', '.servers-list']) {
    const el = part(selector);
    el.hidden = true;
    el.replaceChildren();
    written.delete(el);
  }
  const host = part('.servers-state');
  host.hidden = false;
  const key = `statement\n${message}\n${detail}`;
  if (written.get(host) === key) return;
  renderState(host, {
    icon: 'cpu',
    message,
    detail,
    action: { label: 'Retry', onClick: () => void poll() },
  });
  written.set(host, key);
}

function render() {
  if (!view?.snapshot) return;
  const { snapshot } = view;
  const host = part('.servers-state');
  host.replaceChildren();
  host.hidden = true;
  written.delete(host);

  const header = part('.servers-header');
  header.hidden = false;
  writePart(header, headerHtml(snapshot));
  tick(header);

  const band = part('.servers-band');
  band.hidden = false;
  if (writePart(band, bandHtml(snapshot, { pressed: view.tile, grouped: view.grouped }))) {
    band.querySelector('[data-action="refresh"]')?.append(icon('refresh'));
  }
  renderList();
}

/** The table, or the centred state when the filters leave no row. */
function renderList() {
  if (!view?.snapshot) return;
  const all = view.snapshot.servers ?? [];
  const rows = filterRows(all, { tile: view.tile, text: view.text });
  setResultCount(countHtml(rows.length, all.length));
  const list = part('.servers-list');
  list.hidden = false;
  if (!rows.length) {
    const filtered = view.tile !== null || view.text.trim() !== '';
    const message = filtered
      ? `No ${LABELS.server} for this filter`
      : `The snapshot lists no ${LABELS.server}`;
    const key = `state\n${message}`;
    if (written.get(list) === key) return;
    renderState(list, {
      icon: 'cpu',
      message,
      ...(filtered ? { action: { label: 'Clear filter', onClick: clearFilters } } : {}),
    });
    written.set(list, key);
    return;
  }
  writePart(list, tableHtml(rows, view.grouped ? view.snapshot.group_by : null));
  tick(list);
}

// ---- Actions ----

function clearFilters() {
  if (!view) return;
  view.tile = null;
  view.text = '';
  const input = searchInput();
  if (input) input.value = '';
  render();
}

/** @param {Event} event */
function onClick(event) {
  if (!view) return;
  const target = event.target instanceof Element ? event.target : null;
  if (!target) return;
  const tile = /** @type {HTMLElement|null} */ (target.closest('[data-tile]'));
  if (tile) {
    const key = tile.dataset.tile ?? '';
    view.tile = view.tile === key ? null : key;
    render();
    return;
  }
  const action = /** @type {HTMLElement|null} */ (target.closest('[data-action]'));
  switch (action?.dataset.action) {
    case 'group':
      view.grouped = !view.grouped;
      render();
      return;
    case 'refresh':
      void poll();
      return;
    default:
  }
}
