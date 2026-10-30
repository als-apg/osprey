// @ts-check
/* OSPREY Web Terminal — the sidebar's artifact lists (layout F)
 *
 * The hub owns the artifact list. The sidebar's Artifacts row lists the
 * artifacts of the current session's scope and its Script runs row lists the
 * run records (the approval hook's review notebook, the executor's notebook
 * and result record): two listings of the artifacts panel, split by the
 * server's `run` filter (sidebar-artifact-listing.js) and drawn by
 * sidebar-artifact-rows.js. The server stamps every entry with `run_record`,
 * so an artifact the event stream announces lands in the right list without
 * a second copy of the rule here.
 *
 * The artifacts iframe is a viewer of ONE artifact. This module decides which
 * one and tells the iframe with `osprey-artifact-select`. It never changes the
 * agent's focus: that endpoint belongs to the agent and to the chat's artifact
 * links, and a click in the list is only the operator looking at something.
 *
 * Which artifact is shown (the rules the gallery applied before the list moved
 * here):
 *   - Expert: the selection. A row click picks; an agent focus frame jumps the
 *     pick to the focused artifact.
 *   - Simple: the operator's pick while it is still listed, else the agent's
 *     focus unless it is a run record, else the fallback. A focus frame the
 *     agent's tooling sent on its own (`automatic`: the approval hook showing
 *     its review notebook, which the chat's approval card already shows) only
 *     records the focus; any other focus frame also takes the pick.
 *   - Fallback, both views: the newest result, then the newest script run,
 *     then the shipped example.
 * A page that opens after the agent focused (a reload, a second tab) or a
 * stream that reconnects reads the focus the panel kept, as an automatic one.
 *
 * The module also draws the artifacts pane header (Artifacts › title,
 * ‹ N of M ›, Open full size, Save) as the tile bar's local contribution
 * (tile-header-contrib.js), and lends the tile's ⋯ menu the artifact verbs,
 * which the iframe runs on `osprey-artifact-action`.
 *
 * The DOM is the sidebar skeleton of index.html and sidebar.js. Each slot is
 * looked up when it is drawn and a missing one is skipped, so on a page
 * without the skeleton the module runs and draws nothing.
 *
 * @module sidebar-artifacts
 */

import { closePopover, togglePopover } from '/design-system/js/popover.js';
import { scopedStorageKey } from '/design-system/js/storage-scope.js';
import { onModeChange } from '/design-system/js/frame-params.js';
import { createEventSource } from './api.js';
import { getCurrentSessionId, onSessionChange } from './terminal.js';
import { contributeLocal } from './tile-header-contrib.js';
import { sendArtifactActionToIframe, sendArtifactSelectToIframe } from './panel-iframe-sync.js';
import { activateTab } from './panel-manager.js';
import { hasDockedTile } from './panel-placement.js';
import { setServiceMenuExtras } from './panel-menu-policy.js';
import {
  ARTIFACTS_BASE,
  createListing,
  fetchArtifact,
  fetchFocusedId,
} from './sidebar-artifact-listing.js';
import {
  artifactRows,
  drawnOrder,
  fillArtifactMenu,
  rowIn,
  runRows,
} from './sidebar-artifact-rows.js';
import { headerItems, menuItems, openArtifact, saveArtifact } from './artifacts-pane-header.js';

/**
 * @typedef {import('./sidebar-artifact-listing.js').ArtifactEntry} ArtifactEntry
 * @typedef {import('./sidebar-artifact-listing.js').Listing} Listing
 * @typedef {import('./sidebar-artifact-rows.js').ListView} ListView
 */

/** The panel id of the artifact gallery. */
const ARTIFACTS_PANEL_ID = 'artifacts';

/** The panel's event stream; createEventSource puts the per-user prefix on it. */
const EVENTS_PATH = `${ARTIFACTS_BASE}/api/events`;

/** Base key of the persisted grouping, scoped per user through scopedStorageKey. */
const VIEW_STORAGE_KEY = 'osprey-sidebar-artifacts-view';

/** @type {readonly ListView[]} */
const VIEWS = ['types', 'activity'];

/** Above this many artifacts the filter field stays open. */
const FILTER_THRESHOLD = 8;

/** How close to the end of a list (two rows) a scroll asks for the next page. */
const NEAR_END_PX = 52;

const ARTIFACT_LIST_IDS = ['sidebar-artifacts-list', 'sidebar-artifacts-flyout-list'];
const RUNS_LIST_ID = 'sidebar-runs-list';
const MENU_ID = 'sidebar-artifacts-menu';
const LIST_SELECTOR = [...ARTIFACT_LIST_IDS, RUNS_LIST_ID].map((id) => `#${id}`).join(', ');

/**
 * The view state: the grouping, the filter, the scope, and the three ids the
 * shown artifact is resolved from.
 * @returns {{view: ListView, filter: string, filterOpen: boolean, allSessions: boolean,
 *   pickId: string | null, focusId: string | null, shownId: string | null}}
 */
function freshState() {
  return {
    view: 'types',
    filter: '',
    filterOpen: false,
    allSessions: false,
    pickId: null,
    focusId: null,
    shownId: null,
  };
}

let state = freshState();
/** @type {{artifacts: Listing, runs: Listing} | null} */
let listings = null;
/** Listeners on shared targets, removed together by teardown. */
let abort = new AbortController();
/** @type {{stop: () => void} | null} */
let stream = null;
let streamOpened = false;
let active = false;
/** The last header contribution, so an unchanged one is not sent again. */
let lastHeader = '';
/** Elements already wired, so a re-render never stacks listeners. */
let wired = new WeakSet();

/** @returns {string | null} the session both listings are scoped to */
function scopeSessionId() {
  return state.allSessions ? null : getCurrentSessionId();
}

/** @returns {{artifacts: Listing, runs: Listing}} */
function lists() {
  if (!listings) throw new Error('sidebar-artifacts: initSidebarArtifacts() has not run');
  return listings;
}

// ---- Init / teardown ----

/**
 * Start the artifact lists: fetch both, follow the panel's event stream, the
 * session and the view, and draw into whatever part of the skeleton exists.
 * Idempotent.
 */
export function initSidebarArtifacts() {
  if (active) return;
  active = true;
  state.view = readView();
  listings = {
    artifacts: createListing({ run: false, sessionId: scopeSessionId, onChange: render }),
    runs: createListing({ run: true, sessionId: scopeSessionId, onChange: render }),
  };
  const signal = abort.signal;
  document.addEventListener('click', onClick, { signal });
  document.addEventListener('input', onInput, { signal });
  document.addEventListener('keydown', onListKeydown, { signal });
  document.addEventListener('osprey-sidebar-rendered', render, { signal });
  window.addEventListener('message', onMessage, { signal });
  stream = createEventSource(EVENTS_PATH, { onMessage: onFrame, onOpen: onStreamOpen });
  onSessionChange(() => {
    if (!active) return;
    state.allSessions = false;
    render();
    void refresh();
  });
  onModeChange(() => {
    if (active) render();
  });
  setServiceMenuExtras(ARTIFACTS_PANEL_ID, menuExtras);
  void restoreFocus();
  render();
}

/**
 * Stop everything initSidebarArtifacts started and forget the lists. The
 * session and mode subscriptions cannot be removed from their owners; they
 * check `active` and do nothing once this has run.
 */
export function teardownSidebarArtifacts() {
  if (!active) return;
  active = false;
  abort.abort();
  abort = new AbortController();
  stream?.stop();
  stream = null;
  streamOpened = false;
  listings?.artifacts.stop();
  listings?.runs.stop();
  listings = null;
  setServiceMenuExtras(ARTIFACTS_PANEL_ID, null);
  state = freshState();
  lastHeader = '';
  wired = new WeakSet();
}

/**
 * Fetch the first page of both lists again (scope change, Refresh, a reconnect).
 * @returns {Promise<unknown>} settled once both first pages have landed or failed
 */
function refresh() {
  return Promise.all([lists().artifacts.loadFirst(), lists().runs.loadFirst()]);
}

/**
 * Read the lists and the agent's persisted focus: a focus frame only reaches a
 * page that is open when it is sent, so a reload, a second tab or a reconnect
 * learns the focus here. It is applied once the first pages have landed (a
 * first page replaces what the lists hold, the fetched focus included) and
 * only when it changed, so a reconnect does not take back the operator's pick.
 * It counts as automatic: the panel does not keep whether the agent's tooling
 * sent it, and an automatic focus never puts a run record in Simple's viewer.
 */
async function restoreFocus() {
  const [, id] = await Promise.all([refresh(), fetchFocusedId()]);
  if (active && id && id !== state.focusId) await onFocus(id, true);
}

// ---- Which artifact is shown ----

/**
 * @param {ArtifactEntry} entry
 * @returns {Listing}
 */
function listingFor(entry) {
  return entry.run_record === true ? lists().runs : lists().artifacts;
}

/**
 * @param {string | null} id
 * @returns {ArtifactEntry | null}
 */
function held(id) {
  if (!id) return null;
  return lists().artifacts.find(id) ?? lists().runs.find(id);
}

/**
 * Whether an entry belongs in the lists the current scope fetches: the
 * server's session test admits an untagged artifact.
 * @param {ArtifactEntry} entry
 * @returns {boolean}
 */
function inScope(entry) {
  const sessionId = scopeSessionId();
  return !sessionId || !entry.session_id || entry.session_id === sessionId;
}

/** @returns {boolean} */
function isSimple() {
  return document.documentElement.dataset.uiMode === 'simple';
}

/**
 * The artifact the viewer shows (see the module docstring for the rules).
 * @returns {ArtifactEntry | null}
 */
function resolveShown() {
  const pick = held(state.pickId);
  if (pick) return pick;
  if (isSimple()) {
    const focus = held(state.focusId);
    if (focus && focus.run_record !== true) return focus;
  }
  const results = lists().artifacts.entries;
  const runs = lists().runs.entries;
  const own = (/** @type {ArtifactEntry} */ a) => a.origin !== 'demo';
  return results.find(own) ?? runs.find(own) ?? results[0] ?? runs[0] ?? null;
}

/** @returns {ArtifactEntry | null} */
function currentShown() {
  return held(state.shownId);
}

/**
 * The order ‹ › steps through: the shown artifact's list as it is drawn
 * (its grouping, not the filter, which only narrows what is visible).
 * @param {Listing} listing
 * @returns {ArtifactEntry[]}
 */
function stepOrder(listing) {
  return listing.run ? listing.entries : drawnOrder(listing.entries, state.view);
}

/**
 * Step the pick through the shown artifact's list, fetching the next page
 * when the step runs past the last one held.
 * @param {1 | -1} delta
 */
async function step(delta) {
  const shown = currentShown();
  if (!shown) return;
  const listing = listingFor(shown);
  if (delta > 0 && stepOrder(listing).at(-1)?.id === shown.id) await listing.loadNext();
  const order = stepOrder(listing);
  const index = order.findIndex((a) => a.id === shown.id);
  // A scope change while the page was in flight can leave the shown artifact
  // out of the list: the pick stays, and there is nothing to step to.
  const target = index < 0 ? undefined : order[index + delta];
  if (!target || !active) return;
  state.pickId = target.id;
  render();
}

// ---- The event stream ----

/** A reconnect: the stream does not replay, so the lists and the focus are read again. */
function onStreamOpen() {
  if (!active) return;
  if (streamOpened) void restoreFocus();
  streamOpened = true;
}

/** @param {any} frame */
function onFrame(frame) {
  if (!active || !frame || typeof frame !== 'object' || typeof frame.id !== 'string') return;
  const entry = { ...frame };
  delete entry.type;
  if (frame.type === 'artifact') {
    if (!inScope(entry)) return;
    listingFor(entry).add(entry);
  } else if (frame.type === 'artifact_updated') {
    if (!lists().artifacts.update(entry)) lists().runs.update(entry);
  } else if (frame.type === 'artifact_deleted') {
    lists().artifacts.remove(frame.id);
    lists().runs.remove(frame.id);
    if (state.pickId === frame.id) state.pickId = null;
    if (state.focusId === frame.id) state.focusId = null;
  } else if (frame.type === 'focus') {
    void onFocus(frame.id, frame.automatic === true);
    return;
  } else {
    return;
  }
  render();
}

/**
 * The agent focused an artifact. One the lists do not hold yet (an older one)
 * is fetched and held; one outside the current scope is not focused at all.
 * @param {string} id
 * @param {boolean} automatic
 */
async function onFocus(id, automatic) {
  if (!held(id)) {
    const entry = await fetchArtifact(id);
    if (!active || entry?.id !== id || !inScope(entry)) return;
    listingFor(entry).add(entry, { counted: false });
  }
  state.focusId = id;
  if (!automatic || !isSimple()) state.pickId = id;
  render();
}

// ---- The viewer, the pane header and the tile menu ----

/** @returns {HTMLIFrameElement | null} */
function artifactsIframe() {
  return /** @type {HTMLIFrameElement | null} */ (
    document.querySelector(`iframe.panel-iframe[data-panel-id="${ARTIFACTS_PANEL_ID}"]`)
  );
}

/** Tell the viewer what to show. */
function sendSelect() {
  const iframe = artifactsIframe();
  if (iframe) sendArtifactSelectToIframe(iframe, state.shownId);
}

/**
 * The viewer announced itself (a first load, a reload, a re-created frame):
 * it shows nothing until told, so it is told again.
 * @param {MessageEvent} event
 */
function onMessage(event) {
  if (!active || event.origin !== window.location.origin) return;
  if (event.data?.type !== 'osprey-artifact-ready') return;
  const iframe = artifactsIframe();
  if (iframe && event.source === iframe.contentWindow) sendSelect();
}

/**
 * Where the shown artifact sits in the list ‹ › steps through.
 * @param {ArtifactEntry | null} shown
 * @returns {import('./artifacts-pane-header.js').PagerPosition | null}
 */
function pagerPosition(shown) {
  if (!shown) return null;
  const listing = listingFor(shown);
  const order = stepOrder(listing);
  const index = order.findIndex((a) => a.id === shown.id);
  // Not in its list (the scope changed under it): no position to show.
  if (index < 0) return null;
  return {
    index,
    count: Math.max(listing.total, order.length),
    last: index === order.length - 1 && !listing.hasMore,
  };
}

/**
 * A header control was used. Open full size runs here, in the click's own
 * task, so the browser lets the new tab open.
 * @param {string} id
 */
function onHeaderAction(id) {
  const shown = currentShown();
  if (!shown) return;
  if (id === 'prev') void step(-1);
  else if (id === 'next') void step(1);
  else if (id === 'open') openArtifact(shown);
  else if (id === 'save') saveArtifact(shown);
}

/**
 * The artifact verbs of the tile's ⋯ menu, for the shown artifact; none
 * while no viewer is there to run them.
 */
function menuExtras() {
  const shown = active ? currentShown() : null;
  if (!shown || !artifactsIframe()) return [];
  return menuItems(shown, (action) =>
    sendArtifactActionToIframe(artifactsIframe(), shown.id, action)
  );
}

// ---- Drawing ----

/** Draw everything from the current state; tell the viewer when what it shows changed. */
function render() {
  if (!active) return;
  const { artifacts, runs } = lists();
  const filterShown = artifacts.total > FILTER_THRESHOLD || state.filterOpen;
  if (!filterShown) state.filter = '';
  const shown = resolveShown();
  const changed = (shown?.id ?? null) !== state.shownId;
  state.shownId = shown?.id ?? null;
  for (const listId of ARTIFACT_LIST_IDS) {
    drawList(listId, artifacts, () =>
      artifactRows(artifacts.entries, {
        ...rowsState(artifacts),
        view: state.view,
        filter: state.filter,
      })
    );
  }
  drawList(RUNS_LIST_ID, runs, () => runRows(runs.entries, rowsState(runs)));
  drawCounts();
  drawControls(filterShown);
  const header = headerItems(shown, pagerPosition(shown));
  const key = JSON.stringify(header);
  if (key !== lastHeader) {
    lastHeader = key;
    contributeLocal(ARTIFACTS_PANEL_ID, header, onHeaderAction);
  }
  if (!changed) return;
  sendSelect();
  for (const listId of [...ARTIFACT_LIST_IDS, RUNS_LIST_ID]) {
    const list = document.getElementById(listId);
    const row = list && state.shownId ? rowIn(list, state.shownId) : null;
    row?.scrollIntoView?.({ block: 'nearest' });
  }
}

/** @param {Listing} listing */
function rowsState(listing) {
  return { shownId: state.shownId, loaded: listing.loaded, failed: listing.failed };
}

/**
 * Rebuild one list, keeping keyboard focus on the row it was on. A visible
 * list too short to scroll to its end (a narrow filter, a small page size)
 * fetches the next page itself, as scrolling would.
 * @param {string} listId
 * @param {Listing} listing
 * @param {() => Node[]} build
 */
function drawList(listId, listing, build) {
  const list = document.getElementById(listId);
  if (!list) return;
  wireList(list, listing);
  const focused = document.activeElement;
  const focusedId =
    focused instanceof HTMLElement && list.contains(focused) ? (focused.dataset.id ?? null) : null;
  list.replaceChildren(...build());
  list.setAttribute('aria-busy', String(!listing.loaded));
  if (focusedId) rowIn(list, focusedId)?.focus({ preventScroll: true });
  if (list.clientHeight > 0 && nearEnd(list)) void listing.loadNext();
}

/**
 * @param {HTMLElement} list
 * @returns {boolean} whether the list is scrolled to within two rows of its end
 */
function nearEnd(list) {
  return list.scrollHeight - list.scrollTop - list.clientHeight <= NEAR_END_PX;
}

/** The counts on the Artifacts row, in the flyout and on the Script runs row. */
function drawCounts() {
  const { artifacts, runs } = lists();
  each('[data-art-count]', (el) => showCount(el, artifacts));
  each('[data-runs-count]', (el) => showCount(el, runs));
  // The collapsed sidebar names its rows in tooltips; these carry the count.
  each(`.osprey-nav-row[data-panel-id="${ARTIFACTS_PANEL_ID}"]`, (row) => {
    row.dataset.tip = `Artifacts · ${artifacts.total}`;
  });
  each('#sidebar-runs-toggle', (row) => {
    row.dataset.tip = `Script runs · ${runs.total}`;
  });
}

/**
 * @param {HTMLElement} el
 * @param {Listing} listing
 */
function showCount(el, listing) {
  el.textContent = String(listing.total);
  el.hidden = !listing.loaded || listing.total === 0;
}

/**
 * The row actions, in the row and in the flyout alike: the filter field and
 * its toggle, the scope mark on ⋯, and the ⋯ menu's Group by radios and
 * All sessions checkbox.
 * @param {boolean} filterShown
 */
function drawControls(filterShown) {
  each('[data-art-filter]', (label) => {
    label.hidden = !filterShown;
    const input = label.querySelector('input');
    if (!input) return;
    wireFilterInput(input);
    if (input !== document.activeElement) input.value = state.filter;
  });
  each('[data-art-filter-toggle]', (button) =>
    button.setAttribute('aria-pressed', String(filterShown))
  );
  // All sessions is the non-default scope: ⋯ carries a mark while it is on,
  // so the wider scope shows with the menu closed (sidebar-artifacts.css).
  each('[data-art-menu]', (button) => button.toggleAttribute('data-art-scope', state.allSessions));
  const menu = document.getElementById(MENU_ID);
  if (!menu) return;
  fillArtifactMenu(menu);
  each('[data-art-view]', (item) =>
    item.setAttribute('aria-checked', String(item.dataset.artView === state.view))
  );
  const allSessions = menu.querySelector('[data-art-all-sessions]');
  allSessions?.setAttribute('aria-checked', String(state.allSessions));
}

/**
 * Run `fn` on every element of the page that matches.
 * @param {string} selector
 * @param {(el: HTMLElement) => void} fn
 */
function each(selector, fn) {
  for (const el of document.querySelectorAll(selector)) {
    if (el instanceof HTMLElement) fn(el);
  }
}

// ---- Input ----

/**
 * Load more when a list is scrolled to its end. Wired once per element: the
 * sidebar keeps its list elements across its own re-renders.
 * @param {HTMLElement} list
 * @param {Listing} listing
 */
function wireList(list, listing) {
  if (wired.has(list)) return;
  wired.add(list);
  const onScroll = () => {
    if (nearEnd(list)) void listing.loadNext();
  };
  list.addEventListener('scroll', onScroll, { signal: abort.signal });
}

/**
 * Escape clears a filter that holds text, and stops there: in the collapsed
 * sidebar's flyout it would otherwise close the flyout too.
 * @param {HTMLInputElement} input
 */
function wireFilterInput(input) {
  if (wired.has(input)) return;
  wired.add(input);
  const onKeydown = (/** @type {KeyboardEvent} */ event) => {
    if (event.key !== 'Escape' || !input.value) return;
    event.stopPropagation();
    input.value = '';
    state.filter = '';
    render();
  };
  input.addEventListener('keydown', onKeydown, { signal: abort.signal });
}

/** @param {MouseEvent} event */
function onClick(event) {
  const target = event.target instanceof Element ? event.target : null;
  if (!target) return;
  const row = target.closest('.osprey-list-row[data-id]');
  if (row instanceof HTMLElement && row.closest(LIST_SELECTOR)) {
    pickRow(/** @type {string} */ (row.dataset.id));
    return;
  }
  const viewButton = target.closest('[data-art-view]');
  if (viewButton instanceof HTMLElement) {
    setView(viewButton.dataset.artView);
    return;
  }
  const filterToggle = target.closest('[data-art-filter-toggle]');
  if (filterToggle instanceof HTMLElement) {
    toggleFilter(filterToggle);
    return;
  }
  const menu = document.getElementById(MENU_ID);
  const menuButton = target.closest('[data-art-menu]');
  if (menuButton instanceof HTMLElement && menu) {
    fillArtifactMenu(menu);
    // From the sidebar row the menu opens beside the sidebar, as every other
    // sidebar popover does; inside the flyout it hangs under its button.
    const sidebar = menuButton.closest('#sidebar');
    togglePopover(
      menuButton,
      menu,
      sidebar ? { place: 'beside-right', edge: sidebar } : { place: 'below-end' }
    );
  } else if (target.closest('[data-art-all-sessions]')) {
    state.allSessions = !state.allSessions;
    render();
    void refresh();
  } else if (target.closest('[data-art-refresh]') && menu) {
    closePopover(menu);
    void refresh();
  }
}

/**
 * The operator picked an artifact in a list: show it, and bring the
 * artifacts tile up when it is not on screen.
 * @param {string} id
 */
function pickRow(id) {
  state.pickId = id;
  render();
  if (!hasDockedTile(ARTIFACTS_PANEL_ID)) activateTab(ARTIFACTS_PANEL_ID, { userInitiated: true });
}

/** @param {Event} event */
function onInput(event) {
  const input = event.target;
  if (!(input instanceof HTMLInputElement) || !input.closest('[data-art-filter]')) return;
  state.filter = input.value;
  render();
}

/**
 * Up and Down move between the rows of a list; the rows are buttons, so Enter
 * and Space pick.
 * @param {KeyboardEvent} event
 */
function onListKeydown(event) {
  if (event.key !== 'ArrowDown' && event.key !== 'ArrowUp') return;
  const row = event.target instanceof HTMLElement ? event.target.closest('.osprey-list-row') : null;
  const list = row?.closest(LIST_SELECTOR);
  if (!row || !list) return;
  const rows = [...list.querySelectorAll('.osprey-list-row')];
  const next = rows[rows.indexOf(row) + (event.key === 'ArrowDown' ? 1 : -1)];
  if (!(next instanceof HTMLElement)) return;
  event.preventDefault();
  next.focus();
}

/**
 * Persist and apply a grouping.
 * @param {string | undefined} view
 */
function setView(view) {
  if (!isView(view) || view === state.view) return;
  state.view = view;
  try {
    localStorage.setItem(scopedStorageKey(VIEW_STORAGE_KEY), view);
  } catch {
    /* storage blocked: the grouping holds for this page */
  }
  render();
}

/** @returns {ListView} the persisted grouping, Types by default */
function readView() {
  try {
    const stored = localStorage.getItem(scopedStorageKey(VIEW_STORAGE_KEY));
    return isView(stored) ? stored : 'types';
  } catch {
    return 'types';
  }
}

/**
 * @param {unknown} value
 * @returns {value is ListView}
 */
function isView(value) {
  return VIEWS.includes(/** @type {ListView} */ (value));
}

/**
 * The filter toggle opens or closes the field while the list is short (above
 * the threshold the field is always open), then puts the caret in the field
 * of the same surface: the row's or the flyout's.
 * @param {HTMLElement} toggle
 */
function toggleFilter(toggle) {
  if (lists().artifacts.total <= FILTER_THRESHOLD) {
    state.filterOpen = !state.filterOpen;
    if (!state.filterOpen) state.filter = '';
  }
  render();
  const scope = toggle.closest('.sidebar-row, .osprey-popover');
  const input = scope?.querySelector('[data-art-filter]:not([hidden]) input');
  if (input instanceof HTMLElement) input.focus();
}
