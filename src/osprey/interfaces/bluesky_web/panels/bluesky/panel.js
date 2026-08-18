// @ts-check
/**
 * BLUESKY panel — the shell.
 *
 * One panel, three views: Plans (compose a plan), Queue (what the queue server
 * is holding) and Results (the selected run's record, table and trace). This
 * module owns only what is common to all three — which one is showing, the
 * run the Results view is following, the tile-bar contribution, and the fan-out
 * of the hub's Expert/Simple flips. Each view is a factory that takes the panel
 * root and its callbacks; nothing here reaches into a view's internals.
 *
 * TWO tab strips, deliberately, and both are load-bearing:
 *
 * - The panel describes the three views to the web-terminal hub as a `nav`
 *   contribution, so an embedded panel gets ONE header (the tile bar) rather
 *   than a bar under a bar.
 * - The panel ALSO renders its own strip in the body. `contributeHeader` is a
 *   strict no-op standalone, and the hub collapses a service tile's bar to
 *   zero height in Simple mode — in both cases the contributed strip does not
 *   exist, and the panel would be unnavigable without its own. The in-body
 *   strip is therefore hidden by CSS in exactly one case: embedded AND Expert.
 *
 * - No `innerHTML`, anywhere. Plan names, parameter values, and refusal
 *   sentences all originate off-panel (an agent's draft, the bridge, another
 *   operator), and they reach the DOM only as text nodes. Row identity travels
 *   in `data-*` attributes read by delegated listeners, never interpolated
 *   into markup.
 * - No local copy of the bridge's arming policy. The panel never decides
 *   whether a deployment may execute; it asks, and when the bridge refuses it
 *   shows the bridge's own sentence VERBATIM — that sentence carries the
 *   remedy (which token is missing, or the `osprey set connector=…` flip
 *   command for a browse-only deployment).
 *
 * The persistent status strip (queue state, "Stop after current item", "Abort
 * running plan") is NEVER contributed, for the same reason read in reverse:
 * anything safety-bearing has to survive Simple mode, and the tile bar does
 * not. It renders in the body on every tab — see index.html.
 *
 * @module panel
 */

import { panelApiPrefix } from '/design-system/js/dom.js';
import { onModeChange } from '/design-system/js/frame-params.js';
import { contributeHeader, isSimpleMode, onHeaderAction } from '/design-system/js/header-contrib.js';

import { createPlansView } from './plans-view.js';
import { createQueueView } from './queue-view.js';
import { createResultsView } from './results-view.js';

/** @typedef {'plans'|'queue'|'results'} ViewId */

/** The contributed nav item's id, and the prefix of every in-body tab id. */
const VIEW_ITEM_ID = 'view';
/** The contributed search item's id (the Plans filter). */
const FILTER_ITEM_ID = 'plan-filter';

/**
 * Appended to the Results nav label while data is arriving for a run the
 * operator is not currently watching. The contract's nav entries carry a label
 * and an active flag and nothing else, so a suffix on the label is the honest
 * way to say "there is something here" — no new item kind, no hub-side state.
 */
const ACTIVITY_MARKER = ' •';

const PREFIX = panelApiPrefix();

/**
 * @param {string} path
 * @returns {string}
 */
function api(path) {
  return `${PREFIX}${path}`;
}

/**
 * The panel owns its own shell, so every id below is present by construction —
 * a missing one is a bundle bug, not a runtime condition to branch on.
 *
 * @param {string} id
 * @returns {HTMLElement}
 */
function byId(id) {
  return /** @type {HTMLElement} */ (document.getElementById(id));
}

const root = byId('panel-root');
const tabStrip = byId('view-tabs');

/** @type {Array<{id: ViewId, label: string, tab: HTMLElement, view: HTMLElement}>} */
const VIEWS = [
  { id: 'plans', label: 'Plans', tab: byId('view-tab-plans'), view: byId('view-plans') },
  { id: 'queue', label: 'Queue', tab: byId('view-tab-queue'), view: byId('view-queue') },
  { id: 'results', label: 'Results', tab: byId('view-tab-results'), view: byId('view-results') },
];

/** @type {ViewId} */
let activeView = 'plans';
/**
 * The run the Results view is following (`null` = none picked yet).
 * @type {string|null}
 */
let selectedRunId = null;
/** Whether the Results view is polling live data right now. */
let resultsPolling = false;

// ---------------------------------------------------------------------------
// Views
// ---------------------------------------------------------------------------

const resultsView = createResultsView({
  api,
  elements: {
    statusBadge: byId('run-status-badge'),
    meta: byId('run-meta'),
    note: byId('run-note'),
    emptyState: byId('results-empty'),
    tableCard: byId('data-card'),
    tableDetails: /** @type {HTMLDetailsElement} */ (byId('table-details')),
    tableSummaryCount: byId('table-summary-count'),
    tableNote: byId('table-note'),
    tableHeadRow: /** @type {HTMLTableRowElement} */ (byId('table-head-row')),
    tableBody: /** @type {HTMLTableSectionElement} */ (byId('table-body')),
    exportButton: /** @type {HTMLButtonElement} */ (byId('export-btn')),
    exportNote: byId('export-note'),
    figureCard: byId('figure-card'),
    figurePanels: byId('figure-panels'),
    figureNote: byId('figure-note'),
  },
  onPollingChange(polling) {
    if (polling === resultsPolling) return;
    resultsPolling = polling;
    publishContribution();
  },
});

const queueView = createQueueView({
  root,
  api,
  onSelectRun(runId, activate) {
    selectRun(runId);
    // An operator's click on a run is a request to SEE it; the Simple-mode
    // auto-pick is the panel's own housekeeping and must not move anyone off
    // the tab they chose.
    if (activate) setActiveView('results');
  },
});

const plansView = createPlansView({
  root,
  api,
  onOpenRun(runId) {
    selectRun(runId);
    setActiveView('results');
  },
});

// ---------------------------------------------------------------------------
// Selection
// ---------------------------------------------------------------------------

/**
 * Follow `runId` in the Results view and highlight it in the queue/history
 * lists. Selecting does NOT switch tabs on its own — callers decide, because
 * "the operator asked to see this run" and "the panel picked one for them" are
 * different events.
 *
 * @param {string|null} runId
 */
function selectRun(runId) {
  if (runId === selectedRunId) return;
  selectedRunId = runId;
  resultsView.follow(runId);
  queueView.setSelected(runId);
}

// ---------------------------------------------------------------------------
// View switching
// ---------------------------------------------------------------------------

/**
 * Show one view and hide the other two.
 *
 * The ONE function behind both tab strips: the in-body buttons call it, and so
 * does the hub's header action, so the two can never drive different code.
 *
 * @param {string} id A view id; anything else is ignored (the hub's action
 *   value is data from another frame, not a promise).
 */
function setActiveView(id) {
  if (!VIEWS.some((entry) => entry.id === id)) return;
  activeView = /** @type {ViewId} */ (id);
  for (const entry of VIEWS) {
    const active = entry.id === activeView;
    entry.tab.classList.toggle('active', active);
    entry.tab.setAttribute('aria-selected', active ? 'true' : 'false');
    entry.view.hidden = !active;
  }
  publishContribution();
}

// ---------------------------------------------------------------------------
// Tile-bar contribution
// ---------------------------------------------------------------------------

/**
 * Whether the Results tab should carry its activity marker: data is arriving
 * for a run the operator is not looking at. Activating the tab, or the poll
 * settling, clears it — both are already inputs to this expression, so there is
 * no marker state to forget to reset.
 *
 * @returns {boolean}
 */
function resultsHaveUnwatchedActivity() {
  return resultsPolling && activeView !== 'results';
}

/**
 * (Re)send the WHOLE contribution. The hub renders only what the last one
 * says — a contribution is an idempotent replace, never a diff — so every
 * state change republishes all of it.
 */
function publishContribution() {
  /** @type {import('/design-system/js/header-contrib.js').HeaderItem[]} */
  const items = [];
  // ORDER IS THE LAYOUT (see header-contrib.js): the hub right-anchors every
  // interactive item into one cluster against the close button, so an item's
  // arrival or departure moves everything contributed BEFORE it and nothing
  // after it.
  //
  // The filter is the conditional item here — it belongs to the Plans view, so
  // it is contributed only while that view is showing, and never in Simple
  // mode, where the hub collapses the tile bar and Simple deliberately
  // ENLARGES a search box rather than hiding it (in both excluded cases the
  // in-body box is the one on screen). The view switcher is the item an
  // operator aims at over and over. So the filter goes FIRST and the switcher
  // LAST: the switcher stays pinned to the close button and the filter appears
  // and disappears to its left, in the slack.
  //
  // The other order would put the tab strip on the moving side of its own
  // effect — clicking away from Plans drops the filter, and the strip you just
  // clicked slides right by the filter's whole width. Do not swap these.
  if (activeView === 'plans' && !isSimpleMode()) {
    items.push({ kind: 'search', id: FILTER_ITEM_ID, placeholder: 'Filter plans…' });
  }
  items.push({
    kind: 'nav',
    id: VIEW_ITEM_ID,
    items: VIEWS.map((entry) => ({
      id: entry.id,
      label:
        entry.id === 'results' && resultsHaveUnwatchedActivity()
          ? `${entry.label}${ACTIVITY_MARKER}`
          : entry.label,
      active: entry.id === activeView,
    })),
  });
  contributeHeader(items);
}

// ---------------------------------------------------------------------------
// Wiring
// ---------------------------------------------------------------------------

tabStrip.addEventListener('click', (event) => {
  const target = event.target;
  if (!(target instanceof Element)) return;
  const button = target.closest('button[data-view]');
  if (!(button instanceof HTMLElement) || !button.dataset.view) return;
  setActiveView(button.dataset.view);
});

// The hub's round-trip drives the SAME functions the in-body controls do.
onHeaderAction((id, value) => {
  if (id === VIEW_ITEM_ID && value) setActiveView(value);
  else if (id === FILTER_ITEM_ID) plansView.setFilter(value ?? '');
});

// One subscription for the whole panel, fanned out to the views. Re-publishing
// matters as much as the fan-out: the Plans filter is contributed only outside
// Simple mode, so a flip has to add or drop that item.
onModeChange((mode) => {
  plansView.onModeChange(mode);
  queueView.onModeChange(mode);
  publishContribution();
});

// ---------------------------------------------------------------------------
// Boot
// ---------------------------------------------------------------------------

/**
 * The `?run_id=` deep link. Panels inside this bundle no longer need it — a
 * queued run is one tab away — but it is still how an outside link (an agent
 * message, a bookmark, another interface) names a run, so it is still honored:
 * the run is selected and the Results tab opens on it.
 *
 * @returns {string|null}
 */
function initialRunIdFromUrl() {
  try {
    return new URLSearchParams(window.location.search).get('run_id');
  } catch {
    return null;
  }
}

const initialRunId = initialRunIdFromUrl();
if (initialRunId) {
  selectRun(initialRunId);
  setActiveView('results');
} else {
  publishContribution();
}
