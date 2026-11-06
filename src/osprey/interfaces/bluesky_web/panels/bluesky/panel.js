// @ts-check
/**
 * BLUESKY panel — the shell.
 *
 * One panel, three views: Plans (compose a plan), Queue (what the queue server
 * is holding) and Results (the selected run's record, table and trace). This
 * module owns only what is common to all three — which one is showing, the
 * run the Results view is following, the toolbar's view switcher and lane
 * picker, and the panel's ⋯ entries. Each view is a factory that takes the
 * panel root and its callbacks; nothing here reaches into a view's internals.
 *
 * The panel toolbar (DESIGN.md §12, "Panel chrome standard") is the panel's
 * one header row, embedded and standalone alike: the plan sidebar's toggle
 * and the Plans | Queue | Results switcher at its start; the lane picker
 * (with more than one lane), the queue status chip, Start (Queue view only)
 * and the two halts at its end. The chip and the halts are queue-view.js's.
 * Anything safety-bearing has to be on screen wherever the panel runs, so
 * none of it is contributed to the tile bar, and nothing about the toolbar's
 * width folds or hides the halts or the lane picker.
 *
 * The tile bar gets ONE contribution: a v3 `menu` with the two Clears, which
 * the hub lists in the tile ⋯ under "This panel". A standalone or popped-out
 * page has no tile bar, so it carries its own ⋯ at the toolbar's far right
 * with the same entries and the same handler.
 *
 * - No `innerHTML`, anywhere. Plan names, parameter values, and refusal
 *   sentences all originate off-panel (an agent's draft, the bridge, another
 *   operator), and they reach the DOM only as text nodes. Row identity travels
 *   in `data-*` attributes read by delegated listeners, never interpolated
 *   into markup.
 * - No local copy of the bridge's arming policy. The panel never decides
 *   whether a deployment may execute; it asks, and when the bridge refuses it
 *   shows the bridge's own sentence VERBATIM — that sentence carries the
 *   remedy (which token is missing, or the
 *   `osprey set config.control_system.connector.virtual_accelerator.serving=served`
 *   flip command for a deployment serving the simulator in process).
 *
 * @module panel
 */

import { openContextMenu } from '/design-system/js/context-menu.js';
import { panelApiPrefix } from '/design-system/js/dom.js';
import { contributeHeader, onHeaderAction } from '/design-system/js/header-contrib.js';
import { icon } from '/design-system/js/icons.js';
import { initTooltips } from '/design-system/js/tooltip.js';

import {
  laneIsKnown,
  laneOptions,
  laneSearch,
  parseLaneRoster,
  resolveLaneFromSearch,
  withLane,
} from './lane-client.js';
import { createPlansView } from './plans-view.js';
import { MENU_ID } from './queue-client.js';
import { createQueueView } from './queue-view.js';
import { createResultsView } from './results-view.js';
import { setChannelCatalog } from './schema-form.js';

/** @typedef {'plans'|'queue'|'results'} ViewId */
/** @typedef {import('/design-system/js/header-contrib.js').HeaderMenuEntry} HeaderMenuEntry */

/**
 * Appended to the Results switcher button while data is arriving for a run the
 * operator is not currently watching. A suffix on the label is the honest way
 * to say "there is something here" without a second control beside it.
 */
const ACTIVITY_MARKER = ' •';

const PREFIX = panelApiPrefix();

/**
 * The PLAN LANE this document is bound to, read from its own URL once at
 * boot. One lane per document, deliberately (see lane-client.js): every
 * request `api()` builds carries it, so the three views, both SSE streams and
 * the channel catalog can never mix two machines' state, and a lane switch is
 * a navigation rather than an in-place mutation.
 *
 * @type {string}
 */
const CURRENT_LANE = (() => {
  try {
    return resolveLaneFromSearch(window.location.search);
  } catch {
    return resolveLaneFromSearch('');
  }
})();

/**
 * @param {string} path
 * @returns {string}
 */
function api(path) {
  return `${PREFIX}${withLane(path, CURRENT_LANE)}`;
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

// The design system's tooltip layer: the field-help Info icons' `data-tip`,
// the sidebar toggle slot's reason, and every control's `title`, in one look.
initTooltips(document.body);

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
/**
 * The ⋯ entries as the queue view last reported them (`onClearsChange`): what
 * the tile ⋯ and the standalone ⋯ both list, and what `runMenuEntry` accepts.
 *
 * @type {HeaderMenuEntry[]}
 */
let menuEntries = [];

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
    removeButton: /** @type {HTMLButtonElement} */ (byId('run-remove-btn')),
  },
  onPollingChange(polling) {
    if (polling === resultsPolling) return;
    resultsPolling = polling;
    renderActivityMarker();
  },
  onRemove(runId) {
    // The write is the queue view's (`DELETE /runs/{id}`, banner, history
    // re-read); a success deselects through `onSelectRun(null)` below.
    void queueView.removeRun(runId);
  },
});

/**
 * The Plans view, once built. The queue view is created first (it owns the
 * toolbar's queue controls every view shares) and reports queue state to
 * Plans through this reference, so the enqueue button's label can say what
 * the click does. Frames arrive after boot; the boot render itself reports
 * nothing.
 *
 * @type {import('./plans-view.js').PlansView|null}
 */
let plansViewRef = null;

const queueView = createQueueView({
  root,
  api,
  onSelectRun(runId, activate) {
    selectRun(runId);
    // An operator's click on a run is a request to SEE it; following a finished
    // run is the panel's own housekeeping and must not move anyone off the view
    // they chose.
    if (activate) setActiveView('results');
  },
  onStatus(status) {
    if (plansViewRef) plansViewRef.setQueueStatus(status);
  },
  onClearsChange(entries) {
    menuEntries = entries;
    publishContribution();
  },
});

/**
 * How long a plan form will wait for the channel catalog before rendering
 * without suggestions.
 *
 * The catalog is read *synchronously* when a field is built and a form is
 * never retrofitted afterwards (see schema-form.js), so a form built while the
 * fetch is still in flight loses its comboboxes for good. Both fetches start
 * together at boot and the form is two round trips deep against the catalog's
 * one, so this deadline is normally never reached — it exists only so a
 * `/channels` endpoint that hangs (as opposed to 404ing, which resolves fast)
 * degrades to "no suggestions" instead of withholding the form entirely.
 *
 * @type {number}
 */
const CHANNEL_CATALOG_DEADLINE_MS = 2000;

/**
 * Resolve once the channel catalog has been installed, or once the deadline
 * passes — whichever comes first. Never rejects: `loadChannelCatalog` already
 * treats every failure as "no catalog", and a form must render regardless.
 *
 * @type {Promise<void>}
 */
const channelCatalogReady = Promise.race([
  loadChannelCatalog(),
  new Promise((resolve) => {
    setTimeout(resolve, CHANNEL_CATALOG_DEADLINE_MS);
  }),
]);

const plansView = createPlansView({
  root,
  api,
  channelCatalogReady,
  onOpenRun(runId) {
    selectRun(runId);
    setActiveView('results');
  },
  // Retry on a connecting or unreachable state: the Plans view re-reads its
  // own plans and capability; the queue stream is dialled again here.
  onRetry() {
    queueView.reconnect();
  },
});
plansViewRef = plansView;

// ---------------------------------------------------------------------------
// Selection
// ---------------------------------------------------------------------------

/**
 * Follow `runId` in the Results view and highlight it in the queue/history
 * lists. Selecting does NOT switch views on its own — callers decide, because
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
 * Show one view and hide the other two, and press its switcher button.
 *
 * @param {string} id A view id; anything else is ignored.
 */
function setActiveView(id) {
  if (!VIEWS.some((entry) => entry.id === id)) return;
  activeView = /** @type {ViewId} */ (id);
  for (const entry of VIEWS) {
    const active = entry.id === activeView;
    entry.tab.setAttribute('aria-pressed', active ? 'true' : 'false');
    entry.view.hidden = !active;
  }
  // The toolbar's queue controls are shown per view (stripControls): the
  // Queue view is the control panel, the others show a dormant queue as its
  // chip alone. The plan sidebar opens on Plans only, and its toggle stays in
  // its slot, disabled, everywhere else.
  queueView.setView(activeView);
  plansView.setShown(activeView === 'plans');
  renderActivityMarker();
}

/**
 * Whether the Results button should carry its activity marker: data is
 * arriving for a run the operator is not looking at. Activating the view, or
 * the poll settling, clears it — both are already inputs to this expression,
 * so there is no marker state to forget to reset.
 *
 * @returns {boolean}
 */
function resultsHaveUnwatchedActivity() {
  return resultsPolling && activeView !== 'results';
}

function renderActivityMarker() {
  for (const entry of VIEWS) {
    if (entry.id !== 'results') continue;
    entry.tab.textContent = resultsHaveUnwatchedActivity()
      ? `${entry.label}${ACTIVITY_MARKER}`
      : entry.label;
  }
}

// ---------------------------------------------------------------------------
// The panel's ⋯ entries
// ---------------------------------------------------------------------------

/**
 * (Re)send the WHOLE contribution: one v3 `menu`, the tile ⋯'s "This panel"
 * section. The hub renders only what the last one says — a contribution is an
 * idempotent replace, never a diff. A no-op standalone.
 */
function publishContribution() {
  contributeHeader([{ kind: 'menu', id: MENU_ID, items: menuEntries }]);
}

/**
 * Run one ⋯ entry, from the hub's tile ⋯ or the standalone ⋯ alike. The two
 * Clears only ARM: the panel switches to the Queue view, where the confirm
 * note sits at the top of the card the Clear empties.
 *
 * @param {string} entry An entry id; anything the menu did not list is ignored
 *   (the hub's action value is data from another frame, not a promise).
 */
function runMenuEntry(entry) {
  if (!menuEntries.some((item) => item.id === entry)) return;
  setActiveView('queue');
  queueView.armClear(entry);
}

/**
 * The standalone page's own ⋯: a popped-out or directly opened page has no
 * tile bar, and the entries must stay reachable there. The same entries,
 * through the design system's context menu, to the same handler. Never built
 * embedded — a tile has exactly one ⋯, the hub's.
 */
function mountStandaloneMenu() {
  if (document.body.classList.contains('embedded')) return;
  const button = document.createElement('button');
  button.type = 'button';
  button.id = 'panel-menu-btn';
  button.className = 'osprey-icon-btn';
  button.setAttribute('aria-label', 'Panel menu');
  button.setAttribute('aria-haspopup', 'menu');
  button.setAttribute('aria-expanded', 'false');
  button.title = 'Panel menu';
  button.append(icon('more'));
  button.addEventListener('click', () => {
    /** @type {import('/design-system/js/context-menu.js').MenuItem[]} */
    const rows = [];
    for (const entry of menuEntries) {
      if (entry.separator) rows.push({ divider: true });
      rows.push({
        label: entry.label,
        checked: entry.checked,
        disabled: entry.disabled,
        danger: entry.danger,
        run: () => runMenuEntry(entry.id),
      });
    }
    openContextMenu({
      x: 0,
      y: 0,
      triggerEl: button,
      anchorEl: button,
      ariaLabel: 'Panel menu',
      items: rows,
    });
  });
  byId('panel-toolbar-end').append(button);
}

// ---------------------------------------------------------------------------
// Wiring
// ---------------------------------------------------------------------------

byId('view-switch').addEventListener('click', (event) => {
  const target = event.target;
  if (!(target instanceof Element)) return;
  const button = target.closest('button[data-view]');
  if (!(button instanceof HTMLElement) || !button.dataset.view) return;
  setActiveView(button.dataset.view);
});

onHeaderAction((id, value) => {
  if (id === MENU_ID && value) runMenuEntry(value);
});

mountStandaloneMenu();

// ---------------------------------------------------------------------------
// Boot
// ---------------------------------------------------------------------------

/**
 * The `?run_id=` deep link. Panels inside this bundle no longer need it — a
 * queued run is one view away — but it is still how an outside link (an agent
 * message, a bookmark, another interface) names a run, so it is still honored:
 * the run is selected and the Results view opens on it.
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
  // Through setActiveView even for the default, so the switcher, the queue
  // controls and the sidebar agree with the view from first paint.
  setActiveView('plans');
}

/**
 * Fetch the deployment's channel catalog — once per panel load — and hand it
 * to the schema-form module, so plan forms offer suggestions on
 * channel-tagged fields. The endpoint is optional: a 404 (no catalog
 * deployed), a network failure, a malformed payload, and an empty list all
 * mean the same thing — no suggestions, forms exactly as they always were —
 * so none of them is worth a console line.
 *
 * Awaited (via `channelCatalogReady`) before a plan form is rendered rather
 * than left to race it: the catalog is read synchronously at field-build time
 * and a form is never retrofitted, so losing that race cost the form its
 * comboboxes for the rest of the panel's life.
 *
 * @returns {Promise<void>} Resolves when the catalog is installed or skipped.
 */
async function loadChannelCatalog() {
  try {
    const response = await fetch(api('/channels'));
    if (!response.ok) return;
    const channels = await response.json();
    if (
      Array.isArray(channels) &&
      channels.length > 0 &&
      channels.every((entry) => typeof entry === 'string')
    ) {
      setChannelCatalog(channels);
    }
  } catch {
    // Optional endpoint: absence is a normal deployment state, not an error.
  }
}

/**
 * Render the lane picker from one roster: a small select in the toolbar,
 * left of the queue chip, one option per lane labelled by the control target
 * it drives, the current one selected. Picking another lane NAVIGATES —
 * `laneSearch` rewrites only the `lane` parameter, so the host's
 * `?embedded=`/`?mode=`/`?theme=` survive — because a lane is a different
 * machine and this panel binds one lane per document (see lane-client.js).
 *
 * In the toolbar, deliberately: which machine this panel is pointed at is
 * safety-bearing context for the queue chip and the two halts beside it, must
 * be visible from every view, and never folds.
 *
 * @param {import('./lane-client.js').LaneEntry[]} roster
 */
function renderLanePicker(roster) {
  const select = /** @type {HTMLSelectElement} */ (byId('lane-select'));
  // Text nodes, never markup: the labels originate in deployment config,
  // which is off-panel input like everything else this bundle renders.
  select.replaceChildren(
    ...laneOptions(roster, CURRENT_LANE).map((option) => {
      const node = document.createElement('option');
      node.value = option.value;
      node.textContent = option.label;
      node.selected = option.selected;
      node.disabled = option.disabled;
      return node;
    })
  );
  select.title = `This panel is on the '${CURRENT_LANE}' plan lane.`;
  select.addEventListener('change', () => {
    if (select.value === CURRENT_LANE) return;
    window.location.search = laneSearch(window.location.search, select.value);
  });
  select.hidden = false;
}

/**
 * Fetch the sidecar's lane roster — once per panel load — and show the lane
 * picker when there is more than one lane to pick. A single-lane deployment
 * (every deployment until a second lane is opted in) takes the early return
 * and renders exactly the panel it always has; so does any failure to read
 * the roster, which is the direction that cannot invent a lane.
 *
 * A document pinned to a lane the roster does not know is said out loud
 * instead of being silently rerouted: every request is already 404ing at the
 * sidecar (`unknown bluesky lane`), and this banner is the one place that
 * explains why. Wrong-machine silence is the failure mode; a loud refusal is
 * recoverable.
 *
 * @returns {Promise<void>}
 */
async function loadLaneRoster() {
  let roster;
  try {
    const response = await fetch(`${PREFIX}/lanes`);
    if (!response.ok) return;
    roster = parseLaneRoster(await response.json());
  } catch {
    return;
  }
  if (!laneIsKnown(roster, CURRENT_LANE)) {
    const banner = byId('lane-banner');
    banner.textContent =
      `This deployment renders no '${CURRENT_LANE}' plan lane — every request from ` +
      'this panel is being refused. Pick a rendered lane in the toolbar.';
    banner.hidden = false;
  } else if (roster.length < 2) {
    return;
  }
  renderLanePicker(roster);
}

void loadLaneRoster();
