// @ts-check
/**
 * BLUESKY panel — DOM wiring.
 *
 * Two halves over one bridge. The top half is the live queue: the SSE relay
 * `GET /queue/events` pushes a full snapshot on every change (~1 s), pending
 * items can be reordered or removed, the queue can be started or stopped, and
 * a plan already moving hardware can be aborted.
 * The bottom half follows one run's results (`results-view.js`), driven by
 * clicking any queued, running, or completed run — or by a `?run_id=` deep
 * link, which is how the PLAN panel hands a freshly-queued run over.
 *
 * Every fetch is prefix-relative (`panelApiPrefix()`), so the panel works
 * unmodified whether it is opened directly on the sidecar or embedded through
 * the web terminal's reverse proxy.
 *
 * Two house rules this file keeps deliberately:
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
 */

import { panelApiPrefix } from '/design-system/js/dom.js';
import { onModeChange } from '/design-system/js/frame-params.js';

import {
  abortButtonClass,
  abortButtonLabel,
  abortControl,
  abortOutcomeTone,
  abortRefusalTone,
  abortSuccessMessage,
  classifyQueueRefusal,
  createInitialQueueState,
  createQueueStream,
  describeProgress,
  describeQueueStatus,
  historyChanged,
  historyEmptyState,
  historyRecords,
  itemParamSummary,
  itemRunId,
  moveDownBody,
  moveUpBody,
  queueControls,
  queueEmptyState,
  reduceQueueFrame,
  refusalTone,
  stopButtonClass,
  stopButtonLabel,
  writeOutcomeTone,
} from './queue-client.js';
import { badgeToneForStatus, createResultsView } from './results-view.js';

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

const queueBadge = byId('queue-state-badge');
const queueNote = byId('queue-note');
const queueBanner = byId('queue-banner');
const startBtn = /** @type {HTMLButtonElement} */ (byId('start-btn'));
const stopBtn = /** @type {HTMLButtonElement} */ (byId('stop-btn'));
const stopNote = byId('stop-note');
const abortBtn = /** @type {HTMLButtonElement} */ (byId('abort-btn'));
const abortNote = byId('abort-note');
const runningCard = byId('running-card');
const runningSelect = /** @type {HTMLButtonElement} */ (byId('running-select'));
const runningName = byId('running-name');
const runningParams = byId('running-params');
const runningProgressBar = byId('running-progress-bar');
const runningProgressFill = byId('running-progress-fill');
const runningProgressLabel = byId('running-progress-label');
const queueList = /** @type {HTMLUListElement} */ (byId('queue-items'));
const queueEmpty = byId('queue-empty');
const historyList = /** @type {HTMLUListElement} */ (byId('history-items'));
const historyEmpty = byId('history-empty');

const resultsView = createResultsView({
  api,
  elements: {
    statusBadge: byId('run-status-badge'),
    meta: byId('run-meta'),
    note: byId('run-note'),
    emptyState: byId('results-empty'),
    tableCard: byId('table-card'),
    tableNote: byId('table-note'),
    tableHeadRow: /** @type {HTMLTableRowElement} */ (byId('table-head-row')),
    tableBody: /** @type {HTMLTableSectionElement} */ (byId('table-body')),
    chartCard: byId('chart-card'),
    chartSvg: /** @type {SVGSVGElement} */ (
      /** @type {unknown} */ (document.getElementById('results-chart'))
    ),
    chartLegend: byId('chart-legend'),
  },
});

/** @type {{
 *   queue: ReturnType<typeof createInitialQueueState>,
 *   history: ReturnType<typeof historyRecords>,
 *   historyLoaded: boolean,
 *   selected: string|null,
 * }} */
const state = {
  queue: createInitialQueueState(),
  history: [],
  // Whether a `GET /runs` fetch has EVER succeeded. Until one has, the panel
  // has not read history and must not claim there is none.
  historyLoaded: false,
  selected: null,
};

/**
 * Transient "the withdrawal has been clicked once" flag — panel-local UI
 * state, never derived from the queue. Only meaningful while the stop control
 * is on its arming branch.
 *
 * Cleared by the stop button's own handler, and whenever the pending stop it
 * refers to stops being observable: a frame reporting `queue_stop_pending` no
 * longer true, or the stream dropping. Deliberately NOT cleared by the other
 * queue writes — reordering or removing an item while a stop is pending says
 * nothing about the withdrawal, and disarming there would make the operator
 * click twice again for no reason.
 */
let stopConfirmArmed = false;

/**
 * The same transient flag for the ABORT control's second click.
 *
 * Separate from `stopConfirmArmed` on purpose: the two buttons sit next to
 * each other, and sharing one flag would let a click on one arm the other.
 *
 * Cleared by the abort button's own handler, whenever a frame reports no
 * running item (there is nothing left for the confirm to refer to), and
 * whenever the stream drops (the panel can no longer see what it would be
 * aborting). Every clearing path is in the SAFE direction — it costs a second
 * click, it never fires an abort nobody asked for.
 */
let abortConfirmArmed = false;

// ---------------------------------------------------------------------------
// Banner
// ---------------------------------------------------------------------------

/**
 * @param {'ok'|'warn'|'err'|'info'|null} tone Pass `null` to clear.
 * @param {string} [message]
 */
function setBanner(tone, message = '') {
  if (tone === null) {
    queueBanner.hidden = true;
    queueBanner.textContent = '';
    return;
  }
  queueBanner.className = `banner banner-${tone}`;
  queueBanner.textContent = message;
  queueBanner.hidden = false;
}

// ---------------------------------------------------------------------------
// Queue writes
// ---------------------------------------------------------------------------

/**
 * Issue one queue write and surface its outcome.
 *
 * A refused write is not an exception path — an unarmed deployment refusing a
 * start is the system working — so the bridge's sentence is shown as-is and
 * the panel state is left alone. The next SSE frame is the truth about what
 * actually changed; nothing here optimistically re-renders.
 *
 * @param {string} method
 * @param {string} path
 * @param {unknown} [body]
 * @param {string|((parsed: any) => string)} [okMessage] A function receives the
 *   parsed success body, so a write whose outcome the bridge describes (the
 *   abort) can relay the bridge's own `msg` instead of a local substitute.
 * @param {'ok'|'warn'} [okTone] Tone for the success banner, decided by the
 *   pure module (`writeOutcomeTone` / `abortOutcomeTone`) and never by the
 *   request itself.
 * @param {(code: string|null) => 'warn'|'err'} [toneForRefusal] Tone for a
 *   REFUSED write, same rule: decided by the pure module. Defaults to
 *   `refusalTone`; the emergency abort passes `abortRefusalTone`, under which a
 *   refusal means the machine was not stopped rather than a policy said no.
 * @returns {Promise<boolean>}
 */
async function queueWrite(
  method,
  path,
  body,
  okMessage,
  okTone = 'ok',
  toneForRefusal = refusalTone
) {
  setBanner(null);
  /** @type {Response} */
  let response;
  try {
    response = await fetch(api(path), {
      method,
      headers: body === undefined ? {} : { 'content-type': 'application/json' },
      body: body === undefined ? undefined : JSON.stringify(body),
    });
  } catch {
    setBanner('err', 'The panel could not reach its own sidecar. Check that the service is up.');
    return false;
  }

  let parsed = null;
  try {
    parsed = await response.json();
  } catch {
    parsed = null;
  }

  if (response.ok) {
    const message = typeof okMessage === 'function' ? okMessage(parsed) : okMessage;
    if (message) setBanner(okTone, message);
    return true;
  }

  const refusal = classifyQueueRefusal(response.status, parsed);
  setBanner(toneForRefusal(refusal.code), refusal.message);
  return false;
}

// ---------------------------------------------------------------------------
// Rendering: queue half
// ---------------------------------------------------------------------------

function renderQueue() {
  const { queue } = state;
  const described = describeQueueStatus(queue.status);
  queueBadge.textContent = queue.connected ? described.label : 'connecting…';
  queueBadge.className = `badge ${queue.connected ? described.tone : 'warn'}`;

  const notes = [];
  if (!queue.connected) notes.push('Waiting for the queue stream.');
  if (described.note) notes.push(described.note);
  queueNote.textContent = notes.join(' ');
  queueNote.hidden = notes.length === 0;

  const controls = queueControls(queue);
  startBtn.disabled = controls.start.disabled;
  startBtn.title = controls.start.reason || '';

  // The stop button is never disabled — see `queueControls`. Whatever this
  // panel believes about the manager is a tooltip, not a gate.
  stopBtn.textContent = stopButtonLabel(controls.stop, stopConfirmArmed);
  stopBtn.className = stopButtonClass(controls.stop, stopConfirmArmed);
  stopBtn.title = controls.stop.note || '';
  stopBtn.dataset.cancel = controls.stop.body.cancel ? 'true' : 'false';
  setNote(stopNote, stopConfirmArmed ? controls.stop.note : null);

  // The abort button is never disabled either — see `abortControl`. It carries
  // no `disabled` assignment here for exactly that reason, and its note is
  // shown once the confirm is armed, when the operator is about to commit.
  const abort = abortControl(queue);
  abortBtn.textContent = abortButtonLabel(abortConfirmArmed);
  abortBtn.className = abortButtonClass(abortConfirmArmed);
  abortBtn.title = abort.note || '';
  setNote(abortNote, abortConfirmArmed ? abort.note : null);

  renderRunningItem();
  renderPendingItems();
}

/**
 * @param {HTMLElement} node
 * @param {string|null} message
 */
function setNote(node, message) {
  node.textContent = message || '';
  node.hidden = !message;
}

function renderRunningItem() {
  const item = state.queue.runningItem;
  if (item === null) {
    runningCard.hidden = true;
    return;
  }
  runningCard.hidden = false;
  runningName.textContent = typeof item.name === 'string' ? item.name : '(unnamed plan)';
  runningParams.textContent = itemParamSummary(item);

  const runId = itemRunId(item);
  // An item enqueued out of band carries no OSPREY run id, so there is no run
  // to open below — it is shown, but not selectable.
  if (runId === null) runningSelect.removeAttribute('data-run-id');
  else runningSelect.dataset.runId = runId;
  runningSelect.disabled = runId === null;
  runningCard.classList.toggle('selected', runId !== null && runId === state.selected);

  const progress = describeProgress(item.progress);
  runningProgressLabel.textContent = progress.label;
  // An unknown denominator renders as an indeterminate bar, never as 0% — a
  // scan with no predicted point count (the common case for an agent-authored
  // session plan) is running fine, and showing it at zero reads as stalled.
  const indeterminate = progress.mode === 'indeterminate';
  runningProgressBar.classList.toggle('indeterminate', indeterminate);
  runningProgressFill.style.width = indeterminate ? '100%' : `${progress.percent ?? 0}%`;
  // An ABSENT aria-valuenow is how ARIA spells "indeterminate" — assistive
  // tech announces a busy bar rather than a number the panel does not have.
  if (indeterminate) runningProgressBar.removeAttribute('aria-valuenow');
  else runningProgressBar.setAttribute('aria-valuenow', String(progress.percent ?? 0));
}

function renderPendingItems() {
  const items = state.queue.items;
  queueList.replaceChildren();
  const empty = queueEmptyState(state.queue);
  queueEmpty.textContent = empty.message;
  queueEmpty.hidden = empty.hidden;

  items.forEach((item, index) => {
    const runId = itemRunId(item);
    const row = document.createElement('li');
    row.className = 'queue-row';
    if (runId !== null && runId === state.selected) row.classList.add('selected');

    const position = document.createElement('span');
    position.className = 'queue-position';
    position.textContent = String(index + 1);

    const label = selectLabel(
      runId,
      typeof item.name === 'string' ? item.name : '(unnamed plan)',
      itemParamSummary(item)
    );

    const actions = document.createElement('span');
    actions.className = 'queue-actions';
    const uid = typeof item.item_uid === 'string' ? item.item_uid : '';
    actions.append(
      iconButton('up', '↑', 'Move towards the front', uid, index, index === 0),
      iconButton('down', '↓', 'Move towards the back', uid, index, index === items.length - 1),
      iconButton('remove', '✕', 'Remove from the queue', uid, index, uid === '')
    );

    row.append(position, label, actions);
    queueList.appendChild(row);
  });
}

/**
 * A row's selectable label. A real `<button>` rather than a click handler on
 * the row: that is what makes the queue keyboard-reachable and screen-reader
 * announceable for free, and it matches the PLAN panel's plan rows.
 *
 * @param {string|null} runId `null` for an item with no run to open.
 * @param {string} name
 * @param {string} detail
 * @returns {HTMLElement}
 */
function selectLabel(runId, name, detail) {
  const host = document.createElement(runId === null ? 'span' : 'button');
  host.className = 'queue-label';
  if (host instanceof HTMLButtonElement) {
    host.type = 'button';
    host.dataset.runId = runId || '';
    host.setAttribute('aria-pressed', runId === state.selected ? 'true' : 'false');
  }

  const nameEl = document.createElement('span');
  nameEl.className = 'queue-name';
  nameEl.textContent = name;
  const detailEl = document.createElement('span');
  detailEl.className = 'queue-params';
  detailEl.textContent = detail;
  host.append(nameEl, detailEl);
  return host;
}

/**
 * @param {string} action
 * @param {string} glyph
 * @param {string} title
 * @param {string} uid
 * @param {number} index
 * @param {boolean} disabled
 * @returns {HTMLButtonElement}
 */
function iconButton(action, glyph, title, uid, index, disabled) {
  const button = document.createElement('button');
  button.type = 'button';
  button.className = `icon-btn icon-${action}`;
  button.textContent = glyph;
  button.title = title;
  button.setAttribute('aria-label', title);
  button.disabled = disabled;
  button.dataset.action = action;
  button.dataset.uid = uid;
  button.dataset.index = String(index);
  return button;
}

// ---------------------------------------------------------------------------
// Rendering: completed runs
// ---------------------------------------------------------------------------

function renderHistory() {
  historyList.replaceChildren();
  const empty = historyEmptyState(state.history, state.historyLoaded);
  historyEmpty.textContent = empty.message;
  historyEmpty.hidden = empty.hidden;

  for (const record of state.history) {
    const row = document.createElement('li');
    row.className = 'queue-row';
    if (record.id === state.selected) row.classList.add('selected');

    const badge = document.createElement('span');
    badge.className = `badge ${badgeToneForStatus(record.status)}`;
    badge.textContent = record.status;

    row.append(badge, selectLabel(record.id, record.planName || '(unnamed plan)', record.error || record.id));
    historyList.appendChild(row);
  }
}

async function refreshHistory() {
  try {
    const response = await fetch(api('/runs'));
    if (!response.ok) {
      renderHistory();
      return;
    }
    state.history = historyRecords(await response.json());
    state.historyLoaded = true;
  } catch {
    // Leave the last known list up — a transient failure after a good read is
    // still best answered with what was last read. What must NOT happen is the
    // list silently reading as "no completed runs" when it was never loaded,
    // so re-render either way and let `historyEmptyState` say which it is.
    renderHistory();
    return;
  }
  renderHistory();
  maybeAutoSelectLatest();
}

// ---------------------------------------------------------------------------
// Selection
// ---------------------------------------------------------------------------

/** @param {string|null} runId */
function selectRun(runId) {
  state.selected = runId;
  resultsView.follow(runId);
  renderQueue();
  renderHistory();
}

/**
 * In Simple mode the operator shouldn't have to hunt for the newest run:
 * surface the running one, or failing that the newest completed one, as soon
 * as either exists. No-op in Expert, and no-op once anything is selected (a
 * deep link or a manual pick).
 */
function maybeAutoSelectLatest() {
  if (document.documentElement.dataset.uiMode !== 'simple') return;
  if (state.selected) return;
  const running = itemRunId(state.queue.runningItem);
  const candidate = running || (state.history.length > 0 ? state.history[0].id : null);
  if (candidate) selectRun(candidate);
}

// ---------------------------------------------------------------------------
// Wiring
// ---------------------------------------------------------------------------

/**
 * Row clicks (select a run) and item-action clicks (reorder/remove) share one
 * delegated listener, so rows re-rendered by an SSE frame need no re-binding.
 *
 * @param {Event} event
 * @param {HTMLElement} root
 */
function onListClick(event, root) {
  const target = event.target;
  if (!(target instanceof Element)) return;

  const button = target.closest('button[data-action]');
  if (button instanceof HTMLButtonElement && root.contains(button)) {
    void onItemAction(button);
    return;
  }

  const row = target.closest('[data-run-id]');
  if (row instanceof HTMLElement && root.contains(row) && row.dataset.runId) {
    selectRun(row.dataset.runId);
  }
}

/** @param {HTMLButtonElement} button */
async function onItemAction(button) {
  const uid = button.dataset.uid || '';
  const index = Number(button.dataset.index);
  if (uid === '' || Number.isNaN(index)) return;
  const encoded = encodeURIComponent(uid);

  if (button.dataset.action === 'remove') {
    await queueWrite('DELETE', `/queue/items/${encoded}`);
    return;
  }

  const body =
    button.dataset.action === 'up'
      ? moveUpBody(state.queue.items, index)
      : moveDownBody(state.queue.items, index);
  // Already at the end it can move to: the button is disabled for this case,
  // but the queue may have shifted under a click that was already in flight.
  if (body === null) return;
  await queueWrite('POST', `/queue/items/${encoded}/move`, body);
}

queueList.addEventListener('click', (event) => onListClick(event, queueList));
historyList.addEventListener('click', (event) => onListClick(event, historyList));
runningSelect.addEventListener('click', () => {
  if (runningSelect.dataset.runId) selectRun(runningSelect.dataset.runId);
});

startBtn.addEventListener('click', () => {
  if (startBtn.disabled) return;
  void queueWrite(
    'POST',
    '/queue/start',
    {},
    'Queue started — it is now draining toward hardware.',
    writeOutcomeTone(true)
  );
});

// Two-step ONLY on the withdrawal branch, matching the PLAN panel's
// Add-to-queue and discard-draft buttons. A plain stop fires on the first
// click: friction in front of a halt is friction in exactly the wrong place.
stopBtn.addEventListener('click', () => {
  const cancel = stopBtn.dataset.cancel === 'true';
  if (!cancel) {
    stopConfirmArmed = false;
    void queueWrite(
      'POST',
      '/queue/stop',
      { cancel: false },
      'The queue will stop after the running item.',
      writeOutcomeTone(false)
    );
    return;
  }
  if (!stopConfirmArmed) {
    stopConfirmArmed = true;
    renderQueue();
    return;
  }
  stopConfirmArmed = false;
  void queueWrite(
    'POST',
    '/queue/stop',
    { cancel: true },
    'Pending stop withdrawn — the queue keeps draining after the current item.',
    writeOutcomeTone(true)
  );
});

// Two-step, unlike the plain stop one button to its left — see
// `abortButtonLabel` for why the two halts differ. The button itself is live in
// every state; the confirm is friction, not a gate, and no state removes it.
abortBtn.addEventListener('click', () => {
  if (!abortConfirmArmed) {
    abortConfirmArmed = true;
    renderQueue();
    return;
  }
  abortConfirmArmed = false;
  renderQueue();
  void queueWrite(
    'POST',
    '/queue/abort',
    {},
    abortSuccessMessage,
    abortOutcomeTone(),
    // A refused abort is never routine: see `abortRefusalTone`.
    abortRefusalTone
  );
});

onModeChange((mode) => {
  if (mode === 'simple') maybeAutoSelectLatest();
});

/** @returns {string|null} */
function initialRunIdFromUrl() {
  try {
    return new URLSearchParams(window.location.search).get('run_id');
  } catch {
    return null;
  }
}

function init() {
  const initial = initialRunIdFromUrl();
  selectRun(initial);

  void refreshHistory();

  createQueueStream(api('/queue/events'), {
    onFrame(frame) {
      const previousStatus = state.queue.status;
      const next = reduceQueueFrame(state.queue, frame);
      if (next === state.queue) return;
      state.queue = next;
      // An armed withdrawal confirm refers to a specific pending stop. Once
      // that stop is gone the confirm has nothing left to confirm, so it must
      // not sit armed and fire against whatever the queue does next.
      if (next.status?.queue_stop_pending !== true) stopConfirmArmed = false;
      // Same rule for the abort confirm: with no running item there is nothing
      // for it to refer to, so it must not sit armed over whatever runs next.
      if (next.runningItem === null) abortConfirmArmed = false;
      renderQueue();
      // The history list is re-fetched on a manager *history* move rather
      // than on a timer: the SSE summary is the change detector, so a settled
      // queue costs no polling at all.
      if (historyChanged(previousStatus, next.status)) void refreshHistory();
      maybeAutoSelectLatest();
    },
    onConnectionChange(connected) {
      if (connected === state.queue.connected) return;
      state.queue = { ...state.queue, connected };
      // A dropped stream means the pending stop this confirm refers to can no
      // longer be observed. Disarm rather than leave a live withdrawal sitting
      // over a queue whose state is now unknown.
      if (!connected) {
        stopConfirmArmed = false;
        // The abort confirm refers to a run this panel can no longer see.
        abortConfirmArmed = false;
      }
      renderQueue();
    },
  });
}

init();
