// @ts-check
/**
 * BLUESKY panel — the queue view (Queue view) plus the toolbar's queue
 * controls.
 *
 * One bridge, two surfaces. The SSE relay `GET /queue/events` pushes a full
 * snapshot on every change (~1 s); pending items can be reordered or removed,
 * the queue can be started or stopped, and a plan already moving hardware can
 * be aborted. Selecting any queued, running or completed run hands the run id
 * up to the panel shell (`onSelectRun`), which follows it in the Results view.
 *
 * The queue status chip and the two halts do NOT live inside the Queue view —
 * they render into the panel toolbar, which is on screen for every view. This
 * view owns them all the same: they are driven by the same queue frames as
 * the list below them, so splitting their wiring across two modules would be
 * splitting one state machine in half. The same holds for the two Clears: the
 * tile ⋯ only arms them through `armClear`,
 * and their confirm notes sit at the top of the cards they empty.
 *
 * Every fetch is prefix-relative (`api()` from the shell), so the panel works
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
 *   remedy (which token is missing, or the
 *   `osprey set config.control_system.connector.virtual_accelerator.serving=served`
 *   flip command for a deployment serving the simulator in process).
 *
 * @module queue-view
 */

import { openInfoPopover } from '/design-system/js/info-popover.js';
import { writeTitle } from '/design-system/js/tooltip.js';

import {
  abortButtonClass,
  abortButtonLabel,
  abortControl,
  abortOutcomeTone,
  abortRefusalTone,
  abortSuccessMessage,
  classifyQueueRefusal,
  clearControl,
  clearMenuEntries,
  createInitialQueueState,
  createQueueStream,
  describeProgress,
  finishedRunId,
  historyChanged,
  historyClearConfirmNote,
  historyClearControl,
  historyEmptyState,
  historyRecords,
  itemParamSummary,
  itemRunId,
  moveDownBody,
  moveUpBody,
  queueChip,
  queueChipDetail,
  queueClearConfirmNote,
  queueControls,
  queueEmptyState,
  reduceQueueFrame,
  refusalTone,
  removalRecords,
  stopButtonClass,
  stopButtonLabel,
  stripControls,
  writeOutcomeTone,
  CANCEL_CLEAR_LABEL,
  CONFIRM_CLEAR_LABEL,
  CONTROL_HINTS,
  MENU_ENTRY,
  START_LABEL,
} from './queue-client.js';
import { badgeToneForStatus } from './results-view.js';

/**
 * @typedef {object} QueueViewDeps
 * @property {HTMLElement} root  The panel root; every element this view drives
 *   is looked up beneath it, including the toolbar controls that sit OUTSIDE
 *   the Queue view's own container.
 * @property {(path: string) => string} api  Prefix-relative URL builder.
 * @property {(runId: string|null, activate: boolean) => void} onSelectRun  A run
 *   was picked, or (`null`) the selected run was removed and nothing is
 *   selected now. `activate` is true for an operator's click — which opens the
 *   Results view — and false for following a finished run, which fills the
 *   Results view without yanking the operator off the view they are on.
 * @property {(status: import('./queue-client.js').QueueStatus|null) => void} [onStatus]
 *   The manager summary changed (or the stream dropped: `null`). The shell
 *   forwards it to the Plans view, whose enqueue button says what the click
 *   does in the queue's current state — from the same frames as the chip.
 * @property {(entries: ReturnType<typeof clearMenuEntries>) => void} [onClearsChange]
 *   What the two ⋯ Clear entries say changed (a count, or whether they are
 *   live). The shell republishes its menu from these entries.
 */

/**
 * @typedef {object} QueueView
 * @property {(runId: string|null) => void} setSelected  Adopt the shell's
 *   current selection so the queue and history rows highlight it.
 * @property {(view: string) => void} setView  The view now showing. Decides
 *   which of the toolbar's queue controls are on screen (`stripControls`).
 * @property {(runId: string) => Promise<boolean>} removeRun  Drop one finished
 *   run from history; the Results view's "Remove from history" lands here.
 * @property {(entry: string) => void} armClear  A ⋯ Clear entry was picked
 *   (`MENU_ENTRY`): arm that Clear's confirm at the top of its card.
 * @property {() => ReturnType<typeof clearMenuEntries>} menuEntries  The two
 *   ⋯ Clear entries as they stand now.
 * @property {() => void} reconnect  Dial the queue stream again now (Retry).
 */

/**
 * @param {QueueViewDeps} deps
 * @returns {QueueView}
 */
export function createQueueView({
  root,
  api,
  onSelectRun,
  onStatus = () => {},
  onClearsChange = () => {},
}) {
  /**
   * The panel owns its own shell, so every id below is present by
   * construction — a missing one is a bundle bug, not a runtime condition to
   * branch on.
   *
   * @param {string} id
   * @returns {HTMLElement}
   */
  function byId(id) {
    return /** @type {HTMLElement} */ (root.querySelector(`#${id}`));
  }

  const queueChipEl = byId('queue-state-chip');
  const queueChipLabel = byId('queue-state-label');
  const queueBanner = byId('queue-banner');
  const startBtn = /** @type {HTMLButtonElement} */ (byId('start-btn'));
  const stopBtn = /** @type {HTMLButtonElement} */ (byId('stop-btn'));
  const stopNote = byId('stop-note');
  const abortBtn = /** @type {HTMLButtonElement} */ (byId('abort-btn'));
  const abortNote = byId('abort-note');
  const clearNote = byId('clear-note');
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
  const historyClearNote = byId('history-clear-note');

  /** @type {{
   *   queue: ReturnType<typeof createInitialQueueState>,
   *   history: ReturnType<typeof historyRecords>,
   *   historyLoaded: boolean,
   *   removals: ReturnType<typeof removalRecords>,
   *   selected: string|null,
   * }} */
  const state = {
    queue: createInitialQueueState(),
    history: [],
    // Whether a `GET /runs` fetch has EVER succeeded. Until one has, the panel
    // has not read history and must not claim there is none.
    historyLoaded: false,
    // Queue work somebody withdrew. No "loaded" flag beside it: an empty list
    // and an unread one say the same thing here, which is nothing at all.
    removals: [],
    selected: null,
  };

  /**
   * Transient "the withdrawal has been clicked once" flag — panel-local UI
   * state, never derived from the queue. Only meaningful while the stop
   * control is on its arming branch.
   *
   * Cleared by the stop button's own handler, and whenever the pending stop it
   * refers to stops being observable: a frame reporting `queue_stop_pending`
   * no longer true, or the stream dropping. Deliberately NOT cleared by the
   * other queue writes — reordering or removing an item while a stop is
   * pending says nothing about the withdrawal, and disarming there would make
   * the operator click twice again for no reason.
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
   * aborting). Every clearing path is in the SAFE direction — it costs a
   * second click, it never fires an abort nobody asked for.
   */
  let abortConfirmArmed = false;

  /**
   * The view on screen, as the shell reports it (`setView`). Decides which of
   * the toolbar's queue controls show (`stripControls`): all of them on the
   * Queue view, only what is moving on the other two. Starts as Plans, the
   * shell's boot default.
   *
   * @type {string}
   */
  let activeView = 'plans';

  /**
   * The queue Clear's "armed" flag — set by its ⋯ entry (`armClear`), and the
   * Confirm in its note sends the clear. Separate from the abort's for the
   * same reason the two halts' flags are separate. Cleared by Confirm and
   * Cancel, whenever a frame shows nothing waiting (there is nothing left to
   * clear), whenever the clear goes dead, and when the stream drops.
   */
  let clearConfirmArmed = false;

  /** The history Clear's "armed" flag — the same two-step as the queue's. */
  let historyClearConfirmArmed = false;

  /**
   * A success banner leaves by itself; a refusal stays until the next write.
   * The queue's status chip is the durable answer to "did it work" — the
   * banner is the receipt, and a receipt that never goes away reads as a
   * state of its own.
   */
  const SUCCESS_BANNER_MS = 5000;
  /** @type {ReturnType<typeof setTimeout>|null} */
  let bannerTimer = null;

  /** The ⋯ entries as last reported, so a frame that changes neither sends nothing. */
  let reportedEntries = '';

  /**
   * Build a Clear's confirm note into `host`, at the top of the card it
   * empties: the sentence saying what goes, the Confirm that sends it, and
   * Cancel. The ⋯ entry only arms; this Confirm is the second step.
   *
   * @param {HTMLElement} host
   * @param {() => void} onConfirm
   * @param {() => void} onCancel
   * @returns {(sentence: string|null) => void} Show the note with `sentence`,
   *   or hide it on `null`.
   */
  function confirmNote(host, onConfirm, onCancel) {
    const text = document.createElement('span');
    text.className = 'confirm-note-text';
    const confirm = document.createElement('button');
    confirm.type = 'button';
    confirm.className = 'osprey-btn confirm';
    confirm.textContent = CONFIRM_CLEAR_LABEL;
    confirm.addEventListener('click', onConfirm);
    const cancel = document.createElement('button');
    cancel.type = 'button';
    cancel.className = 'osprey-btn';
    cancel.textContent = CANCEL_CLEAR_LABEL;
    cancel.addEventListener('click', onCancel);
    host.replaceChildren(text, confirm, cancel);
    return (sentence) => {
      text.textContent = sentence || '';
      host.hidden = !sentence;
    };
  }

  // -------------------------------------------------------------------------
  // Banner
  // -------------------------------------------------------------------------

  /**
   * @param {'ok'|'warn'|'err'|'info'|null} tone Pass `null` to clear.
   * @param {string} [message]
   * @param {boolean} [transient] Leave by itself after `SUCCESS_BANNER_MS`.
   */
  function setBanner(tone, message = '', transient = false) {
    if (bannerTimer !== null) {
      clearTimeout(bannerTimer);
      bannerTimer = null;
    }
    if (tone === null) {
      queueBanner.hidden = true;
      queueBanner.textContent = '';
      return;
    }
    queueBanner.className = `banner banner-${tone}`;
    queueBanner.textContent = message;
    queueBanner.hidden = false;
    if (transient) bannerTimer = setTimeout(() => setBanner(null), SUCCESS_BANNER_MS);
  }

  // -------------------------------------------------------------------------
  // Queue writes
  // -------------------------------------------------------------------------

  /**
   * Issue one queue write and surface its outcome.
   *
   * A refused write is not an exception path — an unarmed deployment refusing
   * a start is the system working — so the bridge's sentence is shown as-is
   * and the panel state is left alone. The next SSE frame is the truth about
   * what actually changed; nothing here optimistically re-renders.
   *
   * @param {string} method
   * @param {string} path
   * @param {unknown} [body]
   * @param {string|((parsed: any) => string)} [okMessage] A function receives
   *   the parsed success body, so a write whose outcome the bridge describes
   *   (the abort) can relay the bridge's own `msg` instead of a local
   *   substitute.
   * @param {'ok'|'warn'} [okTone] Tone for the success banner, decided by the
   *   pure module (`writeOutcomeTone` / `abortOutcomeTone`) and never by the
   *   request itself.
   * @param {(code: string|null) => 'warn'|'err'} [toneForRefusal] Tone for a
   *   REFUSED write, same rule: decided by the pure module. Defaults to
   *   `refusalTone`; the emergency abort passes `abortRefusalTone`, under
   *   which a refusal means the machine was not stopped rather than a policy
   *   said no.
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
      // A receipt, not a state: it leaves by itself. The status chip is the state.
      if (message) setBanner(okTone, message, true);
      return true;
    }

    const refusal = classifyQueueRefusal(response.status, parsed);
    setBanner(toneForRefusal(refusal.code), refusal.message);
    return false;
  }

  // -------------------------------------------------------------------------
  // Rendering: queue
  // -------------------------------------------------------------------------

  function renderQueue() {
    const { queue } = state;
    // The chip is the operator's word for the state; its tooltip adds one
    // plain sentence with the manager's own word, and a click opens the rest
    // (what could not be read, and what each control does) in its popover.
    const chip = queueChip(queue);
    queueChipLabel.textContent = chip.label;
    queueChipEl.className = chip.tone ? `osprey-chip osprey-chip--${chip.tone}` : 'osprey-chip';
    queueChipEl.setAttribute('aria-label', `Queue status: ${chip.label}`);
    writeTitle(queueChipEl, chip.hint);

    const controls = queueControls(queue);
    startBtn.textContent = START_LABEL;
    startBtn.disabled = controls.start.disabled;
    writeTitle(startBtn, controls.start.reason || CONTROL_HINTS.start);

    // The stop button is never disabled — see `queueControls`. Whatever this
    // panel believes about the manager is a tooltip, not a gate.
    stopBtn.textContent = stopButtonLabel(controls.stop, stopConfirmArmed);
    stopBtn.className = stopButtonClass(controls.stop, stopConfirmArmed);
    writeTitle(stopBtn, controls.stop.note || CONTROL_HINTS.stop);
    stopBtn.dataset.cancel = controls.stop.body.cancel ? 'true' : 'false';
    setNote(stopNote, stopConfirmArmed ? controls.stop.note : null);

    // The abort button is never disabled either — see `abortControl`. It
    // carries no `disabled` assignment here for exactly that reason, and its
    // note is shown once the confirm is armed, when the operator is about to
    // commit.
    const abort = abortControl(queue);
    abortBtn.textContent = abortButtonLabel(abortConfirmArmed);
    abortBtn.className = abortButtonClass(abortConfirmArmed);
    writeTitle(abortBtn, abortConfirmArmed ? abort.note || '' : CONTROL_HINTS.abort);
    setNote(abortNote, abortConfirmArmed ? abort.note : null);

    // Clear: a usability gate like Start's (dead with nothing waiting), and
    // two-step like the abort, since the plans it drops are staged work. A
    // dead clear cannot stay armed: its Confirm would send a refused write.
    if (clearControl(queue).disabled) clearConfirmArmed = false;
    showClearNote(clearConfirmArmed ? queueClearConfirmNote(queue.items.length) : null);

    // Which controls are on screen is decided per view (`stripControls`): the
    // Queue view shows all three; the other two hide the halts only while the
    // panel affirmatively knows they are dormant (`haltsPinned`), and every
    // doubtful state pins them on. This is `hidden`, never `disabled` — a
    // visible halt is always live, and the states that permit hiding are a
    // subset of the states that already cleared both confirm-armed flags, so
    // no half-armed confirm can vanish. Nothing about the toolbar's width
    // hides them: they never fold.
    const shown = stripControls(queue, activeView);
    startBtn.hidden = !shown.start;
    stopBtn.hidden = !shown.halts;
    abortBtn.hidden = !shown.halts;
    // Quiet halts while nothing is moving — a visual weight, never a gate:
    // both stay live, and the armed confirm colour still wins.
    stopBtn.classList.toggle('halt-idle', shown.dormant && !stopConfirmArmed);
    abortBtn.classList.toggle('halt-idle', shown.dormant && !abortConfirmArmed);

    renderRunningItem();
    renderPendingItems();
    reportClears();
  }

  /**
   * The two ⋯ Clear entries as they stand: live or dead, and how many rows
   * each would remove, for the hub's menu.
   *
   * @returns {ReturnType<typeof clearMenuEntries>}
   */
  function menuEntries() {
    return clearMenuEntries({
      queue: clearControl(state.queue),
      waiting: state.queue.items.length,
      history: historyClearControl(state.history, state.removals, state.historyLoaded),
      listed: state.history.length + state.removals.length,
    });
  }

  /** Tell the shell when what the ⋯ entries say has changed, and only then. */
  function reportClears() {
    const entries = menuEntries();
    const key = JSON.stringify(entries);
    if (key === reportedEntries) return;
    reportedEntries = key;
    onClearsChange(entries);
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
    // An item enqueued out of band carries no OSPREY run id, so there is no
    // run to open — it is shown, but not selectable.
    if (runId === null) runningSelect.removeAttribute('data-run-id');
    else runningSelect.dataset.runId = runId;
    runningSelect.disabled = runId === null;
    runningCard.classList.toggle('selected', runId !== null && runId === state.selected);

    const progress = describeProgress(item.progress);
    runningProgressLabel.textContent = progress.label;
    // An unknown denominator renders as an indeterminate bar, never as 0% — a
    // plan with no predicted point count (the common case for an
    // agent-authored session plan) is running fine, and showing it at zero
    // reads as stalled.
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
   * announceable for free, and it matches the Plans view's plan rows.
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

  // -------------------------------------------------------------------------
  // Rendering: completed runs
  // -------------------------------------------------------------------------

  function renderHistory() {
    historyList.replaceChildren();
    const empty = historyEmptyState(state.history, state.removals, state.historyLoaded);
    historyEmpty.textContent = empty.message;
    historyEmpty.hidden = empty.hidden;

    // Clear: a usability gate (dead with nothing listed) and two-step, like
    // the queue's. It empties both halves of this card — the manager's whole
    // history and the bridge's record of withdrawn work; run data is kept.
    const clear = historyClearControl(state.history, state.removals, state.historyLoaded);
    if (clear.disabled) historyClearConfirmArmed = false;
    showHistoryClearNote(
      historyClearConfirmArmed
        ? historyClearConfirmNote(state.history.length, state.removals.length)
        : null
    );

    state.history.forEach((record, index) => {
      const row = document.createElement('li');
      row.className = 'queue-row';
      if (record.id === state.selected) row.classList.add('selected');

      const badge = document.createElement('span');
      badge.className = `badge ${badgeToneForStatus(record.status)}`;
      badge.textContent = record.status;

      // One action per finished run: drop it from the list. Single-click, like
      // a pending row's remove — it forgets a record, it moves nothing.
      const actions = document.createElement('span');
      actions.className = 'queue-actions';
      actions.append(
        iconButton('remove-run', '✕', CONTROL_HINTS.removeRun, record.id, index, false)
      );

      row.append(
        badge,
        selectLabel(record.id, record.planName || '(unnamed plan)', record.error || record.id),
        actions
      );
      historyList.appendChild(row);
    });

    state.removals.forEach((record) => historyList.appendChild(removalRow(record)));
    reportClears();
  }

  /**
   * One withdrawn queue row: when it happened, and what somebody did.
   *
   * Deliberately not a run: no status badge, no action buttons, no run id, so
   * there is nothing to select and nothing to open in Results. It is the
   * history list's quiet half, and it reads as one.
   *
   * The withdrawals sit BELOW the completed runs rather than interleaved with
   * them. A run record carries no time — `GET /runs` orders by relevance, not
   * by clock — so there is no key the two halves could be merged on, and each
   * half stays in its own newest-first order under a heading that is the same
   * question: what already happened.
   *
   * @param {{time: string, text: string}} record
   * @returns {HTMLLIElement}
   */
  function removalRow(record) {
    const row = document.createElement('li');
    row.className = 'queue-row removal';

    const time = document.createElement('span');
    time.className = 'queue-time';
    time.textContent = record.time;

    const label = document.createElement('span');
    label.className = 'queue-label';
    const text = document.createElement('span');
    text.className = 'queue-name';
    text.textContent = record.text;
    label.append(text);

    row.append(time, label);
    return row;
  }

  /**
   * Drop one finished run from the history view. The bridge records the
   * removal on its side (the manager cannot drop a single history entry),
   * so the list is re-read here rather than waiting for the next frame — and
   * a run that was on screen in Results is deselected, since its record is
   * gone.
   *
   * @param {string} runId
   * @returns {Promise<boolean>}
   */
  async function removeRun(runId) {
    const ok = await queueWrite(
      'DELETE',
      `/runs/${encodeURIComponent(runId)}`,
      undefined,
      'Run removed from history.',
      writeOutcomeTone(false)
    );
    if (!ok) return false;
    if (state.selected === runId) onSelectRun(null, false);
    await refreshHistory();
    return true;
  }

  /**
   * Re-read the withdrawals listed under the completed runs.
   *
   * Its own fetch, and its own failure: the bridge serves this log without
   * touching the manager, so it answers when `GET /runs` cannot, and a failure
   * on either side must not blank the other. A failed read leaves the last
   * list up — the records are append-only, so what was shown is still true.
   */
  async function refreshRemovals() {
    try {
      const response = await fetch(api('/queue/removals'));
      if (!response.ok) return;
      state.removals = removalRecords(await response.json());
    } catch {
      // Transient: keep what was last read rather than emptying the list.
    }
  }

  async function refreshHistory() {
    await refreshRemovals();
    try {
      const response = await fetch(api('/runs'));
      if (!response.ok) {
        renderHistory();
        return;
      }
      state.history = historyRecords(await response.json());
      state.historyLoaded = true;
    } catch {
      // Leave the last known list up — a transient failure after a good read
      // is still best answered with what was last read. What must NOT happen
      // is the list silently reading as "no completed runs" when it was never
      // loaded, so re-render either way and let `historyEmptyState` say which
      // it is.
      renderHistory();
      return;
    }
    renderHistory();
  }

  // -------------------------------------------------------------------------
  // Wiring
  // -------------------------------------------------------------------------

  /**
   * Row clicks (select a run) and item-action clicks (reorder/remove) share
   * one delegated listener, so rows re-rendered by an SSE frame need no
   * re-binding.
   *
   * @param {Event} event
   * @param {HTMLElement} listRoot
   */
  function onListClick(event, listRoot) {
    const target = event.target;
    if (!(target instanceof Element)) return;

    const button = target.closest('button[data-action]');
    if (button instanceof HTMLButtonElement && listRoot.contains(button)) {
      void onItemAction(button);
      return;
    }

    const row = target.closest('[data-run-id]');
    if (row instanceof HTMLElement && listRoot.contains(row) && row.dataset.runId) {
      onSelectRun(row.dataset.runId, true);
    }
  }

  /** @param {HTMLButtonElement} button */
  async function onItemAction(button) {
    const uid = button.dataset.uid || '';
    const index = Number(button.dataset.index);
    if (uid === '' || Number.isNaN(index)) return;
    const encoded = encodeURIComponent(uid);

    if (button.dataset.action === 'remove-run') {
      // A history row: `uid` is the run id, not a queue item uid.
      await removeRun(uid);
      return;
    }

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
    if (runningSelect.dataset.runId) onSelectRun(runningSelect.dataset.runId, true);
  });

  startBtn.addEventListener('click', () => {
    if (startBtn.disabled) return;
    // Bind the start to the queue that is on screen. The uid is read HERE, at
    // the click, so it is always the last frame's — the stream is the panel's
    // only reader of the queue, and a queue that moved has already delivered
    // the frame carrying its new uid.
    //
    // The bridge compares this uid under its arming lock and refuses
    // `queue_changed_since_approval` when the list moved between the glance
    // and the click. That refusal arrives like every other: `queueWrite` puts
    // the bridge's own sentence in the notice slot, re-renders nothing, and
    // never retries — the panel waits, the stream brings the queue that is
    // actually there, and the operator's next click binds to that one.
    //
    // A summary with no uid (a manager the bridge could not read) sends no
    // expectation at all, which the bridge takes as "start what is there":
    // the panel must not invent a token it never saw.
    const uid = state.queue.status?.plan_queue_uid;
    const body = typeof uid === 'string' && uid !== '' ? { expected_plan_queue_uid: uid } : {};
    void queueWrite(
      'POST',
      '/queue/start',
      body,
      'Queue started — it runs what is queued, and whatever is added.',
      writeOutcomeTone(true)
    );
  });

  // Two-step ONLY on the withdrawal branch, matching the Plans view's
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
  // `abortButtonLabel` for why the two halts differ. The button itself is live
  // in every state; the confirm is friction, not a gate, and no state removes
  // it.
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

  // Two-step, like the abort: the plans this drops are somebody's staged
  // work. Never touches the running item. The ⋯ entry arms it (`armClear`);
  // this Confirm, at the top of the Queued card, sends it.
  const showClearNote = confirmNote(
    clearNote,
    () => {
      if (!clearConfirmArmed) return;
      clearConfirmArmed = false;
      renderQueue();
      void queueWrite(
        'DELETE',
        '/queue/items',
        undefined,
        'Waiting plans removed.',
        writeOutcomeTone(false)
      );
    },
    () => {
      clearConfirmArmed = false;
      renderQueue();
    }
  );

  // Two-step as well: the records it drops are the operator's own log of what
  // ran. The manager forgets them; Tiled keeps every run's data.
  const showHistoryClearNote = confirmNote(
    historyClearNote,
    () => {
      if (!historyClearConfirmArmed) return;
      historyClearConfirmArmed = false;
      renderHistory();
      void (async () => {
        const ok = await queueWrite(
          'DELETE',
          '/history',
          undefined,
          'History cleared.',
          writeOutcomeTone(false)
        );
        if (!ok) return;
        if (state.selected && state.history.some((record) => record.id === state.selected)) {
          onSelectRun(null, false);
        }
        await refreshHistory();
      })();
    },
    () => {
      historyClearConfirmArmed = false;
      renderHistory();
    }
  );

  /**
   * A ⋯ Clear entry was picked: arm that Clear and show its note at the top
   * of the card it empties. Only arms — the note's Confirm is the second
   * step. A dead Clear arms nothing: the menu the click came from may be a
   * frame old.
   *
   * @param {string} entry A `MENU_ENTRY` value.
   */
  function armClear(entry) {
    if (entry === MENU_ENTRY.clearQueue) {
      if (clearControl(state.queue).disabled) return;
      clearConfirmArmed = true;
      renderQueue();
      clearNote.scrollIntoView({ block: 'nearest' });
    } else if (entry === MENU_ENTRY.clearHistory) {
      if (historyClearControl(state.history, state.removals, state.historyLoaded).disabled) return;
      historyClearConfirmArmed = true;
      renderHistory();
      historyClearNote.scrollIntoView({ block: 'nearest' });
    }
  }

  // The chip opens what it summarises: the state in a sentence, what could
  // not be read, and one plain sentence per control.
  queueChipEl.addEventListener('click', () => {
    openInfoPopover(queueChipEl, queueChipDetail(state.queue));
  });

  // -------------------------------------------------------------------------
  // Boot
  // -------------------------------------------------------------------------

  renderQueue();
  renderHistory();
  void refreshHistory();

  const stream = createQueueStream(api('/queue/events'), {
    onFrame(frame) {
      const previousStatus = state.queue.status;
      const previousRunning = state.queue.runningItem;
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
      // And the clear confirm: with nothing waiting there is nothing to clear.
      if (next.items.length === 0) clearConfirmArmed = false;
      renderQueue();
      if (previousStatus !== next.status) onStatus(next.status);
      // The history list is re-fetched on a manager *history* move rather
      // than on a timer: the SSE summary is the change detector, so a settled
      // queue costs no polling at all.
      if (historyChanged(previousStatus, next.status)) void refreshHistory();
      // A run the operator just watched finish must not vanish silently into
      // the Completed-runs card: on a fast machine the whole run fits between
      // two glances, and with nothing selected the Results view would sit on
      // "Select a queued or completed run" while the finished run's row waits
      // one tab away. So when the running item leaves the slot and NOTHING is
      // selected yet the finished run is followed. `activate: false`: this
      // is the panel's own housekeeping, so it fills the Results view without
      // moving anyone off the tab they chose. A manual pick or deep link
      // (state.selected non-null) always wins.
      const finished = finishedRunId(previousRunning, next.runningItem);
      if (finished !== null && state.selected === null) onSelectRun(finished, false);
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
        clearConfirmArmed = false;
      }
      renderQueue();
      // A dropped stream is an unknown queue as far as the Plans view is
      // concerned: its button must not keep promising "Runs now".
      onStatus(connected ? state.queue.status : null);
    },
  });

  return {
    /** @param {string|null} runId */
    setSelected(runId) {
      if (runId === state.selected) return;
      state.selected = runId;
      renderQueue();
      renderHistory();
    },
    /** @param {string} view */
    setView(view) {
      if (view === activeView) return;
      activeView = view;
      renderQueue();
    },
    /**
     * Drop one finished run from the history view — the Results view's
     * "Remove from history" lands here, so the write and its banner live in
     * one place.
     * @param {string} runId
     * @returns {Promise<boolean>}
     */
    removeRun,
    armClear,
    menuEntries,
    reconnect() {
      stream.reconnect();
    },
  };
}
